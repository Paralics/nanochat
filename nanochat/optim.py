"""
A nice and efficient mixed AdamW/Muon Combined Optimizer.
Usually the embeddings and scalars go into AdamW, and the matrix parameters go into Muon.
Two versions are provided (MuonAdamW, DistMuonAdamW), for single GPU and distributed.
Adapted from: https://github.com/KellerJordan/modded-nanogpt
Further contributions from @karpathy and @chrisjmccormick.
"""
import torch
import torch.distributed as dist
from torch import Tensor

# -----------------------------------------------------------------------------
"""
Good old AdamW optimizer, fused kernel.
https://arxiv.org/abs/1711.05101
"""
@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_fused(
    p: Tensor,              # (32768, 768) - parameter tensor
    grad: Tensor,           # (32768, 768) - gradient, same shape as p
    exp_avg: Tensor,        # (32768, 768) - first moment, same shape as p
    exp_avg_sq: Tensor,     # (32768, 768) - second moment, same shape as p
    step_t: Tensor,          # () - 0-D CPU tensor, step count
    lr_t: Tensor,           # () - 0-D CPU tensor, learning rate
    beta1_t: Tensor,        # () - 0-D CPU tensor, beta1
    beta2_t: Tensor,        # () - 0-D CPU tensor, beta2
    eps_t: Tensor,          # () - 0-D CPU tensor, epsilon
    wd_t: Tensor,           # () - 0-D CPU tensor, weight decay
) -> None:
    """
    Fused AdamW step: weight_decay -> momentum_update -> bias_correction -> param_update
    All in one compiled graph to eliminate Python overhead between ops.
    The 0-D CPU tensors avoid recompilation when hyperparameter values change.
    """
    # Weight decay (decoupled, applied before the update)
    p.mul_(1 - lr_t * wd_t)
    # Update running averages (lerp_ is cleaner and fuses well)
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    # Bias corrections
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t
    # Compute update and apply
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    p.add_(exp_avg / denom, alpha=-step_size)

# -----------------------------------------------------------------------------
"""
Muon optimizer adapted and simplified from modded-nanogpt.
https://github.com/KellerJordan/modded-nanogpt
"""
polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

@torch.compile(dynamic=False, fullgraph=True)
def muon_step_fused(
    stacked_grads: Tensor,          # (12, 768, 3072) - stacked gradients
    stacked_params: Tensor,         # (12, 768, 3072) - stacked parameters
    momentum_buffer: Tensor,        # (12, 768, 3072) - first moment buffer
    second_momentum_buffer: Tensor, # (12, 768, 1) or (12, 1, 3072) - factored second moment
    momentum_t: Tensor,             # () - 0-D CPU tensor, momentum coefficient
    lr_t: Tensor,                   # () - 0-D CPU tensor, learning rate
    wd_t: Tensor,                   # () - 0-D CPU tensor, weight decay
    beta2_t: Tensor,                # () - 0-D CPU tensor, beta2 for second moment
    ns_steps: int,                  # 5 - number of Newton-Schulz/Polar Express iterations
    red_dim: int,                   # -1 or -2 - reduction dimension for variance
) -> None:
    """Fused Muon step: momentum -> polar_express -> variance_reduction -> cautious_update"""
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)

    X = g.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)
    if g.size(-2) > g.size(-1): # Tall matrix
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else: # Wide matrix (original math)
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    g = X

    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)

    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)

# -----------------------------------------------------------------------------
class MuonAdamW(torch.optim.Optimizer):
    """Combined optimizer: Muon for 2D matrix params, AdamW for others, single GPU version."""
    def __init__(self, param_groups: list[dict]):
        super().__init__(param_groups, defaults={})
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    def _step_adamw(self, group: dict) -> None:
        for p in group['params']:
            if p.grad is None: continue
            grad = p.grad
            state = self.state[p]
            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
            state['step'] += 1
            self._adamw_step_t.fill_(state['step'])
            self._adamw_lr_t.fill_(group['lr'])
            self._adamw_beta1_t.fill_(group['betas'][0])
            self._adamw_beta2_t.fill_(group['betas'][1])
            self._adamw_eps_t.fill_(group['eps'])
            self._adamw_wd_t.fill_(group['weight_decay'])
            adamw_step_fused(p, grad, state['exp_avg'], state['exp_avg_sq'],
                             self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                             self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t)

    def _step_muon(self, group: dict) -> None:
        params = group['params']
        if not params: return
        p = params[0]
        state = self.state[p]
        num_params = len(params)
        shape, device, dtype = p.shape, p.device, p.dtype
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(num_params, *shape, dtype=dtype, device=device)
        momentum_buffer = state["momentum_buffer"]
        if "second_momentum_buffer" not in state:
            state_shape = (num_params, shape[-2], 1) if shape[-2] >= shape[-1] else (num_params, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        second_momentum_buffer = state["second_momentum_buffer"]
        red_dim = -1 if shape[-2] >= shape[-1] else -2

        stacked_grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)

        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
        self._muon_wd_t.fill_(group["weight_decay"])

        muon_step_fused(
            stacked_grads, stacked_params, momentum_buffer, second_momentum_buffer,
            self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t, self._muon_beta2_t,
            group["ns_steps"], red_dim,
        )
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if group['kind'] == 'adamw':
                self._step_adamw(group)
            elif group['kind'] == 'muon':
                self._step_muon(group)
            else:
                raise ValueError(f"Unknown optimizer kind: {group['kind']}")

# -----------------------------------------------------------------------------
class DistMuonAdamW(torch.optim.Optimizer):
    """Combined distributed optimizer: Muon for 2D matrix params, AdamW for others."""
    def __init__(self, param_groups: list[dict]):
        super().__init__(param_groups, defaults={})
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    def _reduce_adamw(self, group: dict, world_size: int) -> dict:
        param_infos = {}
        for p in group['params']:
            if p.grad is None: continue
            grad = p.grad
            if p.numel() < 1024:
                future = dist.all_reduce(grad, op=dist.ReduceOp.AVG, async_op=True).get_future()
                param_infos[p] = dict(future=future, grad_slice=grad, is_small=True)
            else:
                assert grad.shape[0] % world_size == 0, f"AdamW reduce_scatter requires shape[0] ({grad.shape[0]}) divisible by world_size ({world_size})"
                rank_size = grad.shape[0] // world_size
                grad_slice = torch.empty_like(grad[:rank_size])
                future = dist.reduce_scatter_tensor(grad_slice, grad, op=dist.ReduceOp.AVG, async_op=True).get_future()
                param_infos[p] = dict(future=future, grad_slice=grad_slice, is_small=False)
        return dict(param_infos=param_infos)

    def _reduce_muon(self, group: dict, world_size: int) -> dict:
        params = group['params']
        chunk_size = (len(params) + world_size - 1) // world_size
        padded_num_params = chunk_size * world_size
        p = params[0]
        shape, device, dtype = p.shape, p.device, p.dtype
        grad_stack = torch.stack([p.grad for p in params])
        stacked_grads = torch.empty(padded_num_params, *shape, dtype=dtype, device=device)
        stacked_grads[:len(params)].copy_(grad_stack)
        if len(params) < padded_num_params:
            stacked_grads[len(params):].zero_()
        grad_chunk = torch.empty(chunk_size, *shape, dtype=dtype, device=device)
        future = dist.reduce_scatter_tensor(grad_chunk, stacked_grads, op=dist.ReduceOp.AVG, async_op=True).get_future()
        return dict(future=future, grad_chunk=grad_chunk, stacked_grads=stacked_grads, chunk_size=chunk_size)

    def _compute_adamw(self, group: dict, info: dict, gather_list: list, rank: int, world_size: int) -> None:
        param_infos = info['param_infos']
        for p in group['params']:
            if p.grad is None: continue
            pinfo = param_infos[p]
            pinfo['future'].wait()
            grad_slice = pinfo['grad_slice']
            state = self.state[p]
            if pinfo['is_small']:
                p_slice = p
            else:
                rank_size = p.shape[0] // world_size
                p_slice = p[rank * rank_size:(rank + 1) * rank_size]
            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p_slice)
                state['exp_avg_sq'] = torch.zeros_like(p_slice)
            state['step'] += 1
            self._adamw_step_t.fill_(state['step'])
            self._adamw_lr_t.fill_(group['lr'])
            self._adamw_beta1_t.fill_(group['betas'][0])
            self._adamw_beta2_t.fill_(group['betas'][1])
            self._adamw_eps_t.fill_(group['eps'])
            self._adamw_wd_t.fill_(group['weight_decay'])
            adamw_step_fused(p_slice, grad_slice, state['exp_avg'], state['exp_avg_sq'],
                             self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                             self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t)

    def _compute_muon(self, group: dict, info: dict, gather_list: list, rank: int) -> None:
        info['future'].wait()
        params = group['params']
        chunk_size = info['chunk_size']
        grad_chunk = info['grad_chunk']
        p = params[0]
        shape, device, dtype = p.shape, p.device, p.dtype
        start_idx = rank * chunk_size
        num_owned = min(chunk_size, max(0, len(params) - start_idx))
        state = self.state[p]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(chunk_size, *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            state_shape = (chunk_size, shape[-2], 1) if shape[-2] >= shape[-1] else (chunk_size, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2
        updated_params = torch.empty(chunk_size, *shape, dtype=dtype, device=device)
        if num_owned > 0:
            owned_params = [params[start_idx + i] for i in range(num_owned)]
            stacked_owned = torch.stack(owned_params)
            self._muon_momentum_t.fill_(group["momentum"])
            self._muon_beta2_t.fill_(group["beta2"])
            self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
            self._muon_wd_t.fill_(group["weight_decay"])
            muon_step_fused(
                grad_chunk[:num_owned], stacked_owned,
                state["momentum_buffer"][:num_owned], state["second_momentum_buffer"][:num_owned],
                self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t, self._muon_beta2_t,
                group["ns_steps"], red_dim,
            )
            updated_params[:num_owned].copy_(stacked_owned)
        if num_owned < chunk_size:
            updated_params[num_owned:].zero_()
        stacked_params = info["stacked_grads"]
        future = dist.all_gather_into_tensor(stacked_params, updated_params, async_op=True).get_future()
        gather_list.append(dict(future=future, stacked_params=stacked_params, params=params))

    def _finish_gathers(self, gather_list: list) -> None:
        for info in gather_list:
            info["future"].wait()
            if info["params"] is not None:
                torch._foreach_copy_(info["params"], list(info["stacked_params"][:len(info["params"])].unbind(0)))

    @torch.no_grad()
    def step(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        reduce_infos = []
        for group in self.param_groups:
            if group['kind'] == 'adamw':
                reduce_infos.append(self._reduce_adamw(group, world_size))
            elif group['kind'] == 'muon':
                reduce_infos.append(self._reduce_muon(group, world_size))
            else:
                raise ValueError(f"Unknown optimizer kind: {group['kind']}")
        gather_list = []
        for group, info in zip(self.param_groups, reduce_infos):
            if group['kind'] == 'adamw':
                self._compute_adamw(group, info, gather_list, rank, world_size)
            elif group['kind'] == 'muon':
                self._compute_muon(group, info, gather_list, rank)
        self._finish_gathers(gather_list)

# -----------------------------------------------------------------------------
@torch.compile(dynamic=False, fullgraph=True)
def galore_adam_step_fused(
    p: Tensor,          # (m, n) - full parameter
    grad: Tensor,       # (m, n) - gradient
    proj: Tensor,       # (n, r) - low-rank projection matrix
    exp_avg: Tensor,    # (m, r) - first moment (low-rank)
    exp_avg_sq: Tensor, # (m, r) - second moment (low-rank)
    step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t
) -> None:
    """Fused GaLore-Adam step: project -> AdamW(low-rank) -> unproject & apply."""
    # Project gradient to low-rank subspace
    g_low = torch.matmul(grad, proj)
    # Decoupled weight decay on full param
    p.mul_(1 - lr_t * wd_t)
    # AdamW in low-rank space
    exp_avg.lerp_(g_low, 1 - beta1_t)
    exp_avg_sq.lerp_(g_low.square(), 1 - beta2_t)
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    # Unproject update and apply to full parameter
    update_full = torch.matmul(exp_avg / denom, proj.T)
    p.add_(update_full, alpha=-step_size)

class GaLoreAdam(torch.optim.Optimizer):
    """Combined optimizer: GaLore-Adam for 2D matrix params, AdamW for others."""
    def __init__(self, param_groups: list[dict]):
        super().__init__(param_groups, defaults={})
        self._step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    def _step_galore(self, group: dict) -> None:
        rank = group.get('rank', 128)
        update_proj_gap = group.get('update_proj_gap', 200)
        scale = group.get('scale', 0.25)
        betas = group.get('betas', (0.9, 0.999))
 
        for p in group['params']:
            if p.grad is None: continue
            grad = p.grad
            state = self.state[p]

            if not state:
                # FIX: Safely cap rank to matrix dimensions to prevent QR shape collapse
                eff_rank = min(rank, p.shape[0], p.shape[-1])
                state['rank'] = eff_rank
                state['step'] = 0
                # Safe orthogonal initialization
                P = torch.empty(p.shape[-1], eff_rank, dtype=p.dtype, device=p.device)
                torch.nn.init.orthogonal_(P)
                state['proj'] = P
                state['exp_avg'] = torch.zeros(p.shape[0], eff_rank, dtype=p.dtype, device=p.device)
                state['exp_avg_sq'] = torch.zeros(p.shape[0], eff_rank, dtype=p.dtype, device=p.device)
                state['proj_last_step'] = 0

            proj = state['proj']
            exp_avg = state['exp_avg']
            exp_avg_sq = state['exp_avg_sq']
            eff_rank = state['rank']
            state['step'] += 1

            # Update projection subspace periodically via SVD
            if state['step'] % update_proj_gap == 0:
                _, _, Vh = torch.linalg.svd(grad.float(), full_matrices=False)
                proj.data.copy_(Vh.T[:, :eff_rank].to(p.dtype))
                state['proj_last_step'] = state['step']

            # Fused low-rank AdamW step
            self._step_t.fill_(state['step'])
            # FIX: Apply scale to LR, NOT to the parameter tensor
            self._lr_t.fill_(group['lr'] * scale)
            self._beta1_t.fill_(betas[0])
            self._beta2_t.fill_(betas[1])
            self._eps_t.fill_(group.get('eps', 1e-8))
            self._wd_t.fill_(group.get('weight_decay', 0.0))

            galore_adam_step_fused(
                p, grad, proj, exp_avg, exp_avg_sq,
                self._step_t, self._lr_t, self._beta1_t,
                self._beta2_t, self._eps_t, self._wd_t
            )

    def _step_adamw(self, group: dict) -> None:
        for p in group['params']:
            if p.grad is None: continue
            grad = p.grad
            state = self.state[p]
            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
            state['step'] += 1
            self._step_t.fill_(state['step'])
            self._lr_t.fill_(group['lr'])
            self._beta1_t.fill_(group['betas'][0])
            self._beta2_t.fill_(group['betas'][1])
            self._eps_t.fill_(group['eps'])
            self._wd_t.fill_(group['weight_decay'])
            adamw_step_fused(p, grad, state['exp_avg'], state['exp_avg_sq'],
                             self._step_t, self._lr_t, self._beta1_t,
                             self._beta2_t, self._eps_t, self._wd_t)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if group['kind'] == 'galore_adam':
                self._step_galore(group)
            elif group['kind'] == 'adamw':
                self._step_adamw(group)
            else:
                raise ValueError(f"Unknown optimizer kind: {group['kind']}")

class DistGaLoreAdamW(torch.optim.Optimizer):
    """Distributed GaLore-Adam following 3-phase async pattern."""
    def __init__(self, param_groups: list[dict]):
        super().__init__(param_groups, defaults={})
        self._step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    def _reduce_galore(self, group: dict, world_size: int) -> dict:
        params = group['params']
        chunk_size = (len(params) + world_size - 1) // world_size
        padded_num_params = chunk_size * world_size
        p = params[0]
        shape, device, dtype = p.shape, p.device, p.dtype
        grad_stack = torch.stack([p.grad for p in params])
        stacked_grads = torch.empty(padded_num_params, *shape, dtype=dtype, device=device)
        stacked_grads[:len(params)].copy_(grad_stack)
        if len(params) < padded_num_params:
            stacked_grads[len(params):].zero_()
        grad_chunk = torch.empty(chunk_size, *shape, dtype=dtype, device=device)
        future = dist.reduce_scatter_tensor(grad_chunk, stacked_grads, op=dist.ReduceOp.AVG, async_op=True).get_future()
        return dict(future=future, grad_chunk=grad_chunk, stacked_grads=stacked_grads, chunk_size=chunk_size)

    def _compute_galore(self, group: dict, info: dict, gather_list: list, rank: int) -> None:
        info['future'].wait()
        params = group['params']
        chunk_size = info['chunk_size']
        grad_chunk = info['grad_chunk']
        p = params[0]
        shape, device, dtype = p.shape, p.device, p.dtype
        start_idx = rank * chunk_size
        num_owned = min(chunk_size, max(0, len(params) - start_idx))
        updated_params = torch.empty(chunk_size, *shape, dtype=dtype, device=device)

        if num_owned > 0:
            owned_params = [params[start_idx + i] for i in range(num_owned)]
            stacked_owned = torch.stack(owned_params)
            rank_proj = []
            rank_exp_avg = []
            rank_exp_avg_sq = []

            for i, param in enumerate(owned_params):
                state = self.state[param]
                if not state:
                    # FIX: Safely cap rank to matrix dimensions
                    eff_rank = min(group.get('rank', 128), shape[0], shape[-1])
                    state['rank'] = eff_rank
                    state['step'] = 0
                    P = torch.empty(shape[-1], eff_rank, device=device, dtype=dtype)
                    torch.nn.init.orthogonal_(P)
                    state['proj'] = P
                    state['exp_avg'] = torch.zeros(shape[0], eff_rank, dtype=dtype, device=device)
                    state['exp_avg_sq'] = torch.zeros(shape[0], eff_rank, dtype=dtype, device=device)
                    state['proj_last_step'] = 0

                rank_proj.append(state['proj'])
                rank_exp_avg.append(state['exp_avg'])
                rank_exp_avg_sq.append(state['exp_avg_sq'])
                state['step'] += 1

                if state['step'] % group.get('update_proj_gap', 200) == 0:
                    _, _, Vh = torch.linalg.svd(grad_chunk[i].float(), full_matrices=False)
                    state['proj'].data.copy_(Vh.T[:, :state['rank']].to(dtype))

            for i in range(num_owned):
                self._step_t.fill_(self.state[owned_params[i]]['step'])
                # FIX: Scale LR instead of parameters
                self._lr_t.fill_(group['lr'] * group.get('scale', 0.25))
                self._beta1_t.fill_(group['betas'][0])
                self._beta2_t.fill_(group['betas'][1])
                self._eps_t.fill_(group.get('eps', 1e-8))
                self._wd_t.fill_(group.get('weight_decay', 0.0))

                galore_adam_step_fused(
                    stacked_owned[i], grad_chunk[i], rank_proj[i],
                    rank_exp_avg[i], rank_exp_avg_sq[i],
                    self._step_t, self._lr_t, self._beta1_t,
                    self._beta2_t, self._eps_t, self._wd_t
                )
            updated_params[:num_owned].copy_(stacked_owned)

        if num_owned < chunk_size:
            updated_params[num_owned:].zero_()

        stacked_params = info["stacked_grads"]
        future = dist.all_gather_into_tensor(stacked_params, updated_params, async_op=True).get_future()
        gather_list.append(dict(future=future, stacked_params=stacked_params, params=params))

    def _reduce_adamw(self, group: dict, world_size: int) -> dict:
        param_infos = {}
        for p in group['params']:
            if p.grad is None: continue
            grad = p.grad
            if p.numel() < 1024:
                future = dist.all_reduce(grad, op=dist.ReduceOp.AVG, async_op=True).get_future()
                param_infos[p] = dict(future=future, grad_slice=grad, is_small=True)
            else:
                assert grad.shape[0] % world_size == 0, f"AdamW reduce_scatter requires shape[0] divisible by world_size"
                rank_size = grad.shape[0] // world_size
                grad_slice = torch.empty_like(grad[:rank_size])
                future = dist.reduce_scatter_tensor(grad_slice, grad, op=dist.ReduceOp.AVG, async_op=True).get_future()
                param_infos[p] = dict(future=future, grad_slice=grad_slice, is_small=False)
        return dict(param_infos=param_infos)

    def _compute_adamw(self, group: dict, info: dict, rank: int, world_size: int) -> None:
        param_infos = info['param_infos']
        for p in group['params']:
            if p.grad is None: continue
            pinfo = param_infos[p]
            pinfo['future'].wait()
            grad_slice = pinfo['grad_slice']
            state = self.state[p]
            if pinfo['is_small']:
                p_slice = p
            else:
                rank_size = p.shape[0] // world_size
                p_slice = p[rank * rank_size:(rank + 1) * rank_size]
            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p_slice)
                state['exp_avg_sq'] = torch.zeros_like(p_slice)
            state['step'] += 1
            self._step_t.fill_(state['step'])
            self._lr_t.fill_(group['lr'])
            self._beta1_t.fill_(group['betas'][0])
            self._beta2_t.fill_(group['betas'][1])
            self._eps_t.fill_(group['eps'])
            self._wd_t.fill_(group['weight_decay'])
            adamw_step_fused(p_slice, grad_slice, state['exp_avg'], state['exp_avg_sq'],
                             self._step_t, self._lr_t, self._beta1_t,
                             self._beta2_t, self._eps_t, self._wd_t)

    def _finish_gathers(self, gather_list: list) -> None:
        for info in gather_list:
            info["future"].wait()
            if info["params"] is not None:
                torch._foreach_copy_(info["params"], list(info["stacked_params"][:len(info["params"])].unbind(0)))

    @torch.no_grad()
    def step(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        reduce_infos = []
        for group in self.param_groups:
            if group['kind'] == 'galore_adam':
                reduce_infos.append(self._reduce_galore(group, world_size))
            elif group['kind'] == 'adamw':
                reduce_infos.append(self._reduce_adamw(group, world_size))
            else:
                raise ValueError(f"Unknown optimizer kind: {group['kind']}")

        gather_list = []
        for group, info in zip(self.param_groups, reduce_infos):
            if group['kind'] == 'galore_adam':
                self._compute_galore(group, info, gather_list, rank)
            elif group['kind'] == 'adamw':
                self._compute_adamw(group, info, rank, world_size)
        self._finish_gathers(gather_list)
