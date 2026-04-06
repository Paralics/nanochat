#!/bin/bash

# Shared defaults. You can override any value before sourcing this file.
export HF_HOME="${HF_HOME:-/var/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
HF_MODULES_CACHE_DEFAULT="${HF_MODULES_CACHE:-$HF_HOME/modules}"
HF_MODULES_CACHE_FALLBACK="${XDG_CACHE_HOME:-$HOME/.cache}/huggingface/modules"
HF_MODULES_CACHE="$HF_MODULES_CACHE_DEFAULT"
HF_MODULES_LOCK_PROBE_REL="datasets_modules/datasets/openwebtext.lock"

# Keep heavyweight dataset cache shared, but avoid shared lock-file failures for modules.
if ! mkdir -p "$HF_MODULES_CACHE" 2>/dev/null; then
  HF_MODULES_CACHE="$HF_MODULES_CACHE_FALLBACK"
elif ! touch "$HF_MODULES_CACHE/.hf_modules_write_test.$$" 2>/dev/null; then
  HF_MODULES_CACHE="$HF_MODULES_CACHE_FALLBACK"
elif ! mkdir -p "$(dirname "$HF_MODULES_CACHE/$HF_MODULES_LOCK_PROBE_REL")" 2>/dev/null; then
  HF_MODULES_CACHE="$HF_MODULES_CACHE_FALLBACK"
elif ! touch "$HF_MODULES_CACHE/$HF_MODULES_LOCK_PROBE_REL" 2>/dev/null; then
  HF_MODULES_CACHE="$HF_MODULES_CACHE_FALLBACK"
else
  rm -f "$HF_MODULES_CACHE/.hf_modules_write_test.$$"
fi

if [[ "$HF_MODULES_CACHE" == "$HF_MODULES_CACHE_FALLBACK" ]]; then
  mkdir -p "$(dirname "$HF_MODULES_CACHE/$HF_MODULES_LOCK_PROBE_REL")"
  touch "$HF_MODULES_CACHE/$HF_MODULES_LOCK_PROBE_REL" 2>/dev/null || true
fi

if [[ -f "$HF_MODULES_CACHE/.hf_modules_write_test.$$" ]]; then
  rm -f "$HF_MODULES_CACHE/.hf_modules_write_test.$$"
fi

export HF_MODULES_CACHE
export HF_DATASETS_TRUST_REMOTE_CODE="${HF_DATASETS_TRUST_REMOTE_CODE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export MASTER_ADDR="${MASTER_ADDR:-192.168.3.130}"

host="$(hostname -s)"

case "$host" in
  igor-tower)
    export MACHINE_RANK="${MACHINE_RANK:-0}"
    export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-enp10s0}"
    ;;
  AMD-7700)
    export MACHINE_RANK="${MACHINE_RANK:-1}"
    export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-enp2s0}"
    ;;
  *)
    # Unknown host: require manual override to avoid accidental wrong rank/interface.
    : "${MACHINE_RANK:?Set MACHINE_RANK for host '$host'}"
    : "${NCCL_SOCKET_IFNAME:?Set NCCL_SOCKET_IFNAME for host '$host'}"
    ;;
esac
