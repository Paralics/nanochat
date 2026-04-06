#!/bin/bash
pid=`ps aux | grep base_train | grep -v grep`
kill $pid
