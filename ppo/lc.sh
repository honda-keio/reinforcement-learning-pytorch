#!/bin/bash
for ((i=0;i<4;i++));do
for ((j=1;j<100;j++));do
CUDA_VISIBLE_DEVICES=0 python -u cartpole.py --ep_len 200 --seed $i
done
done