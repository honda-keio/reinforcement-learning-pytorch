#!/bin/bash
for ((i=0;i<5;i++));do
for ((j=1;j<10;j++));do
CUDA_VISIBLE_DEVICES=0 python -u cartpole.py --ep_len 200 --algo ppo --seed $i >> lc-p.out&
CUDA_VISIBLE_DEVICES=1 python -u cartpole.py --ep_len 200 --algo dqn --seed $i >> lc-d.out
done
done