#!/bin/bash
for ((j=1;j<10;j++));do
lr=$j"e-4"
CUDA_VISIBLE_DEVICES=0 python -u pong.py --lr $lr
done