#!/bin/bash
for ((j=1;j<10;j++));do
lr=$j"e-5"
CUDA_VISIBLE_DEVICES=1 python -u pong.py --lr $lr
done