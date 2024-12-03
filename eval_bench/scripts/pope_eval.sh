#!/bin/bash

seed=42
dataset_name="coco" # coco | aokvqa | gqa
type="random" # random | popular | adversarial

# llava
# model="llava"
# model_path="/data/ce/model/llava-v1.5-7b"

# instructblip
model="instructblip"
model_path=None

pope_path="/data/ce/data/POPE/${dataset_name}/${dataset_name}_pope_${type}.json"
data_path="/data/ce/data/coco/val2014"

# data_path="/data/ce/data/gqa/images"

log_path="./logs"

use_ritual=True
use_vcd=False
use_m3id=False
use_diffusion=False

degf_alpha_pos=3.0
degf_alpha_neg=1.0
degf_beta=0.25

experiment_index=3

#####################################
# Run single experiment
#####################################
export CUDA_VISIBLE_DEVICES=2
torchrun --nnodes=1 --nproc_per_node=1 --master_port 1236 eval_bench/pope_eval_${model}.py \
--seed ${seed} \
--model_path ${model_path} \
--model_base ${model} \
--pope_path ${pope_path} \
--data_path ${data_path} \
--log_path ${log_path} \
--use_ritual ${use_ritual} \
--use_vcd ${use_vcd} \
--use_m3id ${use_m3id} \
--use_diffusion ${use_diffusion} \
--degf_alpha_pos ${degf_alpha_pos} \
--degf_alpha_neg ${degf_alpha_neg} \
--degf_beta ${degf_beta} \
--experiment_index ${experiment_index}
