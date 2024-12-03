#!/bin/bash

seed=42 # 12 # 42

# llava
model="llava"
model_path="/data/ce/model/llava-v1.5-7b"

# instructblip
# model="instructblip"
# model_path=None

coco_path="/data/ce/data/coco"
img_path="${coco_path}/val2014/"
anno_path="${coco_path}/annotations/instances_val2014.json"
log_path="./logs/chair"
out_path="./chair_results/${model}"

use_ritual=False
use_vcd=False
use_m3id=False
use_diffusion=True

degf_alpha_pos=3.0
degf_alpha_neg=1.0
degf_beta=0.1

experiment_index=1

num_eval_samples=500
max_new_tokens=64

#####################################
# Run experiment
#####################################
export CUDA_VISIBLE_DEVICES=1
torchrun --nnodes=1 --nproc_per_node=1 --master_port 2222 eval_bench/chair_eval_${model}.py \
--seed ${seed} \
--model_path ${model_path} \
--model_base ${model} \
--data_path ${img_path} \
--anno_path ${anno_path} \
--log_path ${log_path} \
--out_path ${out_path} \
--use_ritual ${use_ritual} \
--use_vcd ${use_vcd} \
--use_m3id ${use_m3id} \
--use_diffusion ${use_diffusion} \
--degf_alpha_pos ${degf_alpha_pos} \
--degf_alpha_neg ${degf_alpha_neg} \
--degf_beta ${degf_beta} \
--num_eval_samples ${num_eval_samples} \
--max_new_tokens ${max_new_tokens} \
--experiment_index ${experiment_index}

#####################################
# Run evaluation
#####################################
experiment_index=001
cap_json_path="${out_path}/exp_${experiment_index}.jsonl"
echo ${cap_json_path}
python eval_bench/chair.py \
--cap_file ${cap_json_path} \
--coco_path ${coco_path}/annotations \
--save_path ${out_path}/exp_${experiment_index}_result.jsonl \
--image_id_key image_id \
--caption_key caption