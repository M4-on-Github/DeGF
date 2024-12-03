seed=2
dataset_name="llava-bench"
question_file="./experiments/llava-bench/questions.jsonl"
image_folder="./experiments/llava-bench/images"

# llava
model="llava"
model_path="//data/ce/model/llava-v1.5-7b"

# instructblip
# model="instructblip"
# model_path=None

gpu=1
export CUDA_VISIBLE_DEVICES=${gpu}

use_ritual=False
use_vcd=False
use_m3id=False
use_diffusion=True

degf_alpha_pos=3.0
degf_alpha_neg=1.0
degf_beta=0.1

log_path="./logs/llavabench/${model}"

python ./experiments/eval/llavabench_${model}.py \
--seed ${seed} \
--model-path ${model_path} \
--question-file ${question_file} \
--image-folder ${image_folder} \
--answers-file ./experiments/output/${model}_${dataset_name}_answers_ours.jsonl \
--use_ritual ${use_ritual} \
--use_vcd ${use_vcd} \
--use_m3id ${use_m3id} \
--use_diffusion ${use_diffusion} \
--degf_alpha_pos ${degf_alpha_pos} \
--degf_alpha_neg ${degf_alpha_neg} \
--degf_beta ${degf_beta} \