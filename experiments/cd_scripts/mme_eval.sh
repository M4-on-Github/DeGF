seed=0
dataset_name="mme"
question_file="./experiments/data/MME_Benchmark_release_version/mme_hallucination.jsonl"
image_folder="./experiments/data/MME_Benchmark_release_version"

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
degf_beta=0.5

log_path="./logs/mme/${model}"

python ./experiments/eval/mme_${model}.py \
--seed ${seed} \
--model-path ${model_path} \
--question-file ${question_file} \
--image-folder ${image_folder} \
--answers-file ./experiments/output/${model}_${dataset_name}_answers_seed${seed}.jsonl \
--use_ritual ${use_ritual} \
--use_vcd ${use_vcd} \
--use_m3id ${use_m3id} \
--use_diffusion ${use_diffusion} \
--degf_alpha_pos ${degf_alpha_pos} \
--degf_alpha_neg ${degf_alpha_neg} \
--degf_beta ${degf_beta} \

python ./experiments/eval/convert_answer_to_mme.py \
--output_path ./experiments/output/${model}_${dataset_name}_answers_seed${seed}.jsonl \
--seed ${seed} \
--model ${model} \
--log_path ${log_path}

python ./experiments/eval/eval_mme.py \
--results_dir ${log_path}/mme_answers \
--log_path ${log_path}
