# [ICLR 2025] DeGF

[![Website](https://img.shields.io/badge/Project-Website-green)](https://zhangce01.github.io/DeGF/) [![arXiv](https://img.shields.io/badge/arXiv-2502.06130-red)](http://arxiv.org/abs/2502.06130) [![Conference](https://img.shields.io/badge/ICLR-2025-blue)](https://iclr.cc/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 👀Introduction

This repository contains the code for our ICLR 2025 paper `Self-Correcting Decoding with Generative Feedback for Mitigating Hallucinations in Large Vision-Language Models`. 

![](fig/intro3.png)

![](fig/overview.png)

## 💡Environment

We test our codebase with PyTorch 2.0.1. Please install corresponding PyTorch and CUDA versions according to your computational resources.

```
conda create -n DeGF python=3.10
conda activate DeGF
git clone https://github.com/zhangce01/DeGF.git
cd DeGF
pip install -r requirements.txt
```

Please also download the model checkpoints:

- [**LLaVA-1.5**](https://github.com/haotian-liu/LLaVA): Download [LLaVA-1.5 merged 7B](https://huggingface.co/liuhaotian/llava-v1.5-7b)
- [**InstructBLIP**](https://github.com/salesforce/LAVIS/tree/main/projects/instructblip): Download [InstructBLIP](https://huggingface.co/Salesforce/instructblip-vicuna-7b)

As for the datasets and benchmarks:

- For **MSCOCO** dataset, see [this link](https://cocodataset.org/).
- For **MME**, see [this link](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation).

## 📦Usage

We provide the code for evaluating our DeGF on POPE, CHAIR, and MME-Hallucination benchmark. You can simply run the following code to run the experiments:

- POPE: `bash eval_bench/scripts/pope_eval.sh`
- CHAIR:`bash eval_bench/scripts/chair_eval.sh`
- MME:`bash experiments/cd_scripts/mme_eval.sh`

## 🙏Acknowledgements

Our codebase is adapted from  [RITUAL](https://github.com/sangminwoo/RITUAL), [VCD](https://github.com/DAMO-NLP-SG/VCD), [OPERA](https://github.com/shikiw/OPERA), and [LLaVA](https://github.com/haotian-liu/LLaVA). We thank the authors for releasing their code!

## 📧Contact

If you have any questions, please  contact at [cezhang@cs.cmu.edu](mailto:cezhang@cs.cmu.edu).

## 📌 BibTeX & Citation

If you find this code useful, please consider citing our work:

```bibtex
@inproceedings{zhang2025selfcorrecting,
  title={Self-Correcting Decoding with Generative Feedback for Mitigating Hallucinations in Large Vision-Language Models},
  author={Ce Zhang and Zifu Wan and Zhehan Kan and Martin Q. Ma and Simon Stepputtis and Deva Ramanan and Russ Salakhutdinov and Louis-Philippe Morency and Katia P. Sycara and Yaqi Xie},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=tTBXePRKSx}
}
```

## CASTOR Application (ONR Research)

This repository is also configured for the **CASTOR** maritime disaster classification task: classifying shipwreck images (`aground / capsized / on_fire / sunken`) using LLaVA-1.5-7B and Stable Diffusion v1.5 on the AART Lab `pleiades` SLURM cluster.

### Cluster Setup

```bash
ssh head1.condo.cs.cmu.edu
# Interactive GPU node (RTX6000Ada required)
srun -p pleiades --time=1:00:00 --cpus-per-task=4 --gpus=1 --mem=40G --constraint=RTX6000ADA --pty bash
```

### Running CASTOR Inference

From `~/DeGF/` on the cluster:

```bash
# Full sweep — baseline + DeGF with SD reference, all prompts
bash CASTOR/submit.sh

# Baseline only (no Stable Diffusion)
bash CASTOR/submit.sh --no-diffusion

# DeGF only (with Stable Diffusion reference image)
bash CASTOR/submit.sh --use-diffusion

# With a run tag (appended to output filenames)
bash CASTOR/submit.sh --run-name my_run

# Monitor
squeue -u $USER
tail -f /data/$USER/logs/castor_<ARRAYJOBID>_<TASKID>.out
```

Results land in `/data/$USER/castor_results/answers_{mode}[_{run_name}]_{prompt}_{jobid}.jsonl`. Runs are resumable — resubmitting skips already-written lines.

See `CLAUDE.md` for full architecture details, cluster storage layout, container build process, and the SLURM array task ID mapping.

