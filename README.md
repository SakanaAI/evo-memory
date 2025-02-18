<h1 align="center">
  <a href="https://github.com/SakanaAI/universal-transformer-memory/">
<img src="figures/logo.png" width="300" /></a><br>
<b>An Evolved Universal Transformer Memory</b><br>
</h1>

<p align="center">
  📄 <a href="http://arxiv.org/abs/2410.13166">[Paper]</a> |
  🤗 <a href="https://huggingface.co/SakanaAI">[Hugging Face]</a>
  📁 <a href="https://huggingface.co/datasets/SakanaAI/ChouBun">[Dataset]</a>
</p>

## Installation

We provide means to install this repository with [conda](https://docs.conda.io/projects/conda/en/latest/index.html):

For the full set of dependencies with fixed versions (provided to ensure some level of long-term reproducibility):

```bash
conda env create --file=env.yaml
```

For a more minimal and less constrained set of dependencies (for future development/extensions):

```bash
conda env create --file=env_minimal.yaml
```

## Usage

### Training

Training following the incremental setup described in our work can be replicated via the following [hydra](https://hydra.cc/) commands:

stage 1 training:
```bash
torchrun --standalone --nproc_per_node=$NUM_OF_GPUs main.py run@_global_=namm_bam_i1.yaml
```

stage 2 training:
```bash
torchrun --standalone --nproc_per_node=$NUM_OF_GPUs main.py run@_global_=namm_bam_i2.yaml init_from='path/to/stage1/results/ckpt.pt'
```

stage 3 training:
```bash
torchrun --standalone --nproc_per_node=$NUM_OF_GPUs main.py run@_global_=namm_bam_i3.yaml init_from='path/to/stage2/results/ckpt.pt'
```

### Evaluation

Evaluating trained NAMMs on the full set of LongBench tasks can be replicated for both NAMMs with the following command:

```bash
torchrun --standalone --nproc_per_node=$NUM_OF_GPUs main.py run@_global_=namm_bam_eval.yaml init_from='path/to/results/ckpt.pt'
```

Evaluating trained NAMMs on the full set of ChouBun tasks can be replicated with the following command:

```bash
torchrun --standalone --nproc_per_node=$NUM_OF_GPUs main.py run@_global_=namm_bam_eval_choubun.yaml init_from='path/to/results/ckpt.pt'
```

### Additional notes

Using [wandb](https://wandb.ai/) to log the results (through the hydra setting wandb_log=true) requires authenticating to the wandb server via the following command:

```bash
wandb login
```

and using your account's API key (which you should be able to find [here](https://wandb.ai/authorize))

### Gated models (e.g., Llama)

Using gated models requires authenticating to the hugging face hub by running:

```bash
huggingface-cli login
```

and using your account's access tokens (which you should be able to find [here](https://huggingface.co/settings/tokens))


## Bibtex

To cite our work, you can use the following:

```
@article{sakana2024memory,
title={An Evolved Universal Transformer Memory}, 
       author={Edoardo Cetin and Qi Sun and Tianyu Zhao and Yujin Tang},
       year={2024},
       eprint={2410.13166},
       archivePrefix={arXiv},
       primaryClass={cs.LG},
       url={https://arxiv.org/abs/2410.13166},
}
```

