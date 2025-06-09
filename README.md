# Model State Arithmetic for Machine Unlearning

[![arXiv](https://img.shields.io/badge/arXiv-TODO-b31b1b.svg)](TODO)

<!-- Add your method’s cover image here -->

![MSA Cover Image](./assets/MSA.png)

This repository contains the official implementation of **Model State Arithmetic (MSA)**, introduced in the paper:

> 📄 *“Model State Arithmetic for Machine Unlearning”*
> Large language models are trained on massive corpora of web data, which may include private data, copyrighted material, factually inaccurate data, or data that degrades model performance. Eliminating the influence of such problematic datapoints through complete retraining—by repeatedly pretraining the model on datasets that exclude these specific instances—is computationally prohibitive. For this reason, unlearning algorithms have emerged that aim to eliminate the influence of particular datapoints, while otherwise preserving the model—at a low computational cost. However, precisely estimating and undoing the influence of individual datapoints has proved to be challenging. In this work, we propose a new algorithm, MSA, for estimating and undoing the influence of datapoints—by leveraging model checkpoints, i.e., artifacts capturing model states at different stages of pretraining. Our experimental results demonstrate that MSA consistently outperforms existing machine unlearning algorithms across multiple benchmarks, models, and evaluation metrics, suggesting that MSA could be an effective approach towards more flexible large language models that are capable of data erasure.




## ⚙️ Setup

Create and activate a dedicated conda environment, then install dependencies:

```bash
conda create -n MSA python=3.11 -c conda-forge
conda activate MSA
python -m pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
MAX_JOBS=64 python -m pip -v install flash-attn --no-build-isolation
```


If you get errors while installing flash-attn, try setting up cuda and gcc modules before its installation. Example:

```
module load gcc/11.2.0
module load cuda/12.4.1
export CUDA_HOME=/opt/common/cuda/cuda-12.4.1
export CUDA_PATH=$CUDA_HOME
export PATH=$CUDA_HOME/bin:$PATH
```

---

## 🚀 Quick Start

### Unlearning with MSA

The MSA approach performs unlearning on a target model $\theta_\mathcal{D}$ that has been trained on the entire corpus, including the forget documents $\mathcal{D}_f$ that are subject to removal—for example, a model trained on the full TOFU[^1] or RESTOR[^2] dataset.

To apply unlearning, we leverage an earlier model checkpoint $\theta_0$, obtained before the model is exposed to the forget documents. This checkpoint is used to compute forget and retain vectors, which are subsequently used to adjust the target model.
Specifically, $\theta_0$ is finetuned on the forget set $\mathcal{D}_f$ to produce $\theta_1$, and the forget vector is defined as:
$$\vec{\theta}_f = \theta_1 - \theta_0.$$

 If a retain set $\mathcal{D}_r$ is available, a retain vector is computed similarly: $\theta_0$ is finetuned on $\mathcal{D}_r$ to yield $\theta_2$, and the retain vector is:

$$\vec{\theta}_r = \theta_2 - \theta_0.$$

Finally, the parameters of the unlearned model are computed by modifying the target model using both directions:

$$\theta_{\text{unlearn}} = \theta_{\mathcal{D}} - \alpha \vec{\theta}_f + \beta \vec{\theta}_r.$$

To support this process, the repository provides two key functions:  
1. finetuning;
2. applying MSA to compute the unlearned model.

---

[^1]: Maini, Pratyush, et al. *"Tofu: A task of fictitious unlearning for LLMs."* arXiv preprint [arXiv:2401.06121](https://arxiv.org/abs/2401.06121), 2024.  
[^2]: Rezaei, Keivan, et al. *"RESTOR: Knowledge Recovery through Machine Unlearning."* arXiv preprint [arXiv:2411.00204](https://arxiv.org/abs/2411.00204), 2024.

#### 1. Finetuning

Code in [`src/finetune.py`](src/finetune.py) supports finetuning a model on a given dataset  
(e.g., `TOFU_QA_forget01`, `TOFU_QA_retain99_ft`, or `TOFU_QA_full`).

Key arguments:

- `model_name`: Name of the model to begin finetuning with (e.g., `Llama-3.2-1B-Instruct`)
- `local_model_path`: If set, loads model weights from this path
- `dataset`: Name of the dataset used for finetuning
- `trainer_config`: Path to the trainer config YAML (e.g., `configs/trainer/trainer_config.yaml`)
- `per_device_train_batch_size`: Batch size per CUDA device

See [`scripts/finetune.sh`](scripts/finetune.sh) for example usage.

---

#### 2. Unlearning with MSA

Code in [`src/tv_unlearn.py`](src/tv_unlearn.py) supports computing the unlearned model using MSA.  
It requires the following arguments:

- `clean_model_name`: Name of $\theta_0$
- `corrupted_model_name`: Name of $\theta_\mathcal{D}$
- `forget_model_name`: Name of $\theta_1$
- `retain_model_name`: Name of $\theta_2$
- `local_forget_model_path`: Path to the saved model $\theta_1$
- `local_retain_model_path`: Path to the saved model $\theta_2$
- `local_corrupted_model_path`: Path to the model $\theta_\mathcal{D}$
- `alpha`: Scalar for forget direction $\vec{\theta}_f$
- `beta`: Scalar for retain direction $\vec{\theta}_r$  
  (If $\beta = 0$, retain model arguments can be omitted)
- `save_path`: Path where the unlearned model parameters will be saved

See [`scripts/tv_unlearn.sh`](scripts/tv_unlearn.sh) for example usage.

### 🤝 Acknowledgements
The code in this repo is inspired by [Open-Unlearning](https://github.com/locuslab/open-unlearning). 

## 📖 Citation

If you find our work useful, please consider citing us via:

```bibtex

```
