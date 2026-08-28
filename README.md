<h1 align="center">Understanding Evolution Strategies for LLM Reasoning</h1>

<p align="center"><strong>Broader Reasoning Coverage than GRPO</strong></p>

<p align="center">
  <a href="https://arxiv.org/abs/2608.27351">Paper</a> ·
  <a href="README_zh-CN.md">简体中文</a> ·
  <a href="https://github.com/yunpengba7/understanding-es">Code</a>
</p>

<p align="center"><strong>Yunpeng Ba¹<sup>*</sup> · Zhi Zheng²<sup>*</sup> · Yue Xie¹ · Jiaqing Li⁵ · Xialiang Tong³ · Tao Zhong³ · Mingxuan Yuan³ · Zhichao Lu⁴ · Xuyang Wu¹ · Zhenkun Wang¹</strong></p>

<p align="center">
  ¹ Southern University of Science and Technology<br>
  ² National University of Singapore<br>
  ³ Huawei Noah's Ark Lab<br>
  ⁴ City University of Hong Kong<br>
  ⁵ Harbin Institute of Technology, Weihai
</p>

<p align="center"><sup>*</sup> Equal contribution.</p>

<p align="center"><strong>Correspondence:</strong> <a href="mailto:zhi.zheng@u.nus.edu">zhi.zheng@u.nus.edu</a> · <a href="mailto:wangzhenkun90@gmail.com">wangzhenkun90@gmail.com</a></p>

## Abstract

Evolution Strategies (ES) have recently emerged as a memory-efficient post-training paradigm for LLM reasoning. However, the optimization behavior of ES remains understudied, making it hard to define its advantage scope compared to mainstream post-training paradigms (e.g., Group Relative Policy Optimization (GRPO)). By systematically investigating ES dynamics and mechanisms, this paper **first identifies a performance advantage of ES over GRPO**, theoretically and empirically showing that ES can lead to broader reasoning coverage, thereby better exploiting the reasoning capabilities of pretrained LLMs. Theoretically, we show that verifier-projected Jensen–Shannon diversity across the ES population is helpful to higher Pass@$K$ performances. Empirically, unlike GRPO, which exhibits entropy collapse, ES improves Pass@1 while attaining higher Pass@$K$ than GRPO. We further develop a sequential GRPO–ES training strategy that combines GRPO's strength in Pass@1 with ES's gains in Pass@$K$. **Second,** we find that despite substantial whole-model parameter drift, the task-performance gains of ES are only contributed to a sparse subset of larger-magnitude updates. This functional sparsity suggests that large parameter movement need not imply widespread functional change, and held-out evaluations further show that it does not necessarily lead to catastrophic forgetting. **Finally,** we study how hyperparameter design affects the effectiveness of ES, demonstrating that ES requires a smaller population size in a larger LLM. These findings position ES as a distinct reasoning post-training paradigm rather than a less effective, memory-efficient alternative to GRPO.

## Why Evolution Strategies? 💡

[![Overview of the paper's three research questions: the Pass-at-K advantage of ES, functional sparsity and catastrophic forgetting, and effective ES settings.](assets/readme/rq123_overview.png)](assets/readme/rq123_overview.pdf)

*Figure 1. Overview of the three research questions and main findings. Panel (a) contrasts ES and GRPO post-training behavior; Panel (b) shows that keeping larger ES updates preserves target-task performance and reports held-out Maj@32 changes; Panel (c) summarizes normalization, population-size, and estimator choices. Click the preview to open the PDF version of the figure.*

Here, **Maj@32** denotes majority-vote accuracy over 32 sampled responses.

The paper studies ES through three research questions:

- **RQ1: Does ES exhibit the same post-training characteristics as GRPO?** We find that ES maintains broader reasoning coverage than GRPO. Across models post-trained on GSM8K and DeepScaleR, ES improves Pass@1 while achieving higher Pass@$K$ than GRPO, without exhibiting the same entropy collapse. Theoretically, we show that verifier-projected Jensen–Shannon diversity across the ES population improves repeated-sampling success and can translate into higher Pass@$K$ in the ES-updated policy. We further develop two sequential compositions, ES$\rightarrow$GRPO and GRPO$\rightarrow$ES, that combine GRPO's strength in Pass@1 with ES's gains in Pass@$K$.
- **RQ2: Does ES necessarily cause catastrophic forgetting?** By examining the distribution of parameter changes, we find that the task-relevant effects of ES are concentrated in a small subset of larger-magnitude updates, while most parameter changes contribute little after perturbation cancellation. This functional sparsity suggests that substantial whole-model drift need not correspond to widespread functional change. Consistently, held-out capabilities remain largely preserved under appropriate training settings, indicating that large parameter movement alone does not imply catastrophic forgetting and that prior observations are better explained by training-set overfitting.
- **RQ3: What hyperparameter settings and estimators make ES effective and scalable?** We systematically evaluate ES hyperparameters and estimator designs to identify stable and effective configurations. We find that z-score reward normalization is a key ingredient for effective ES training. Due to the discrete reward in reasoning, the two-point estimator commonly favored in zeroth-order SFT provides no advantage for ES. We further find that the population size required for effective optimization decreases as pretrained model scale increases.

### Results at a glance

| Research question | Main finding reported in the paper | Evidence in this repository |
| --- | --- | --- |
| RQ1: ES versus GRPO | ES preserves broader reasoning coverage, improves Pass@1, achieves higher Pass@$K$ than GRPO, and supports complementary sequential compositions | `Qwen2.5-1.5B-Instruct` Easy Setting base/ES endpoint only; GRPO, sequential compositions, and the full Pass@$K$ suite are not included |
| RQ2: drift and forgetting | ES gains are concentrated in a sparse subset of larger-magnitude updates; large whole-model drift does not necessarily imply catastrophic forgetting | Checkpoint export and recorded run provenance only; update-sparsity and held-out evaluations are not included |
| RQ3: ES design choices | Z-score normalization is important, two-point estimation provides no advantage in the matched GSM8K experiment, and the required population decreases with model scale | The selected z-score, population-32, one-point configuration only; the full hyperparameter and estimator ablations are not included |

The reference endpoint results reported by this package are:

| Model | Greedy | Sampled Pass@1 (`mean@32`) |
| --- | ---: | ---: |
| `Qwen2.5-1.5B-Instruct` (base checkpoint) | 988 / 1,319 | 0.702710 |
| `Qwen2.5-1.5B-Instruct` (ES checkpoint, step 234) | 966 / 1,319 | 0.731994 |

The evaluator directly reports two endpoint metrics: temperature-0 greedy accuracy (together with its correct and total counts) and sampled Pass@1 (`mean@32`). It retains the correctness of all 32 sampled responses per question in `sampled_samples.jsonl`, but it does not directly report Pass@32 or Maj@32.

The sampled protocol retains exactly $n=32$ responses per question at temperature 0.6. For a question with $c$ correct responses:

- sampled **Pass@1** is $c/n$; averaging it over questions is exactly the `mean@32` value reported by this package;
- **Pass@32** is 1 if at least one of the 32 responses is correct and 0 otherwise, averaged over questions. More generally, the paper uses the standard without-replacement Pass@$K$ estimator.

The paper's statement that ES improves Pass@1 refers to the sampled metric: `mean@32` rises from `0.702710` to `0.731994`. The separate temperature-0 greedy audit declines from 988 to 966 correct, so it must not be substituted for sampled Pass@1. vLLM must return all 32 candidates or evaluation aborts; an unparseable response is retained and scored as incorrect.

## How it works ⚙️

For model parameters $\theta$, the implementation samples seeded perturbations $\epsilon_i$, evaluates $\theta + \sigma\epsilon_i$ on a GSM8K batch, normalizes the population rewards, and applies

$$
\theta \leftarrow \theta
+ \frac{\texttt{learning\_rate}}{N}
\sum_{i=1}^{N}\hat r_i\epsilon_i.
$$

The conventional $1/\sigma$ factor is already absorbed into `training.learning_rate`; the implementation must not divide by $\sigma$ again.

Each perturbation is represented by a random seed rather than a stored full-model noise tensor. Every vLLM worker regenerates the same per-parameter noise from the seed and a stable tensor identifier. This keeps perturbations replayable and lets eight single-GPU model replicas evaluate the population in parallel without transferring 32 copies of full-parameter noise. “One-point” means the reward is evaluated at $\theta+\sigma\epsilon_i$ only; the negative perturbation is used to restore the center, not as a second reward sample.

At each training step:

1. Shuffle GSM8K deterministically and select a batch of up to 64 questions.
2. Generate 32 perturbation seeds and shard them across eight one-GPU vLLM engines.
3. Apply one perturbation, generate greedy answers, average the per-question rewards into its scalar $R_i$, and revert the perturbation.
4. Z-score the 32 scalar rewards and replay their noise as one reward-weighted parameter update. If every reward is identical, all normalized weights are zero and that step makes no parameter update.
5. Apply the same update locally on every replica at each step; at steps 117 and 234, synchronize replicas from rank 0 and export merged Hugging Face checkpoints.

Training rewards are `1.0` for a correct boxed answer, `0.1` for an incorrect boxed answer, and `0.0` when the required `\boxed{}` format is missing.

### Canonical protocol

The public command surface fixes the paper protocol in [`configs/easy_qwen25_1p5b.yaml`](configs/easy_qwen25_1p5b.yaml).

| Item | Canonical value |
| --- | ---: |
| Base model | `Qwen/Qwen2.5-1.5B-Instruct` |
| GSM8K train/test rows | 7,473 / 1,319 |
| Population / batch size | 32 / 64 |
| Perturbation scale $\sigma$ | 0.0015 |
| Learning rate | 0.00025 |
| Training | 2 epochs, 234 steps, seed 42 |
| Full-run hardware | 8 visible GPUs, one vLLM engine per GPU |
| Checkpoints | `step_117/`, `step_234/` |
| Training decoding | greedy, up to 2,048 new tokens |
| Sampled evaluation | 32 samples, temperature 0.6 |

## What is included 🧭

This repository releases the focused `Qwen2.5-1.5B-Instruct` member of the paper's Easy Setting. The full paper studies GSM8K Easy Setting experiments with `Qwen2.5-1.5B-Instruct`, `Llama-3.2-3B-Instruct`, and `Qwen2.5-7B-Instruct`; a DeepScaleR Hard Setting experiment with `DeepSeek-R1-Distill-Qwen-1.5B`; and population-size experiments with `Qwen2.5-0.5B-Instruct`, `Qwen2.5-1.5B-Instruct`, and `Qwen2.5-3B-Instruct`. These experiments include GRPO comparisons, sequential compositions, held-out evaluations, update sparsity, and ES design ablations.

| Workflow | Included |
| --- | :---: |
| Eight-GPU, two-epoch `Qwen2.5-1.5B-Instruct` ES training | Yes |
| One-GPU resource smoke test | Yes |
| `Qwen2.5-1.5B-Instruct` base-checkpoint GSM8K evaluation | Yes |
| `Qwen2.5-1.5B-Instruct` ES step-234-checkpoint GSM8K evaluation | Yes |
| Machine-readable reference endpoint results | Yes |
| GRPO training and evaluation | No |
| Sequential ES$\rightarrow$GRPO and GRPO$\rightarrow$ES training | No |
| Cross-task forgetting experiments | No |
| Update-sparsity experiments | No |
| Hyperparameter and estimator ablations | No |

## Repository map 📦

```text
src/es_reproduction/          ES training, evaluation, scoring, and release audit
configs/                      fixed paper reproduction protocol
data/gsm8k/                   exact MIT-licensed GSM8K snapshots
reference/results.json
                              model metadata, dataset row counts, and reference endpoint results
scripts/                      training, evaluation, and free-GPU scheduling commands
tests/                        CPU tests for configuration, rewards, ES updates, and evaluation
assets/readme/                paper overview figure in PDF and PNG formats
```

## Requirements 🧰

- Linux with NVIDIA GPUs.
- Python 3.13 and [`uv`](https://docs.astral.sh/uv/).
- A CUDA-capable NVIDIA driver compatible with the pinned PyTorch/vLLM stack.
- Network access and permission to download the Hugging Face `Qwen/Qwen2.5-1.5B-Instruct` model, unless the exact snapshot is already available locally.

The reference experiments used NVIDIA A100-SXM4-80GB GPUs. Full training requires exactly eight homogeneous GPUs on one host; smoke testing and each evaluation task require exactly one visible GPU. The release does not claim a lower VRAM bound, mixed-GPU compatibility, or a fixed wall-clock/runtime profile, so use the smoke test to validate another machine before committing to a full run.

## Quick start ⚡

```bash
git clone https://github.com/yunpengba7/understanding-es.git
cd understanding-es
uv sync --extra dev --locked
uv run pytest -q
```

`Qwen/Qwen2.5-1.5B-Instruct` is downloaded at the pinned revision when its Hugging Face repository ID is passed. A local directory containing the same snapshot can be used instead. Model weights are never stored in this repository.

Run the inexpensive one-GPU resource check:

```bash
CUDA_VISIBLE_DEVICES=0 \
  bash scripts/smoke_test.sh \
  Qwen/Qwen2.5-1.5B-Instruct \
  outputs/smoke
```

The smoke test uses two perturbations, two questions, one update, and at most 128 generated tokens. It verifies the execution path but makes no numerical reproduction claim.

## Reproduce the workflow 🧪

### Two-epoch `Qwen2.5-1.5B-Instruct` ES training

Expose exactly eight GPUs and run:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  bash scripts/train_two_epochs.sh \
  Qwen/Qwen2.5-1.5B-Instruct \
  outputs/two-epochs
```

### Evaluate the `Qwen2.5-1.5B-Instruct` base and ES checkpoints

Each evaluation task must see exactly one GPU. Run the tasks directly:

```bash
CUDA_VISIBLE_DEVICES=0 \
  bash scripts/evaluate_gsm8k.sh \
  Qwen/Qwen2.5-1.5B-Instruct \
  base \
  outputs/evaluations/base

CUDA_VISIBLE_DEVICES=0 \
  bash scripts/evaluate_gsm8k.sh \
  outputs/two-epochs/step_234 \
  es_step_234 \
  outputs/evaluations/es_step_234
```

The machine-readable label must be either `base` for the `Qwen2.5-1.5B-Instruct` base checkpoint or `es_step_234` for the `Qwen2.5-1.5B-Instruct` ES checkpoint at step 234. It identifies the evaluated checkpoint and does not change decoding parameters.

Alternatively, schedule the independent `Qwen2.5-1.5B-Instruct` base-checkpoint and ES step-234-checkpoint jobs on currently idle GPUs:

```bash
uv run python scripts/run_two_evaluations.py \
  --base-model Qwen/Qwen2.5-1.5B-Instruct \
  --es-model outputs/two-epochs/step_234 \
  --output-root outputs/evaluations
```

The scheduler runs both tasks concurrently when two GPUs are idle and sequentially when only one is available.

### Reference results

[`reference/results.json`](reference/results.json) provides the paper run's model metadata, dataset row counts, greedy results, and sampled Pass@1 (`mean@32`) as a machine-readable reference. These values are provided for comparison only; the package does not impose an automated pass/fail acceptance criterion on a user's run.

## Outputs

Training writes `run_manifest.json`, `metrics.jsonl`, `tensorboard/`, `step_117/`, and `step_234/` beneath `outputs/two-epochs/`. The checkpoints are merged Hugging Face models for independent evaluation; this package does not produce resumable trainer or optimizer state.

The two-task scheduler writes `outputs/evaluations/base/result.json` and `outputs/evaluations/es_step_234/result.json`. Each evaluation directory also contains `greedy_samples.jsonl` and `sampled_samples.jsonl`.

Each result records dataset row counts, model metadata, software versions, and the visible GPU. Users may compare the reported metrics with [`reference/results.json`](reference/results.json); no automated result acceptance is required. Training records the resolved model directory name together with the canonical base-model metadata in `run_manifest.json`, but it does not reject a caller-supplied local snapshot before the run.

The entire `outputs/` tree is ignored by Git.

## Checks ✅

Fast checks do not require a model server or GPU:

```bash
uv run pytest -q
uv run ruff check .
uv run es-audit .
```

`es-audit` checks release hygiene, including accidental local paths and tracked model weights, generated responses, results, or logs. Author metadata and the Git remote are expected and allowed.

## Citation

```bibtex
@misc{ba2026understandinges,
  title  = {Understanding Evolution Strategies for LLM Reasoning: Broader Reasoning Coverage than GRPO},
  author = {Ba, Yunpeng and Zheng, Zhi and Xie, Yue and Li, Jiaqing and Tong, Xialiang and Zhong, Tao and Yuan, Mingxuan and Lu, Zhichao and Wu, Xuyang and Wang, Zhenkun},
  year   = {2026},
  eprint = {2608.27351},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG},
  url    = {https://arxiv.org/abs/2608.27351}
}
```

## License

The implementation is released under the [MIT License](LICENSE). The included GSM8K snapshots retain their upstream MIT license; external model artifacts remain governed by their respective providers' terms.
