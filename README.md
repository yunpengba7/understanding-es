<h1 align="center">
  Understanding Evolution Strategies for LLM Reasoning:<br>
  Broader Reasoning Coverage than GRPO
</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2608.27351">Paper</a> ·
  <a href="https://huggingface.co/papers/2608.27351">Hugging Face</a> ·
  <a href="https://www.alphaxiv.org/pdf/2608.27351">alphaXiv</a> ·
  <a href="https://github.com/yunpengba7/understanding-es">Code</a>
</p>

<p align="center">
  ⭐ If you find our work useful, please consider <a href="https://www.alphaxiv.org/pdf/2608.27351"><strong>liking it on alphaXiv</strong></a>. Thank you for your support!
</p>

<p align="center"><strong>Yunpeng Ba<sup>1,*</sup>, Zhi Zheng<sup>2,*</sup>, Yue Xie<sup>1</sup>, Jiaqing Li<sup>5</sup>, Xialiang Tong<sup>3</sup>, Tao Zhong<sup>3</sup>,<br>Mingxuan Yuan<sup>3</sup>, Zhichao Lu<sup>4</sup>, Xuyang Wu<sup>1</sup>, Zhenkun Wang<sup>1</sup></strong></p>

<p align="center">
  <sup>1</sup>Southern University of Science and Technology,
  <sup>2</sup>National University of Singapore,
  <sup>3</sup>Huawei Noah's Ark Lab,<br>
  <sup>4</sup>City University of Hong Kong,
  <sup>5</sup>Harbin Institute of Technology, Weihai
</p>

<p align="center"><sup>*</sup> Equal contribution.</p>

<p align="center"><strong>Correspondence:</strong> <a href="mailto:zhi.zheng@u.nus.edu">zhi.zheng@u.nus.edu</a>, <a href="mailto:wangzhenkun90@gmail.com">wangzhenkun90@gmail.com</a></p>

## Why Evolution Strategies? 💡

[![Overview of the paper's three research questions: the Pass-at-K advantage of ES, functional sparsity and catastrophic forgetting, and effective ES settings.](assets/readme/rq123_overview.png)](assets/readme/rq123_overview.pdf)

*Figure 1. Overview of the three research questions and main findings. Panel (a) contrasts ES and GRPO post-training behavior; Panel (b) shows that keeping larger ES updates preserves target-task performance and reports held-out Maj@32 changes; Panel (c) summarizes normalization, population-size, and estimator choices. Click the preview to open the PDF version of the figure.*

Here, **Maj@32** denotes majority-vote accuracy over 32 sampled responses.

The paper studies ES through three research questions:

- **RQ1: Does ES exhibit the same post-training characteristics as GRPO?** We find that ES maintains broader reasoning coverage than GRPO. Across models post-trained on GSM8K and DeepScaleR, ES improves Pass@1 while achieving higher Pass@K than GRPO, without exhibiting the same entropy collapse. Theoretically, we show that verifier-projected Jensen–Shannon diversity across the ES population improves repeated-sampling success and can translate into higher Pass@K in the ES-updated policy. We further develop two sequential compositions, ES→GRPO and GRPO→ES, that combine GRPO's strength in Pass@1 with ES's gains in Pass@K.
- **RQ2: Does ES necessarily cause catastrophic forgetting?** By examining the distribution of parameter changes, we find that the task-relevant effects of ES are concentrated in a small subset of larger-magnitude updates, while most parameter changes contribute little after perturbation cancellation. This functional sparsity suggests that substantial whole-model drift need not correspond to widespread functional change. Consistently, held-out capabilities remain largely preserved under appropriate training settings, indicating that large parameter movement alone does not imply catastrophic forgetting and that prior observations are better explained by training-set overfitting.
- **RQ3: What hyperparameter settings and estimators make ES effective and scalable?** We systematically evaluate ES hyperparameters and estimator designs to identify stable and effective configurations. We find that z-score reward normalization is a key ingredient for effective ES training. Due to the discrete reward in reasoning, the two-point estimator commonly favored in zeroth-order SFT provides no advantage for ES. We further find that the population size required for effective optimization decreases as pretrained model scale increases.

### Results at a glance

| Research question | Main finding reported in the paper |
| --- | --- |
| RQ1: ES versus GRPO | ES preserves broader reasoning coverage, improves Pass@1, achieves higher Pass@K than GRPO, and supports complementary sequential compositions |
| RQ2: drift and forgetting | ES gains are concentrated in a sparse subset of larger-magnitude updates; large whole-model drift does not necessarily imply catastrophic forgetting |
| RQ3: ES design choices | Z-score normalization is important, two-point estimation provides no advantage in the matched GSM8K experiment, and the required population decreases with model scale |

The reference endpoint results reported by this package are:

| Model | Greedy | Pass@1 | Pass@16 | Pass@32 |
| --- | ---: | ---: | ---: | ---: |
| `Qwen2.5-1.5B-Instruct` (base checkpoint) | 988 / 1,319 | 0.702710 | 0.948300 | 0.963609 |
| `Qwen2.5-1.5B-Instruct` (ES checkpoint, step 234) | 966 / 1,319 | 0.731994 | 0.948958 | 0.965883 |

The evaluator reports temperature-0 greedy accuracy together with sampled Pass@1, Pass@16, and Pass@32. It retains all 32 sampled responses, their correctness, and the per-question `hits` count in `sampled_samples.jsonl`.

The sampled protocol retains exactly `n = 32` responses per question at temperature 0.6. For a question with `c` correct responses:

- **Pass@1** is `c/n`;
- **Pass@K** uses the standard without-replacement estimator

```math
\mathrm{Pass@K}=1-\frac{\binom{n-c}{K}}{\binom{n}{K}},
```

computed per question and then macro-averaged. Pass@32 is therefore the fraction of questions with at least one correct response among the 32 retained samples.

The paper's statement that ES improves Pass@1 refers to the sampled metric: Pass@1 rises from `0.702710` to `0.731994`. The separate temperature-0 greedy audit declines from 988 to 966 correct, so it must not be substituted for sampled Pass@1. vLLM must return all 32 candidates or evaluation aborts; an unparseable response is retained and scored as incorrect.

## How it works ⚙️

For model parameters θ, the implementation samples seeded perturbations εᵢ, evaluates θ + σεᵢ on a GSM8K batch, normalizes the population rewards, and applies

```math
\theta \leftarrow \theta
+ \frac{\text{learning rate}}{N}
\sum_{i=1}^{N}\hat r_i\epsilon_i.
```

The conventional `1/σ` factor is already absorbed into `training.learning_rate`; the implementation must not divide by σ again.

Each perturbation is represented by a random seed rather than a stored full-model noise tensor. Every vLLM worker regenerates the same per-parameter noise from the seed and a stable tensor identifier. This keeps perturbations replayable and lets eight single-GPU model replicas evaluate the population in parallel without transferring 32 copies of full-parameter noise. “One-point” means the reward is evaluated at θ + σεᵢ; the negative perturbation is used to restore the center, not as a second reward sample.

At each training step:

1. Shuffle GSM8K deterministically and select a batch of up to 64 questions.
2. Generate 32 perturbation seeds and shard them across eight one-GPU vLLM engines.
3. Apply one perturbation, generate greedy answers, average the per-question rewards into its scalar Rᵢ, and revert the perturbation.
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
| Perturbation scale σ | 0.0015 |
| Learning rate | 0.00025 |
| Training | 2 epochs, 234 steps, seed 42 |
| Full-run hardware | 8 visible GPUs, one vLLM engine per GPU |
| Checkpoints | `step_117/`, `step_234/` |
| Training decoding | greedy, up to 2,048 new tokens |
| Sampled evaluation | 32 samples, temperature 0.6 |

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

`Qwen/Qwen2.5-1.5B-Instruct` is downloaded at the pinned revision when its Hugging Face repository ID is passed. A local directory containing the same snapshot can be used instead.

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

[`reference/results.json`](reference/results.json) provides the paper run's model metadata, dataset row counts, greedy results, Pass@1, Pass@16, and Pass@32 as a machine-readable reference. These values are provided for comparison only; the package does not impose an automated pass/fail acceptance criterion on a user's run.

## Outputs

Training writes `run_manifest.json`, `metrics.jsonl`, `tensorboard/`, `step_117/`, and `step_234/` beneath `outputs/two-epochs/`. The checkpoints are merged Hugging Face models for independent evaluation.

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
