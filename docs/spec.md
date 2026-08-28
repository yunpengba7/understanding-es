# Artifact Specification

## Scope

The artifact reproduces the paper's easy-setting ES experiment with
Qwen2.5-1.5B-Instruct and evaluates the Qwen2.5-1.5B-Instruct base checkpoint
and two-epoch ES checkpoint on GSM8K.
No other optimizer, benchmark, or model family is part of the public contract.

## Training acceptance

- Fixed model revision and fixed GSM8K train snapshot.
- Population 32, batch size 64, two epochs, seed 42.
- Eight independent one-GPU vLLM engines.
- Full-parameter seeded perturbations and decomposed in-place updates.
- Greedy generation with a 2048-token response limit.
- Full runs synchronize engines at the configured periodic interval and before
  exporting merged Hugging Face checkpoints at completed steps 117 and 234.

## Evaluation acceptance

- The Qwen2.5-1.5B-Instruct base checkpoint and two-epoch ES checkpoint are
  separate single-GPU tasks.
- GSM8K uses the included fixed test snapshot.
- Greedy evaluation matches the exact reference correct-question count.
- Sampling retains 32 responses per problem.
- mean@32 matches within absolute error `0.005`.

## Public release requirements

- Author names, affiliations, correspondence information, and repository URL
  match the paper metadata.
- No machine-specific information or private paths.
- No model weights, generated responses, run logs, checkpoints, or actual result
  files.
- Exact dependency lock and CPU-only continuous integration.
- Code and dataset licensing documented separately.
