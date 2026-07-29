# Anonymous ES Reproduction Artifact

This context defines the vocabulary used to package and verify the anonymous code artifact accompanying the AAAI-27 submission.

## Language

**Anonymous artifact**:
A standalone code repository containing only the material needed to reproduce the submission's declared ES training and GSM8K evaluation claims, without author-identifying information.
_Avoid_: Full research repository, source-repository mirror

**Clean publication history**:
A Git history created specifically for the anonymous artifact that contains no commits, authorship metadata, remotes, or internal paths inherited from the research repository.
_Avoid_: Squashed source history, filtered source history

**Greedy-and-mean GSM8K evaluation**:
The evaluation contract reporting exact greedy accuracy together with mean@32 from the declared 32-sample run. Greedy reproduction requires the same correct-question count; mean@32 permits an absolute error of at most `0.005`.
_Avoid_: GSM8K score, multi-metric evaluation

**Compact reference**:
The committed greedy and mean@32 evaluation metrics plus small scorer fixtures used for verification without retaining full historical model generations.
_Avoid_: Reference outputs, archived generations

**Fixed GSM8K snapshot**:
The MIT-licensed GSM8K `main` train and test parquet files whose contents, row order, sizes, and SHA-256 identities are fixed by the artifact.
_Avoid_: Downloaded GSM8K, latest GSM8K

**Paper reproduction run**:
The fixed two-epoch training run using eight single-GPU Ray/vLLM engines.
_Avoid_: Full run, standard run

**Resource smoke test**:
A single-GPU reduced run that verifies installation and execution flow but carries no claim of matching the paper's reward or evaluation values.
_Avoid_: Reproduction run, quick reproduction

**Single-GPU evaluation job**:
One model-and-task evaluation process restricted to one visible GPU. Independent Base and ES jobs may use any available GPUs concurrently, but no individual evaluation may span multiple GPUs.
_Avoid_: Evaluation batch, multi-GPU evaluation

**Base model**:
The unmodified Qwen2.5-1.5B-Instruct model evaluated under the greedy-and-mean GSM8K evaluation contract.
_Avoid_: Initial checkpoint

**Two-epoch ES model**:
The Qwen2.5-1.5B-Instruct model produced after 234 ES training steps under the easy-setting protocol.
_Avoid_: Local model, final model

**Merged epoch checkpoint**:
A standalone Hugging Face model exported after step 117 or step 234 for independent evaluation; the artifact does not define a resumable trainer-state checkpoint.
_Avoid_: Raw checkpoint, resume checkpoint

**External model artifact**:
A base model or trained checkpoint supplied by model identifier or caller-provided path and verified by identity metadata, but never stored in the anonymous code repository.
_Avoid_: Bundled model, repository checkpoint

**Submission paper**:
The anonymous AAAI-27 submission titled “Understanding Evolution Strategies for LLM Reasoning: Broader Reasoning Coverage than GRPO.”
_Avoid_: Project paper, ES paper

**ES reproduction package**:
The standalone implementation containing only the GSM8K-based training, checkpoint export, GSM8K inference, scoring, and verification behavior required by the artifact's declared claims.
_Avoid_: Trimmed research repository, multi-method framework

**Canonical protocol**:
The version-controlled easy-setting configuration whose experimental parameters cannot be overridden through the public command line; callers may supply only artifact locations and device visibility.
_Avoid_: Default configuration, example configuration

**Locked reproduction environment**:
The exact Python dependency set captured by the artifact's project metadata and lockfile, supplemented at runtime with the effective CUDA, driver, and GPU identity.
_Avoid_: Recommended environment, latest dependencies
