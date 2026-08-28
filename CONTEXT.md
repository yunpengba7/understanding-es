# Public ES Reproduction Package

This context defines the vocabulary used to package and verify the public code release accompanying “Understanding Evolution Strategies for LLM Reasoning: Broader Reasoning Coverage than GRPO.”

## Language

**Public reproduction package**:
A standalone code repository containing the material needed to reproduce the released easy-setting ES training and GSM8K endpoint evaluation claims. Its author metadata matches the paper, while generated model and evaluation artifacts remain external.
_Avoid_: Full research repository, source-repository mirror

**Focused publication history**:
A Git history created specifically for the reproduction package rather than copied from the full research repository. Public authorship and the official remote are expected; private paths and unrelated internal history are excluded.
_Avoid_: Squashed source history, filtered source history

**Greedy-and-mean GSM8K evaluation**:
The evaluation contract reporting exact greedy accuracy together with mean@32 from the declared 32-sample run. Greedy reproduction requires the same correct-question count; mean@32 permits an absolute error of at most `0.005`.
_Avoid_: GSM8K score, multi-metric evaluation

**Compact reference**:
The committed greedy and mean@32 evaluation metrics plus small scorer fixtures used for verification without retaining full historical model generations.
_Avoid_: Reference outputs, archived generations

**Fixed GSM8K snapshot**:
The bundled MIT-licensed GSM8K `main` train and test parquet files used by the artifact, with expected row counts recorded in the evaluation contract.
_Avoid_: Downloaded GSM8K, latest GSM8K

**Paper reproduction run**:
The fixed two-epoch training run using eight single-GPU Ray/vLLM engines.
_Avoid_: Full run, standard run

**Resource smoke test**:
A single-GPU reduced run that verifies installation and execution flow but carries no claim of matching the paper's reward or evaluation values.
_Avoid_: Reproduction run, quick reproduction

**Single-GPU evaluation job**:
One model-and-task evaluation process restricted to one visible GPU. Independent Qwen2.5-1.5B-Instruct base-checkpoint and ES-checkpoint jobs may use any available GPUs concurrently, but no individual evaluation may span multiple GPUs.
_Avoid_: Evaluation batch, multi-GPU evaluation

**Qwen2.5-1.5B-Instruct base checkpoint**:
The unmodified Qwen2.5-1.5B-Instruct model evaluated under the greedy-and-mean GSM8K evaluation contract.
_Avoid_: Initial checkpoint

**Qwen2.5-1.5B-Instruct two-epoch ES checkpoint**:
The Qwen2.5-1.5B-Instruct model produced after 234 ES training steps under the easy-setting protocol.
_Avoid_: Local model, final model

**Merged epoch checkpoint**:
A standalone Hugging Face model exported after step 117 or step 234 for independent evaluation; the artifact does not define a resumable trainer-state checkpoint.
_Avoid_: Raw checkpoint, resume checkpoint

**External model artifact**:
A base model or trained checkpoint supplied by model identifier or caller-provided path and recorded in run provenance, but never stored in the code repository.
_Avoid_: Bundled model, repository checkpoint

**Paper**:
The research paper titled “Understanding Evolution Strategies for LLM Reasoning: Broader Reasoning Coverage than GRPO.”
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
