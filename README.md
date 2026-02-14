# AI4Math - AIMO Progress Prize 3

Solutions for the [AI Mathematical Olympiad Progress Prize 3](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3) Kaggle competition.

## Overview

This project uses the GPT-OSS-120B model served via vLLM to solve IMO-level math problems. The model runs on an H100 GPU with tool-augmented generation — it can write and execute Python code mid-reasoning using sandboxed Jupyter kernels.

## Notebooks

| Notebook | Description |
|----------|-------------|
| `aimo-3-gpt-oss-120b-with-tools-0265e8.ipynb` | Original source notebook (LB 41) |
| `aimo-3-42-50-stable-lb-gpt-oss-120b.ipynb` | Improved version with adaptive retry + time banking (LB 42-50) |
| `aimo3-gpt-oss-120b-finetuning.ipynb` | Finetuning variant with adjusted prompts and hyperparameters |

## Architecture

1. **vLLM Server** — serves GPT-OSS-120B with FP8 KV cache and prefix caching
2. **Parallel Sampling** — 8 attempts per problem with different seeds
3. **Tool Use** — 16 sandboxed Jupyter kernels for Python code execution (math, numpy, sympy)
4. **Majority Voting** — answers selected by vote count with Python call tiebreaker
5. **Adaptive Retry** — if no consensus after round 1, runs additional attempts with a different prompt
6. **Time Banking** — unused time from easy problems is saved for harder ones

## Key Improvements (42-50 Stable)

- **Adaptive retry**: When fewer than 2 attempts agree, runs 4 more attempts with a verification-focused prompt that includes candidate answers from round 1
- **Time banking**: Easy problems deposit unused time into a bank; hard problems can withdraw up to 300s extra
- **Larger search window** (1024 tokens) for answer detection
- **More generous timeouts** for code execution (10s) and sandbox acquisition (5s)
- **Stream-level error handling** for graceful recovery from API failures

## Requirements

- Kaggle competition environment with H100 GPU
- GPT-OSS-120B model weights
- Python packages: vllm, unsloth, trl, openai_harmony, transformers, openai

## Reference Problems

See `sample_problems.md` for the 10 reference problems with descriptions and results (9/10 correct).

## License

See [LICENSE](LICENSE) for details.
