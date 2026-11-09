# BalCorpus Balochi Assistant — experimental v1

[Try the demo](https://huggingface.co/spaces/shah-bakhsh/balcorpus-balochi-demo) · [Public adapter and model card](https://huggingface.co/shah-bakhsh/balcorpus-qwen3-0.6b-v1)

This is a small **continued-pretraining experiment**, based on Qwen3-0.6B. It is not a native-verified Balochi assistant. Initial generation tests show wrong-script answers and repetition. It needs reviewed Balochi instruction examples and independent native evaluation before reliable use.

The public LoRA was trained on **15,818 word occurrences across 16 training records**, all from sources marked CC BY 4.0. One separate validation record has 1,746 words and 6,301 predicted tokens. Validation perplexity decreased from **67.91 to 65.81**, and reloading the saved adapter reproduced the metrics. This does not establish conversational accuracy or improvement over other Balochi models. Native-reviewed training records: **0**. Verified instruction pairs: **0**.

The larger noncommercial experiment is kept private. Neither its weights nor any raw corpus is in this repository or the public demo. See [source credits](SOURCE_CREDITS.md) for attribution and source terms. Code is MIT; the published public adapter is Apache 2.0 and retains its separate model card and notices.

## Run the public demo locally

Python 3.12 is recommended. The first run downloads the pinned base model (about 1.2 GB) and adapter. FP32 inference needs several GB of available RAM. There is no API key requirement for these public weights.

```sh
python -m venv .venv
# Activate the environment using your operating system's command.
python -m pip install -r requirements.txt
python app.py
```

Local inference defaults to CPU. Set `BAL_DEVICE=cuda` for a suitable CUDA GPU. Hugging Face ZeroGPU initializes CUDA at startup, then assigns a GPU to the decorated inference request. The app limits input to 4,000 characters, model context to 1,024 input tokens, and responses to 16–128 tokens. Only the last three complete conversation turns are retained; older turns are removed to fit. The app does not save chat history to disk. Hugging Face hosting has its own access and usage policies.

`release.json` pins the public adapter to an immutable Hub commit. The loader verifies its weight checksum, file manifest, commercial purpose and pinned base revision before loading it. It never loads custom remote Python code. The private research model cannot be substituted into this public app.

## Training and verification code

`training/` contains the original preparation, LoRA training and evaluation code and configuration used for the experiments, with no raw text or prepared data. Its preparation command expects the separate frozen `BalCorpus-Final` directory next to that training kit; this public code checkout alone does not contain the corpus. The receipt configuration contains hashes and provenance, not training text. Do not replace those receipts with unverified data and assume equivalent results.

The measured commercial run used one TITAN X Pascal GPU, FP32, rank 8 / alpha 16 LoRA on `q_proj` and `v_proj`, learning rate 1e-5, block size 512, batch 1, gradient accumulation 8, and one epoch (15 optimizer steps). The base is `Qwen/Qwen3-0.6B` at `c1899de289a04d12100db370d81485cdf75e47ca`. Test rows were not used for optimization or validation. See `reports/commercial-validation.json` and the model card.

```sh
python -m pip install pytest
python -m pytest tests -q
python -m pytest training/tests -q
```

The tests verify software behavior using small engineering fixtures, not Balochi fluency. `tools/build_release.py` creates an explicit model-only export from the original private result bundle and rejects a public noncommercial export.

## What is still needed

Recruit native reviewers across the intended dialects and scripts; build consented, licensed instruction examples; reserve an independent evaluation set; and compare this checkpoint against its base using fluency, meaning, script following and factuality scores. This v1 is a reproducible starting point, not a claim of a perfect or state-of-the-art Balochi model.

## Release checks

51 software and training-code tests passed. The published adapter was downloaded again and its SHA256 matched the trained file. The live ZeroGPU API returned nonempty responses to two script-specific greeting prompts, and three browser chat turns completed. The Latin-script request produced Arabic script, so script following **failed**. Samples in `reports/live-generation-smoke.json` are synthetic, unreviewed diagnostic outputs and must not be treated as training examples.

Further anonymous inference reached the provider's free GPU runs limit. Open the demo through its Hugging Face Space page and sign in for the quota available to your account; free hosting does not provide unlimited inference. No paid GPU or subscription was selected. Native quality remains unverified.

To deploy the same source to a Gradio Space, copy `app.py`, `inference.py`, `serving.py`, `release.json`, `requirements.txt`, `LICENSE` and `SOURCE_CREDITS.md` to its root, and use `SPACE_README.md` as its `README.md`. The tested deployment uses ZeroGPU and initializes CUDA before entering the GPU worker.
