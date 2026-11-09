---
title: Balochi Writing Lab
emoji: 📚
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: 6.5.1
python_version: 3.12.12
app_file: app.py
pinned: false
license: mit
short_description: Experimental Balochi model; native review pending.
models:
  - shah-bakhsh/balcorpus-qwen3-0.6b-v1
---

# Balochi Writing Lab

An experimental Balochi writing demo using the public commercial-data LoRA on Qwen3-0.6B. [Model card and measured results](https://huggingface.co/shah-bakhsh/balcorpus-qwen3-0.6b-v1) · [Source code and tests](https://github.com/shah-bakhsh/balcorpus-balochi-assistant).

The model was trained on 15,818 word occurrences of licensed text, with no native-reviewed instruction examples. It can repeat words, choose the wrong script, and give incorrect answers. **Native-speaker quality is unverified.** This is a small continued-pretraining experiment, not a production assistant.

Only the pinned public adapter is loaded. The separate noncommercial research adapter and raw corpus are excluded. See SOURCE_CREDITS.md for source attribution. Code is MIT; the public adapter and Qwen base have their own Apache 2.0 model notices.

The demo uses free ZeroGPU with short, serialized requests. Availability and per-user GPU quotas apply. Chat content is not saved to disk by this application. The Hugging Face platform has its own hosting policies.
