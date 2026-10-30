# Third-party notices

## ToonComposer — `nodes/tooncomposer/`

- Source: [TencentARC/ToonComposer](https://github.com/TencentARC/ToonComposer) at commit `53dc3df95eca4f6a2e3e4c9dea0160a339b12f1c` (2025-08-20).
- Copyright (C) 2025 Tencent. MIT License, except the parts based on Wan2.1 (Copyright 2024-2025 The Alibaba Wan Team Authors), which are Apache-2.0. Full text: [`nodes/tooncomposer/LICENSE.txt`](nodes/tooncomposer/LICENSE.txt).
- Copied files: `tooncomposer.py`, `model/`, `pipeline/`, `scheduler/`, `util/model_util.py`. Not copied: `app.py` (Gradio app — its inference path is re-implemented in `nodes/main.py`), `util/stylesheets.py`, `samples/`.
- PromBoard changes (each marked with a `PromBoard:` comment): relative imports so the folder works as a package (`tooncomposer.py`, `model/dit.py`, `model/prompter.py`, `pipeline/i2v_pipeline.py`); `ToonComposer` subclasses `torch.nn.Module` instead of `lightning.LightningModule` (inference only); `torch.load(..., mmap=True)`; the missing-flash-attn warning in `model/dit.py` no longer prints a `pip install` command (PromBoard installs from lock files only; the code already falls back to PyTorch SDPA); new empty `__init__.py` and `util/__init__.py`.

## Model weights (downloaded by PromBoard at setup, not included in this repository or the plugin zip)

- [TencentARC/ToonComposer](https://huggingface.co/TencentARC/ToonComposer) `480p/` · `608p/` — MIT (Tencent).
- [Wan-AI/Wan2.1-I2V-14B-480P](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P) — Apache-2.0 (Alibaba Wan team).

## Python packages (installed by PromBoard from PyPI with hashes, not included)

See `nodes/requirements-win.txt` · `nodes/requirements-linux.txt`. `diffsynth` 1.1.7 is Apache-2.0 ([modelscope/DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)).

## Example pictures — `cards/`

`color-key.jpg` and `line-key.png` were drawn for PromBoard with an image generation model (Codex CLI `image_generation`), as a new character and scene. The ToonComposer sample `samples/2_image1.jpg` was attached to the prompt only as a drawing-style reference (the prompt told the model not to copy its people or layout); `line-key.png` is a line-art version of a second generated color picture. They are not frames from any dataset or from the ToonComposer repository.
