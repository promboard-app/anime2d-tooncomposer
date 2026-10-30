"""상류 코드가 잠금 파일 환경에서 모델 없이 불러와지는지(diffsynth · transformers 4.46 · torch 2.8). 그 환경이 아니면 건너뛴다.

PromBoard 와 같은 가상환경(Python 3.12, torch 2.8.0, nodes/requirements-*.txt)에서: `python -m pytest tests`
"""
import importlib
import sys

import pytest

pytest.importorskip("torch")
pytest.importorskip("diffsynth")

MODULES = [
    "nodes.tooncomposer.util.model_util",
    "nodes.tooncomposer.scheduler.flow_match",
    "nodes.tooncomposer.model.text_encoder",
    "nodes.tooncomposer.model.dit",
    "nodes.tooncomposer.model.dera",
    "nodes.tooncomposer.model.vae",
    "nodes.tooncomposer.model.image_encoder",
    "nodes.tooncomposer.model.prompter",
    "nodes.tooncomposer.pipeline.i2v_pipeline",
    "nodes.tooncomposer.tooncomposer",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    importlib.import_module(name)


def test_unused_dependencies_stay_unloaded():
    # 잠금 파일에서 뺀 diffsynth 의존성(controlnet-aux · cupy)과 상류가 쓰던 lightning · gradio 를 부르지 않는다
    for name in MODULES:
        importlib.import_module(name)
    loaded = {m.split(".")[0] for m in sys.modules}
    assert not loaded & {"controlnet_aux", "cupy", "lightning", "gradio", "clip"}


def test_pipeline_builds_without_weights():
    import torch

    from nodes.tooncomposer.pipeline.i2v_pipeline import WanVideoPipeline

    pipe = WanVideoPipeline(device="cpu", torch_dtype=torch.bfloat16)
    assert pipe.dit is None and pipe.scheduler is not None
