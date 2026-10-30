"""노드가 모델을 불러오기 전에 멈추는 길(RAM)과 메모리 부족 뒤 정리 — 가짜 torch · psutil 로 본다(그래픽카드 · 모델 없이)."""
import contextlib
import gc
import sys
import types
import weakref

import pytest

import nodes.main as nm

INPUTS = {"color": ["c.png"], "line": ["l.png"], "prompt": "A girl jumps. Anime."}
SETTINGS = {"length": "61", "resolution": "480p", "color-at": "", "line-at": "", "guidance": 7.5, "sketch-strength": 1.0, "steps": 15, "seed": 1}
GiB = 2**30


class Ctx:
    device = "cuda"

    def progress(self, *a, **k):
        pass

    def check_cancel(self):
        pass

    def log(self, *a, **k):
        pass


class FakeOOM(RuntimeError):
    pass


def fake_torch(vram_gib):
    torch = types.ModuleType("torch")
    torch.OutOfMemoryError = FakeOOM
    torch.bfloat16 = "bf16"
    torch.no_grad = contextlib.nullcontext
    torch.amp = types.SimpleNamespace(autocast=lambda **k: contextlib.nullcontext())
    torch.cuda = types.SimpleNamespace(
        get_device_properties=lambda i: types.SimpleNamespace(total_memory=vram_gib * GiB),
        is_available=lambda: False,
        empty_cache=lambda: None,
    )
    return torch


@pytest.fixture
def fakes(monkeypatch):
    def install(vram_gib=80, ram_gib=128):
        monkeypatch.setitem(sys.modules, "torch", fake_torch(vram_gib))
        psutil = types.ModuleType("psutil")
        psutil.virtual_memory = lambda: types.SimpleNamespace(total=ram_gib * GiB)
        monkeypatch.setitem(sys.modules, "psutil", psutil)

    monkeypatch.setattr(nm, "_CACHE", {"model": None, "ckpt": None})
    return install


def test_small_ram_stops_before_loading(fakes, monkeypatch):
    fakes(ram_gib=64)
    monkeypatch.setattr(nm, "_load", lambda *a: pytest.fail("모델을 불러오면 안 된다"))
    with pytest.raises(ValueError) as err:
        nm.compose(Ctx(), INPUTS, SETTINGS)
    msg = str(err.value)
    assert "RAM 은 64GB" in msg and "76GB" in msg and "이슈 #6" in msg


def test_small_gpu_stops_before_ram_check(fakes, monkeypatch):
    fakes(vram_gib=12, ram_gib=64)
    monkeypatch.setattr(nm, "_load", lambda *a: pytest.fail("모델을 불러오면 안 된다"))
    with pytest.raises(ValueError, match="그래픽카드 메모리는 12GB"):
        nm.compose(Ctx(), INPUTS, SETTINGS)


def test_out_of_memory_releases_the_model(fakes, monkeypatch):
    fakes()

    class Pipe:
        device = None

        def __call__(self, **kwargs):
            raise FakeOOM("CUDA out of memory")

    class Model:
        negative_prompt = ""

        def __init__(self):
            self.pipe = Pipe()

        def update_height_width(self, h, w):
            pass

    model = Model()
    ref = weakref.ref(model)

    def load(ctx, resolution):
        nm._CACHE.update(model=model, ckpt="x")
        return model

    monkeypatch.setattr(nm, "_load", load)
    tensor = types.SimpleNamespace(to=lambda **k: None)
    monkeypatch.setattr(nm, "_conditions", lambda *a: (tensor, tensor))
    monkeypatch.setattr(nm, "_sketch_mask", lambda *a: object())
    with pytest.raises(RuntimeError) as err:
        nm.compose(Ctx(), INPUTS, SETTINGS)
    assert "그래픽카드 메모리가 모자랍니다" in str(err.value)
    assert nm._CACHE["model"] is None
    del model, load
    gc.collect()
    # 오류의 traceback 이 compose 프레임을 쥐고 있어도 그 안의 model 은 이미 None 이라 모델이 풀린다
    assert ref() is None
