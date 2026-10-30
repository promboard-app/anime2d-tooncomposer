"""promboard.anime2d-tooncomposer — ToonComposer(TencentARC) 키프레임 영상 노드 코드.
🔴 검증 안 됨 · 서버급: 이 코드는 실제로 돌려 보지 않았다(공식 안내 그래픽카드 메모리 57GB, 이 PC 는 12GB). 모델 없이 import 만 확인했다.

노드 하나(compose): 컬러 키프레임 1~4장 + 선화 키프레임 1~4장(+ 움직일 곳 마스크) + 프롬프트 → 중간 프레임과 채색을 한 번에 만든 영상.
  상류 코드 nodes/tooncomposer/ = TencentARC/ToonComposer 53dc3df(MIT, Wan2.1 부분 Apache-2.0). 고친 곳은 그 파일의 'PromBoard:' 주석.
  상류 app.py(gradio 화면 · 시작할 때 모델 받기)는 담지 않고, 그 추론 경로(initialize_model · process_conditions ·
  process_sketch_masks · tooncomposer_inference)를 이 파일에 옮겼다 — 모델은 hf_hub 대신 ctx.model_path(받아 둔 파일)에서 읽는다.
  공식 앱 값: 480p 832×480 · 608p 1088×608, 15단계, CFG 7.5, 위치 반영 정도 1.0, 시드 42, 결과 mp4 20fps.
  상류와 다른 것: 마스크는 노드에 한 장만 받아 모든 선화 키프레임에 쓰고(흰 곳 = 움직일 곳), 그림과 같은 방법(짧은 변 맞춤 + 가운데
  자르기)으로 크기를 맞춘다(상류는 마스크만 늘려 맞춰 자른 선화와 어긋날 수 있다). 투명한 곳은 흰 바탕으로 채운다.

노드 코드 약속(NODES · ctx · 입출력 모양)은 PromBoard docs/Scrum4/02_스키마-v0.md §6.
"""
import gc
import json
import os
import random
import re
import time

TOON_REPO = "TencentARC/ToonComposer"
WAN_REPO = "Wan-AI/Wan2.1-I2V-14B-480P"
# app.py build_checkpoints_by_resolution — (높이, 넓이)
SIZES = {"480p": (480, 832), "608p": (608, 1088)}
FPS = 20  # app.py 가 결과 mp4 를 쓰는 fps
MAX_KEYFRAMES = 4  # 공식 config max_num_cond_images · max_num_cond_sketches, app.py 화면도 4장까지
# 글 인코더 토크나이저 — 파이프라인이 T5 파일 옆 google/umt5-xxl 폴더에서 읽는다(i2v_pipeline fetch_models_from_checkpoints)
UMT5_TOKENIZER = ("special_tokens_map.json", "spiece.model", "tokenizer.json", "tokenizer_config.json")
# 그래픽카드에 올리는 모델 무게(bf16) — 받는 파일 크기로 셌다: DiT 65.6GB(fp32) → 32.8GB, T5 11.4GB(bf16), CLIP 4.8GB(fp32) → 2.4GB,
# VAE 0.5GB → 0.25GB, ToonComposer 0.18GB → 0.09GB. 공식 앱은 모두 그래픽카드에 올린다(model.to(device)). 이보다 작으면 무게도 못 올린다.
WEIGHTS_BYTES = 46_880_000_000
OFFICIAL_VRAM_GB = 57  # 공식 README: 480p 61프레임에 약 57GB

PORT = {"color": "컬러 키프레임", "line": "선화 키프레임", "mask": "움직일 곳", "prompt": "프롬프트"}
CARD = {"color": "예제 — 컬러 키프레임", "line": "예제 — 선화 키프레임"}
NODE = "키프레임 영상"

# app.py _merge_with_defaults 의 기본값 — 받은 config.json 에 없는 칸만 채운다
CONFIG_DEFAULTS = {
    "base_model_name": "Wan2.1-I2V-14B-480P",
    "learning_rate": 1e-5,
    "use_gradient_checkpointing": True,
    "tiled": False,
    "tile_size_height": 34,
    "tile_size_width": 34,
    "tile_stride_height": 18,
    "tile_stride_width": 16,
    "output_path": "./",
    "use_dera": False,
    "dera_rank": None,
    "use_dera_spatial": True,
    "use_dera_temporal": True,
    "use_sequence_cond": True,
    "sequence_cond_mode": "sparse",
    "use_channel_cond": False,
    "use_sequence_cond_position_aware_residual": True,
    "use_sequence_cond_loss": False,
    "fast_dev": False,
    "max_num_cond_images": 1,
    "max_num_cond_sketches": 2,
    "use_sketch_mask": True,
    "sketch_mask_ratio": 0.2,
    "no_first_sketch": False,
}

# 그래픽카드 모델은 한 벌만 둔다 — 해상도를 바꾸면 ToonComposer 조각만 갈아 끼운다(app.py update_resolution 과 같다).
_CACHE = {"model": None, "ckpt": None}


# ── 입력 검사(모델을 불러오기 전, torch 없이) ─────────────────────────────────────


def even_positions(count, length, kind):
    """비운 위치 칸 — app.py 화면이 장 수를 바꿀 때 넣는 값과 같다. 컬러는 0 부터, 선화는 끝 쪽부터 채운다."""
    step = (length - 1) // max(count, 1)
    return [step * (i if kind == "color" else i + 1) for i in range(count)]


def parse_positions(text, count, length, kind):
    """위치 글('0' · '30,60')을 프레임 번호 목록으로. 비우면 고르게 나눈다. 틀리면 한국어 ValueError."""
    raw = (text or "").strip()
    if not raw:
        return even_positions(count, length, kind)
    name = PORT[kind]
    parts = [p for p in re.split(r"[,\s，、]+", raw) if p]
    try:
        values = [int(p) for p in parts]
    except ValueError:
        raise ValueError(f"'{name} 위치'는 프레임 번호를 쉼표로 적습니다(예: 0,30). 지금 글은 '{raw}' 입니다.") from None
    if len(values) != count:
        raise ValueError(
            f"'{name} 위치'에 번호가 {len(values)}개 있는데 '{name}'은 {count}장 이어져 있습니다. 이은 장 수만큼 적거나 칸을 비워 고르게 나누세요."
        )
    for v in values:
        if v < 0 or v > length - 1:
            raise ValueError(f"'{name} 위치'의 {v} 는 0 부터 {length - 1} 사이여야 합니다(지금 길이 {length}프레임).")
    if len(set(values)) != len(values):
        raise ValueError(f"'{name} 위치'에 같은 번호가 두 번 있습니다. 한 프레임에는 한 장만 둡니다.")
    return values


def check_inputs(inputs, settings):
    """연결 · 설정 검사. 맞으면 (컬러 위치, 선화 위치, 길이), 아니면 한국어 ValueError(지금 이어진 것까지 말한다)."""
    colors = list(inputs.get("color") or [])
    lines = list(inputs.get("line") or [])
    if not colors or not lines:
        missing = [k for k, v in (("color", colors), ("line", lines)) if not v]
        joined = "'과 '".join(PORT[k] for k in missing)
        connected = [PORT[k] for k, v in (("color", colors), ("line", lines)) if v]
        now = f"지금은 '{connected[0]}'만 이어져 있습니다." if connected else "지금은 둘 다 비어 있습니다."
        cards = "' · '".join(CARD[k] for k in missing)
        raise ValueError(
            f"'{NODE}'은 '{PORT['color']}'과 '{PORT['line']}'을 한 장 이상씩 이어야 합니다. {now} "
            f"빠진 '{joined}'에는 에셋의 '{cards}' 카드를 소스 노드 위에 끌어 놓아 보세요."
        )
    for kind, items in (("color", colors), ("line", lines)):
        if len(items) > MAX_KEYFRAMES:
            raise ValueError(f"'{PORT[kind]}'은 {MAX_KEYFRAMES}장까지 받습니다. 지금 {len(items)}장 이어져 있습니다.")
    if not (inputs.get("prompt") or "").strip():
        raise ValueError(f"'{NODE}'은 '{PORT['prompt']}'가 있어야 합니다. 장면과 움직임을 영어로 적어 이어 주세요(예: A girl waves her hand and smiles. Anime. High quality.).")
    length = int(settings["length"])
    color_at = parse_positions(settings.get("color-at"), len(colors), length, "color")
    line_at = parse_positions(settings.get("line-at"), len(lines), length, "line")
    both = sorted(set(color_at) & set(line_at))
    if both:
        raise ValueError(
            f"'{PORT['color']}'과 '{PORT['line']}'을 같은 프레임({', '.join(map(str, both))})에 둘 수 없습니다. 위치 설정을 바꿔 주세요."
        )
    return color_at, line_at, length


def cover_box(h, w, target_h, target_w):
    """app.py process_conditions 의 크기 맞추기 — 짧은 변을 맞춰 늘리고(비율 유지) 가운데를 자른다. (새 높이, 새 넓이, 위, 왼쪽)."""
    if h / w < target_h / target_w:
        new_h, new_w = target_h, int(w * (target_h / h))
    else:
        new_w, new_h = target_w, int(h * (target_w / w))
    return new_h, new_w, max(0, (new_h - target_h) // 2), max(0, (new_w - target_w) // 2)


# ── 모델 ─────────────────────────────────────────────────────────────────


def _repo_dir(ctx, repo, probe):
    path = os.path.normpath(ctx.model_path(f"{repo}/{probe}"))
    for _ in probe.split("/"):
        path = os.path.dirname(path)
    return path


def _free():
    import torch

    _CACHE.update(model=None, ckpt=None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load(ctx, resolution):
    import torch

    from .tooncomposer.pipeline import i2v_pipeline
    from .tooncomposer.tooncomposer import ToonComposer, get_base_model_paths

    ckpt = ctx.model_path(f"{TOON_REPO}/{resolution}/tooncomposer.ckpt")
    if _CACHE["model"] is not None:
        if _CACHE["ckpt"] != ckpt:
            ctx.progress(0.04, "load-model")
            _CACHE["model"].load_tooncomposer_checkpoint(ckpt)
            _CACHE["ckpt"] = ckpt
        return _CACHE["model"]

    root = _repo_dir(ctx, WAN_REPO, "Wan2.1_VAE.pth")
    wanted = get_base_model_paths("Wan2.1-I2V-14B-480P", format="list", model_root=".")
    for rel in [os.path.basename(p) for p in wanted[0] + wanted[1:]] + [f"google/umt5-xxl/{f}" for f in UMT5_TOKENIZER]:
        ctx.model_path(f"{WAN_REPO}/{rel}")  # 받아 둔 파일인지 본다(없으면 model-missing)
    with open(ctx.model_path(f"{TOON_REPO}/{resolution}/config.json"), "r", encoding="utf-8") as fh:
        args = {**CONFIG_DEFAULTS, **json.load(fh)}

    # 모델 파일 10개(DiT 7조각 · CLIP · T5 · VAE)를 읽을 때마다 load-model 을 다시 알린다 — 한 파일이 11GB 라 느린 디스크에서 오래 걸린다.
    original = i2v_pipeline.WanVideoPipeline._load_single_file
    count = {"n": 0}

    def load_single_file(pipe, file_path):
        count["n"] += 1
        ctx.check_cancel()
        ctx.progress(0.02 + 0.06 * min(1.0, count["n"] / 10), "load-model")
        return original(pipe, file_path)

    _free()
    ctx.progress(0.02, "load-model")
    i2v_pipeline.WanVideoPipeline._load_single_file = load_single_file
    try:
        # app.py initialize_model 과 같은 인자
        model = ToonComposer(
            base_model_name=args["base_model_name"],
            model_root=root,
            learning_rate=args["learning_rate"],
            use_gradient_checkpointing=args["use_gradient_checkpointing"],
            checkpoint_path=ckpt,
            tiled=args["tiled"],
            tile_size=(args["tile_size_height"], args["tile_size_width"]),
            tile_stride=(args["tile_stride_height"], args["tile_stride_width"]),
            output_path=args["output_path"],
            use_dera=args["use_dera"],
            dera_rank=args["dera_rank"],
            use_dera_spatial=args["use_dera_spatial"],
            use_dera_temporal=args["use_dera_temporal"],
            use_sequence_cond=args["use_sequence_cond"],
            sequence_cond_mode=args["sequence_cond_mode"],
            use_channel_cond=args["use_channel_cond"],
            use_sequence_cond_position_aware_residual=args["use_sequence_cond_position_aware_residual"],
            use_sequence_cond_loss=args["use_sequence_cond_loss"],
            fast_dev=False,
            max_num_cond_images=args["max_num_cond_images"],
            max_num_cond_sketches=args["max_num_cond_sketches"],
            random_spaced_cond_frames=False,
            use_sketch_mask=args["use_sketch_mask"],
            sketch_mask_ratio=args["sketch_mask_ratio"],
            no_first_sketch=args["no_first_sketch"],
        )
    finally:
        i2v_pipeline.WanVideoPipeline._load_single_file = original
    ctx.progress(0.09, "load-model")
    model = model.to(ctx.device, dtype=torch.bfloat16).eval()
    _CACHE.update(model=model, ckpt=ckpt)
    return model


# ── 입력 그림 → 조건 텐서 (app.py process_conditions · process_sketch_masks) ───────────────


def _open_rgb(path):
    from PIL import Image, ImageOps

    with Image.open(path) as im:
        im.load()
        im = ImageOps.exif_transpose(im)
        if im.mode in ("RGBA", "LA", "P"):
            im = im.convert("RGBA")
            white = Image.new("RGBA", im.size, (255, 255, 255, 255))
            return Image.alpha_composite(white, im).convert("RGB")
        return im.convert("RGB")


def _fit(tensor, target_h, target_w, mode):
    """(1, C, h, w) → (1, C, target_h, target_w). cover_box 대로 늘리고 가운데를 자른다."""
    import torch

    _, _, h, w = tensor.shape
    new_h, new_w, top, left = cover_box(h, w, target_h, target_w)
    tensor = torch.nn.functional.interpolate(tensor, size=(new_h, new_w), mode=mode)
    return tensor[:, :, top:top + target_h, left:left + target_w]


def _conditions(paths, positions, num_frames, target_h, target_w, device, is_sketch):
    import numpy as np
    import torch

    video = torch.zeros((1, 3, num_frames, target_h, target_w), device=device)
    mask = torch.zeros((1, num_frames), device=device)
    for path, frame_idx in zip(paths, positions):
        img = torch.from_numpy(np.array(_open_rgb(path))).permute(2, 0, 1).float() / 127.5 - 1.0
        if is_sketch:
            img = -img  # app.py 와 같다 — 선화는 흰 바탕 검은 선을 뒤집어 넣는다
        video[:, :, frame_idx] = _fit(img.unsqueeze(0).to(device), target_h, target_w, "bilinear")[0]
        mask[:, frame_idx] = 1.0
    return video, mask


def _sketch_mask(path, line_at, num_frames, target_h, target_w, device):
    """1 = 선화를 따른다 · 0 = 자유롭게 움직인다(app.py process_sketch_masks). 마스크의 흰 곳(밝기 128 이상)이 0 이다."""
    import numpy as np
    import torch
    from PIL import Image, ImageOps

    local = torch.ones((1, 1, num_frames, target_h, target_w), device=device)
    if not path:
        return local
    with Image.open(path) as im:
        im.load()
        im = ImageOps.exif_transpose(im)
        if im.mode in ("RGBA", "LA", "P"):
            im = im.convert("RGBA")
            black = Image.new("RGBA", im.size, (0, 0, 0, 255))
            im = Image.alpha_composite(black, im)
        gray = np.array(im.convert("L"), dtype=np.float32) / 255.0
    moving = _fit(torch.from_numpy(gray)[None, None].to(device), target_h, target_w, "nearest")[0, 0] >= 0.5
    keep = 1.0 - moving.float()
    for idx in line_at:
        local[0, 0, idx] = keep
    return local


class _Steps:
    """파이프라인 progress_bar_cmd 자리 — 단계마다 취소를 보고 진행률을 알린다."""

    def __init__(self, ctx, iterable):
        self.ctx, self.items = ctx, list(iterable)

    def __iter__(self):
        total = max(1, len(self.items))
        for i, item in enumerate(self.items):
            self.ctx.check_cancel()
            self.ctx.progress(0.12 + 0.78 * i / total, "generate")
            yield item
        self.ctx.progress(0.9, "decode")


# ── 노드 ─────────────────────────────────────────────────────────────────


def compose(ctx, inputs, settings):
    t0 = time.monotonic()
    color_at, line_at, num_frames = check_inputs(inputs, settings)
    if ctx.device != "cuda":
        raise ValueError(f"'{NODE}'은 NVIDIA 그래픽카드(CUDA)에서만 돕니다. 지금은 그래픽카드 없이(CPU) 돌도록 되어 있습니다.")

    os.environ.setdefault("MPLBACKEND", "Agg")  # 상류 util/model_util.py 가 맨 위에서 matplotlib.pyplot 을 부른다
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch

    total = torch.cuda.get_device_properties(0).total_memory  # 바이트. 글에는 카드 표기처럼 GiB 로 보인다(12GB 카드 → 12)
    if total < WEIGHTS_BYTES:
        raise ValueError(
            f"이 PC 그래픽카드 메모리는 {total / 2**30:.0f}GB 라 이 모델을 올릴 수 없습니다. 모델 무게만 약 {WEIGHTS_BYTES / 1e9:.0f}GB, "
            f"공식 안내로 480p 61프레임에 약 {OFFICIAL_VRAM_GB}GB 가 필요합니다. 일반 PC 에서는 '2D 애니메이션 팩 — 프레임 사이 채우기'를 쓰세요."
        )

    resolution = settings["resolution"]
    target_h, target_w = SIZES[resolution]
    seed = int(settings["seed"])
    if seed < 0:
        seed = random.randint(0, 2**31 - 1)
    prompt = inputs["prompt"].strip()

    failure = None
    try:
        model = _load(ctx, resolution)
        model.update_height_width(target_h, target_w)
        device = ctx.device
        ctx.progress(0.1, "prepare")
        with torch.no_grad():
            cond_video, cond_mask = _conditions(inputs["color"], color_at, num_frames, target_h, target_w, device, False)
            sketch_video, sketch_mask = _conditions(inputs["line"], line_at, num_frames, target_h, target_w, device, True)
            local_mask = _sketch_mask(inputs.get("mask"), line_at, num_frames, target_h, target_w, device)
            dtype = torch.bfloat16
            ctx.check_cancel()
            with torch.amp.autocast(dtype=dtype, device_type="cuda"):
                model.pipe.device = device
                generated = model.pipe(
                    prompt=[prompt],
                    negative_prompt=[model.negative_prompt],
                    input_image=None,
                    num_inference_steps=int(settings["steps"]),
                    num_frames=num_frames,
                    seed=seed,
                    tiled=True,
                    input_condition_video=cond_video.to(device=device, dtype=dtype),
                    input_condition_preserved_mask=cond_mask.to(device=device, dtype=dtype),
                    input_condition_video_sketch=sketch_video.to(device=device, dtype=dtype),
                    input_condition_preserved_mask_sketch=sketch_mask.to(device=device, dtype=dtype),
                    sketch_local_mask=local_mask,
                    cfg_scale=float(settings["guidance"]),
                    sequence_cond_residual_scale=float(settings["sketch-strength"]),
                    height=target_h,
                    width=target_w,
                    progress_bar_cmd=lambda it: _Steps(ctx, it),
                )
            frames = model.pipe.tensor2video(generated[0].cpu())
        del generated, cond_video, sketch_video, local_mask
    except torch.OutOfMemoryError:
        failure = "vram"
    except MemoryError:
        failure = "ram"
    except RuntimeError as exc:
        if "not enough memory" not in str(exc) and "DefaultCPUAllocator" not in str(exc):
            raise
        failure = "ram"
    if failure:
        # except 블록 밖에서 정리하고 새 오류를 낸다 — 블록 안에서 내면 원래 오류(와 그 프레임이 쥔 텐서)가 새 오류의 __context__ 로 남아
        # 사이드카가 다음 실행까지 메모리를 놓지 못한다.
        _free()
        if failure == "vram":
            raise RuntimeError(
                f"그래픽카드 메모리가 모자랍니다(공식 안내 480p 61프레임에 약 {OFFICIAL_VRAM_GB}GB 필요, 이 PC {total / 2**30:.0f}GB). "
                "길이를 줄이거나 480p 로 다시 해 보세요."
            )
        raise RuntimeError("컴퓨터 메모리(RAM)가 모자랍니다. 이 모델은 불러오는 동안 RAM 을 76GB 넘게 씁니다. 다른 프로그램을 닫고 다시 해 보세요.")

    ctx.check_cancel()
    ctx.progress(0.95, "encode")
    out = ctx.out_path("tooncomposer.mp4")
    _write_mp4(out, frames, FPS)
    pictures = []
    for i, frame in enumerate(frames):
        path = ctx.out_path(f"frame-{i:03d}.png")
        frame.save(path)
        pictures.append(path)
    ctx.log(f"done in {time.monotonic() - t0:.1f}s ({len(frames)} frames, {resolution}, seed {seed})")
    ctx.progress(1.0, "done")
    return {"video": out, "frames": pictures}


def _write_mp4(path, frames, fps):
    """PIL 그림 목록 → H.264 yuv420p mp4(소리 없음)."""
    import av
    import numpy as np

    w, h = frames[0].size
    w2, h2 = w - w % 2, h - h % 2
    container = av.open(path, "w", options={"movflags": "+faststart"})
    try:
        stream = container.add_stream("libx264", rate=fps)
        stream.width, stream.height = w2, h2
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "16", "preset": "medium"}
        for frame in frames:
            arr = np.ascontiguousarray(np.asarray(frame.convert("RGB"))[:h2, :w2])
            for packet in stream.encode(av.VideoFrame.from_ndarray(arr, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    finally:
        container.close()


NODES = {"compose": compose}
