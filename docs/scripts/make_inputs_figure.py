"""README 대표 그림 — 노드에 넣는 두 키프레임(예제 카드 그림)을 프레임 위치와 함께 보인다. 결과 그림은 없다(돌려 보지 않았다).

사용: python docs/scripts/make_inputs_figure.py   (플러그인 폴더에서, Pillow · 윈도우 '맑은 고딕' 글꼴)
만드는 것: docs/inputs.ko.png · docs/inputs.en.png
"""
import pathlib

from PIL import Image, ImageDraw, ImageFont

ROOT = pathlib.Path(__file__).resolve().parents[2]
FONT = r"C:\Windows\Fonts\malgun.ttf"
FONT_BOLD = r"C:\Windows\Fonts\malgunbd.ttf"

TEXT = {
    "ko": {
        "color": "컬러 키프레임 · 0번 프레임",
        "line": "선화 키프레임 · 60번 프레임",
        "middle": "사이 59프레임을\nAI가 그리고 칠한다",
        "foot": "결과 영상: 61프레임 · 832×480 · 20fps (실제로 돌려 보지 않아 결과 그림은 없습니다)",
    },
    "en": {
        "color": "Color keyframe · frame 0",
        "line": "Sketch keyframe · frame 60",
        "middle": "AI draws and\ncolors the 59\nframes between",
        "foot": "Output: 61 frames · 832×480 · 20 fps (not run yet, so no result picture)",
    },
}

BG = (250, 249, 246)
INK = (34, 34, 38)
MUTED = (110, 110, 118)
ACCENT = (214, 120, 40)
W_IMG, PAD, GAP = 520, 28, 230


def figure(lang):
    t = TEXT[lang]
    color = Image.open(ROOT / "cards" / "color-key.jpg").convert("RGB")
    line = Image.open(ROOT / "cards" / "line-key.png").convert("RGB")
    h_img = round(W_IMG * color.height / color.width)
    color = color.resize((W_IMG, h_img), Image.LANCZOS)
    line = line.resize((W_IMG, h_img), Image.LANCZOS)

    label = ImageFont.truetype(FONT_BOLD, 24)
    middle = ImageFont.truetype(FONT_BOLD, 22)
    foot = ImageFont.truetype(FONT, 20)
    top = PAD + 44
    width = PAD * 2 + W_IMG * 2 + GAP
    height = top + h_img + 74
    canvas = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(canvas)

    x_line = PAD + W_IMG + GAP
    for x, img, text in ((PAD, color, t["color"]), (x_line, line, t["line"])):
        canvas.paste(img, (x, top))
        draw.rectangle((x - 1, top - 1, x + W_IMG, top + h_img), outline=(200, 198, 192), width=1)
        draw.text((x, PAD), text, font=label, fill=INK)

    # 가운데 화살표와 설명
    mid_y = top + h_img // 2
    x0, x1 = PAD + W_IMG + 18, x_line - 18
    draw.line((x0, mid_y, x1 - 14, mid_y), fill=ACCENT, width=5)
    draw.polygon([(x1, mid_y), (x1 - 20, mid_y - 12), (x1 - 20, mid_y + 12)], fill=ACCENT)
    box = draw.multiline_textbbox((0, 0), t["middle"], font=middle, spacing=6, align="center")
    tw = box[2] - box[0]
    draw.multiline_text(((x0 + x1) / 2 - tw / 2, mid_y - (box[3] - box[1]) - 18), t["middle"], font=middle, fill=ACCENT, spacing=6, align="center")

    draw.text((PAD, top + h_img + 22), t["foot"], font=foot, fill=MUTED)
    out = ROOT / "docs" / f"inputs.{lang}.png"
    canvas.save(out, optimize=True)
    print(out, canvas.size, out.stat().st_size)


for lang in TEXT:
    figure(lang)
