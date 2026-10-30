"""입력 검사 · 위치 · 크기 맞추기 순수 함수 시험(모델 · torch 없이). 플러그인 폴더에서 `python -m pytest tests`."""
import pytest

from nodes.main import check_inputs, cover_box, even_positions, parse_positions

SETTINGS = {"length": "61", "color-at": "", "line-at": ""}


def inputs(colors=1, lines=1, prompt="A girl jumps. Anime."):
    out = {}
    if colors:
        out["color"] = [f"c{i}.png" for i in range(colors)]
    if lines:
        out["line"] = [f"l{i}.png" for i in range(lines)]
    if prompt is not None:
        out["prompt"] = prompt
    return out


def test_even_positions_match_official_app():
    # app.py update_visibility_images · update_visibility 의 값
    assert even_positions(1, 61, "color") == [0]
    assert even_positions(2, 61, "color") == [0, 30]
    assert even_positions(1, 61, "line") == [60]
    assert even_positions(2, 61, "line") == [30, 60]
    assert even_positions(3, 61, "line") == [20, 40, 60]
    assert even_positions(1, 17, "line") == [16]


def test_parse_positions_accepts_commas_and_spaces():
    assert parse_positions("0,30", 2, 61, "color") == [0, 30]
    assert parse_positions(" 30 60 ", 2, 61, "line") == [30, 60]
    assert parse_positions("30，60", 2, 61, "line") == [30, 60]
    assert parse_positions("", 2, 61, "line") == [30, 60]


@pytest.mark.parametrize(
    "text,count,want",
    [
        ("0,30", 1, "1장 이어져"),
        ("61", 1, "0 부터 60 사이"),
        ("-1", 1, "0 부터 60 사이"),
        ("10,10", 2, "같은 번호"),
        ("첫,끝", 2, "쉼표로 적습니다"),
    ],
)
def test_parse_positions_errors_are_korean(text, count, want):
    with pytest.raises(ValueError) as err:
        parse_positions(text, count, 61, "color")
    assert want in str(err.value)
    assert "컬러 키프레임 위치" in str(err.value)


def test_check_inputs_defaults():
    assert check_inputs(inputs(), SETTINGS) == ([0], [60], 61)
    assert check_inputs(inputs(2, 3), SETTINGS) == ([0, 30], [20, 40, 60], 61)


@pytest.mark.parametrize("length", [17, 33, 49, 61])
@pytest.mark.parametrize("colors", [1, 2, 3, 4])
@pytest.mark.parametrize("lines", [1, 2, 3, 4])
def test_empty_positions_never_overlap(colors, lines, length):
    # 두 칸이 다 비면 선화가 컬러를 피해 고르게 놓인다(공식 화면 기본값은 2장씩부터 겹친다)
    color_at, line_at, got = check_inputs(inputs(colors, lines), {**SETTINGS, "length": str(length)})
    assert got == length
    assert len(color_at) == colors and len(line_at) == lines
    assert len(set(color_at) | set(line_at)) == colors + lines
    assert all(0 <= f < length for f in color_at + line_at)
    assert color_at == even_positions(colors, length, "color")  # 컬러는 공식 값 그대로


def test_empty_positions_move_only_the_clash():
    assert check_inputs(inputs(2, 2), SETTINGS) == ([0, 30], [31, 60], 61)  # 공식 값 30,60 중 30 만 옮김
    assert check_inputs(inputs(4, 4), {**SETTINGS, "length": "17"}) == ([0, 4, 8, 12], [5, 9, 13, 16], 17)
    assert check_inputs(inputs(2, 3), SETTINGS) == ([0, 30], [20, 40, 60], 61)  # 안 겹치면 공식 값 그대로


def test_empty_side_avoids_the_written_side():
    assert check_inputs(inputs(2, 2), {**SETTINGS, "color-at": "0,30"})[1] == [31, 60]
    assert check_inputs(inputs(1, 1), {**SETTINGS, "line-at": "0"})[0] == [1]
    assert check_inputs(inputs(1, 2), {**SETTINGS, "line-at": "30,60"})[0] == [0]


def test_written_positions_that_overlap_are_rejected():
    # 공식 app.py 도 같은 프레임에 컬러와 선화를 두면 거절한다(validate_inputs)
    with pytest.raises(ValueError) as err:
        check_inputs(inputs(2, 2), {**SETTINGS, "color-at": "0,30", "line-at": "30,60"})
    assert "같은 프레임(30)" in str(err.value)


def test_check_inputs_names_missing_port_and_card():
    with pytest.raises(ValueError) as err:
        check_inputs(inputs(lines=0), SETTINGS)
    msg = str(err.value)
    assert "지금은 '컬러 키프레임'만 이어져 있습니다" in msg
    assert "'예제 — 선화 키프레임' 카드" in msg
    with pytest.raises(ValueError) as err:
        check_inputs(inputs(colors=0, lines=0), SETTINGS)
    assert "둘 다 비어" in str(err.value)
    assert "'예제 — 컬러 키프레임' · '예제 — 선화 키프레임'" in str(err.value)


def test_check_inputs_limits_and_prompt():
    with pytest.raises(ValueError, match="4장까지"):
        check_inputs(inputs(colors=5), SETTINGS)
    with pytest.raises(ValueError, match="프롬프트"):
        check_inputs(inputs(prompt="  "), SETTINGS)
    with pytest.raises(ValueError, match="프롬프트"):
        check_inputs(inputs(prompt=None), SETTINGS)


def test_cover_box_matches_official_resize():
    # app.py process_conditions — 짧은 변을 맞추고 가운데를 자른다
    assert cover_box(640, 1024, 480, 832) == (520, 832, 20, 0)  # 16:10 예제 그림
    assert cover_box(1080, 1920, 480, 832) == (480, 853, 0, 10)  # 16:9
    assert cover_box(512, 512, 480, 832) == (832, 832, 176, 0)  # 정사각
    assert cover_box(480, 832, 480, 832) == (480, 832, 0, 0)
