"""workflows/workflows.json 을 manifest.json 노드 정의로 다시 만든다(플러그인 노드 snapshot 이 manifest 와 어긋나지 않게).

사용: python docs/scripts/build_workflows.py   (플러그인 폴더에서)
노드 · 선 모양은 PromBoard 앱이 워크플로우를 저장하는 모양(WorkflowDoc 의 WorkspaceNode[] · Wire[]) 그대로다. 소스 노드는
비워 둔다 — 앱은 플러그인 그림을 소스 노드에 미리 넣지 못해, 사용자가 에셋 탭의 예제 카드를 소스 노드 위에 끌어 놓는다.
ToonComposer 는 돌려 보지 않아 결과 썸네일이 없다.
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
node = manifest["nodes"][0]
SNAPSHOT_KEYS = ("id", "label", "layout", "showSources", "sources", "settingsPanel", "views", "compare", "menuGroup", "inputs", "outputs", "settings")
snapshot = {"plugin": manifest["id"], **{k: node[k] for k in SNAPSHOT_KEYS if k in node}}
plugin_type = f"{manifest['id']}/{node['id']}"


def source(node_id, x, y):
    return {
        "id": node_id, "position": {"x": x, "y": y}, "selectedTermIds": [], "isStale": False,
        "type": "sourceImage", "images": [], "nodeType": "sourceImage", "pendingUploads": [], "inputWires": [], "outputWires": [],
    }


def wire(wire_id, src, src_port, dst, dst_port, data_type):
    return {"id": wire_id, "fromNodeId": src, "fromPortId": src_port, "toNodeId": dst, "toPortId": dst_port, "dataType": data_type}


PROMPT = (
    "A girl with a long dark braid in a blue pinafore dress stands on a grassy hill under a blue sky with white clouds, "
    "then happily jumps into the air with both arms raised. Anime. High quality."
)

workflow = {
    "id": "keyframes-to-video",
    "name": "컬러 키프레임 + 선화 키프레임 → 키프레임 영상",
    "description": (
        "에셋 탭의 '예제 — 컬러 키프레임' 카드를 맨 위 소스 노드에, '예제 — 선화 키프레임' 카드를 그 아래 소스 노드에 끌어 놓고 "
        "키프레임 영상 노드 ▷ 를 누른다. 컬러는 첫 프레임, 선화는 마지막 프레임에 들어가 그 사이를 AI 가 그리고 칠한다. "
        "움직임은 프롬프트 노드 글(영어)로 바꾼다. 서버급 그래픽카드(57GB 급)에서만 돈다(검증 안 됨)."
    ),
    "nodes": [
        source("color", 0, 0),
        source("line", 0, 340),
        {
            "id": "prompt", "position": {"x": 0, "y": 680}, "selectedTermIds": [], "isStale": False,
            "type": "prompt", "promptType": "video", "promptText": PROMPT, "promptHistory": [],
            "nodeType": "prompt", "inputWires": [], "outputWires": [],
        },
        {
            "id": "compose", "position": {"x": 380, "y": 160}, "selectedTermIds": [], "isStale": False,
            "type": plugin_type, "plugin": {"id": manifest["id"], "nodeId": node["id"]}, "snapshot": snapshot,
            "pluginSettings": {s["id"]: s["default"] for s in node["settings"]},
            "nodeType": plugin_type, "inputWires": [], "outputWires": [], "outputPick": {},
        },
    ],
    "wires": [
        wire("color-to-compose", "color", "source-image-out", "compose", "in-color", "image"),
        wire("line-to-compose", "line", "source-image-out", "compose", "in-line", "image"),
        wire("prompt-to-compose", "prompt", "prompt-text-out", "compose", "in-prompt", "text"),
    ],
}
assert len(workflow["description"]) <= 300, len(workflow["description"])
out = ROOT / "workflows" / "workflows.json"
out.write_text(json.dumps({"schema": 1, "workflows": [workflow]}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(out, len(workflow["description"]))
