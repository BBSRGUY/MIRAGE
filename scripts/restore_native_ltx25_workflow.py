from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


MIRAGE_NODE_IDS = {
    5011,
    5012,
    6100,
    6101,
    6102,
    6103,
    6104,
    6105,
    7039,
    7040,
    7041,
    7042,
    7043,
    7044,
    7045,
    7046,
    7047,
    7048,
    7049,
    7057,
    7065,
    7066,
}


def _input_slot(node: dict[str, Any], name: str) -> int:
    return next(index for index, item in enumerate(node.get("inputs", [])) if item["name"] == name)


def _first_widget(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, list):
        return value[0] if value else default
    return value


def _rebuild_links(workflow: dict[str, Any]) -> None:
    nodes = {node["id"]: node for node in workflow["nodes"]}
    for node in workflow["nodes"]:
        for item in node.get("inputs", []):
            item["link"] = None
        for item in node.get("outputs", []):
            item["links"] = []
    for link_id, source_id, source_slot, target_id, target_slot, _kind in workflow["links"]:
        nodes[source_id]["outputs"][source_slot]["links"].append(link_id)
        nodes[target_id]["inputs"][target_slot]["link"] = link_id


def restore_native(
    official: dict[str, Any], edited: dict[str, Any], fps_override: int | None = None
) -> dict[str, Any]:
    workflow = copy.deepcopy(official)
    official_nodes = {node["id"]: node for node in workflow["nodes"]}
    edited_nodes = {node["id"]: node for node in edited["nodes"]}

    mirage_node_ids = set(MIRAGE_NODE_IDS)
    image_input_slot = _input_slot(edited_nodes[6100], "image")
    image_source = next(
        (link[1] for link in edited["links"] if link[3] == 6100 and link[4] == image_input_slot),
        None,
    )
    if image_source is not None:
        mirage_node_ids.add(image_source)

    missing = sorted(mirage_node_ids - edited_nodes.keys())
    if missing:
        raise ValueError(f"edited workflow is missing MIRAGE nodes: {missing}")
    workflow["nodes"].extend(copy.deepcopy(edited_nodes[node_id]) for node_id in sorted(mirage_node_ids))
    nodes = {node["id"]: node for node in workflow["nodes"]}

    # Preserve the user's controls while retaining the native LTX-2.5 topology.
    resolution = edited_nodes.get(7068)
    if resolution is not None:
        official_nodes[409]["widgets_values"] = copy.deepcopy(resolution["widgets_values"])
    fps_node = edited_nodes.get(5098)
    fps = int(round(float(_first_widget(fps_node.get("widgets_values") if fps_node else None, 24))))
    if fps_override is not None:
        fps = int(fps_override)
    official_nodes[449]["widgets_values"] = [fps, "fixed"]
    gemma_widgets = edited_nodes[6102].get("widgets_values", [])
    duration = int(gemma_widgets[14]) if len(gemma_widgets) > 14 else 5
    official_nodes[450]["widgets_values"] = [duration, "fixed"]

    # Keep every link wholly inside the user's prompt/image/ZiT selection branch.
    internal_links = [
        copy.deepcopy(link)
        for link in edited["links"]
        if link[1] in mirage_node_ids and link[3] in mirage_node_ids
    ]
    for index, link in enumerate(internal_links):
        link[0] = 17000 + index

    # Remove only the native edges replaced by reference-aware routing.
    replaced_native_links = {805, 806, 822, 823, 827, 853, 854}
    workflow["links"] = [link for link in workflow["links"] if link[0] not in replaced_native_links]
    workflow["links"].extend(internal_links)

    gemma = nodes[6102]
    additions = [
        [18000, 455, 0, 5011, _input_slot(nodes[5011], "model"), "MODEL"],
        [18001, 440, 0, 5011, _input_slot(nodes[5011], "vae"), "VAE"],
        [18002, 6103, 0, 5011, _input_slot(nodes[5011], "reference_sheet"), "IMAGE"],
        [18003, 6103, 1, 5011, _input_slot(nodes[5011], "has_reference"), "BOOLEAN"],
        [18004, 5011, 0, 427, _input_slot(nodes[427], "model"), "MODEL"],
        [18005, 5011, 0, 419, _input_slot(nodes[419], "model"), "MODEL"],
        [18006, 430, 0, 5012, _input_slot(nodes[5012], "positive"), "CONDITIONING"],
        [18007, 430, 1, 5012, _input_slot(nodes[5012], "negative"), "CONDITIONING"],
        [18008, 440, 0, 5012, _input_slot(nodes[5012], "vae"), "VAE"],
        [18009, 434, 0, 5012, _input_slot(nodes[5012], "latent"), "LATENT"],
        [18010, 6103, 0, 5012, _input_slot(nodes[5012], "image"), "IMAGE"],
        [18011, 6103, 1, 5012, _input_slot(nodes[5012], "has_reference"), "BOOLEAN"],
        [18012, 5012, 0, 427, _input_slot(nodes[427], "positive"), "CONDITIONING"],
        [18013, 5012, 1, 427, _input_slot(nodes[427], "negative"), "CONDITIONING"],
        [18014, 5012, 0, 419, _input_slot(nodes[419], "positive"), "CONDITIONING"],
        [18015, 5012, 1, 419, _input_slot(nodes[419], "negative"), "CONDITIONING"],
        [18016, 5012, 2, 431, _input_slot(nodes[431], "video_latent"), "LATENT"],
        [18017, 6102, 2, 432, _input_slot(nodes[432], "text"), "STRING"],
        [18018, 6102, 3, 433, _input_slot(nodes[433], "text"), "STRING"],
        [18019, 450, 0, 6102, _input_slot(gemma, "duration_seconds"), "INT"],
        [18020, 449, 0, 6102, _input_slot(gemma, "fps"), "INT"],
    ]
    workflow["links"].extend(additions)

    _rebuild_links(workflow)
    workflow["last_node_id"] = max(node["id"] for node in workflow["nodes"])
    workflow["last_link_id"] = max(link[0] for link in workflow["links"])
    workflow.setdefault("extra", {})["mirage_backend"] = "native-ltx-2.5-two-stage-editanything"
    workflow["extra"]["mirage_native_source"] = "LTX 2.5 T 2 V.json"
    return workflow


def main() -> None:
    parser = argparse.ArgumentParser(description="Restore native LTX-2.5 sampling around MIRAGE reference controls")
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--edited", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int)
    args = parser.parse_args()
    official = json.loads(args.official.read_text(encoding="utf-8"))
    edited = json.loads(args.edited.read_text(encoding="utf-8"))
    restored = restore_native(official, edited, fps_override=args.fps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(restored, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
