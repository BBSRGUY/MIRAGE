from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROLE_INPUTS = (
    "character_1",
    "character_2",
    "wardrobe",
    "environment",
    "object_detail",
    "style_reference",
)


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


def upgrade(workflow: dict[str, Any]) -> dict[str, Any]:
    nodes = {node["id"]: node for node in workflow["nodes"]}
    contact = nodes[6100]
    old_role_links = [link for link in workflow["links"] if link[3] == 6100]
    first_image_link = old_role_links[0] if old_role_links else None

    contact.update(
        {
            "type": "MIRAGEMultiReferenceContactSheet",
            "size": [620, 430],
            "title": "Optional references: characters / wardrobe / environment / details / style",
            "inputs": [
                {"name": "sheet_width", "type": "INT", "widget": {"name": "sheet_width"}, "link": None},
                {"name": "sheet_height", "type": "INT", "widget": {"name": "sheet_height"}, "link": None},
                {"name": "variation_count", "type": "INT", "widget": {"name": "variation_count"}, "link": None},
                *[
                    {"name": name, "type": "IMAGE", "shape": 7, "link": None}
                    for name in ROLE_INPUTS
                ],
            ],
            "outputs": [
                {"name": "contact_sheet", "type": "IMAGE", "links": []},
                {"name": "primary_image", "type": "IMAGE", "links": []},
                {"name": "has_reference", "type": "BOOLEAN", "links": []},
                {"name": "reference_manifest", "type": "STRING", "links": []},
                {"name": "reference_count", "type": "INT", "links": []},
                {"name": "zit_denoise", "type": "FLOAT", "links": []},
                {"name": "variation_count", "type": "INT", "links": []},
            ],
            "properties": {
                **contact.get("properties", {}),
                "Node name for S&R": "MIRAGEMultiReferenceContactSheet",
            },
            "widgets_values": [1024, 576, 3],
            "widgets_values_named": {
                "sheet_width": 1024,
                "sheet_height": 576,
                "variation_count": 3,
            },
        }
    )

    # Remove only old links whose endpoint/source slots change below.
    workflow["links"] = [
        link
        for link in workflow["links"]
        if not (
            link[3] == 6100
            or (link[1] == 6100 and link[3] in {6102, 6103})
            or (link[3] == 7044 and link[4] == 9)
            or (link[3] == 7066 and link[4] == 1)
        )
    ]
    if first_image_link is not None:
        workflow["links"].append(
            [first_image_link[0], first_image_link[1], first_image_link[2], 6100, 3, "IMAGE"]
        )

    gemma = nodes[6102]
    gemma["inputs"] = [
        item for item in gemma["inputs"] if item["name"] != "reference_manifest"
    ]
    manifest_slot = len(gemma["inputs"])
    gemma["inputs"].append(
        {"name": "reference_manifest", "type": "STRING", "link": None}
    )

    selector = nodes[6103]
    selector["inputs"] = [
        item for item in selector["inputs"] if item["name"] != "reference_count"
    ]
    reference_count_slot = len(selector["inputs"])
    selector["inputs"].append(
        {"name": "reference_count", "type": "INT", "link": None}
    )
    selector["title"] = "ZiT composes contact sheet; select clearest generated variation"

    workflow["links"].extend(
        [
            [18100, 6100, 0, 6102, next(i for i, x in enumerate(gemma["inputs"]) if x["name"] == "input_image"), "IMAGE"],
            [18101, 6100, 2, 6102, next(i for i, x in enumerate(gemma["inputs"]) if x["name"] == "has_input_image"), "BOOLEAN"],
            [18102, 6100, 3, 6102, manifest_slot, "STRING"],
            [18103, 6100, 1, 6103, next(i for i, x in enumerate(selector["inputs"]) if x["name"] == "original_image"), "IMAGE"],
            [18104, 6100, 2, 6103, next(i for i, x in enumerate(selector["inputs"]) if x["name"] == "has_input_image"), "BOOLEAN"],
            [18105, 6100, 4, 6103, reference_count_slot, "INT"],
            [18106, 6100, 5, 7044, next(i for i, x in enumerate(nodes[7044]["inputs"]) if x["name"] == "denoise"), "FLOAT"],
            [18107, 6100, 6, 7066, next(i for i, x in enumerate(nodes[7066]["inputs"]) if x["name"] == "amount"), "INT"],
        ]
    )
    workflow.setdefault("extra", {})["mirage_reference_mode"] = "role-aware-contact-sheet-zit"
    _rebuild_links(workflow)
    workflow["last_node_id"] = max(nodes)
    workflow["last_link_id"] = max(link[0] for link in workflow["links"])
    return workflow


def main() -> None:
    parser = argparse.ArgumentParser(description="Upgrade MIRAGE Ref2V to role-aware contact sheets")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    workflow = json.loads(args.input.read_text(encoding="utf-8"))
    upgraded = upgrade(workflow)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(upgraded, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
