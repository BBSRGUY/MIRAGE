from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from typing import Any

from .composer import ComposedReference, ReferencePipelineConfig


_INGREDIENTS_TEMPLATE = (
    "custom_nodes/ComfyUI-LTXVideo/example_workflows/2.3/"
    "LTX-2.3_ICLoRA_Ingredients_Single_Stage_Distilled.json"
)
_LTX25_LOADER_TEMPLATE = "user/default/workflows/LTX 2.5 T 2 V.json"
_ZIT_IMAGE_TEMPLATE = "user/default/workflows/ZiT_img2img_Updated_photorealism.json"


def build_comfy_workflow(
    config: ReferencePipelineConfig,
    composed: ComposedReference,
    output_path: str | Path,
) -> dict[str, Any]:
    """Adapt the local IC-LoRA graph to LTX-2.5 plus the MIRAGE EditAnything node."""
    comfy_root = Path(config.runtime.comfy_root).expanduser().resolve()
    models_root = Path(config.runtime.models_root).expanduser().resolve()
    ingredient_path = comfy_root / _INGREDIENTS_TEMPLATE
    ltx25_path = comfy_root / _LTX25_LOADER_TEMPLATE
    zit_path = comfy_root / _ZIT_IMAGE_TEMPLATE
    if not ingredient_path.is_file():
        raise FileNotFoundError(f"ComfyUI-LTXVideo Ingredients workflow not found: {ingredient_path}")
    if not ltx25_path.is_file():
        raise FileNotFoundError(f"local LTX-2.5 workflow not found: {ltx25_path}")
    if not zit_path.is_file():
        raise FileNotFoundError(f"local ZiT img2img workflow not found: {zit_path}")

    workflow = json.loads(ingredient_path.read_text(encoding="utf-8"))
    ltx25 = json.loads(ltx25_path.read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in workflow["nodes"]}
    source_nodes = {node["id"]: node for node in ltx25["nodes"]}
    _expect(nodes, 3940, "CheckpointLoaderSimple")
    _expect(nodes, 4922, "LoraLoaderModelOnly")
    _expect(nodes, 5011, "LTXICLoRALoaderModelOnly")
    _expect(nodes, 5012, "LTXAddVideoICLoRAGuide")
    _expect(nodes, 5023, "LTXAVTextEncoderLoader")
    _expect(source_nodes, 455, "DiffusionModelLoaderKJ")
    _expect(source_nodes, 442, "CLIPLoader")
    _expect(source_nodes, 440, "VAELoader")
    _expect(source_nodes, 441, "VAELoader")

    model_node = _loader_clone(source_nodes[455], nodes[3940], 3940)
    model_node["widgets_values"][0] = _model_name(
        config.runtime.transformer_path, models_root / "diffusion_models"
    )
    model_node["outputs"][0]["links"] = [13217]

    clip_node = _loader_clone(source_nodes[442], nodes[5023], 5023)
    clip_node["widgets_values"][0] = _model_name(
        config.runtime.text_encoder_path, models_root / "text_encoders"
    )
    clip_node["outputs"][0]["links"] = [13459, 13460]

    video_vae_node = copy.deepcopy(source_nodes[440])
    video_vae_node.update(
        {
            "id": 6001,
            "pos": [nodes[3940]["pos"][0], nodes[3940]["pos"][1] + 130],
            "order": nodes[3940].get("order", 0) + 1,
        }
    )
    video_vae_node["widgets_values"][0] = _model_name(
        config.runtime.video_vae_path, models_root / "vae"
    )
    video_vae_node["outputs"][0]["links"] = [13279, 13405, 13543]

    audio_node = _loader_clone(source_nodes[441], nodes[4010], 4010)
    audio_node["widgets_values"][0] = _model_name(
        config.runtime.audio_vae_path, models_root / "vae"
    )
    audio_node["outputs"][0]["links"] = nodes[4010]["outputs"][0]["links"]

    workflow["nodes"] = [
        node for node in workflow["nodes"] if node["id"] not in {3940, 4922, 5023, 4010}
    ]
    workflow["nodes"].extend([model_node, clip_node, video_vae_node, audio_node])
    nodes = {node["id"]: node for node in workflow["nodes"]}

    nodes[2483]["widgets_values"] = [""]
    nodes[2612]["widgets_values"] = [""]
    reference_lora = Path(config.runtime.reference_lora_path).expanduser().resolve()
    lora_roots = [models_root / "loras", reference_lora.parent]
    lora_name = next(
        (
            reference_lora.relative_to(root.resolve()).as_posix()
            for root in lora_roots
            if reference_lora.is_relative_to(root.resolve())
        ),
        reference_lora.name,
    )
    nodes[5011].update(
        {
            "type": "MIRAGELTXEditAnythingReference",
            "size": [560, 300],
            "inputs": [
                {"name": "model", "type": "MODEL", "link": 13217},
                {"name": "vae", "type": "VAE", "link": 16000},
            ],
            "outputs": [
                {"name": "model", "type": "MODEL", "links": [13401]},
                {"name": "reference_latent", "type": "LATENT", "links": []},
            ],
            "properties": {
                "Node name for S&R": "MIRAGELTXEditAnythingReference",
                "mirage_adapter": "wan2gp-editanything-local-port",
            },
            "widgets_values": [
                lora_name,
                str(Path(config.runtime.reference_module_path).expanduser().resolve()),
                config.runtime.lora_strength,
                0.01,
                0.25,
                2.0,
                12,
                35,
            ],
        }
    )
    nodes[5098]["widgets_values"] = config.frame_rate
    nodes[4832]["widgets_values"][0] = config.seed

    # The default graph is genuinely text-only. A user may add one Load Image
    # node and connect its IMAGE output to the optional image sockets on the
    # Gemma director, EditAnything adapter, and reference guide.
    removed_image_nodes = {2004, 3159, 5019, 5067, 5068, 5069, 5072, 5093, 5095, 5100}
    workflow["nodes"] = [node for node in workflow["nodes"] if node["id"] not in removed_image_nodes]
    nodes = {node["id"]: node for node in workflow["nodes"]}

    nodes[3059]["inputs"][0]["link"] = None
    nodes[3059]["inputs"][1]["link"] = None
    nodes[3059]["widgets_values"] = [config.width, config.height, config.num_frames, 1]
    nodes[5012].update(
        {
            "type": "MIRAGELTXOptionalReferenceGuide",
            "size": [390, 270],
            "inputs": [
                {"name": "positive", "type": "CONDITIONING", "link": 13403},
                {"name": "negative", "type": "CONDITIONING", "link": 13404},
                {"name": "vae", "type": "VAE", "link": 13405},
                {"name": "latent", "type": "LATENT", "link": 13402},
            ],
            "properties": {
                "Node name for S&R": "MIRAGELTXOptionalReferenceGuide",
                "mirage_mode": "optional-image-pass-through",
            },
            "widgets_values": [0, 1.0, 1.0, "disabled", False, 256, 64],
            "title": "Optional reference image (unconnected = text-only)",
        }
    )

    prompt_node = {
        "id": 6101,
        "type": "MIRAGEPromptOrSurprise",
        "pos": [-4200, 2920],
        "size": [560, 180],
        "flags": {},
        "order": 0,
        "mode": 0,
        "inputs": [],
        "outputs": [
            {"name": "prompt_for_gemma4", "type": "STRING", "links": [16001]}
        ],
        "properties": {"Node name for S&R": "MIRAGEPromptOrSurprise"},
        "widgets_values": ["", config.seed],
        "title": "Your prompt (leave blank for a Gemma4 surprise)",
    }
    input_image_node = {
        "id": 6100,
        "type": "MIRAGEMultiReferenceContactSheet",
        "pos": [-4200, 2630],
        "size": [620, 360],
        "flags": {},
        "order": 0,
        "mode": 0,
        "inputs": [
            {"name": "sheet_width", "type": "INT", "widget": {"name": "sheet_width"}, "link": None},
            {"name": "sheet_height", "type": "INT", "widget": {"name": "sheet_height"}, "link": None},
            {"name": "variation_count", "type": "INT", "widget": {"name": "variation_count"}, "link": None},
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
        "properties": {"Node name for S&R": "MIRAGEMultiReferenceContactSheet"},
        "widgets_values": [1024, 576, 3],
        "title": "Optional references: characters / wardrobe / environment / details / style",
    }
    gemma_node = {
        "id": 6102,
        "type": "MIRAGEGemmaDirector",
        "pos": [-3500, 2820],
        "size": [620, 850],
        "flags": {},
        "order": 1,
        "mode": 0,
        "inputs": [
            {"name": "prompt", "type": "STRING", "widget": {"name": "prompt"}, "link": 16001},
            {"name": "input_image", "type": "IMAGE", "link": 16004},
            {"name": "has_input_image", "type": "BOOLEAN", "link": 16005},
            {"name": "reference_manifest", "type": "STRING", "link": 16015},
            {"name": "fps", "type": "INT", "widget": {"name": "fps"}, "link": 16204},
        ],
        "outputs": [
            {"name": "DIRECTOR_MANIFEST_JSON", "type": "STRING", "links": []},
            {"name": "FIRST_FRAME_PROMPT", "type": "STRING", "links": []},
            {"name": "VIDEO_PROMPT", "type": "STRING", "links": [16002]},
            {"name": "NEGATIVE_PROMPT", "type": "STRING", "links": [16003]},
            {"name": "NEEDS_START_FRAME", "type": "BOOLEAN", "links": []},
            {"name": "MODE", "type": "STRING", "links": []},
            {"name": "DURATION_SECONDS", "type": "INT", "links": [16200]},
            {"name": "SUMMARY", "type": "STRING", "links": []},
        ],
        "properties": {"Node name for S&R": "MIRAGEGemmaDirector"},
        "widgets_values": [
            "",
            "http://192.168.1.107:7788/v1/chat/completions",
            "unsloth/gemma-4-12B-it-GGUF:Q4_K_M",
            "auto (let Gemma decide)",
            "auto (let Gemma decide)",
            "auto (let Gemma decide)",
            "auto (let Gemma decide)",
            "auto (let Gemma decide)",
            "auto (let Gemma decide)",
            "auto (let Gemma decide)",
            "none - no dialogue",
            "single continuous shot",
            "auto (Gemma decides)",
            "16:9",
            max(1, round(config.num_frames / config.frame_rate)),
            config.frame_rate,
            max(config.width, config.height),
            "",
            0.7,
            8192,
            600,
            2,
            1024,
            "raise the error",
            config.seed,
        ],
        "title": "Gemma4 director (sees the real input image)",
    }
    timing_node = {
        "id": 6110,
        "type": "MIRAGELTXFrameCount",
        "pos": [-2760, 3040],
        "size": [390, 120],
        "flags": {},
        "order": 9,
        "mode": 0,
        "inputs": [
            {"name": "duration_seconds", "type": "INT", "link": 16200},
            {"name": "fps", "type": "INT", "link": 16201},
        ],
        "outputs": [
            {"name": "LTX_FRAME_COUNT", "type": "INT", "links": [16202, 16203]},
        ],
        "properties": {"Node name for S&R": "MIRAGELTXFrameCount"},
        "widgets_values": [],
        "title": "Duration -> actual LTX video/audio length (8n+1)",
    }
    selector_node = {
        "id": 6103,
        "type": "MIRAGEClearStartFrame",
        "pos": [-1400, 2600],
        "size": [520, 260],
        "flags": {},
        "order": 30,
        "mode": 0,
        "inputs": [
            {"name": "original_image", "type": "IMAGE", "link": 16007},
            {"name": "has_input_image", "type": "BOOLEAN", "link": 16008},
            {"name": "needs_start_frame", "type": "BOOLEAN", "link": 16006},
            {"name": "reference_count", "type": "INT", "link": 16016},
            {"name": "edited_image", "type": "IMAGE", "link": 16117},
            {"name": "generated_image", "type": "IMAGE", "link": 16121},
        ],
        "outputs": [
            {"name": "start_frame", "type": "IMAGE", "links": []},
            {"name": "has_frame", "type": "BOOLEAN", "links": []},
            {"name": "selected_source", "type": "STRING", "links": []},
            {"name": "comparison", "type": "STRING", "links": []},
        ],
        "properties": {"Node name for S&R": "MIRAGEClearStartFrame"},
        "widgets_values": [0.84, 0.05],
        "title": "Compare original vs ZiT edit; preserve the clearest faithful frame",
    }
    selected_preview_node = {
        "id": 6104,
        "type": "PreviewImage",
        "pos": [-820, 2490],
        "size": [360, 420],
        "flags": {},
        "order": 31,
        "mode": 0,
        "inputs": [{"name": "images", "type": "IMAGE", "link": 16013}],
        "outputs": [],
        "properties": {"Node name for S&R": "PreviewImage"},
        "widgets_values": [],
        "title": "Selected start frame sent to LTX",
    }
    comparison_preview_node = {
        "id": 6105,
        "type": "PreviewAny",
        "pos": [-1400, 2910],
        "size": [520, 140],
        "flags": {},
        "order": 32,
        "mode": 0,
        "inputs": [{"name": "source", "type": "*", "link": 16014}],
        "outputs": [{"name": "STRING", "type": "STRING", "links": []}],
        "properties": {"Node name for S&R": "PreviewAny"},
        "widgets_values": [None, None, None],
        "title": "Start-frame fidelity / clarity decision",
    }
    workflow["nodes"].extend(
        [
            input_image_node,
            prompt_node,
            gemma_node,
            timing_node,
            selector_node,
            selected_preview_node,
            comparison_preview_node,
        ]
    )
    _add_zit_frame_candidates(workflow, zit_path, config)
    nodes = {node["id"]: node for node in workflow["nodes"]}
    nodes[2483]["inputs"].append(
        {"name": "text", "type": "STRING", "widget": {"name": "text"}, "link": 16002}
    )
    nodes[2612]["inputs"].append(
        {"name": "text", "type": "STRING", "widget": {"name": "text"}, "link": 16003}
    )

    patched_links = []
    for link in workflow["links"]:
        if link[0] in {13400, 13406}:
            continue
        if link[0] == 13217:
            link = [13217, 3940, 0, 5011, 0, "MODEL"]
        elif link[1] == 3940 and link[2] == 2:
            link = [link[0], 6001, 0, link[3], link[4], link[5]]
        patched_links.append(link)
    workflow["links"] = patched_links
    workflow["links"].extend(
        [
            [16000, 6001, 0, 5011, 1, "VAE"],
            [16001, 6101, 0, 6102, 0, "STRING"],
            [16002, 6102, 2, 2483, 1, "STRING"],
            [16003, 6102, 3, 2612, 1, "STRING"],
            [16004, 6100, 0, 6102, 1, "IMAGE"],
            [16005, 6100, 2, 6102, 2, "BOOLEAN"],
            [16006, 6102, 4, 6103, 2, "BOOLEAN"],
            [16007, 6100, 1, 6103, 0, "IMAGE"],
            [16008, 6100, 2, 6103, 1, "BOOLEAN"],
            [16009, 6103, 0, 5011, 2, "IMAGE"],
            [16010, 6103, 1, 5011, 3, "BOOLEAN"],
            [16011, 6103, 0, 5012, 4, "IMAGE"],
            [16012, 6103, 1, 5012, 5, "BOOLEAN"],
            [16013, 6103, 0, 6104, 0, "IMAGE"],
            [16014, 6103, 3, 6105, 0, "*"],
            [16015, 6100, 3, 6102, 3, "STRING"],
            [16016, 6100, 4, 6103, 3, "INT"],
            [16200, 6102, 6, 6110, 0, "INT"],
            [16201, 5099, 0, 6110, 1, "INT"],
            [16202, 6110, 0, 3059, 2, "INT"],
            [16203, 6110, 0, 3980, 1, "INT"],
            [16204, 5099, 0, 6102, 4, "INT"],
        ]
    )
    nodes = {node["id"]: node for node in workflow["nodes"]}
    nodes[5011]["inputs"].extend(
        [
            {"name": "reference_sheet", "type": "IMAGE", "link": 16009},
            {"name": "has_reference", "type": "BOOLEAN", "link": 16010},
        ]
    )
    nodes[5012]["inputs"].extend(
        [
            {"name": "image", "type": "IMAGE", "link": 16011},
            {"name": "has_reference", "type": "BOOLEAN", "link": 16012},
        ]
    )
    nodes[6001]["outputs"][0]["links"].append(16000)
    # The text-only base latent now feeds the optional guide directly.
    workflow["links"].append([13402, 3059, 0, 5012, 3, "LATENT"])
    nodes[3059]["outputs"][0]["links"] = [13402]
    # Some published workflow templates retain UI-only links to nodes that are no
    # longer serialized. Remove those dangling links so API validation is strict.
    node_ids = {node["id"] for node in workflow["nodes"]}
    workflow["links"] = [
        link for link in workflow["links"] if link[1] in node_ids and link[3] in node_ids
    ]
    _rebuild_serialized_links(workflow)
    # Reroutes are optional canvas decoration. The upstream template contains
    # reroutes for links replaced above; stale reroute parents can make the
    # frontend abort canvas restoration and display an empty graph.
    workflow.setdefault("extra", {}).pop("reroutes", None)
    workflow["extra"].pop("linkExtensions", None)
    workflow["last_node_id"] = max(node["id"] for node in workflow["nodes"])
    workflow["last_link_id"] = max(link[0] for link in workflow["links"])
    workflow.setdefault("extra", {})["mirage_reference_manifest"] = str(
        composed.manifest_path.resolve()
    )
    workflow["extra"]["mirage_backend"] = "ltx-2.5-nvfp4-editanything-local-port"

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(workflow, indent=2), encoding="utf-8")
    return workflow


def deploy_comfy_assets(
    config: ReferencePipelineConfig,
    composed: ComposedReference,
    workflow_path: str | Path,
) -> dict[str, str]:
    comfy_root = Path(config.runtime.comfy_root).expanduser().resolve()
    input_root = Path(config.runtime.input_root).expanduser().resolve()
    input_target = input_root / Path(config.runtime.comfy_input_name)
    workflow_target = comfy_root / "user" / "default" / "workflows" / "MIRAGE LTX2.5 Ref2V.json"
    input_target.parent.mkdir(parents=True, exist_ok=True)
    workflow_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(composed.sheet_path, input_target)
    shutil.copy2(workflow_path, workflow_target)
    repo_root = Path(__file__).resolve().parents[3]
    node_source = repo_root / "comfy_nodes" / "mirage_ltx_reference" / "__init__.py"
    node_target = comfy_root / "custom_nodes" / "mirage_ltx_reference" / "__init__.py"
    node_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(node_source, node_target)
    return {
        "input_sheet": str(input_target),
        "workflow": str(workflow_target),
        "custom_node": str(node_target),
    }


def _add_zit_frame_candidates(
    workflow: dict[str, Any],
    template_path: Path,
    config: ReferencePipelineConfig,
) -> None:
    """Add lazy ZiT img2img and text-to-image candidates using the user's local graph."""
    template = json.loads(template_path.read_text(encoding="utf-8"))
    source = {node["id"]: node for node in template["nodes"]}
    expected = {
        39: "CLIPLoader",
        40: "VAELoader",
        41: "EmptySD3LatentImage",
        42: "ConditioningZeroOut",
        43: "VAEDecode",
        44: "KSampler",
        45: "CLIPTextEncode",
        46: "UNETLoader",
        47: "ModelSamplingAuraFlow",
        57: "VAEEncode",
        65: "ImageScaleToMaxDimension",
        66: "RepeatLatentBatch",
    }
    for node_id, node_type in expected.items():
        _expect(source, node_id, node_type)

    id_map = {node_id: 7000 + node_id for node_id in expected}
    cloned: dict[int, dict[str, Any]] = {}
    for old_id, new_id in id_map.items():
        node = copy.deepcopy(source[old_id])
        node["id"] = new_id
        node["mode"] = 0
        node["order"] = node.get("order", 0) + 40
        node["pos"] = [node["pos"][0] - 2900, node["pos"][1] + 1800]
        for item in node.get("inputs", []):
            item["link"] = None
        for item in node.get("outputs", []):
            item["links"] = []
        cloned[new_id] = node

    cloned[7039]["widgets_values"] = ["qwen_3_4b_fp8_mixed.safetensors", "lumina2", "default"]
    cloned[7040]["widgets_values"] = ["ae.safetensors"]
    cloned[7041]["widgets_values"] = [config.width, config.height, 1]
    cloned[7045]["widgets_values"] = [""]
    cloned[7046]["widgets_values"] = ["z_image_turbo_int8_convrot.safetensors", "default"]
    cloned[7047]["widgets_values"] = [3]
    cloned[7065]["widgets_values"] = ["area", max(config.width, config.height)]
    cloned[7066]["widgets_values"] = [1]
    # Preservation edit: denoise=0.30. The source workflow used 1.0, which
    # regenerated the picture and discarded the supplied reference structure.
    cloned[7044]["widgets_values"] = [config.seed, "fixed", 9, 1.0, "euler", "simple", 0.30]
    cloned[7044]["title"] = "ZiT img2img preservation edit (denoise 0.30)"

    generated_sampler = copy.deepcopy(cloned[7044])
    generated_sampler.update(
        {
            "id": 7048,
            "pos": [cloned[7044]["pos"][0], cloned[7044]["pos"][1] + 540],
            "title": "ZiT generated first frame (only when Gemma requests it)",
            "widgets_values": [config.seed + 1, "fixed", 9, 1.0, "euler", "simple", 1.0],
        }
    )
    generated_decode = copy.deepcopy(cloned[7043])
    generated_decode.update(
        {
            "id": 7049,
            "pos": [cloned[7043]["pos"][0], cloned[7043]["pos"][1] + 540],
            "title": "Decode Gemma-requested ZiT first frame",
        }
    )
    cloned[7048] = generated_sampler
    cloned[7049] = generated_decode
    workflow["nodes"].extend(cloned.values())

    workflow["links"].extend(
        [
            [16100, 7046, 0, 7047, 0, "MODEL"],
            [16101, 7047, 0, 7044, 0, "MODEL"],
            [16102, 7047, 0, 7048, 0, "MODEL"],
            [16103, 7039, 0, 7045, 0, "CLIP"],
            [16104, 6102, 1, 7045, 1, "STRING"],
            [16105, 7045, 0, 7042, 0, "CONDITIONING"],
            [16106, 7045, 0, 7044, 1, "CONDITIONING"],
            [16107, 7045, 0, 7048, 1, "CONDITIONING"],
            [16108, 7042, 0, 7044, 2, "CONDITIONING"],
            [16109, 7042, 0, 7048, 2, "CONDITIONING"],
            [16110, 7040, 0, 7057, 1, "VAE"],
            [16111, 6100, 0, 7065, 0, "IMAGE"],
            [16112, 7065, 0, 7057, 0, "IMAGE"],
            [16113, 7057, 0, 7066, 0, "LATENT"],
            [16114, 7066, 0, 7044, 3, "LATENT"],
            [16115, 7044, 0, 7043, 0, "LATENT"],
            [16116, 7040, 0, 7043, 1, "VAE"],
            [16117, 7043, 0, 6103, 4, "IMAGE"],
            [16118, 7041, 0, 7048, 3, "LATENT"],
            [16119, 7048, 0, 7049, 0, "LATENT"],
            [16120, 7040, 0, 7049, 1, "VAE"],
            [16121, 7049, 0, 6103, 5, "IMAGE"],
            [16122, 6100, 5, 7044, 9, "FLOAT"],
            [16123, 6100, 6, 7066, 1, "INT"],
        ]
    )
    workflow.setdefault("groups", []).append(
        {
            "id": 6100,
            "title": "MIRAGE | Gemma-directed ZiT start-frame candidates (lazy)",
            "bounding": [-2810, 1980, 1660, 1720],
            "color": "#6b4f82",
            "flags": {},
        }
    )


def _rebuild_serialized_links(workflow: dict[str, Any]) -> None:
    """Make node socket metadata agree exactly with the canonical links array."""
    nodes = {node["id"]: node for node in workflow["nodes"]}
    for node in workflow["nodes"]:
        for item in node.get("inputs", []):
            item["link"] = None
        for item in node.get("outputs", []):
            item["links"] = []
    for link_id, source_id, source_slot, target_id, target_slot, _link_type in workflow["links"]:
        source_outputs = nodes[source_id].get("outputs", [])
        target_inputs = nodes[target_id].get("inputs", [])
        if source_slot >= len(source_outputs):
            raise ValueError(f"link {link_id} has invalid source slot {source_id}:{source_slot}")
        if target_slot >= len(target_inputs):
            raise ValueError(f"link {link_id} has invalid target slot {target_id}:{target_slot}")
        source_outputs[source_slot]["links"].append(link_id)
        target_inputs[target_slot]["link"] = link_id


def _loader_clone(source: dict[str, Any], target: dict[str, Any], node_id: int) -> dict[str, Any]:
    result = copy.deepcopy(source)
    result.update(
        {
            "id": node_id,
            "pos": target.get("pos", source.get("pos")),
            "size": target.get("size", source.get("size")),
            "order": target.get("order", source.get("order", 0)),
            "mode": target.get("mode", 0),
        }
    )
    return result


def _model_name(raw_path: str, category_root: Path) -> str:
    path = Path(raw_path).expanduser().resolve()
    try:
        return path.relative_to(category_root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"model must be inside {category_root}: {path}") from error


def _expect(nodes: dict[int, dict[str, Any]], node_id: int, expected: str) -> None:
    actual = nodes.get(node_id, {}).get("type")
    if actual != expected:
        raise ValueError(f"workflow node {node_id} changed: expected {expected}, found {actual}")
