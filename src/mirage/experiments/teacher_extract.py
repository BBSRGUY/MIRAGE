from __future__ import annotations

from typing import Any

from ..datasets import FeatureStore, load_prompt_split
from ..m2_config import M2Config
from ..teacher import get_teacher_adapter
from ..teacher.extraction import TeacherExtractor


def run_teacher_extraction(config: M2Config) -> dict[str, Any]:
    prompts = load_prompt_split(
        config.data.prompts_file,
        config.data.num_train_prompts,
        config.data.num_eval_prompts,
        config.data.seed,
    )
    adapter = get_teacher_adapter(config.teacher.type, config.teacher)
    return TeacherExtractor(adapter, FeatureStore(config.output_dir), config).run(prompts)
