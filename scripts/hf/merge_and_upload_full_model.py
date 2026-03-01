#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.6.0",
#   "transformers @ git+https://github.com/huggingface/transformers",
#   "peft>=0.18.1",
#   "huggingface-hub>=1.3.0",
#   "mistral-common>=1.8.6",
# ]
# ///

"""Merge a PEFT adapter into Ministral-3 language model and upload merged weights."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import HfApi
from peft import PeftModel
from transformers import AutoTokenizer, Mistral3ForConditionalGeneration


def default_output_repo(adapter_model_id: str) -> str:
    if "/" in adapter_model_id:
        namespace, name = adapter_model_id.split("/", 1)
        return f"{namespace}/{name}-full"
    return f"{adapter_model_id}-full"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-model-id",
        default="mistralai/Ministral-3-3B-Instruct-2512",
        help="Base model repo id.",
    )
    parser.add_argument(
        "--adapter-model-id",
        default="Haruk1y/atelier-kotone-ministral3b-ft-selfimprove1",
        help="Adapter model repo id.",
    )
    parser.add_argument(
        "--output-model-id",
        default=None,
        help="Destination full model repo id. Default: <adapter>-full",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/hf/merged_models",
        help="Local folder for merged model.",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
        help="Torch dtype for model load/merge.",
    )
    parser.add_argument(
        "--max-shard-size",
        default="5GB",
        help="Max shard size for save_pretrained.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create destination repo as private (if new).",
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Only save locally, skip Hub upload.",
    )
    parser.add_argument(
        "--commit-message",
        default="Upload merged full model (base + PEFT adapter)",
        help="Commit message for hub upload.",
    )
    return parser.parse_args()


def resolve_dtype(name: str):
    if name == "auto":
        return "auto"
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def ensure_peft_generation_compat(model: torch.nn.Module) -> None:
    # Some Transformers/PEFT combinations expect this method on the base module.
    if hasattr(model, "prepare_inputs_for_generation"):
        return

    def _prepare_inputs_for_generation(
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        prepared: dict[str, Any] = {"input_ids": input_ids}
        if attention_mask is not None:
            prepared["attention_mask"] = attention_mask
        prepared.update(kwargs)
        return prepared

    setattr(model, "prepare_inputs_for_generation", _prepare_inputs_for_generation)


def main() -> None:
    args = parse_args()
    output_model_id = args.output_model_id or default_output_repo(args.adapter_model_id)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    local_dir = Path(args.output_dir) / f"{output_model_id.replace('/', '__')}-{timestamp}"
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"[merge] base={args.base_model_id}")
    print(f"[merge] adapter={args.adapter_model_id}")
    print(f"[merge] output_repo={output_model_id}")
    print(f"[merge] local_dir={local_dir}")

    model = Mistral3ForConditionalGeneration.from_pretrained(
        args.base_model_id,
        torch_dtype=resolve_dtype(args.dtype),
        attn_implementation="eager",
    )
    ensure_peft_generation_compat(model.model.language_model)

    model.model.language_model = PeftModel.from_pretrained(
        model.model.language_model,
        args.adapter_model_id,
        is_trainable=False,
    )

    print("[merge] applying LoRA weights into base language model")
    merged_language_model = model.model.language_model.merge_and_unload()

    # Save merged language model weights only (the adapter was trained on this module).
    print("[save] writing merged language model weights")
    merged_language_model.save_pretrained(
        str(local_dir),
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_id)
        tokenizer.save_pretrained(str(local_dir))
        print("[save] tokenizer saved")
    except Exception as error:  # noqa: BLE001
        raise RuntimeError("failed to save tokenizer") from error

    readme_path = local_dir / "README.md"
    readme_path.write_text(
        "\n".join(
            [
                f"# {output_model_id}",
                "",
                "Merged full-weights model generated from:",
                f"- Base model: `{args.base_model_id}`",
                f"- Adapter model: `{args.adapter_model_id}`",
                "",
                "This repo contains merged language-model weights (no PEFT runtime required).",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    if args.no_upload:
        print("[done] upload skipped (--no-upload)")
        return

    api = HfApi()
    api.create_repo(output_model_id, repo_type="model", private=args.private, exist_ok=True)
    print("[upload] uploading merged model to Hugging Face Hub")
    api.upload_folder(
        repo_id=output_model_id,
        repo_type="model",
        folder_path=str(local_dir),
        commit_message=args.commit_message,
    )
    print(f"[done] https://huggingface.co/{output_model_id}")


if __name__ == "__main__":
    main()
