import asyncio
import json
import os
from typing import Dict, List

import weave
from datasets import load_dataset
from peft import AutoPeftModelForCausalLM, PeftModel
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer
from weave import Evaluation, Model

PROJECT_NAME = os.getenv("WEAVE_PROJECT", "atelier-kotone-ft")
BASE_MODEL_REPO = os.getenv(
    "EVAL_LOCAL_BASE_MODEL_ID",
    os.getenv("HF_BASE_MODEL_ID", "mistralai/Ministral-3-3B-Instruct-2512"),
)
ADAPTER_MODEL_REPO = os.getenv(
    "EVAL_LOCAL_ADAPTER_MODEL_ID",
    os.getenv(
        "EVAL_FINE_TUNED_MODEL_ID",
        os.getenv("HF_FT_OUTPUT_MODEL_ID", "uzumibi/atelier-kotone-ministral3b-ft"),
    ),
)
DATASET_NAME = os.getenv("EVAL_DATASET_REPO_ID", "Haruk1y/atelier-kotone-ft-request-hidden")
DATASET_SPLIT = os.getenv("EVAL_DATASET_SPLIT", "test")
NUM_SAMPLES = int(os.getenv("EVAL_DATASET_MAX_SAMPLES", "20")) or 20
TOLERANCE = 2.0
PARAM_MIN = 0.0
PARAM_MAX = 10.0
MAX_NEW_TOKENS = int(os.getenv("EVAL_LOCAL_MAX_NEW_TOKENS", "128"))
TEMPERATURE = 0.0

PROMPT = """
You estimate 6 hidden music parameters.,
Each value must be integer between 0 and 10.

- energy: silent(0) ↔ intense(10)
- warmth: mechanical(0) ↔ warm(10)
- brightness: dark(0) ↔ bright(10)
- acousticness: electronic(0) ↔ acoustic(10)
- complexity: simple(0) ↔ complex(10)
- nostalgia: futuristic(0) ↔ nostalgic(10)

Return JSON only with exactly these keys:
energy, warmth, brightness, acousticness, complexity, nostalgia
"""

AXES = ["energy", "warmth", "brightness", "acousticness", "complexity", "nostalgia"]


def clamp(value: float) -> float:
    return max(PARAM_MIN, min(PARAM_MAX, value))


def parse_target_hidden_params(value):
    payload = value
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, dict):
        return None

    vector = payload.get("vector")
    if not isinstance(vector, dict):
        return None

    out = {}
    for axis in AXES:
        axis_val = vector.get(axis)
        if axis_val is None:
            return None
        out[axis] = float(axis_val)
    return out


def load_eval_dataset() -> List[Dict]:
    hf_ds = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    rows = []
    for row in hf_ds:
        request_text = row.get("request_text")
        target_raw = row.get("target_hidden_params")
        if not isinstance(request_text, str):
            continue

        target_vector = parse_target_hidden_params(target_raw)
        if target_vector is None:
            continue

        rows.append({"request_text": request_text, "target": target_vector})
        if len(rows) >= NUM_SAMPLES:
            break
    return rows


@weave.op()
def mae_score(output: dict, target: dict) -> dict:
    pred = output.get("vector", {})
    abs_errors = {
        axis: abs(float(pred.get(axis, 0.0)) - float(target.get(axis, 0.0)))
        for axis in AXES
    }
    mae = sum(abs_errors.values()) / len(AXES)
    within_tolerance = all(err <= TOLERANCE for err in abs_errors.values())
    return {
        "mae": round(mae, 4),
        "within_tolerance_all": within_tolerance,
        "abs_errors": abs_errors,
    }


@weave.op()
def axis_match_rate(output: dict, target: dict) -> dict:
    pred = output.get("vector", {})
    matches = 0
    for axis in AXES:
        if abs(float(pred.get(axis, 0.0)) - float(target.get(axis, 0.0))) <= TOLERANCE:
            matches += 1
    return {"axis_match_rate": matches / len(AXES)}


def extract_json_object(text: str) -> dict:
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return {}


def load_base_model(repo_id: str):
    try:
        model = AutoModelForCausalLM.from_pretrained(
            repo_id,
            device_map="auto",
            trust_remote_code=True,
        )
        return model, "causal"
    except Exception:
        model = AutoModelForImageTextToText.from_pretrained(
            repo_id,
            device_map="auto",
            trust_remote_code=True,
        )
        return model, "image_text"

def summarize_lora_weights(model) -> dict:
    lora_tensors = 0
    nonzero_tensors = 0
    total_abs_sum = 0.0

    for name, param in model.named_parameters():
        if ".lora_A." not in name and ".lora_B." not in name:
            continue
        lora_tensors += 1
        tensor_sum = float(param.detach().float().abs().sum().item())
        total_abs_sum += tensor_sum
        if tensor_sum > 0.0:
            nonzero_tensors += 1

    return {
        "lora_param_tensors": lora_tensors,
        "nonzero_lora_tensors": nonzero_tensors,
        "lora_abs_sum": round(total_abs_sum, 6),
    }


class HFLocalModel(Model):
    repo_id: str = BASE_MODEL_REPO
    adapter_repo_id: str = ADAPTER_MODEL_REPO
    system_prompt: str = PROMPT

    def __init__(self, **data):
        super().__init__(**data)
        self._tokenizer = None
        self._processor = None
        self._adapter_load_mode = "none"
        self._adapter_summary = None

        tokenizer_repo = self.adapter_repo_id.strip() if self.adapter_repo_id and self.adapter_repo_id.strip() else self.repo_id
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_repo, trust_remote_code=True)
            if self._tokenizer.pad_token_id is None and self._tokenizer.eos_token is not None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
        except Exception:
            self._processor = AutoProcessor.from_pretrained(self.repo_id, trust_remote_code=True)
            self._tokenizer = getattr(self._processor, "tokenizer", None)

        adapter_id = self.adapter_repo_id.strip() if self.adapter_repo_id else ""
        if adapter_id:
            autopeft_error = None
            try:
                self._model = AutoPeftModelForCausalLM.from_pretrained(
                    adapter_id,
                    device_map="auto",
                    trust_remote_code=True,
                )
                self._adapter_load_mode = "autopeft"
            except Exception as error:
                autopeft_error = str(error)
                self._model, _ = load_base_model(self.repo_id)
                self._model = PeftModel.from_pretrained(
                    self._model,
                    adapter_id,
                    trust_remote_code=True,
                )
                self._adapter_load_mode = "peft_fallback"

            self._adapter_summary = summarize_lora_weights(self._model)
            self._adapter_summary["adapter_repo_id"] = adapter_id
            self._adapter_summary["load_mode"] = self._adapter_load_mode
            if autopeft_error:
                self._adapter_summary["autopeft_error"] = autopeft_error

            if self._adapter_summary["lora_param_tensors"] <= 0:
                raise RuntimeError("adapter_load_failed:no_lora_params_found")
            if self._adapter_summary["nonzero_lora_tensors"] <= 0:
                raise RuntimeError("adapter_load_failed:lora_params_all_zero")

            print("[adapter_load_summary]")
            print(json.dumps(self._adapter_summary, ensure_ascii=False))
        else:
            self._model, _ = load_base_model(self.repo_id)

        self._model.eval()

    @weave.op()
    def predict(self, request_text: str):
        user_prompt = f"{self.system_prompt}\n\nRequest text:\n{request_text}"

        if self._tokenizer is not None and hasattr(self._tokenizer, "apply_chat_template"):
            try:
                user_prompt = self._tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": "You output only strict JSON."},
                        {"role": "user", "content": user_prompt},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass

        try:
            model_device = next(self._model.parameters()).device
        except Exception:
            model_device = self._model.device

        if self._processor is not None:
            inputs = self._processor(text=user_prompt, return_tensors="pt").to(model_device)
            input_len = int(inputs["input_ids"].shape[-1]) if "input_ids" in inputs else 0
        else:
            inputs = self._tokenizer(user_prompt, return_tensors="pt").to(model_device)
            input_len = int(inputs["input_ids"].shape[-1])

        generate_kwargs = {
            "max_new_tokens": MAX_NEW_TOKENS,
            "do_sample": TEMPERATURE > 0,
            "temperature": TEMPERATURE if TEMPERATURE > 0 else None,
        }
        generate_kwargs = {k: v for k, v in generate_kwargs.items() if v is not None}

        outputs = self._model.generate(**inputs, **generate_kwargs)

        if self._tokenizer is not None:
            generated_ids = outputs[0][input_len:] if input_len > 0 else outputs[0]
            decoded = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
        elif self._processor is not None:
            decoded = self._processor.batch_decode(outputs, skip_special_tokens=True)[0]
        else:
            decoded = ""

        raw = extract_json_object(decoded)
        normalized = {}
        for axis in AXES:
            value = raw.get(axis, 0.0)
            try:
                value = float(value)
            except (TypeError, ValueError):
                value = 0.0
            normalized[axis] = round(clamp(value), 3)

        return {"vector": normalized, "raw_text": decoded}


async def run():
    weave.init(PROJECT_NAME)
    dataset = load_eval_dataset()
    if not dataset:
        raise ValueError("No valid rows loaded from dataset")

    evaluation = Evaluation(dataset=dataset, scorers=[mae_score, axis_match_rate])
    model_c = HFLocalModel(
        repo_id=BASE_MODEL_REPO,
        adapter_repo_id=ADAPTER_MODEL_REPO,
    )
    result = await evaluation.evaluate(model_c)

    print("=== Evaluation Result: MODEL_C (HF local model + LoRA adapter) ===")
    print(
        json.dumps(
            {
                "project": PROJECT_NAME,
                "base_model": BASE_MODEL_REPO,
                "adapter_model": ADAPTER_MODEL_REPO,
                "dataset": f"{DATASET_NAME}:{DATASET_SPLIT}",
                "num_samples": len(dataset),
                "adapter_load_summary": getattr(model_c, "_adapter_summary", None),
                "result": result,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    asyncio.run(run())
