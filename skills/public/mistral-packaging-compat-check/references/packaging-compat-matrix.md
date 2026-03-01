# Mistral Packaging Compatibility Matrix

## Artifact Types

1. Adapter repo
- Contains LoRA adapter files.
- Must be loaded with base model + PEFT adapter attachment.

2. Merged full repo
- Contains merged model weights.
- Must be loaded directly as a standalone model.

## Required Runtime Alignment

- Base model id must match the one used during adapter training.
- Tokenizer and chat template must match the serving path.
- For Mistral runtime compatibility, preserve metadata files when distributing merged models:
- `tekken.json`
- `params.json`
- `chat_template.jinja`
- `generation_config.json`
- `processor_config.json` when present in base repo

## Common Mismatch Patterns

1. Adapter loaded as full model
- Symptom: load errors or missing expected modules.
- Fix: switch to base+adapter loading path.

2. Full model served with adapter path assumptions
- Symptom: PEFT wrapper errors or wrong module path.
- Fix: load merged full model directly.

3. Missing metadata after merge
- Symptom: tokenizer/backend mismatch, prompt formatting drift.
- Fix: copy required metadata from base model into merged repo before upload.

4. PEFT generation compatibility bug
- Symptom: generation-time method missing on wrapped model.
- Fix: ensure `prepare_inputs_for_generation` compatibility before PEFT wrapping.

## Validation Steps Before Publishing

1. Run smoke inference in the intended mode (adapter or full).
2. Use training-aligned prompt text.
3. Validate strict JSON parse and key/range constraints.
4. Confirm target repo files exist and are readable from Hub.

## Merge Guidance

- Merge only when deployment needs a single standalone repo.
- Treat merge as packaging/distribution step, not a quality optimization step.
