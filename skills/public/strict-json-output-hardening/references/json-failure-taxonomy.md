# JSON Failure Taxonomy

## Contract

Required output shape:

`{"energy":int,"warmth":int,"brightness":int,"acousticness":int,"complexity":int,"nostalgia":int}`

Rules:

- Exactly one JSON object.
- No extra keys, no wrapper key.
- All values are integers in `0..10`.
- No text before `{` or after `}`.

## Failure Categories

1. `no_json_start`
- Model did not emit any JSON start token.
- Typical fix: stronger "JSON only" instruction and deterministic decode.

2. `unterminated_json`
- Open brace without valid close.
- Typical fix: reduce `max_new_tokens`, enforce EOS handling, shorten prompt tail.

3. `json_decode_error:*`
- Emitted malformed JSON syntax.
- Typical fix: prompt variant testing and hard-case retraining rows.

4. `missing_key:*` or schema mismatch
- Some required keys absent.
- Typical fix: explicit key list in prompt, train with exact completion schema only.

5. `extra_keys:*`
- Unexpected keys added.
- Typical fix: "exactly these keys" language and completion-only supervision.

6. `*_not_int` / type mismatch
- Value emitted as float/string/null.
- Typical fix: add examples emphasizing integer-only values.

7. `*_out_of_range`
- Values outside `0..10`.
- Typical fix: add out-of-range negatives to hard-case set and clamp validation in eval.

8. trailing text after JSON
- Valid object plus explanation or repeated object.
- Typical fix: explicit stop rule and stricter decode limits.

## Diagnosis Flow

1. Run parser-based audit on raw completions.
2. Build category histogram.
3. Select top two categories by count.
4. Apply smallest fix set that targets those categories.
5. Re-run same eval set and compare category deltas.

## Promotion Gates

Minimum gate for strict-format tasks:

- `json_valid_rate` must not regress.
- Dominant failure category count must decrease.
- Quality metrics (`mae_raw` etc.) are only compared after format gate passes.
