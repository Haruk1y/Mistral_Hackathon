# Kotone Catalog (Fixed)

Last updated: 2026-02-28

## 1. Source of truth

- Active Kotone catalog is fixed in `lib/catalog.ts` (`CATALOG_PARTS`).
- Training/eval/game UI should reference this only.

## 2. Current fixed counts

- Style: 8
- Instrument: 14
- Mood: 10
- Gimmick: 13
- Total: 45

## 3. Removed duplicate groups (fixed decision)

- Lute family: keep `Sun Lute` only
- Mandolin pair: keep `Round Mandolin` only
- Recorder pair: keep `Tall Recorder` only
- Crowd ambience pair: keep `Chatty Crowd` only
- Insect ambience pair: keep `Cricket Pulse` only
- Moonwind pair: keep `Moonwind Spark` only
- Rise/wind pair: keep `Filter Rise` only
- Whisper stereo pair: keep `Whisper Left` only

## 4. Notes

- Previous ideation-only entries are intentionally removed from active docs.
- If a new Kotone is added, update `lib/catalog.ts` first, then regenerate FT data.
