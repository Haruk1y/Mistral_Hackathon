# Gemini asset generation

This folder provides local scripts to generate MVP assets via direct Gemini API.

## Prerequisites

- `GEMINI_API_KEY` is set in your shell or `.env.local`
- Node.js 18+ (for built-in `fetch`)
- `sharp` package (used by V2 master-sheet pipeline for PNG normalization and chroma-key alpha)

The script automatically loads `.env.local` and `.env` from the project root.

## Usage

```bash
node scripts/assets/generate-gemini-assets.mjs --list
node scripts/assets/generate-gemini-assets.mjs --preset=mvp
node scripts/assets/generate-gemini-assets.mjs --preset=mvp_master_sheets
node scripts/assets/generate-gemini-master-sheets-v2.mjs --preset=mvp_master_sheets
```

By default, real generation asks for confirmation before running.  
For non-interactive execution, pass `--yes`.

Useful options:

- `--dry-run`: prints prompts and output paths only
- `--force`: overwrite existing files
- `--yes`: skip confirmation prompt
- `--only=spr_player_idle,spr_player_walk`: generate specific asset IDs only
- `--preset=extended`: includes additional background assets
- `--preset=mvp_master_sheets`: minimum-sheet mode (currently 3 mega sheets)
- `--preset=legacy_mvp_master_sheets`: previous 11-sheet MVP structure
- `--preset=slot_parts_max`: generate high-density slot part sheets (many icons near sheet capacity)
- `--preset=rich_master_sheets`: legacy 11-sheet set + max slot-parts sheets in one run
- `--model=nano-banana-pro-preview`: override model ID

## V2 direct-Gemini master-sheet pipeline

V2 command:

```bash
npm run assets:generate:master-sheets:v2 -- --yes
```

Key options:

- `--preset=<name>`: default is `mvp_master_sheets`
- `--only=<id1,id2,...>`: run a subset
- `--report-only`: evaluate existing outputs and refresh reports/manifest
- `--max-retries=<n>`: retry generation when quality gate fails (default 2)
- `--image-size=<size>`: Gemini image size hint (`1K` / `2K` / `4K` など)
- `--image-aspect-ratio=<ratio>`: Gemini image aspect ratio hint (`16:9` など)
- `--dry-run`: print plan only

Environment overrides:

- `GEMINI_IMAGE_SIZE`
- `GEMINI_IMAGE_ASPECT_RATIO`

If both CLI and spec define these values, CLI wins.

Master-sheet default recommendations (auto-assigned in `asset-specs.mjs`):

- `bg_sheet`: `imageSize=4K`
- `character_master` / `portrait_sheet` / `ui_sheet` / `fx_sheet`: `imageSize=2K`
- `imageAspectRatio`: inferred from target canvas (`1:1`, `2:1`, `16:9` etc.)

All master-sheet prompts enforce `1 asset = 1 packed sprite-sheet image`.

Optional consistency workflow:

- Set `referenceImagePaths` in an asset spec to send existing images together with the prompt.
- Useful when generating split portrait sheets from a previously generated `character_master_*`.

Generated artifacts (V2):

- `public/assets/master-sheets/_raw/{id}/{runId}_attempt{n}.png`
- `public/assets/master-sheets/{id}_...png` (final)
- `public/assets/master-sheets/_archive/{id}/...png` (previous finals, auto-backed-up before overwrite)
- `public/assets/master-sheets/_reports/{id}.json`
- `data/master-sheets.manifest.json` (schemaVersion 2)

## Split existing 3x4 background sheet

If you already have a manually generated 3x4 sheet (`street/shop/compose` x `day/sunset/rain/night`), split it with:

```bash
npm run assets:split:bg-3x4
```

Defaults:

- Input: `public/assets/master-sheets/bg_3x4.png`
- Raw slices: `public/assets/bg/slices/bg_3x4/*.png` (1584x896 each)
- Resized game backgrounds: `public/assets/bg/bg_<scene>_<time>_960x540.png`

## Extract random street crowd sprites from character master sheet

If you already have `character_master_all_cast_4096x4096.png`, extract crowd sprites with:

```bash
npm run assets:extract:street-crowd
```

Defaults:

- Input: `public/assets/master-sheets/character_master_all_cast_4096x4096.png`
- Output sprites: `public/assets/characters/sprites/street-crowd/crowd_XX_32x48.png`
- Manifest: `public/assets/characters/sprites/street-crowd/manifest.json`

## Extract crowd poses + portraits from generated green-screen sheets

If you have generated sheets like `Generated Image ...png` in `street-crowd/`, extract them with:

```bash
npm run assets:extract:street-crowd:v2
```

Defaults:

- Input dir: `public/assets/characters/sprites/street-crowd/`
- Match files: `Generated Image *.png`
- Sprite outputs: `public/assets/characters/sprites/street-crowd-v2/crowd_XX_{front|diag_left|diag_right}_64x96.png`
- Portrait outputs: `public/assets/characters/portraits/street-crowd-v2/por_crowd_XX_{face|face_alt}_160.png`
- Manifest: `public/assets/characters/sprites/street-crowd-v2/manifest.generated.json`

## Output paths

Generated files are written directly into `public/assets/**` so the game can load them without extra conversion.

Master sheets are written to `public/assets/master-sheets/**`.

## Notes

- Generated sprite assets often need manual pixel cleanup (especially edges and animation consistency).
- Spec validation prevents duplicate `id` and duplicate `outputPath` definitions before generation.
- Keep using `docs/material.md` as the source of truth for MVP-required assets.
