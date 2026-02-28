#!/usr/bin/env python3
"""
Extract selected icons from a green-screen sprite sheet.

Usage:
  python3 scripts/assets/extract-selected-kotone-icons.py \
    --input "/abs/path/to/sheet.png" \
    --mapping "/abs/path/to/mapping.json" \
    --output-dir "/abs/path/to/output-dir"
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image, ImageDraw


@dataclass
class Component:
    min_x: int
    min_y: int
    max_x: int
    max_y: int
    pixels: int

    @property
    def width(self) -> int:
        return self.max_x - self.min_x + 1

    @property
    def height(self) -> int:
        return self.max_y - self.min_y + 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract selected keyed icons from a green background sheet.")
    parser.add_argument("--input", required=True, help="Input sheet PNG path")
    parser.add_argument("--mapping", required=True, help="JSON mapping path: {\"<index>\": \"<output-file>.png\"}")
    parser.add_argument("--output-dir", required=True, help="Output directory for keyed PNG icons")
    parser.add_argument("--padding", type=int, default=2, help="Padding pixels around selected component bbox")
    parser.add_argument("--green-min", type=int, default=180, help="Background green channel floor")
    parser.add_argument(
        "--green-dominance",
        type=int,
        default=70,
        help="Required (green - max(red, blue)) threshold for background",
    )
    parser.add_argument("--min-component-pixels", type=int, default=400, help="Ignore connected components smaller than this")
    parser.add_argument("--debug", action="store_true", help="Write numbered debug image next to input")
    return parser.parse_args()


def is_bg_pixel(rgb: Tuple[int, int, int], green_min: int, green_dominance: int) -> bool:
    r, g, b = rgb
    return g >= green_min and (g - max(r, b)) >= green_dominance


def detect_components(
    img: Image.Image,
    green_min: int,
    green_dominance: int,
    min_component_pixels: int,
) -> List[Component]:
    w, h = img.size
    pix = img.load()
    visited = [[False] * w for _ in range(h)]
    components: List[Component] = []

    for y in range(h):
        for x in range(w):
            if visited[y][x]:
                continue
            visited[y][x] = True
            if is_bg_pixel(pix[x, y], green_min, green_dominance):
                continue

            q = deque([(x, y)])
            min_x = max_x = x
            min_y = max_y = y
            pixels = 0

            while q:
                cx, cy = q.popleft()
                pixels += 1
                if cx < min_x:
                    min_x = cx
                if cx > max_x:
                    max_x = cx
                if cy < min_y:
                    min_y = cy
                if cy > max_y:
                    max_y = cy

                for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                    if nx < 0 or ny < 0 or nx >= w or ny >= h:
                        continue
                    if visited[ny][nx]:
                        continue
                    visited[ny][nx] = True
                    if not is_bg_pixel(pix[nx, ny], green_min, green_dominance):
                        q.append((nx, ny))

            if pixels >= min_component_pixels:
                components.append(Component(min_x, min_y, max_x, max_y, pixels))

    components.sort(key=lambda c: (c.min_y, c.min_x))
    return components


def apply_chroma_key(crop: Image.Image, green_min: int, green_dominance: int) -> Image.Image:
    rgba = crop.convert("RGBA")
    px = rgba.load()
    w, h = rgba.size
    for y in range(h):
        for x in range(w):
            r, g, b, _ = px[x, y]
            if is_bg_pixel((r, g, b), green_min, green_dominance):
                px[x, y] = (0, 0, 0, 0)
            else:
                px[x, y] = (r, g, b, 255)

    # Trim transparent margins after keying.
    alpha = rgba.getchannel("A")
    bbox = alpha.getbbox()
    if bbox:
        rgba = rgba.crop(bbox)
    return rgba


def write_debug_image(input_path: Path, img: Image.Image, components: List[Component]) -> Path:
    ann = img.copy().convert("RGB")
    draw = ImageDraw.Draw(ann)
    for idx, comp in enumerate(components, start=1):
        draw.rectangle([comp.min_x, comp.min_y, comp.max_x, comp.max_y], outline=(255, 0, 255), width=2)
        draw.text((comp.min_x + 2, comp.min_y + 2), str(idx), fill=(255, 255, 255))
    out = input_path.with_suffix("").with_name(input_path.stem + ".debug-numbered.png")
    ann.save(out)
    return out


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    mapping_path = Path(args.mapping).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    img = Image.open(input_path).convert("RGB")
    components = detect_components(
        img=img,
        green_min=args.green_min,
        green_dominance=args.green_dominance,
        min_component_pixels=args.min_component_pixels,
    )

    if args.debug:
        debug_path = write_debug_image(input_path, img, components)
        print(f"[debug] wrote numbered component image: {debug_path}")

    mapping_raw = json.loads(mapping_path.read_text(encoding="utf-8"))
    # Keep mapping order from file for predictable output.
    mapping: Dict[int, str] = {int(k): str(v) for k, v in mapping_raw.items()}

    manifest = {
        "input": str(input_path),
        "mapping": str(mapping_path),
        "params": {
            "padding": args.padding,
            "green_min": args.green_min,
            "green_dominance": args.green_dominance,
            "min_component_pixels": args.min_component_pixels,
        },
        "components_detected": len(components),
        "exports": [],
    }

    for index, filename in mapping.items():
        if index < 1 or index > len(components):
            raise ValueError(f"Mapping index out of range: {index} (detected {len(components)} components)")

        comp = components[index - 1]
        left = max(0, comp.min_x - args.padding)
        top = max(0, comp.min_y - args.padding)
        right = min(img.width - 1, comp.max_x + args.padding)
        bottom = min(img.height - 1, comp.max_y + args.padding)
        crop = img.crop((left, top, right + 1, bottom + 1))
        keyed = apply_chroma_key(crop, green_min=args.green_min, green_dominance=args.green_dominance)

        out_path = output_dir / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        keyed.save(out_path)

        manifest["exports"].append(
            {
                "index": index,
                "filename": filename,
                "source_bbox": [comp.min_x, comp.min_y, comp.max_x, comp.max_y],
                "crop_bbox": [left, top, right, bottom],
                "output_size": list(keyed.size),
            }
        )
        print(f"[ok] {index:>2} -> {out_path.name} ({keyed.size[0]}x{keyed.size[1]})")

    manifest_path = output_dir / "manifest.extracted.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
