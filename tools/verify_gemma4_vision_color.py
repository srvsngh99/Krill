#!/usr/bin/env python3
"""Vision color-parity gate for Gemma-4-12B (and other unified multimodal SKUs).

nvfp4 on the vision patch-embedding attenuates the red input channel, so
red-heavy colors get misread (red->brown, yellow->olive, magenta->purple) while
text quality is untouched. PR #171's multimodal gate ran on the non-requant
checkpoint, so this shipped silently in the registered nvfp4 model. This tool is
the gate that would have caught it: it asks the model to name six solid colors
and FAILS if any is wrong. Run it against any requant before registering.

Requires a running KrillLM (or Ollama-compatible) server with the model loaded.

  tools/verify_gemma4_vision_color.py --url http://127.0.0.1:57455 --model gemma-4-12b

Exit 0 = all colors correct; 1 = at least one miss (or transport error).
"""
import argparse, io, json, sys, base64, urllib.request

try:
    from PIL import Image
except ImportError:
    sys.exit("verify_gemma4_vision_color: Pillow is required (pip install pillow)")

# Solid RGB primaries + secondaries. The three with R=255 (red, yellow, magenta)
# are exactly the ones nvfp4 vision degradation corrupts, so they are the
# load-bearing cases here. `accept` is the set of answers that count as correct;
# it deliberately EXCLUDES each color's known degradation signature (red->brown,
# yellow->olive, magenta->purple) so the gate cannot pass on the very defect it
# guards, while still allowing legitimate synonyms (e.g. fuchsia for magenta).
COLORS = [
    ("red",     (255, 0, 0),   ["red", "crimson", "scarlet"]),
    ("green",   (0, 255, 0),   ["green", "lime"]),
    ("blue",    (0, 0, 255),   ["blue", "navy"]),
    ("yellow",  (255, 255, 0), ["yellow", "gold"]),
    ("cyan",    (0, 255, 255), ["cyan", "aqua", "turquoise"]),
    ("magenta", (255, 0, 255), ["magenta", "fuchsia", "pink"]),
]
FORCED = "What color is this image? Reply with only the color name."
OPEN = "Describe this image in one short sentence."


def solid_png_b64(rgb, size=448):
    img = Image.new("RGB", (size, size), rgb)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def generate(url, model, prompt, image_b64, num_predict):
    body = {
        "model": model, "prompt": prompt, "stream": False,
        "options": {"temperature": 0, "num_predict": num_predict},
        "images": [image_b64],
    }
    req = urllib.request.Request(
        url.rstrip("/") + "/api/generate",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    resp = json.load(urllib.request.urlopen(req, timeout=240))
    text = (resp.get("response") or "").strip()
    # Drop a leading thinking-channel span if the model emits one.
    if "<channel|>" in text:
        text = text.split("<channel|>", 1)[1].strip()
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://127.0.0.1:57455")
    ap.add_argument("--model", default="gemma-4-12b")
    args = ap.parse_args()

    print(f"vision color parity: {args.model} @ {args.url}")
    misses = 0
    for name, rgb, accept in COLORS:
        img = solid_png_b64(rgb)
        try:
            forced = generate(args.url, args.model, FORCED, img, 24)
            opened = generate(args.url, args.model, OPEN, img, 64)
        except Exception as exc:  # transport / server error -> hard fail
            print(f"  {name:8} ERROR: {exc}")
            return 1
        # Pass/fail is decided ONLY on the forced single-word answer (the
        # documented repro). `open` is printed for context, never to widen a
        # pass. A correct answer is any accepted synonym; the degradation
        # signature (brown/olive/purple) is intentionally not accepted.
        fl = forced.lower()
        ok = any(a in fl for a in accept)
        misses += 0 if ok else 1
        print(f"  {name:8} forced={forced[:24]!r:26} open={opened[:60]!r}  "
              f"{'OK' if ok else 'MISS'}")

    total = len(COLORS)
    print(f"\n{total - misses}/{total} colors correct")
    if misses:
        print(f"FAIL: {misses} color(s) misread - vision projector likely "
              f"under-protected (re-requant with vision projectors at 8-bit).")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
