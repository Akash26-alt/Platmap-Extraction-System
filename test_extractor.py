"""
test_extractor.py - Run from project root: python test_extractor.py
"""
import os, json, sys
from pathlib import Path

def load_env_manual(env_path=".env"):
    env_file = Path(env_path)
    if not env_file.exists():
        return
    for encoding in ["utf-8","utf-8-sig","utf-16","utf-16-le","utf-16-be"]:
        try:
            with open(env_file, encoding=encoding) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        key, _, value = line.partition("=")
                        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
            print(f"✅  .env loaded (encoding: {encoding})")
            return
        except (UnicodeDecodeError, UnicodeError):
            continue

load_env_manual()
sys.path.append(str(Path(__file__).parent))
from extractor.platmap_extractoro import PlatMapExtractor

# ── CONFIG ──────────────────────────────────
TEST_FILE    = "plats/plat8.pdf"
LOT_NUMBER   = "127"
BLOCK_NUMBER = None
PAGE_NUMBER  = None  # None = auto-detect page (multi-page PDFs)
TILE_HINT    = (2, 2)  # Optional: (row, col) to help Groq find the lot
               # Set to None to rely purely on Groq detection
# ────────────────────────────────────────────

def run_checks():
    print("\n" + "="*55)
    print("  PLATMAP EXTRACTOR — PRE-FLIGHT CHECKS")
    print("="*55)
    for key, name, url in [
        ("GROQ_API_KEY",      "Groq API Key",   "console.groq.com/keys"),
        ("ANTHROPIC_API_KEY", "Claude API Key",  "console.anthropic.com"),
    ]:
        val = os.getenv(key)
        if not val:
            print(f"❌  {name} not found. Get at: {url}")
            sys.exit(1)
        print(f"✅  {name}: ...{val[-6:]}")

    if not Path(TEST_FILE).exists():
        print(f"❌  File not found: {TEST_FILE}")
        sys.exit(1)
    print(f"✅  Input file: {TEST_FILE}")

    for pkg, name, install in [
        ("groq","groq","groq"),
        ("anthropic","anthropic","anthropic"),
        ("fitz","pymupdf","pymupdf"),
        ("PIL","Pillow","Pillow"),
    ]:
        try:
            __import__(pkg)
            print(f"✅  {name}")
        except ImportError:
            print(f"❌  {name} not installed. Run: pip install {install}")
            sys.exit(1)

    for d in ["outputs/json","outputs/debug_crops"]:
        Path(d).mkdir(parents=True, exist_ok=True)
    print("✅  Output directories ready")
    print("="*55)
    print("  All checks passed!\n")

def run_test():
    run_checks()
    extractor = PlatMapExtractor(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        claude_api_key=os.getenv("ANTHROPIC_API_KEY")
    )

    print(f"📄  File       : {TEST_FILE}")
    print(f"🏠  Lot        : {LOT_NUMBER}")
    print(f"📃  Page       : {PAGE_NUMBER or 'Auto-detect'}")
    print(f"🗺️   Tile hint  : {TILE_HINT or 'None (Groq auto-detects)'}\n")

    result = extractor.extract(
        file_path=TEST_FILE,
        lot_number=LOT_NUMBER,
        block_number=BLOCK_NUMBER,
        page_number=PAGE_NUMBER,
        past_corrections=None,
        tile_hint=TILE_HINT,
    )
    json_path = save_json(result, output_dir="outputs/json")

    print("\n" + "="*55)
    print("  RESULT SUMMARY")
    print("="*55)
    print(f"  Lot          : {result.get('lot_number')}")
    print(f"  Subdivision  : {result.get('subdivision')}")
    print(f"  County/State : {result.get('county')}, {result.get('state')}")
    print(f"  Plat Book    : {result.get('plat_book')}, pg {result.get('plat_page')}")
    print(f"  Confidence   : {result.get('extraction_confidence','?').upper()}")
    print(f"  Page used    : {result.get('page_number')}")
    print(f"  Regions found: {result.get('detected_regions', {})}")
    print(f"  Crops sent   : {result.get('crops_sent', [])}")

    boundaries = result.get("boundaries", {})
    if boundaries:
        print(f"\n📐  BOUNDARIES:")
        for side, data in boundaries.items():
            b = data.get("bearing", "N/A")
            d = data.get("distance_ft", "N/A")
            c = data.get("curve_refs", [])
            s = f"  → curves: {c}" if c else ""
            print(f"    {side.upper():<12}: {b:<28} {d} ft{s}")

    curves = result.get("curves", [])
    if curves:
        print(f"\n📏  CURVES ({len(curves)}):")
        for c in curves:
            print(f"    {c.get('curve_id'):<6}: "
                  f"R={c.get('radius_ft')}  "
                  f"L={c.get('length_ft')}  "
                  f"Chord={c.get('chord_ft')}  "
                  f"Bearing={c.get('bearing')}  "
                  f"Δ={c.get('delta')}")

    line_refs = result.get("line_references", [])
    if line_refs:
        print(f"\n📋  LINE REFERENCES ({len(line_refs)}):")
        for l in line_refs:
            print(f"    {l.get('line_id')}: "
                  f"Bearing={l.get('bearing')}  "
                  f"Length={l.get('length_ft')}ft")

    easements = result.get("easements", [])
    if easements:
        print(f"\n🔧  EASEMENTS:")
        for e in easements:
            print(f"    {e.get('type')}: {e.get('width_ft','?')}ft "
                  f"on {e.get('boundary_side','?')}")

    if result.get("needs_review"):
        print(f"\n⚠️   NEEDS REVIEW: {result['needs_review']}")
    if result.get("error"):
        print(f"\n❌  ERROR: {result['error']}")

    print(f"\n💾  JSON        : {json_path}")
    print(f"🖼️   Debug crops : outputs/debug_crops/")
    print("="*55)
    print("\n📋  FULL JSON:\n")
    print(json.dumps(result, indent=2))
    return result

if __name__ == "__main__":
    run_test()