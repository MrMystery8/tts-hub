
import json
import argparse
from pathlib import Path
from collections import Counter
from watermark.config import N_CLASSES, N_MODELS

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest_path", nargs="?", help="Path to manifest.json")
    args = parser.parse_args()
    
    # helper to find latest manifest if none provided
    if not args.manifest_path:
        base = Path("outputs")
        manifests = list(base.rglob("manifest.json"))
        if not manifests:
            print("No manifest.json found in outputs/")
            return
        # sort by mod time
        manifests.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        target = manifests[0]
        print(f"No path provided, using latest: {target}")
    else:
        target = Path(args.manifest_path)
        
    print(f"Sanity Check B: Manifest Class Distribution")
    print(f"Checking: {target}")
    
    if not target.exists():
        print(f"[FAIL] File not found: {target}")
        return

    try:
        data = json.loads(target.read_text())
    except Exception as e:
        print(f"[FAIL] JSON decode error: {e}")
        return
    
    print(f"Total entries: {len(data)}")
    
    # Handle None/null in model_id by mapping to -1
    model_ids = []
    for d in data:
        mid = d.get("model_id")
        if mid is None:
            model_ids.append(-1)
        else:
            model_ids.append(mid)
            
    counts = Counter(model_ids)
    
    print("\nClass Distribution (model_id):")
    sorted_ids = sorted(counts.keys())
    
    has_error = False
    
    # Check negatives
    if -1 not in counts:
        print("[WARN] No clean samples (model_id=-1) found? (Might be intent depending on mode)")
    else:
        print(f"  Class -1 (Clean/ex-Null): {counts[-1]}")
        
    print(f"Expected Positive Classes: 0..{N_MODELS-1}")
    
    for i in range(N_MODELS):
        c = counts.get(i, 0)
        print(f"  Class {i}: {c}")
        if c == 0:
            print(f"    [FAIL] Class {i} has ZERO samples!")
            # This is critical failure
            has_error = True
            
    # Check for out of bounds
    for k in counts:
        if k != -1 and not (0 <= k < N_MODELS):
            print(f"    [FAIL] Found unexpected class ID: {k}")
            has_error = True
            
    if has_error:
        print("\n[FAIL] Manifest distribution is broken.")
    else:
        print("\n[PASS] Manifest looks plausible (all classes present).")

if __name__ == "__main__":
    main()
