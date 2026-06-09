#!/usr/bin/env python3
"""
balance_ei_music.py

Deletes random music samples from an Edge Impulse project until music
total duration matches the average duration of the other classes.

Usage:
    python balance_ei_music.py --api-key ei_xxx --project-id 993132
    python balance_ei_music.py --api-key ei_xxx --project-id 993132 --dry-run
"""

import argparse
import json
import random
import time
import urllib.error
import urllib.request
from collections import defaultdict

EI_STUDIO = "https://studio.edgeimpulse.com/v1/api"
PAGE_SIZE  = 1000


def ei_get(path: str, api_key: str) -> dict:
    req = urllib.request.Request(f"{EI_STUDIO}{path}")
    req.add_header("x-api-key", api_key)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def ei_delete(path: str, api_key: str) -> bool:
    req = urllib.request.Request(f"{EI_STUDIO}{path}", method="DELETE")
    req.add_header("x-api-key", api_key)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status in (200, 204)
    except urllib.error.HTTPError as e:
        print(f"  DELETE failed: HTTP {e.code}")
        return False


def fetch_all_samples(project_id: str, api_key: str) -> list[dict]:
    """Fetch all samples across both training and testing splits."""
    all_samples = []
    for category in ("training", "testing"):
        offset = 0
        while True:
            data = ei_get(
                f"/{project_id}/raw-data?category={category}&limit={PAGE_SIZE}&offset={offset}",
                api_key,
            )
            batch = data.get("samples", [])
            if not batch:
                break
            for s in batch:
                s["_category"] = category
            all_samples.extend(batch)
            if len(batch) < PAGE_SIZE:
                break
            offset += PAGE_SIZE
    return all_samples


def fmt_duration(ms) -> str:
    h, rem = divmod(int(ms) // 1000, 3600)
    m, s   = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key",    required=True)
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--dry-run",    action="store_true",
                        help="Show what would be deleted without deleting")
    args = parser.parse_args()

    print("Fetching all samples from Edge Impulse…")
    samples = fetch_all_samples(args.project_id, args.api_key)
    print(f"  Total samples: {len(samples)}")

    # Group by label
    by_label: dict[str, list[dict]] = defaultdict(list)
    duration_by_label: dict[str, int] = defaultdict(int)
    for s in samples:
        lbl = s.get("label", "unknown")
        by_label[lbl].append(s)
        duration_by_label[lbl] += s.get("totalLengthMs", 0)

    print("\nCurrent dataset:")
    for lbl in sorted(by_label):
        n   = len(by_label[lbl])
        dur = duration_by_label[lbl]
        print(f"  {lbl:10s}  {n:5d} clips   {fmt_duration(dur)}")

    if "music" not in by_label:
        print("\nNo music samples found — nothing to do.")
        return

    # Target = average duration of non-music classes
    other_labels   = [l for l in by_label if l != "music"]
    other_total_ms = sum(duration_by_label[l] for l in other_labels)
    target_ms      = other_total_ms // len(other_labels)
    music_ms       = duration_by_label["music"]

    print(f"\nOther classes avg duration: {fmt_duration(target_ms)}")
    print(f"Music current duration:     {fmt_duration(music_ms)}")

    if music_ms <= target_ms:
        print("\nMusic is already at or below the target — nothing to delete.")
        return

    excess_ms = music_ms - target_ms
    print(f"Need to remove:             {fmt_duration(excess_ms)}\n")

    # Randomly shuffle music samples and delete until we've freed enough duration
    music_samples = by_label["music"][:]
    random.shuffle(music_samples)

    to_delete = []
    freed_ms  = 0
    for s in music_samples:
        if freed_ms >= excess_ms:
            break
        to_delete.append(s)
        freed_ms += s.get("totalLengthMs", 0)

    print(f"Will delete {len(to_delete)} music clips ({fmt_duration(freed_ms)} freed)")
    if args.dry_run:
        print("\n[dry-run] Sample clips that would be deleted:")
        for s in to_delete[:10]:
            print(f"  id={s['id']}  {fmt_duration(s.get('totalLengthMs',0))}  {s.get('filename','')}")
        if len(to_delete) > 10:
            print(f"  … and {len(to_delete)-10} more")
        print("\nRe-run without --dry-run to actually delete.")
        return

    print("\nDeleting…")
    deleted = failed = 0
    for i, s in enumerate(to_delete, 1):
        ok = ei_delete(f"/{args.project_id}/raw-data/{s['id']}", args.api_key)
        if ok:
            deleted += 1
        else:
            failed += 1
        print(f"\r  {i}/{len(to_delete)}  deleted={deleted}  failed={failed}  ", end="", flush=True)
        time.sleep(0.1)  # gentle rate limiting

    new_music_ms = music_ms - (freed_ms - sum(s.get("totalLengthMs", 0) for s in to_delete[deleted:]))
    print(f"\n\nDone.")
    print(f"  Deleted: {deleted}   Failed: {failed}")
    print(f"  Music duration now ≈ {fmt_duration(music_ms - freed_ms)}")
    print(f"\nRefresh Edge Impulse Data Acquisition to confirm.")


if __name__ == "__main__":
    main()
