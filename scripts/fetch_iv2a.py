"""Resumable fetcher for the remaining BCI IV-2a subjects.

The dataset's only host (lampx.tugraz.at) is frequently unreachable -- it accepts
the TCP connection on 443 and then drops the TLS handshake. There is no working
mirror: bnci-horizon-2020.eu just 302s to the same box.

So: poll it, and grab whatever subjects are available whenever it comes back.
Files already in ~/mne_data are skipped, so this is safe to re-run any time.

    python scripts/fetch_iv2a.py            # one pass
    python scripts/fetch_iv2a.py --watch    # retry every 10 min until complete
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

BASE = "https://lampx.tugraz.at/~bci/database/001-2014"
CACHE = Path.home() / "mne_data" / "MNE-bnci-data" / "~bci" / "database" / "001-2014"
SUBJECTS = range(1, 10)
MIN_BYTES = 10_000_000  # a real A0xT.mat is ~40-50 MB; anything smaller is a stub


def missing() -> list[str]:
    out = []
    for s in SUBJECTS:
        for suf in ("T", "E"):
            f = CACHE / f"A{s:02d}{suf}.mat"
            if not f.exists() or f.stat().st_size < MIN_BYTES:
                out.append(f"A{s:02d}{suf}.mat")
    return out


def fetch(name: str, timeout: int = 120) -> bool:
    import requests

    dest = CACHE / name
    tmp = dest.with_suffix(".part")
    CACHE.mkdir(parents=True, exist_ok=True)
    try:
        with requests.get(f"{BASE}/{name}", stream=True, timeout=timeout) as r:
            r.raise_for_status()
            n = 0
            with open(tmp, "wb") as fh:
                for chunk in r.iter_content(1 << 20):
                    fh.write(chunk)
                    n += len(chunk)
        if n < MIN_BYTES:
            tmp.unlink(missing_ok=True)
            print(f"  {name}: only {n} bytes, discarding")
            return False
        tmp.rename(dest)
        print(f"  {name}: OK ({n/1e6:.0f} MB)")
        return True
    except Exception as e:
        tmp.unlink(missing_ok=True)
        print(f"  {name}: {type(e).__name__}: {str(e)[:90]}")
        return False


def one_pass() -> list[str]:
    todo = missing()
    if not todo:
        print("All 9 subjects present.")
        return []
    print(f"{len(todo)} file(s) missing: {' '.join(todo)}")
    for name in todo:
        fetch(name)
    return missing()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true", help="retry until complete")
    ap.add_argument("--interval", type=int, default=600, help="seconds between passes")
    a = ap.parse_args()

    while True:
        left = one_pass()
        if not left:
            print("\nComplete. Now run:\n"
                  "  python scripts/run_benchmark.py --phase all --dataset iv2a --tag iv2a_full\n"
                  "  python scripts/make_report.py iv2a_full")
            sys.exit(0)
        if not a.watch:
            print(f"\n{len(left)} still missing. Re-run later, or use --watch.")
            sys.exit(1)
        print(f"-- sleeping {a.interval}s --\n", flush=True)
        time.sleep(a.interval)
