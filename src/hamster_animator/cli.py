"""Command line interface: ``hamster-animator --help``."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from . import _hamster as H
from .animation import default_cache
from .gallery import EXAMPLES
from .sim import Cache, Local, Slurm, collect


def _backend(args):
    if args.slurm:
        return Slurm(hamster=args.hamster, partition=args.partition, views_per_job=args.views_per_job)
    return Local(hamster=args.hamster, quiet=not args.verbose)


def cmd_gallery(args) -> int:
    names = list(EXAMPLES) if args.name == "all" else [args.name]
    backend = _backend(args)
    for name in names:
        anim, style = EXAMPLES[name]()
        anim.cache = Cache(args.cache)
        suffix = "_draft" if args.draft else ""
        if args.draft:
            anim = anim.draft()
        print(f"── {name}: {len(anim)} frames ──", flush=True)
        if args.plan:
            print(anim.plan(backend))
            continue
        out = Path(args.out)
        if args.still is not None:
            anim.still(out / f"{name}{suffix}_{args.still:03d}.png", args.still, style, backend)
            continue
        anim.save(out / f"{name}{suffix}.{args.format}", fps=args.fps, style=style, backend=backend)
    return 0


def cmd_info(args) -> int:
    try:
        root = H.find_root(args.hamster)
        print(f"HAMSTER root:   {root}")
        try:
            print(f"HAMSTER binary: {H.find_binary(root)}")
        except H.HamsterNotFound as e:
            print(f"HAMSTER binary: missing ({e})")
    except H.HamsterNotFound as e:
        print(e)
    try:
        H._tools()
        print("hamster-tools:  installed")
    except H.HamsterNotFound:
        print("hamster-tools:  missing")
    print(f"ffmpeg:         {shutil.which('ffmpeg') or 'missing'}")
    root = Path(args.cache).resolve()
    size = sum(f.stat().st_size for f in root.rglob("*") if f.is_file()) if root.exists() else 0
    print(f"cache:          {root} ({size / 1e6:.1f} MB)")
    return 0


def cmd_collect(args) -> int:
    print(f"collected {collect(Cache(args.cache))} detector map(s)")
    return 0


def cmd_clear(args) -> int:
    root = Path(args.cache).resolve()
    if root.exists():
        shutil.rmtree(root)
        print(f"removed {root}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="hamster-animator", description="Animations from HAMSTER simulations.")
    ap.add_argument("--cache", default=str(default_cache()), help="cache directory (default: %(default)s)")
    ap.add_argument("--hamster", default=None, help="HAMSTER repository root (default: auto-detect)")
    sub = ap.add_subparsers(dest="command", required=True)

    g = sub.add_parser("gallery", help="render one of the example animations")
    g.add_argument("name", choices=[*EXAMPLES, "all"])
    g.add_argument("--draft", action="store_true", help="10x fewer photons, half resolution")
    g.add_argument("--plan", action="store_true", help="only show what would be computed")
    g.add_argument("--still", type=int, metavar="N", help="only render frame N as PNG")
    g.add_argument("--out", default="animations", help="output directory (default: %(default)s)")
    g.add_argument("--format", default="mp4", choices=["mp4", "webm", "gif", "mov"])
    g.add_argument("--fps", type=int, default=24)
    g.add_argument("--verbose", action="store_true", help="show HAMSTER output")
    g.add_argument("--slurm", action="store_true", help="submit to SLURM instead of running locally")
    g.add_argument("--partition", default=None)
    g.add_argument("--views-per-job", type=int, default=8)
    g.set_defaults(func=cmd_gallery)

    sub.add_parser("info", help="show detected HAMSTER installation and cache").set_defaults(func=cmd_info)
    sub.add_parser("collect", help="take over results of finished SLURM jobs").set_defaults(func=cmd_collect)
    sub.add_parser("clear-cache", help="delete the cache directory").set_defaults(func=cmd_clear)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
