#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
from importlib.util import find_spec
import argparse, datetime, shutil, sys

def ray_pkg_dir() -> Path:
    spec = find_spec("ray")
    if not spec or not spec.submodule_search_locations:
        print("Ray is not importable in this interpreter.", file=sys.stderr)
        sys.exit(2)
    return Path(list(spec.submodule_search_locations)[0]).resolve()

def latest_backup(backups_root: Path) -> Path | None:
    if not backups_root.exists():
        return None
    candidates = sorted(p for p in backups_root.iterdir() if p.is_dir())
    return candidates[-1] if candidates else None

def apply_patch(overlay_root: Path, expect_ray: str | None = None, dry: bool = False) -> None:
    try:
        import ray
    except Exception as e:
        print(f"Cannot import Ray: {e}", file=sys.stderr)
        sys.exit(2)
    if expect_ray and ray.__version__ != expect_ray:
        print(f"Ray version mismatch: installed={ray.__version__} expected={expect_ray}", file=sys.stderr)
        print("Proceeding anyway. Use --expect-ray to pin strictly.", file=sys.stderr)

    if not overlay_root.exists():
        print(f"Overlay not found: {overlay_root}", file=sys.stderr)
        sys.exit(2)

    src_root = overlay_root / "ray"
    targets = [p for p in src_root.rglob("*.py") if p.is_file()]
    if not targets:
        print(f"No *.py files under {src_root}", file=sys.stderr)
        sys.exit(2)

    dst_root = ray_pkg_dir()
    backups_root = dst_root.parent / "__ray_patch_backups__"
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_dir = backups_root / ts

    print(f"Ray package: {dst_root}")
    print(f"Backup dir : {backup_dir}")
    print(f"Overlay    : {src_root}")

    for src in targets:
        rel = src.relative_to(src_root)
        dst = dst_root / rel
        bkp = backup_dir / rel

        if dry:
            print(f"[DRY] {dst}  <-  {src}")
            continue

        # backup existing file if present
        if dst.exists():
            bkp.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(dst, bkp)

        # ensure destination dir exists, then copy
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"Patched: {dst}")

    if not dry:
        print("Done.")

def restore_latest() -> None:
    dst_root = ray_pkg_dir()
    backups_root = dst_root.parent / "__ray_patch_backups__"
    b = latest_backup(backups_root)
    if not b:
        print("No backups found.", file=sys.stderr)
        sys.exit(2)

    print(f"Restoring from: {b}")
    for src in b.rglob("*.py"):
        rel = src.relative_to(b)
        dst = dst_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"Restored: {dst}")
    print("Done.")

def main(argv=None):
    ap = argparse.ArgumentParser(description="Apply or restore a tiny overlay patch to installed Ray.")
    ap.add_argument("--restore", action="store_true", help="Restore from the latest backup instead of patching.")
    ap.add_argument("--expect-ray", help="Guard against Ray version drift, e.g., 2.31.0")
    ap.add_argument("--dry-run", action="store_true", help="Show what would be copied.")
    ap.add_argument("--overlay", default=str(Path(__file__).parent / "overlay"),
                    help="Path to overlay/ (default: ./overlay)")
    args = ap.parse_args(argv)

    if args.restore:
        restore_latest()
    else:
        apply_patch(Path(args.overlay), expect_ray=args.expect_ray, dry=args.dry_run)

if __name__ == "__main__":
    main()
