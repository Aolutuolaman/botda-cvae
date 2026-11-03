"""Create a source and training-data ZIP for GitHub.

Run from any directory: python tools/build_release.py [--output path.zip]
The explicit file allowlist excludes weights, environments and runtime outputs.
"""

import argparse
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED


ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = ("README.md", "README.zh-CN.md", "requirements.txt",
              ".gitignore", ".gitattributes", "train.py", "generate.py")
SOURCE_DIRS = ("data0", "tests", "tools", ".github")
SUFFIXES = {".py", ".md", ".json", ".txt", ".yml", ".yaml"}


def source_files():
    files = [ROOT / name for name in ROOT_FILES]
    for name in SOURCE_DIRS:
        files.extend(path for path in (ROOT / name).rglob("*") if path.is_file()
                     and "__pycache__" not in path.parts and path.suffix in SUFFIXES)
    for optional in ("LICENSE", "CITATION.cff"):
        if (ROOT / optional).is_file():
            files.append(ROOT / optional)
    return sorted(set(files))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT.parent / "botda-cvae-github.zip")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(args.output, "w", compression=ZIP_DEFLATED) as archive:
        for path in source_files():
            archive.write(path, "botda-cvae/" + path.relative_to(ROOT).as_posix())
    with ZipFile(args.output) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("ZIP integrity check failed")
        print(f"Packaged {len(archive.namelist())} files: {args.output.resolve()}")


if __name__ == "__main__":
    main()
