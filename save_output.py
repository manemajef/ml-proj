from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from io import StringIO
from pathlib import Path

import pandas as pd


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def wait_for_stable_file(
    path: Path, quiet_seconds: float = 1.0, timeout: float = 10.0
) -> None:
    deadline = time.monotonic() + timeout
    last_stat = None
    stable_since = None

    while time.monotonic() < deadline:
        stat = path.stat()
        current_stat = (stat.st_mtime_ns, stat.st_size)

        if current_stat != last_stat:
            last_stat = current_stat
            stable_since = time.monotonic()
        elif (
            stable_since is not None
            and time.monotonic() - stable_since >= quiet_seconds
        ):
            return

        time.sleep(0.1)

    raise TimeoutError(f"{path} did not stabilize after {timeout:.1f}s")


def is_marimo_py(path: Path) -> bool:
    text = path.read_text(errors="ignore")
    return "import marimo" in text and "app = marimo.App" in text


def clean_output(md_file: Path, stem: str, output_md_parent: Path, support_dir: Path) -> None:
    wait_for_stable_file(md_file)
    text = md_file.read_text(errors="ignore")

    def table_to_markdown(match: re.Match[str]) -> str:
        table_html = match.group(1)
        try:
            df = pd.read_html(StringIO(table_html))[0]
            # Optimize float columns for agents reading (round to 4 decimal places)
            for col in df.select_dtypes(include=["float"]):
                df[col] = df[col].round(4)
            return "\n\n" + df.to_markdown(index=False) + "\n\n"
        except Exception:
            # Fallback to original matched content if parsing fails
            return match.group(0)

    # Match tables, optionally wrapped in div and style blocks
    TABLE_RE = re.compile(
        r"(?:<div>\s*(?:<style.*?</style>\s*)*)?(<table.*?</table>)(?:\s*</div>)?",
        re.IGNORECASE | re.DOTALL,
    )
    text = TABLE_RE.sub(table_to_markdown, text)
    
    STYLE_RE = re.compile(r"<style.*?</style>", re.IGNORECASE | re.DOTALL)
    text = STYLE_RE.sub("", text)

    # Rewrite asset paths inside the markdown file to be relative to the markdown file
    rel_support_dir = os.path.relpath(support_dir, output_md_parent)
    rel_support_path = rel_support_dir.replace(os.path.sep, "/")
    
    old_prefix = f"{stem}_files"
    if old_prefix != rel_support_path:
        text = text.replace(f"{old_prefix}/", f"{rel_support_path}/")

    md_file.write_text(text)


def export_marimo_py(py_file: Path, output_dir: Path, assets_dir: Path) -> Path:
    # No check for marimo (simplified guardline code)
    ipynb_file = output_dir / f"{py_file.stem}.ipynb"

    run([
        "marimo",
        "export",
        "ipynb",
        str(py_file),
        "-o",
        str(ipynb_file),
        "--include-outputs",
    ])

    return export_ipynb(ipynb_file, output_dir, assets_dir)


def export_ipynb(ipynb_file: Path, output_dir: Path, assets_dir: Path) -> Path:
    # No check for jupyter (simplified guardline code)
    output_md = output_dir / f"{ipynb_file.stem}.md"
    support_dir = assets_dir / f"{ipynb_file.stem}_files"

    with tempfile.TemporaryDirectory(prefix="notebook-export-") as tmp:
        tmp_dir = Path(tmp)

        run([
            "jupyter",
            "nbconvert",
            "--to",
            "markdown",
            str(ipynb_file),
            "--output-dir",
            str(tmp_dir),
        ])

        tmp_md = tmp_dir / f"{ipynb_file.stem}.md"
        clean_output(tmp_md, ipynb_file.stem, output_md.parent, support_dir)

        tmp_support_dir = tmp_dir / f"{ipynb_file.stem}_files"

        if support_dir.exists():
            shutil.rmtree(support_dir)
        if tmp_support_dir.exists():
            support_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(tmp_support_dir, support_dir)

        shutil.copy2(tmp_md, output_md)

    return output_md


def export_py(py_file: Path, output_dir: Path, assets_dir: Path) -> Path:
    if is_marimo_py(py_file):
        return export_marimo_py(py_file, output_dir, assets_dir)

    raise RuntimeError(
        f"{py_file} is a .py file, but it does not look like a marimo notebook. "
        "Only .ipynb files and marimo .py notebooks are supported."
    )


def to_markdown(input_file: Path, output_dir: Path, assets_dir: Path) -> Path:
    input_file = input_file.resolve()
    output_dir.mkdir(exist_ok=True)

    if not input_file.exists():
        raise FileNotFoundError(input_file)

    suffix = input_file.suffix.lower()

    if suffix == ".ipynb":
        md_file = export_ipynb(input_file, output_dir, assets_dir)
    elif suffix == ".py":
        md_file = export_py(input_file, output_dir, assets_dir)
    else:
        raise ValueError("Expected a .ipynb file or a marimo .py file")

    return md_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a .ipynb or marimo .py notebook to Markdown and clean pandas HTML tables."
    )
    parser.add_argument("file", help="Path to .ipynb or marimo .py file")
    parser.add_argument(
        "--output-dir",
        "-o",
        default=".",
        help="Directory to save the markdown file (default: '.')"
    )
    parser.add_argument(
        "--assets-dir",
        "-a",
        default="assets",
        help="Directory to save assets/images (default: 'assets')"
    )
    args = parser.parse_args()

    try:
        md_file = to_markdown(
            Path(args.file),
            output_dir=Path(args.output_dir),
            assets_dir=Path(args.assets_dir)
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)

    print(f"Saved: {md_file}")


if __name__ == "__main__":
    main()
