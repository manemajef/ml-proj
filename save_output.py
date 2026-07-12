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


def sync_jupytext(py_file: Path) -> None:
    print(f"Syncing code between {py_file.name} and {py_file.stem}.ipynb via Jupytext...")
    run(["jupytext", "--sync", str(py_file)])


def export_ipynb_to_html(ipynb_file: Path, output_html: Path) -> None:
    print(f"Converting {ipynb_file.name} to HTML (fast, no evaluation)...")
    with tempfile.TemporaryDirectory(prefix="notebook-export-html-") as tmp:
        tmp_dir = Path(tmp)
        run([
            "jupyter",
            "nbconvert",
            "--to",
            "html",
            str(ipynb_file),
            "--output-dir",
            str(tmp_dir),
        ])
        tmp_html = tmp_dir / f"{ipynb_file.stem}.html"
        shutil.copy2(tmp_html, output_html)


def export_ipynb(ipynb_file: Path, output_dir: Path, assets_dir: Path) -> Path:
    output_md = output_dir / f"{ipynb_file.stem}.md"
    support_dir = assets_dir / f"{ipynb_file.stem}_files"

    with tempfile.TemporaryDirectory(prefix="notebook-export-md-") as tmp:
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


def sync_pipeline(input_file: Path, output_dir: Path, assets_dir: Path, evaluate: bool, export_html: bool) -> None:
    input_file = input_file.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    if not input_file.exists():
        raise FileNotFoundError(input_file)

    suffix = input_file.suffix.lower()

    if suffix == ".ipynb":
        # Just convert directly using nbconvert (fast, no evaluation)
        md_file = export_ipynb(input_file, output_dir, assets_dir)
        print(f"\n[Fast Sync Success]")
        print(f"Saved Markdown: {md_file}")
        
        if export_html:
            html_file = output_dir / f"{input_file.stem}.html"
            export_ipynb_to_html(input_file, html_file)
            print(f"Saved HTML (nbconvert): {html_file}")

    elif suffix == ".py":
        if not is_marimo_py(input_file):
            raise RuntimeError(
                f"{input_file} is a .py file, but it does not look like a marimo notebook."
            )
        
        # Determine the root ipynb file path
        ipynb_file = input_file.parent / f"{input_file.stem}.ipynb"
        html_file = output_dir / f"{input_file.stem}.html"

        if evaluate:
            print("\n>>> Re-evaluating/running notebook to generate fresh outputs (this may take a few minutes)...")
            # 1. Export .py to .ipynb with evaluation
            run([
                "marimo",
                "export",
                "ipynb",
                str(input_file),
                "-o",
                str(ipynb_file),
                "--include-outputs",
                "-f",
            ])
            # 2. Convert .ipynb to .md
            md_file = export_ipynb(ipynb_file, output_dir, assets_dir)
            print(f"\n[Full Sync/Eval Success]")
        else:
            print("\n>>> Running Fast Sync (updating code without re-evaluation)...")
            # 1. Sync .py and .ipynb using jupytext
            sync_jupytext(input_file)
            # 2. Convert .ipynb to .md
            md_file = export_ipynb(ipynb_file, output_dir, assets_dir)
            print(f"\n[Fast Sync Success]")

        # Run HTML export if requested (always runs marimo evaluation to get styled HTML)
        if export_html:
            print("\n>>> Exporting marimo notebook to HTML (this will run/evaluate the notebook)...")
            run([
                "marimo",
                "export",
                "html",
                str(input_file),
                "-o",
                str(html_file),
                "-f",
            ])
            print(f"Saved HTML: {html_file}")

        print(f"Saved/Updated root notebook: {ipynb_file}")
        print(f"Saved Markdown: {md_file}")
    else:
        raise ValueError("Expected a .ipynb file or a marimo .py file")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sync marimo .py notebooks with .ipynb and export to docs/ folder (Markdown by default)."
    )
    parser.add_argument(
        "file",
        nargs="?",
        default="notebook.py",
        help="Path to .ipynb or marimo .py file (default: 'notebook.py')"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="docs",
        help="Directory to save the markdown and html files (default: 'docs')"
    )
    parser.add_argument(
        "--assets-dir",
        "-a",
        default="docs",
        help="Directory to save assets/images (default: 'docs')"
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Re-run/evaluate the notebook to generate fresh outputs (slower, default: False)"
    )
    parser.add_argument(
        "--html",
        action="store_true",
        help="Export beautiful marimo HTML file (default: False, does not generate ugly Jupyter HTML)"
    )
    args = parser.parse_args()

    try:
        sync_pipeline(
            Path(args.file),
            output_dir=Path(args.output_dir),
            assets_dir=Path(args.assets_dir),
            evaluate=args.eval,
            export_html=args.html
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)



if __name__ == "__main__":
    main()
