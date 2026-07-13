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


def clean_output(
    md_file: Path, stem: str, output_md_parent: Path, support_dir: Path
) -> None:
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


def export_html(py_file: Path, output_html: Path) -> None:
    print(f"Exporting marimo notebook {py_file.name} to HTML...")
    run([
        "marimo",
        "export",
        "html",
        str(py_file),
        "-o",
        str(output_html),
        "-f",
    ])


def run_export_pipeline(
    py_file: Path,
    ipynb_file: Path,
    output_dir: Path,
    assets_dir: Path,
    export_html_flag: bool,
) -> None:
    output_md = output_dir / f"{ipynb_file.stem}.md"
    output_html = output_dir / f"{py_file.stem}.html"

    if export_html_flag:
        needs_export = (
            not ipynb_file.exists()
            or not output_md.exists()
            or not output_html.exists()
            or (py_file.exists() and py_file.stat().st_mtime > ipynb_file.stat().st_mtime)
        )
        if needs_export:
            print("Notebook has changed or outputs are missing. Triggering export...")
            # 1. Export and run the marimo notebook to ipynb to save outputs
            print(f"Running marimo export ipynb on {py_file}...")
            run([
                "marimo",
                "export",
                "ipynb",
                str(py_file),
                "-o",
                str(ipynb_file),
                "--include-outputs",
                "-f",
            ])
            # 2. Export ipynb to markdown with tables converted to markdown format
            print("Converting exported ipynb to Markdown...")
            export_ipynb(ipynb_file, output_dir, assets_dir)
            # 3. Export to HTML using standard marimo export
            print("Converting exported notebook to HTML...")
            export_html(py_file, output_html)
            print("[HTML Mode] Notebook outputs successfully updated.")
        else:
            print("Notebook outputs are already up-to-date. No export needed.")
    else:
        needs_md_export = (
            not output_md.exists()
            or (ipynb_file.exists() and ipynb_file.stat().st_mtime > output_md.stat().st_mtime)
        )
        if needs_md_export:
            if not ipynb_file.exists():
                raise FileNotFoundError(
                    f"Jupyter notebook {ipynb_file} does not exist. Please run with --html to export it from marimo first."
                )
            print(f"Exporting {ipynb_file} -> Markdown...")
            export_ipynb(ipynb_file, output_dir, assets_dir)
            print("[Default Mode] Markdown output successfully updated.")
        else:
            print("Markdown output is already up-to-date. No export needed.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sync marimo .py notebooks with .ipynb and export to docs/ folder."
    )
    parser.add_argument(
        "file",
        nargs="?",
        default=None,
        help="Path to .ipynb or marimo .py file (default: notebook.ipynb / notebook.py)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="docs",
        help="Directory to save the markdown and html files (default: 'docs')",
    )
    parser.add_argument(
        "--assets-dir",
        "-a",
        default="docs",
        help="Directory to save assets/images (default: 'docs')",
    )
    parser.add_argument(
        "--html",
        action="store_true",
        help="Export HTML in addition to Markdown, but only if notebook has changed",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    assets_dir = Path(args.assets_dir)

    target_file = Path(args.file) if args.file is not None else None

    # Resolve py and ipynb files
    if target_file is not None:
        if target_file.suffix == ".py":
            py_file = target_file
            ipynb_file = target_file.with_suffix(".ipynb")
        elif target_file.suffix == ".ipynb":
            ipynb_file = target_file
            py_file = target_file.with_suffix(".py")
        else:
            raise ValueError("Expected a .ipynb or .py file.")
    else:
        py_file = Path("notebook.py")
        ipynb_file = Path("notebook.ipynb")

    try:
        run_export_pipeline(
            py_file=py_file,
            ipynb_file=ipynb_file,
            output_dir=output_dir,
            assets_dir=assets_dir,
            export_html_flag=args.html,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()

