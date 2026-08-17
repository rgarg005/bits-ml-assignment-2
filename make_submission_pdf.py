"""
Builds the single submission PDF required by the assignment brief.

Section 2 of the brief mandates one PDF containing, IN ORDER:
    1. GitHub repository link
    2. Live Streamlit app link
    3. Screenshot of execution on BITS Virtual Lab
    4. The GitHub README content

Usage:
    python make_submission_pdf.py \
        --name "Your Name" --student-id "20xxxxxxxx" \
        --bits-lab-screenshot ~/Desktop/bits_lab.png \
        --app-screenshot ~/Desktop/streamlit_app.png \
        --output ~/Downloads/ML_Assignment_2.pdf

--bits-lab-screenshot is optional only so the document can be previewed before
the screenshot exists; without it the PDF renders a loud placeholder page and
the submission is INCOMPLETE.

Rendering goes through headless Chrome rather than a LaTeX toolchain because it
handles the README's markdown tables and clickable links without extra packages.
"""

from __future__ import annotations

import argparse
import base64
import mimetypes
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import markdown

REPO_ROOT = Path(__file__).resolve().parent
GITHUB_URL = "https://github.com/rgarg005/bits-ml-assignment-2"
APP_URL = "https://bits-ml-assignment-2-dxnqhzykh5yl6dd9bhk4ev.streamlit.app"

CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
]

CSS = """
@page { size: A4; margin: 16mm 14mm; }
* { box-sizing: border-box; }
body { font-family: -apple-system, "Helvetica Neue", Arial, sans-serif;
       font-size: 9.6pt; line-height: 1.5; color: #1a1a1a; margin: 0; }
h1 { font-size: 19pt; margin: 0 0 2mm; letter-spacing: -0.3px; }
h2 { font-size: 12.5pt; margin: 7mm 0 2.5mm; padding-bottom: 1.4mm;
     border-bottom: 1.6px solid #6A2C91; color: #4A1D66; }
h3 { font-size: 10.6pt; margin: 5mm 0 2mm; color: #333; }
h4 { font-size: 9.8pt; margin: 4mm 0 1.5mm; }
p, li { margin: 0 0 2.2mm; }
a { color: #6A2C91; word-break: break-all; }
code { background: #f3f0f7; padding: 0.5mm 1.1mm; border-radius: 2px;
       font-family: "SF Mono", Menlo, monospace; font-size: 8.5pt; }
pre { background: #f7f5fa; border: 1px solid #e3dcec; border-left: 3px solid #6A2C91;
      padding: 2.5mm 3mm; border-radius: 3px; overflow-wrap: break-word;
      white-space: pre-wrap; font-size: 8.2pt; margin: 0 0 3mm; }
pre code { background: none; padding: 0; font-size: 8.2pt; }
table { border-collapse: collapse; width: 100%; margin: 2.5mm 0 4mm; font-size: 8.5pt;
        page-break-inside: avoid; }
th { background: #6A2C91; color: #fff; text-align: left; padding: 1.7mm 2mm;
     font-weight: 600; font-size: 8.4pt; }
td { border-bottom: 1px solid #e6e0ee; padding: 1.6mm 2mm; vertical-align: top; }
tr:nth-child(even) td { background: #faf8fc; }
blockquote { margin: 0 0 3mm; padding: 2mm 3mm; background: #f7f5fa;
             border-left: 3px solid #C9B6DB; }
hr { border: none; border-top: 1px solid #e0d8e8; margin: 5mm 0; }

.cover { text-align: center; padding: 14mm 0 8mm; border-bottom: 2.5px solid #6A2C91;
         margin-bottom: 6mm; }
.cover .course { font-size: 10.5pt; color: #555; margin-bottom: 6mm; }
.cover .who { font-size: 11pt; margin-top: 6mm; }
.cover .who strong { color: #4A1D66; }
.meta { font-size: 8.6pt; color: #666; margin-top: 5mm; }

.linkcard { border: 1.6px solid #6A2C91; border-radius: 4px; padding: 3.5mm 4mm;
            margin: 0 0 4mm; background: #fbf9fd; }
.linkcard .label { font-size: 8.2pt; text-transform: uppercase; letter-spacing: 0.7px;
                   color: #6A2C91; font-weight: 700; margin-bottom: 1.5mm; }
.linkcard a { font-size: 10.5pt; font-weight: 600; }
.linkcard .note { font-size: 8.4pt; color: #555; margin-top: 2mm; }

figure { margin: 0 0 4mm; page-break-inside: avoid; }
figure img { width: 100%; border: 1px solid #ccc; border-radius: 3px; }
figcaption { font-size: 8.2pt; color: #555; margin-top: 1.5mm; font-style: italic; }

.missing { border: 2.5px dashed #C0392B; background: #fdf3f2; color: #8E2A20;
           padding: 8mm 6mm; text-align: center; border-radius: 4px; margin: 4mm 0; }
.missing h3 { color: #C0392B; margin: 0 0 3mm; font-size: 12pt; }
.missing p { margin: 0 auto 2mm; max-width: 135mm; font-size: 9.2pt; }

.pagebreak { page-break-before: always; }
.checklist { font-size: 8.6pt; }
.checklist td:first-child { width: 8mm; text-align: center; font-weight: 700; }
.ok { color: #1E7B34; } .no { color: #C0392B; }
"""


def find_chrome() -> str:
    for candidate in CHROME_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    for name in ("chromium", "google-chrome", "chrome"):
        found = shutil.which(name)
        if found:
            return found
    sys.exit(
        "Could not find Chrome/Chromium to render the PDF. Install Google Chrome, "
        "or render README.md with: pandoc README.md -o out.pdf"
    )


def embed_image(path: Path) -> str:
    """Inline an image as a data URI so headless Chrome needs no file access."""
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def figure(path: Path, caption: str) -> str:
    return (
        f'<figure><img src="{embed_image(path)}" alt="{caption}">'
        f"<figcaption>{caption}</figcaption></figure>"
    )


def build_html(args, readme_html: str, complete: bool) -> str:
    name = args.name or "____________________"
    student_id = args.student_id or "____________________"

    # --- Section 3: the BITS Virtual Lab screenshot -------------------------
    if args.bits_lab_screenshot:
        shot = Path(args.bits_lab_screenshot).expanduser()
        if not shot.exists():
            sys.exit(f"BITS Lab screenshot not found: {shot}")
        section3 = figure(
            shot,
            "Assignment executed on BITS Virtual Lab "
            "(model/ML_Assignment2.ipynb - training and evaluation of all six models).",
        )
    else:
        section3 = """
        <div class="missing">
          <h3>&#9888; REQUIRED SCREENSHOT NOT YET INSERTED</h3>
          <p><strong>This submission is incomplete.</strong> The brief requires one
          screenshot showing the assignment being executed on <strong>BITS Virtual
          Lab</strong>, worth <strong>1 of 15 marks</strong>.</p>
          <p>Run <code>model/ML_Assignment2.ipynb</code> on BITS Virtual Lab, capture
          the screen, then regenerate this PDF with:</p>
          <p><code>python make_submission_pdf.py --bits-lab-screenshot &lt;path&gt;</code></p>
        </div>"""

    # --- Section 2: live app, plus an optional local screenshot -------------
    app_extra = ""
    if args.app_screenshot:
        app_shot = Path(args.app_screenshot).expanduser()
        if not app_shot.exists():
            sys.exit(f"App screenshot not found: {app_shot}")
        app_extra = figure(
            app_shot,
            "The deployed Streamlit app: CSV upload, model dropdown, all six "
            "evaluation metrics and the comparison table. (Supporting illustration "
            "only - this is NOT the BITS Virtual Lab screenshot required in item 3.)",
        )

    status = (
        '<span class="ok">COMPLETE</span>'
        if complete
        else '<span class="no">INCOMPLETE - BITS Lab screenshot missing</span>'
    )

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><style>{CSS}</style></head><body>

<div class="cover">
  <div class="course">Work Integrated Learning Programmes Division<br>
    BITS Pilani &middot; M.Tech (AIML / DSE)</div>
  <h1>Machine Learning Assignment 2</h1>
  <div style="font-size:11pt;color:#4A1D66;margin-top:2mm;">
    Term Deposit Subscription Predictor<br>
    <span style="font-size:9.5pt;color:#666;">Six classification models on the
    UCI Bank Marketing dataset, with a deployed Streamlit dashboard</span></div>
  <div class="who">Submitted by <strong>{name}</strong><br>
    <span style="font-size:9.5pt;">Student ID: {student_id}</span></div>
  <div class="meta">Submission deadline: 18-Aug-2026 &middot;
    Generated {datetime.now().strftime('%d-%b-%Y %H:%M')} &middot; Status: {status}</div>
</div>

<h2>1. GitHub Repository Link</h2>
<div class="linkcard">
  <div class="label">GitHub Repository</div>
  <a href="{GITHUB_URL}">{GITHUB_URL}</a>
  <div class="note">Public repository. Contains complete source code,
    <code>requirements.txt</code>, <code>README.md</code>, the test data
    (<code>test_data.csv</code>, 11,303 rows) and <code>model/</code> with the
    training script, the executed analysis notebook and all six fitted models.</div>
</div>

<h2>2. Live Streamlit App Link</h2>
<div class="linkcard">
  <div class="label">Live Application</div>
  <a href="{APP_URL}">{APP_URL}</a>
  <div class="note">Deployed on Streamlit Community Cloud. Opens an interactive
    frontend with CSV upload, a model-selection dropdown, all six evaluation
    metrics, a confusion matrix, ROC curve and classification report.</div>
</div>
{app_extra}

<h2>3. Screenshot &mdash; Execution on BITS Virtual Lab</h2>
{section3}

<div class="pagebreak"></div>
<h2>4. GitHub README Content</h2>
{readme_html}

</body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="", help="Your full name, for the cover page")
    parser.add_argument("--student-id", default="", help="Your BITS student ID")
    parser.add_argument("--bits-lab-screenshot", default="",
                        help="Screenshot of the notebook running on BITS Virtual Lab")
    parser.add_argument("--app-screenshot", default="",
                        help="Optional screenshot of the deployed Streamlit app")
    parser.add_argument("--output", default=str(Path.home()/"Downloads"/"ML_Assignment_2.pdf"))
    args = parser.parse_args()

    readme_html = markdown.markdown(
        (REPO_ROOT/"README.md").read_text(),
        extensions=["tables", "fenced_code", "sane_lists", "attr_list", "nl2br"],
    )
    complete = bool(args.bits_lab_screenshot)
    html = build_html(args, readme_html, complete)

    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp)/"submission.html"
        source.write_text(html)
        subprocess.run(
            [find_chrome(), "--headless=new", "--disable-gpu", "--no-sandbox",
             "--no-pdf-header-footer", "--virtual-time-budget=20000",
             f"--print-to-pdf={output}", source.as_uri()],
            check=True, capture_output=True, timeout=180,
        )

    size_kb = output.stat().st_size/1024
    print(f"Wrote {output} ({size_kb:,.0f} KB)")
    if not complete:
        print("\n*** INCOMPLETE: the BITS Virtual Lab screenshot is a placeholder. ***")
        print("*** Re-run with --bits-lab-screenshot <path> before submitting.   ***")


if __name__ == "__main__":
    main()
