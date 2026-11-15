"""Convert Assignment2_Report.md -> Assignment2_Report.pdf (pure-Python, no system deps)."""
import sys
from pathlib import Path
import markdown
from xhtml2pdf import pisa

SRC = Path(__file__).resolve().parent / "Assignment2_Report.md"
DST = Path(__file__).resolve().parent / "Assignment2_Report.pdf"

CSS = """
@page { size: A4; margin: 2cm 2cm 2.2cm 2cm; }
body { font-family: Helvetica, Arial, sans-serif; font-size: 10.5pt; color: #1a1a1a; line-height: 1.45; }
h1 { font-size: 19pt; color: #1f3b63; border-bottom: 2px solid #1f3b63; padding-bottom: 4px; margin-top: 4px; }
h2 { font-size: 13.5pt; color: #1f3b63; margin-top: 16px; border-bottom: 1px solid #c9d4e3; padding-bottom: 2px; }
h3 { font-size: 11.5pt; color: #2c4a73; margin-top: 12px; }
p, li { font-size: 10.5pt; }
code { font-family: Courier, monospace; background: #f2f4f7; font-size: 9.5pt; }
th code { background: transparent; color: #ffffff; }
td code { background: transparent; }
pre { background: #f2f4f7; border: 1px solid #d8dee6; padding: 8px; font-size: 9pt;
      font-family: Courier, monospace; }
table { border-collapse: collapse; width: 100%; margin: 8px 0; }
th { background: #1f3b63; color: #ffffff; font-size: 9.5pt; padding: 5px 7px; text-align: left; }
td { border: 1px solid #c9d4e3; font-size: 9.5pt; padding: 4px 7px; }
tr:nth-child(even) td { background: #f5f8fb; }
hr { border: 0; border-top: 1px solid #c9d4e3; margin: 14px 0; }
strong { color: #10233d; }
"""

def main():
    md_text = SRC.read_text(encoding="utf-8")
    html_body = markdown.markdown(md_text, extensions=["tables", "fenced_code", "sane_lists"])
    html = f"<html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{html_body}</body></html>"
    with open(DST, "wb") as f:
        result = pisa.CreatePDF(html, dest=f, encoding="utf-8")
    if result.err:
        print("PDF generation had errors", file=sys.stderr); sys.exit(1)
    print("wrote", DST, f"({DST.stat().st_size/1024:.0f} KB)")

if __name__ == "__main__":
    main()
