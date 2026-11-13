import markdown
from pathlib import Path

css = """
<style>
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    line-height: 1.6;
    max-width: 900px;
    margin: 40px auto;
    padding: 0 20px;
    color: #24292e;
}
h1, h2, h3, h4 {
    color: #1f2328;
    margin-top: 24px;
    margin-bottom: 16px;
    border-bottom: 1px solid #d0d7de;
    padding-bottom: 8px;
}
table {
    border-collapse: collapse;
    width: 100%;
    margin: 16px 0;
}
th, td {
    border: 1px solid #d0d7de;
    padding: 8px 12px;
    text-align: left;
}
th {
    background-color: #f6f8fa;
    font-weight: 600;
}
tr:nth-child(even) {
    background-color: #f6f8fa;
}
code {
    background-color: #eff1f3;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 85%;
    font-family: ui-monospace, SFMono-Regular, monospace;
}
pre {
    background-color: #f6f8fa;
    padding: 16px;
    border-radius: 6px;
    overflow: auto;
    border: 1px solid #d0d7de;
}
pre code {
    background: none;
    padding: 0;
}
blockquote {
    border-left: 4px solid #0969da;
    margin: 0;
    padding-left: 16px;
    color: #57606a;
}
hr {
    border: none;
    border-top: 1px solid #d0d7de;
    margin: 24px 0;
}
@media print {
    body {
        margin: 10mm;
        max-width: 100%;
    }
    pre, table {
        page-break-inside: avoid;
    }
}
</style>
"""

files = ["report.md", "reports/FINAL_REPORT.md", "reports/error_analysis.md"]

for md_file in files:
    p = Path(md_file)
    if p.exists():
        text = p.read_text(encoding="utf-8")
        html_body = markdown.markdown(text, extensions=["tables", "fenced_code"])
        full_html = f"<!DOCTYPE html>\n<html>\n<head>\n<meta charset=\"utf-8\">\n<title>{p.stem}</title>\n{css}\n</head>\n<body>\n{html_body}\n</body>\n</html>"
        out_path = p.with_suffix(".html")
        out_path.write_text(full_html, encoding="utf-8")
        print(f"Rendered: {out_path}")
