"""
AgentNate Manual PDF Builder
Converts AgentNate-Manual.md → AgentNate-Manual.pdf
Uses: markdown (MD→HTML) + Playwright Chromium (HTML→PDF)
"""
import os
import sys
import asyncio
import markdown
from pathlib import Path

MANUAL_DIR = Path(__file__).parent
MD_FILE = MANUAL_DIR / "AgentNate-Manual.md"
HTML_FILE = MANUAL_DIR / "AgentNate-Manual.html"
PDF_FILE = MANUAL_DIR / "AgentNate-Manual.pdf"
SCREENSHOTS_DIR = MANUAL_DIR / "screenshots"

CSS = """
@page {
    size: A4;
    margin: 20mm 18mm 20mm 18mm;
}

* { box-sizing: border-box; }

body {
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, 'Helvetica Neue', Arial, sans-serif;
    font-size: 10.5pt;
    line-height: 1.55;
    color: #1a1a1a;
    max-width: 100%;
    margin: 0;
    padding: 0;
}

/* Title page */
h1 {
    font-size: 28pt;
    font-weight: 700;
    color: #0f172a;
    border-bottom: 3px solid #2563eb;
    padding-bottom: 12px;
    margin-top: 0;
    margin-bottom: 8px;
    page-break-before: avoid;
}

h2 {
    font-size: 18pt;
    font-weight: 700;
    color: #1e293b;
    margin-top: 28px;
    margin-bottom: 10px;
    padding-bottom: 6px;
    border-bottom: 2px solid #e2e8f0;
    page-break-after: avoid;
}

h3 {
    font-size: 13pt;
    font-weight: 600;
    color: #334155;
    margin-top: 20px;
    margin-bottom: 8px;
    page-break-after: avoid;
}

h4 {
    font-size: 11pt;
    font-weight: 600;
    color: #475569;
    margin-top: 16px;
    margin-bottom: 6px;
    page-break-after: avoid;
}

p {
    margin: 6px 0;
    orphans: 3;
    widows: 3;
}

/* Links */
a {
    color: #2563eb;
    text-decoration: none;
}

/* Bold */
strong {
    font-weight: 600;
    color: #0f172a;
}

/* Inline code */
code {
    background: #f1f5f9;
    border: 1px solid #e2e8f0;
    border-radius: 3px;
    padding: 1px 5px;
    font-family: 'Cascadia Code', 'Consolas', 'Monaco', monospace;
    font-size: 9pt;
    color: #be185d;
}

/* Code blocks */
pre {
    background: #1e293b;
    color: #e2e8f0;
    border-radius: 6px;
    padding: 12px 16px;
    margin: 10px 0;
    overflow-x: auto;
    font-size: 8.5pt;
    line-height: 1.45;
    page-break-inside: avoid;
}

pre code {
    background: none;
    border: none;
    padding: 0;
    color: #e2e8f0;
    font-size: 8.5pt;
}

/* Tables */
table {
    border-collapse: collapse;
    width: 100%;
    margin: 10px 0;
    font-size: 9.5pt;
    page-break-inside: auto;
}

thead {
    background: #f8fafc;
}

th {
    border: 1px solid #cbd5e1;
    padding: 7px 10px;
    text-align: left;
    font-weight: 600;
    color: #1e293b;
    background: #f1f5f9;
}

td {
    border: 1px solid #e2e8f0;
    padding: 6px 10px;
    vertical-align: top;
}

tr:nth-child(even) td {
    background: #f8fafc;
}

/* Lists */
ul, ol {
    margin: 6px 0;
    padding-left: 24px;
}

li {
    margin: 3px 0;
}

/* Blockquotes */
blockquote {
    border-left: 4px solid #2563eb;
    margin: 10px 0;
    padding: 8px 16px;
    background: #eff6ff;
    color: #1e40af;
    font-style: italic;
    page-break-inside: avoid;
}

blockquote p {
    margin: 4px 0;
}

blockquote strong {
    color: #1e3a8a;
}

/* Horizontal rules */
hr {
    border: none;
    border-top: 1px solid #e2e8f0;
    margin: 24px 0;
}

/* Images / Screenshots */
img {
    max-width: 100%;
    height: auto;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    margin: 8px 0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    page-break-inside: avoid;
}

/* ASCII art diagrams - keep them contained */
pre code .diagram {
    font-size: 7.5pt;
    line-height: 1.2;
}

/* TOC styling */
h2 + ul {
    column-count: 1;
}

/* Syntax highlighting (Pygments) */
.codehilite .hll { background-color: #49483e }
.codehilite .c { color: #75715e }
.codehilite .k { color: #66d9ef }
.codehilite .n { color: #f8f8f2 }
.codehilite .o { color: #f92672 }
.codehilite .p { color: #f8f8f2 }
.codehilite .s { color: #e6db74 }
.codehilite .m { color: #ae81ff }
.codehilite .l { color: #ae81ff }
.codehilite .nf { color: #a6e22e }
.codehilite .na { color: #a6e22e }
.codehilite .nb { color: #f8f8f2 }
.codehilite .nc { color: #a6e22e }
.codehilite .nd { color: #a6e22e }
.codehilite .ni { color: #f8f8f2 }
.codehilite .ne { color: #a6e22e }
.codehilite .nn { color: #f8f8f2 }
.codehilite .nt { color: #f92672 }
.codehilite .kn { color: #f92672 }
.codehilite .kd { color: #66d9ef }
.codehilite .kc { color: #66d9ef }
.codehilite .kr { color: #66d9ef }
.codehilite .kt { color: #66d9ef }
.codehilite .sd { color: #e6db74 }
.codehilite .s2 { color: #e6db74 }
.codehilite .s1 { color: #e6db74 }
.codehilite .si { color: #e6db74 }
.codehilite .se { color: #ae81ff }
.codehilite .ss { color: #e6db74 }

/* Print-specific */
@media print {
    body { font-size: 10pt; }
    h2 { page-break-before: auto; }
    pre, blockquote, table, img { page-break-inside: avoid; }
    a { color: #2563eb; }
    a[href^="http"]::after { content: none; }
}
"""


def build_html():
    """Convert markdown to styled HTML with embedded images."""
    print(f"Reading {MD_FILE.name}...")
    md_text = MD_FILE.read_text(encoding="utf-8")

    # Fix image paths to absolute file:// URLs for Playwright
    screenshots_url = SCREENSHOTS_DIR.as_uri()
    md_text = md_text.replace("](screenshots/", f"]({screenshots_url}/")

    print("Converting Markdown to HTML...")
    extensions = [
        "tables",
        "fenced_code",
        "codehilite",
        "toc",
        "smarty",
        "attr_list",
        "md_in_html",
    ]
    extension_configs = {
        "codehilite": {
            "css_class": "codehilite",
            "guess_lang": True,
            "noclasses": False,
        },
        "toc": {
            "permalink": False,
        },
    }

    html_body = markdown.markdown(
        md_text,
        extensions=extensions,
        extension_configs=extension_configs,
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AgentNate User Manual</title>
    <style>{CSS}</style>
</head>
<body>
{html_body}
</body>
</html>"""

    HTML_FILE.write_text(html, encoding="utf-8")
    print(f"HTML saved: {HTML_FILE.name} ({len(html):,} chars)")
    return HTML_FILE


async def build_pdf(html_path):
    """Use Playwright Chromium to render HTML as PDF."""
    from playwright.async_api import async_playwright

    print("Launching Chromium for PDF rendering...")
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()

        file_url = html_path.as_uri()
        await page.goto(file_url, wait_until="networkidle")

        # Wait for all images to load
        await page.wait_for_timeout(2000)

        print("Generating PDF (this may take a minute)...")
        await page.pdf(
            path=str(PDF_FILE),
            format="A4",
            margin={
                "top": "20mm",
                "bottom": "20mm",
                "left": "18mm",
                "right": "18mm",
            },
            print_background=True,
            display_header_footer=True,
            header_template='<div style="font-size:8pt;color:#94a3b8;width:100%;text-align:center;padding:4px 0;margin:0 auto;">AgentNate User Manual v2.0</div>',
            footer_template='<table style="width:100%;border:none;"><tr><td style="text-align:center;font-size:8pt;font-family:Arial,sans-serif;color:#94a3b8;border:none;padding:0;"><span class="pageNumber"></span> of <span class="totalPages"></span></td></tr></table>',
        )

        await browser.close()

    size_mb = PDF_FILE.stat().st_size / (1024 * 1024)
    print(f"\nPDF saved: {PDF_FILE.name} ({size_mb:.1f} MB)")
    print(f"Location: {PDF_FILE}")


def main():
    if not MD_FILE.exists():
        print(f"Error: {MD_FILE} not found")
        sys.exit(1)

    if not SCREENSHOTS_DIR.exists():
        print(f"Warning: {SCREENSHOTS_DIR} not found - images will be missing")

    html_path = build_html()
    asyncio.run(build_pdf(html_path))

    # Clean up HTML (keep PDF only)
    if "--keep-html" not in sys.argv:
        HTML_FILE.unlink(missing_ok=True)
        print("Cleaned up intermediate HTML")
    else:
        print(f"HTML kept: {HTML_FILE}")

    print("\nDone!")


if __name__ == "__main__":
    main()
