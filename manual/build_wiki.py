"""
AgentNate Manual Wiki Builder
Splits AgentNate-Manual.md into individual GitHub Wiki pages.

Output: manual/wiki/
  Home.md          - Landing page with intro + TOC links
  _Sidebar.md      - Navigation sidebar
  _Footer.md       - Common footer
  01-Introduction.md
  02-Installation-Getting-Started.md
  ...
  screenshots/     - Copied screenshots folder
"""
import os
import re
import shutil
from pathlib import Path

MANUAL_DIR = Path(__file__).parent
MD_FILE = MANUAL_DIR / "AgentNate-Manual.md"
WIKI_DIR = MANUAL_DIR / "wiki"
SCREENSHOTS_SRC = MANUAL_DIR / "screenshots"


def slugify(title: str) -> str:
    """Convert section title to wiki page filename slug."""
    # Remove markdown formatting
    s = title.strip().rstrip(".")
    # Remove leading ## and number prefix like "1. " or "28. "
    s = re.sub(r"^#+\s*", "", s)
    s = re.sub(r"^\d+\.\s*", "", s)
    # Convert to slug
    s = re.sub(r"[&]+", "and", s)
    s = re.sub(r"[^a-zA-Z0-9\s-]", "", s)
    s = re.sub(r"\s+", "-", s.strip())
    return s


def extract_number(title: str) -> str:
    """Extract section number from '## 1. Introduction' -> '01'."""
    m = re.match(r"^##\s+(\d+)\.", title)
    if m:
        return m.group(1).zfill(2)
    return ""


def split_sections(text: str) -> list[dict]:
    """Split markdown into sections on ## headers."""
    lines = text.split("\n")
    sections = []
    current = None

    for line in lines:
        if re.match(r"^## ", line):
            # Save previous section
            if current:
                sections.append(current)
            current = {
                "header": line,
                "number": extract_number(line),
                "slug": slugify(line),
                "lines": [line],
            }
        elif current:
            current["lines"].append(line)
        # Lines before first ## header are preamble (title, version, hr)
        # We'll use those in Home.md

    if current:
        sections.append(current)

    return sections


def extract_preamble(text: str) -> str:
    """Get everything before the first ## header."""
    lines = text.split("\n")
    preamble = []
    for line in lines:
        if re.match(r"^## ", line):
            break
        preamble.append(line)
    return "\n".join(preamble).strip()


def rewrite_image_paths(content: str) -> str:
    """Rewrite screenshot paths for wiki format."""
    # In the manual: ![alt](screenshots/foo.png) or after build_pdf fixup
    # In wiki: ![alt](screenshots/foo.png) — same relative path since we copy screenshots/
    # But also handle any file:// URLs that build_pdf might have introduced
    content = re.sub(
        r"!\[([^\]]*)\]\(file:///[^)]*screenshots/([^)]+)\)",
        r"![\1](screenshots/\2)",
        content,
    )
    return content


def rewrite_internal_links(content: str, section_map: dict) -> str:
    """Convert #anchor links to wiki page links."""
    def replace_link(m):
        text = m.group(1)
        anchor = m.group(2)

        # Try to find matching page
        anchor_clean = anchor.lstrip("#")
        if anchor_clean in section_map:
            page_name = section_map[anchor_clean]
            return f"[{text}]({page_name})"

        # Fuzzy match: anchor might be a subsection within a page
        # Keep as-is (anchor links work within a single page)
        return m.group(0)

    # Match [text](#anchor)
    content = re.sub(r"\[([^\]]+)\]\((#[^)]+)\)", replace_link, content)
    return content


def build_anchor_id(header: str) -> str:
    """Replicate GitHub's anchor ID generation from a header line."""
    s = re.sub(r"^#+\s*", "", header)
    s = re.sub(r"\d+\.\s*", "", s, count=1)
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s-]", "", s)
    s = re.sub(r"\s+", "-", s)
    return s


def build_section_map(sections: list[dict]) -> dict:
    """Map anchor IDs to wiki page filenames."""
    mapping = {}
    for sec in sections:
        anchor = build_anchor_id(sec["header"])
        filename = sec.get("filename", "")
        if filename:
            # Strip .md for wiki links
            page = filename.replace(".md", "")
            mapping[anchor] = page
    return mapping


def build_wiki():
    """Main build function."""
    print(f"Reading {MD_FILE.name}...")
    text = MD_FILE.read_text(encoding="utf-8")

    # Clean output dir
    if WIKI_DIR.exists():
        shutil.rmtree(WIKI_DIR)
    WIKI_DIR.mkdir()

    # Copy screenshots
    if SCREENSHOTS_SRC.exists():
        dest = WIKI_DIR / "screenshots"
        shutil.copytree(SCREENSHOTS_SRC, dest)
        count = len(list(dest.glob("*.png"))) + len(list(dest.glob("*.jpg")))
        print(f"Copied {count} screenshots")
    else:
        print("Warning: screenshots/ not found")

    # Extract preamble and sections
    preamble = extract_preamble(text)
    sections = split_sections(text)

    # Identify special sections
    toc_section = None
    numbered_sections = []
    unnumbered_sections = []

    for sec in sections:
        title_clean = re.sub(r"^##\s*", "", sec["header"]).strip()
        if title_clean == "Table of Contents":
            toc_section = sec
        elif sec["number"]:
            numbered_sections.append(sec)
        else:
            # Unnumbered sections like "Quick Reference Card" or "Agent Memory (persistent...)"
            # Check if it's a subsection (nested within a numbered section)
            # "Agent Memory (persistent across conversations)" is a sub-heading, skip as standalone
            if "persistent across conversations" in title_clean:
                # This is a sub-heading within Section 11, not a standalone page
                # It was already captured in the previous section's lines
                continue
            unnumbered_sections.append(sec)

    # Assign filenames
    for sec in numbered_sections:
        sec["filename"] = f"{sec['number']}-{sec['slug']}.md"

    for sec in unnumbered_sections:
        sec["filename"] = f"{sec['slug']}.md"

    all_page_sections = numbered_sections + unnumbered_sections

    # Build section anchor -> page mapping
    section_map = build_section_map(all_page_sections)

    # Write individual pages
    for sec in all_page_sections:
        content = "\n".join(sec["lines"])
        content = rewrite_image_paths(content)
        content = rewrite_internal_links(content, section_map)

        filepath = WIKI_DIR / sec["filename"]
        filepath.write_text(content.strip() + "\n", encoding="utf-8")
        print(f"  {sec['filename']}")

    # Build Home.md
    home_lines = [
        "# AgentNate User Manual",
        "",
        "**Version 2.0** | **GitHub**: [github.com/rookiemann/AgentNate](https://github.com/rookiemann/AgentNate)",
        "",
        "---",
        "",
        "AgentNate is a **local-first AI orchestration platform** that brings together large language models, "
        "workflow automation, image/video generation, text-to-speech, and music generation into a single "
        "unified interface. It runs entirely on your machine with no cloud dependencies required.",
        "",
        "## Contents",
        "",
    ]

    for sec in numbered_sections:
        page = sec["filename"].replace(".md", "")
        title = re.sub(r"^##\s*", "", sec["header"]).strip()
        home_lines.append(f"- [{title}]({page})")

    home_lines.append("")
    home_lines.append("### Appendices")
    home_lines.append("")
    for sec in unnumbered_sections:
        page = sec["filename"].replace(".md", "")
        title = re.sub(r"^##\s*", "", sec["header"]).strip()
        home_lines.append(f"- [{title}]({page})")

    home_lines.append("")

    home_path = WIKI_DIR / "Home.md"
    home_path.write_text("\n".join(home_lines), encoding="utf-8")
    print(f"  Home.md")

    # Build _Sidebar.md
    sidebar_lines = [
        "**[Home](Home)**",
        "",
        "---",
        "",
    ]

    # Group sections by category for sidebar
    groups = {
        "Getting Started": [],
        "Chat & Agents": [],
        "Workflows & Automation": [],
        "Creative Tools": [],
        "System & Config": [],
        "Reference": [],
    }

    for sec in numbered_sections:
        n = int(sec["number"])
        page = sec["filename"].replace(".md", "")
        # Short title for sidebar
        title = re.sub(r"^##\s*\d+\.\s*", "", sec["header"]).strip()

        if n <= 3:
            groups["Getting Started"].append((title, page))
        elif n <= 11:
            groups["Chat & Agents"].append((title, page))
        elif n <= 14:
            groups["Workflows & Automation"].append((title, page))
        elif n <= 17:
            groups["Creative Tools"].append((title, page))
        elif n <= 21:
            groups["System & Config"].append((title, page))
        else:
            groups["Reference"].append((title, page))

    for sec in unnumbered_sections:
        page = sec["filename"].replace(".md", "")
        title = re.sub(r"^##\s*", "", sec["header"]).strip()
        groups["Reference"].append((title, page))

    for group_name, items in groups.items():
        if not items:
            continue
        sidebar_lines.append(f"**{group_name}**")
        sidebar_lines.append("")
        for title, page in items:
            sidebar_lines.append(f"- [{title}]({page})")
        sidebar_lines.append("")

    sidebar_path = WIKI_DIR / "_Sidebar.md"
    sidebar_path.write_text("\n".join(sidebar_lines), encoding="utf-8")
    print(f"  _Sidebar.md")

    # Build _Footer.md
    footer = (
        "*AgentNate v2.0 — Local AI Orchestration Platform* | "
        "[GitHub](https://github.com/rookiemann/AgentNate)\n"
    )
    footer_path = WIKI_DIR / "_Footer.md"
    footer_path.write_text(footer, encoding="utf-8")
    print(f"  _Footer.md")

    # Summary
    total_pages = len(all_page_sections) + 3  # +Home, Sidebar, Footer
    print(f"\nWiki built: {total_pages} files in {WIKI_DIR}/")
    print(f"  {len(numbered_sections)} numbered sections")
    print(f"  {len(unnumbered_sections)} appendix sections")
    print(f"  3 special files (Home, Sidebar, Footer)")


if __name__ == "__main__":
    build_wiki()
