#!/usr/bin/env python
"""
构建 lesson new 的单文件离线平板版 (tablet/)
将 assets/course.css 内联进每个课件 HTML 中，生成可完全脱机单文件阅读的 HTML。
"""
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ASSETS_CSS = ROOT / "assets" / "course.css"
OUT = ROOT / "tablet"

def inline(html_path: Path) -> str:
    text = html_path.read_text(encoding="utf-8")
    if not ASSETS_CSS.exists():
        return text
    css = ASSETS_CSS.read_text(encoding="utf-8")
    
    # 替换外链样式表为内联 <style>
    pattern = re.compile(
        r'<link rel="stylesheet" href="(?:\.\./assets/course\.css|assets/course\.css)">', re.I
    )
    if pattern.search(text):
        css_inline = "<style>\n" + css + "\n</style>"
        text = pattern.sub(css_inline, text, count=1)
        
    return text

def main():
    OUT.mkdir(exist_ok=True)
    subdirs = ["01_CFDPython", "02_DeepXDE", "03_FNO", "04_MIT18086_理论导学", "reference"]
    
    total = 0
    for sdir in subdirs:
        src_dir = ROOT / sdir
        if not src_dir.exists(): continue
        dst_dir = OUT / sdir
        dst_dir.mkdir(parents=True, exist_ok=True)
        
        for html_file in sorted(src_dir.glob("*.html")):
            inlined_text = inline(html_file)
            dst_file = dst_dir / html_file.name
            dst_file.write_text(inlined_text, encoding="utf-8")
            total += 1
            
    # Also inline index.html
    master_index = ROOT / "index.html"
    if master_index.exists():
        inlined_index = inline(master_index)
        (OUT / "index.html").write_text(inlined_index, encoding="utf-8")
        total += 1
        
    print(f"Successfully generated {total} offline tablet lessons in lesson new/tablet/")

if __name__ == "__main__":
    main()
