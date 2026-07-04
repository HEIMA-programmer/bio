"""从系统的 Noto CJK .ttc 提取**简体 SC** 面为独立 OTF，供 theme.py 使用。

背景：matplotlib 只能读取 .ttc 集合里的第一个 face（在本机上是 JP 变体），
无法直接选简体 SC。这个脚本用 fonttools 把 SC 面单独抽出来存成 OTF，
theme.py 会优先加载它，保证中文用简体字形。

前置：系统已装 fonts-noto-cjk（`sudo apt-get install -y fonts-noto-cjk`）、
      并 `pip install fonttools`。
用法：cd bio/scripts/figgen && python3 extract_sc_font.py
产出：figgen/fonts/NotoSansCJKsc-{Regular,Bold}.otf（约 33MB，已在 .gitignore 中忽略）
"""
import os
from fontTools.ttLib import TTFont

SRCS = {
    "Regular": "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "Bold": "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
}
OUTDIR = os.path.join(os.path.dirname(__file__), "fonts")


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    for style, path in SRCS.items():
        if not os.path.exists(path):
            print(f"[跳过] 找不到 {path}（先装 fonts-noto-cjk）")
            continue
        saved = False
        for i in range(0, 12):
            try:
                f = TTFont(path, fontNumber=i, lazy=True)
            except Exception:
                break
            fam = (f["name"].getDebugName(1) or "").strip()
            if fam == "Noto Sans CJK SC":
                out = os.path.join(OUTDIR, f"NotoSansCJKsc-{style}.otf")
                f.save(out)
                print(f"[OK] {out}  (来自 face {i})")
                saved = True
                f.close()
                break
            f.close()
        if not saved:
            print(f"[警告] {path} 里没找到 'Noto Sans CJK SC' 面")


if __name__ == "__main__":
    main()
