import json
from pathlib import Path

import pymupdf as fitz
from PIL import Image, ImageDraw

DOCS_DIR = Path(__file__).parent / "docs_test"

with open(DOCS_DIR / "resultat_parsing_2206.01062v1-pages-6.json") as f:
    data = json.load(f)

items = data["items"]["pages"][0]["items"]

doc = fitz.open(DOCS_DIR / "2206.01062v1-pages-6.pdf")
page = doc[0]
zoom = 2
pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
draw = ImageDraw.Draw(img)

colors = {
    "header": "red",
    "footer": "orange",
    "text": "blue",
    "table": "green",
    "link": "purple"
}

for idx, item in enumerate(items):
    color = colors.get(item["type"], "black")
    for box in item["bbox"]:
        x, y, w, h = box["x"], box["y"], box["w"], box["h"]
        rect = [x*zoom, y*zoom, (x+w)*zoom, (y+h)*zoom]
        draw.rectangle(rect, outline=color, width=3)
        draw.text((x*zoom+2, y*zoom+2), f"{idx}:{item['type']}", fill=color)

img.save("verification_bbox_2206.01062v1-pages-6.png")