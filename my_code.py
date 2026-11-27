import re
import json

import cv2
import pymupdf
import numpy as np
import pytesseract
from rapidocr import RapidOCR
from rapid_table import RapidTable, RapidTableInput
from rapid_table_det.inference import TableDetector

ocr = RapidOCR() # load the models once at the start
detector = TableDetector()
table_engine = RapidTable(RapidTableInput())


def upscale(img, rect):
    """Small, blurry images (under 150 dpi) are made bigger, to 200 dpi."""
    dpi = img.shape[1] / (rect.width / 72)
    if dpi >= 150:
        return img
    zoom = 200 / dpi
    return cv2.resize(img, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_CUBIC)


def tesseract(gray, psm=3):
    """Read text with tesseract."""
    return pytesseract.image_to_string(gray, config=f"--psm {psm}").strip()


def dark_sidebars(gray):
    """Find tall dark side columns with light text (like a green or blue left column)."""
    height, width = gray.shape

    background = cv2.medianBlur(gray, max(31, (min(height, width) // 20) | 1)) # blur away the text so only the background colour is left, then look for dark areas
    dark = (background < 100).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))

    count, labels, stats, centers = cv2.connectedComponentsWithStats(dark) # keep only tall dark blocks
    boxes = []
    for x, y, w, h, area in stats[1:]:
        if h > 0.4 * height and w > 80 and area > 0.03 * height * width:
            boxes.append((x, y, x + w, y + h))
    return boxes


def html_to_rows(html): #the radpid ocr gives output in html so it needed to be cleaned so converting it into rowns
    rows = []
    for row_html in re.findall(r"<tr>(.*?)</tr>", html, re.S):
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row_html, re.S)
        rows.append([re.sub(r"<.*?>", "", cell).strip() for cell in cells])
    return rows


def rows_to_json(rows):
    """First row becomes the column names, the other rows become the data."""
    rows = rows[:1] + [row for row in rows[1:] if any(cell.strip() for cell in row)] # drop empty rows, but keep the header row

    columns = max((len(row) for row in rows), default=0) # 1 column or 1 row is not a table
    if columns < 2 or len(rows) < 2:
        return None

    rows = [row + [""] * (columns - len(row)) for row in rows] # give short rows empty cells so every row has the same columns
    headers = [name or f"col{i + 1}" for i, name in enumerate(rows[0])]
    return [dict(zip(headers, row)) for row in rows[1:]]


def is_real_table(box, img, table):
    """The detector also marks page layouts and bullet lists as tables, this rejects those."""
    x1, y1, x2, y2 = box

    if (x2 - x1) * (y2 - y1) > 0.25 * img.shape[0] * img.shape[1]: # a real table is not a quarter of the page
        return False

    cells = [value for row in table for value in row.values()] + list(table[0].keys()) # most cells should have text, and not be bullet points
    filled = [cell for cell in cells if cell and not cell.startswith("col")]
    bullets = sum(cell.lstrip().startswith(("•", "·", "-", "*")) for cell in filled)
    return len(filled) > 0.5 * len(cells) and bullets < 0.3 * len(filled)


def read_tables(img):
    """Returns [(top, bottom, table_json)] for every table on the image."""
    tables, elapsed = detector(img) # find where the tables are
    if not tables:
        return []

    words = ocr(img) # read all the words on the image with their positions
    if words.boxes is None:
        return []
    cx, cy = words.boxes.mean(axis=1).T
    top = words.boxes[:, :, 1].min(axis=1)
    bottom = words.boxes[:, :, 1].max(axis=1)
    txts = np.array(words.txts, dtype=object)
    scores = np.array(words.scores)

    for i in np.where(scores < 0.9)[0]: # words rapidocr is not sure about get read again with tesseract
        x1, y1 = words.boxes[i].min(axis=0).astype(int) - 3
        x2, y2 = words.boxes[i].max(axis=0).astype(int) + 3
        crop = cv2.cvtColor(img[max(0, y1):y2, max(0, x1):x2], cv2.COLOR_BGR2GRAY)
        if crop.size:
            text = pytesseract.image_to_string(crop, config="--psm 7").strip()
            if text:
                txts[i] = text

    found = []
    for t in tables:
        x1, y1, x2, y2 = map(int, t["box"])

        inside = (cx >= x1) & (cx <= x2) & (cy >= y1) & (cy <= y2) # the words that sit inside this table's box
        if not inside.any():
            continue

        ocr_results = [(words.boxes[inside] - [x1, y1], txts[inside], scores[inside])] # build the table from those words
        html = table_engine(img[y1:y2, x1:x2], ocr_results=ocr_results).pred_htmls[0]

        rows = html_to_rows(html) # turn it into json and skip it if it does not look like a real table
        table = rows_to_json(rows)
        if not table or not is_real_table((x1, y1, x2, y2), img, table):
            continue

        cell_text = " ".join(cell for row in rows for cell in row) # if the box also covered a heading that is not part of the table, make the box smaller
        left_out = [i for i in np.where(inside)[0] if txts[i] not in cell_text]
        middle = (y1 + y2) / 2
        below = [top[i] for i in left_out if top[i] > middle]
        above = [bottom[i] for i in left_out if bottom[i] < middle]
        if below:
            y2 = int(min(below)) - 2
        if above:
            y1 = int(max(above)) + 2
        found.append((y1, y2, table))
    return found


def read_image(img, rect):
    """Read one image. Returns [(position on page, "text" or "table", content)]."""
    img = upscale(img, rect) # make small images bigger and keep a gray copy for tesseract
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    scale = rect.height / img.shape[0] # to turn image position into page position
    items = []

    for x1, y1, x2, y2 in dark_sidebars(gray): # read dark side columns with the colours flipped, then paint them white so they are not read twice
        items.append((rect.y0 + y1 * scale, "text", tesseract(cv2.bitwise_not(gray[y1:y2, x1:x2]), psm=6)))
        gray[y1:y2, x1:x2] = 255
        img[y1:y2, x1:x2] = 255

    done_until = 0 # go down the page: text above a table, the table, then text after it
    tables = sorted(read_tables(img), key=lambda t: t[0])
    for y1, y2, table in tables:
        if y1 - done_until > 10:
            items.append((rect.y0 + done_until * scale, "text", tesseract(gray[done_until:y1])))
        items.append((rect.y0 + y1 * scale, "table", table))
        done_until = max(done_until, y2)

    if gray.shape[0] - done_until > 10: # whatever text is left at the bottom
        items.append((rect.y0 + done_until * scale, "text", tesseract(gray[done_until:])))
    return items


def load_image(pdf, xref):
    """Get an image out of the pdf as an opencv image."""
    data = pdf.extract_image(xref)["image"]
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

    if img is None: # some image types opencv can't open, convert them to png first
        pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.Pixmap(pdf, xref))
        img = cv2.imdecode(np.frombuffer(pix.tobytes("png"), np.uint8), cv2.IMREAD_COLOR)
    return img


# ---------------------------------------------------------------- main
def extract_text(pdf_path):
    """PDF -> one text string (tables embedded as JSON between [TABLE] and [/TABLE])."""
    output = []
    pdf = pymupdf.open(pdf_path)

    for page_no, page in enumerate(pdf, start=1):
        print(f"Page {page_no}/{len(pdf)}")

        images = [] # all the images on this page and where they sit
        for image_info in page.get_images():
            xref = image_info[0]
            for rect in page.get_image_rects(xref):
                images.append((xref, rect))
        rects = [rect for xref, rect in images]

        items = [] # normal typed text in the pdf is taken as it is, no ocr needed
        for block in page.get_text("blocks"):
            x0, y0, x1, y1, text, block_no, kind = block
            middle = pymupdf.Point((x0 + x1) / 2, (y0 + y1) / 2)
            on_image = any(middle in rect for rect in rects)
            if kind == 0 and not on_image:
                items.append((y0, "text", text.strip()))

        for xref, rect in images: # read every image on the page, skip tiny ones like icons and lines
            img = load_image(pdf, xref)
            if min(img.shape[:2]) >= 50:
                items += read_image(img, rect)

        items.sort(key=lambda item: item[0]) # put everything in order from top to bottom and join text that sits together
        for position, kind, content in items:
            if not content:
                continue
            last = output[-1] if output else None
            if kind == "text" and last and last["type"] == "text" and last["page"] == page_no:
                last["content"] += "\n\n" + content
            else:
                output.append({"page": page_no, "type": kind, "content": content})

    pdf.close()

    lines = [] # write it all as one text: page by page, tables as json
    last_page = None
    for item in output:
        if item["page"] != last_page:
            lines.append(f"\n===== Page {item['page']} =====")
            last_page = item["page"]
        if item["type"] == "table":
            lines.append("[TABLE]\n" + json.dumps(item["content"], indent=2, ensure_ascii=False) + "\n[/TABLE]")
        else:
            lines.append(item["content"])
    return "\n\n".join(lines).strip() + "\n"
