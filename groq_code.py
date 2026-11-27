
import os
import re
import json
import time
import base64
from pathlib import Path

import groq
import pymupdf
from dotenv import load_dotenv

load_dotenv()

MODEL = "qwen/qwen3.8-27b"
MAX_B64 = 4 * 1024 * 1024 # groq does not take images bigger than this (as base64)
MIN_SIDE = 50 # images smaller than this are icons, lines or logos, so they are skipped

PROMPT = """Extract ALL visible text from this image EXACTLY as written: same spelling, case, punctuation, numbers, symbols and dates. Never guess, correct or add anything. Write "[illegible]" for any text you can't read.
Reading order: top to bottom, left to right. Finish one column completely before starting the next.

Block types:
- "heading": titles and section headings (string)
- "text": paragraphs (string)
- "list": bullet or numbered items (array of strings)
- "key_value": form fields like "Label: Value" (object)
- "table": array of row objects, with the header row as keys

Table rules: empty or missing header -> "col1", "col2", ...; duplicate headers -> "Amount", "Amount_2"; merged cells -> repeat the value; empty cells -> ""; all values are strings; every row has the same keys.

Return ONLY valid JSON, with no markdown or explanation:
{"blocks": [
  {"type": "heading", "content": "INVOICE"},
  {"type": "key_value", "content": {"Invoice No": "INV-042"}},
  {"type": "table", "content": [{"Item": "Pen", "Qty": "2", "Price": "10.00"}]}
]}
If there is no text, return {"blocks": []}."""

TOTAL_KEYS = 0 # count the keys: GROQ_API_KEY_1 ... GROQ_API_KEY_N
while os.getenv(f"GROQ_API_KEY_{TOTAL_KEYS + 1}"):
    TOTAL_KEYS += 1
if not TOTAL_KEYS:
    raise SystemExit("No GROQ_API_KEY_1 found in .env")

key_no = 1 # number of the key we use now
dead = set() # numbers of keys that are invalid or expired


def next_key():
    """Move to the next key number (1, 2, ... N, back to 1), skipping dead ones."""
    global key_no
    for step in range(TOTAL_KEYS):
        key_no = key_no % TOTAL_KEYS + 1
        if key_no not in dead:
            return
    raise SystemExit("All Groq keys are invalid/expired")


def ask_qwen(b64, mime):
    """Send one image. If a key fails use the next one, and wait 30s after every key failed once."""
    for attempt in range(TOTAL_KEYS * 3):
        if key_no in dead:
            next_key()
        key = os.getenv(f"GROQ_API_KEY_{key_no}")
        message = [
            {"type": "text", "text": PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
        ]
        try:
            client = groq.Groq(api_key=key, max_retries=0)
            reply = client.chat.completions.create(
                model=MODEL, temperature=0, messages=[{"role": "user", "content": message}])
            return reply.choices[0].message.content or ""
        except (groq.AuthenticationError, groq.PermissionDeniedError):
            print(f"    GROQ_API_KEY_{key_no} invalid -> skipped")
            dead.add(key_no)
            next_key()
        except groq.BadRequestError:
            raise # the image itself is bad, another key will not help
        except groq.APIError as error:
            print(f"    GROQ_API_KEY_{key_no} {type(error).__name__} -> next key")
            next_key()
            if key_no == 1: # back at key 1 means every key was tried once, so wait a bit
                time.sleep(30)
    raise RuntimeError("All Groq keys keep failing")


def parse_json(text):
    """Remove the model's thinking and ``` marks, then read the JSON."""
    text = re.sub(r"<think>.*?</think>|```(?:json)?", "", text, flags=re.S)
    try:
        return json.loads(text[text.find("{"): text.rfind("}") + 1])
    except json.JSONDecodeError:
        return None


def extract_image(pdf, xref):
    """Get one image as PNG (or a smaller JPEG if it is too big for groq)."""
    pix = pymupdf.Pixmap(pdf, xref)

    if pix.colorspace is None or pix.colorspace.n not in (1, 3): # turn odd colour types (like CMYK) into normal RGB and remove transparency
        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
    if pix.alpha:
        pix = pymupdf.Pixmap(pix, 0)

    data, ext = pix.tobytes("png"), "png"
    while len(base64.b64encode(data)) > MAX_B64:
        pix.shrink(1)
        data, ext = pix.tobytes("jpeg", jpg_quality=90), "jpeg"
    return data, ext, pix


def position(page, xref):
    """Where the image sits on the page (top, left), so we read them in order."""
    rects = page.get_image_rects(xref)
    if not rects:
        return (0, 0)
    return (rects[0].y0, rects[0].x0)


def to_text(output):
    """Turn all blocks into one text: page by page, tables as json, lists as bullets."""
    lines = []
    last_page = None
    for item in output:
        if item["page"] != last_page:
            lines.append(f"\n===== Page {item['page']} =====")
            last_page = item["page"]
        content = item["content"]
        if item["type"] == "table":
            lines.append("[TABLE]\n" + json.dumps(content, indent=2, ensure_ascii=False) + "\n[/TABLE]")
        elif isinstance(content, list):
            lines.append("\n".join(f"• {str(point).lstrip('•- ').strip()}" for point in content))
        else:
            lines.append(str(content))
    return "\n\n".join(lines).strip() + "\n"


#main
def extract_text(pdf_path, images_dir):
    """PDF -> one text string (tables embedded as JSON). Also saves each image to images_dir/<pdf name>/."""
    img_dir = Path(images_dir) / Path(pdf_path).stem
    img_dir.mkdir(parents=True, exist_ok=True)
    pdf = pymupdf.open(pdf_path)
    output = []

    for page_no, page in enumerate(pdf, start=1):
        print(f"Page {page_no}/{len(pdf)}")

        xrefs = {image_info[0] for image_info in page.get_images()} # every image on the page, from top to bottom
        xrefs = sorted(xrefs, key=lambda xref: position(page, xref))

        for img_no, xref in enumerate(xrefs, start=1):
            data, ext, pix = extract_image(pdf, xref)
            if min(pix.width, pix.height) < MIN_SIDE:
                continue

            img_path = img_dir / f"page{page_no}_img{img_no}.{ext}" # save the image
            img_path.write_bytes(data)
            print(f"  {img_path.name} ({pix.width}x{pix.height}) -> {MODEL}")

            reply = ask_qwen(base64.b64encode(data).decode(), f"image/{ext}") # send it to qwen, if the reply is not json keep it as plain text
            result = parse_json(reply)
            if result:
                blocks = result.get("blocks", [])
            else:
                blocks = [{"type": "text", "content": reply.strip()}]

            for block in blocks:
                if block.get("content"):
                    output.append({"page": page_no, "type": block.get("type", "text"), "content": block["content"]})

    pdf.close()
    return to_text(output)
