# Local PDF OCR

`my_code.py` reads text and tables from PDFs, including scanned ones. Everything runs on your own computer, with no internet and no big AI models.

## What you get

One `.txt` file for each PDF:

```
===== Page 1 =====

AARAV SHARMA
Senior Product & Data Analyst

CORE COMPETENCIES & TECHNICAL STACK

[TABLE]
[
  {"col1": "Analytics & BI:", "col2": "SQL, Python, Tableau, Power BI"},
  {"col1": "Data Engineering:", "col2": "Snowflake, BigQuery, Airflow"}
]
[/TABLE]

PROFESSIONAL EXPERIENCE
• Spearheaded end-to-end product analytics ...
```

Normal text stays as text. Tables are saved as JSON between `[TABLE]` and `[/TABLE]`.

## Setup

1. Install Tesseract:
   ```bash
   sudo apt install tesseract-ocr
   ```
2. Install the Python packages:
   ```bash
   pip install pymupdf numpy pytesseract rapidocr onnxruntime rapid_table rapid-table-det
   ```

The table models (about 220 MB) download automatically the first time you run it.

## Run

Start the app, upload your PDFs in the sidebar and press **Process**:
```bash
streamlit run main.py
```

The result is saved to `data/local_doc/<pdf name>.txt`.

## How it works

| Step | What happens | Library |
|---|---|---|
| 1. Open the PDF | Typed text is taken directly, and images are pulled out | PyMuPDF |
| 2. Fix blurry images | Small, blurry images are made bigger so letters are clearer | OpenCV |
| 3. Read dark side columns | Light text on a dark background is read with the colours flipped | OpenCV + Tesseract |
| 4. Find tables | Spots where the tables are on the image | rapid-table-det |
| 5. Read table words | Reads the words inside each table | RapidOCR (+ Tesseract for unclear words) |
| 6. Build the table | Works out the rows and columns, then saves it as JSON | rapid_table |
| 7. Read the rest | Reads all other text, from top to bottom | Tesseract (pytesseract) |
| 8. Save | Puts everything in order from top to bottom and saves one text file | Python |

It takes about 4–15 seconds per page.

## Size

| What | Size |
|---|---|
| Python libraries | ~550 MB (biggest: OpenCV 180, PyMuPDF 120, onnxruntime 80, NumPy 65) |
| Table models (downloaded on first run) | ~220 MB |
| Tesseract | ~15 MB |
| **Total disk** | **~800 MB** |
| **RAM while running** | **~750 MB** |
