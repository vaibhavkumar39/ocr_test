import io
import os
import re
import json
import time
import base64
from pathlib import Path

import groq
import pymupdf
from PIL import Image
from dotenv import load_dotenv
from deepeval.metrics import GEval
from deepeval.models import DeepEvalBaseLLM
from deepeval.models.llms.multimodal import to_openai_content
from deepeval.test_case import LLMTestCase, MLLMImage, SingleTurnParams

load_dotenv() # load the groq keys from the .env file

VISION_MODEL = "qwen/qwen3.8-27b"
TEXT_MODEL = "openai/gpt-oss-120b"

key_no = 1 # number of the key we use now: GROQ_API_KEY_1, GROQ_API_KEY_2 ...


# ---------- judge: a Groq model that DeepEval can use ----------
def ask_groq(model, content):
    """Send one request. If a key fails, try the next one."""
    global key_no
    for attempt in range(60):
        key = os.getenv(f"GROQ_API_KEY_{key_no}")

        if not key: # no key with this number, so start again from key 1
            key_no = 1
            time.sleep(10)
            continue

        try:
            client = groq.Groq(api_key=key, max_retries=0)
            reply = client.chat.completions.create(
                model=model, temperature=0, messages=[{"role": "user", "content": content}])
            return reply.choices[0].message.content or ""
        except groq.BadRequestError:
            raise # the request itself is wrong, another key will not help
        except Exception as error:
            print(f"    GROQ_API_KEY_{key_no} failed ({type(error).__name__}) -> next key")
            key_no += 1

    raise RuntimeError("Groq keys keep failing")


def clean_json(text):
    """Remove the model's thinking and ``` marks, keep only the {...} part."""
    text = re.sub(r"<think>.*?</think>|```(?:json)?", "", text, flags=re.S)
    return text[text.find("{"): text.rfind("}") + 1]


class GroqJudge(DeepEvalBaseLLM): #This will let the code to use groq model as judge

    def generate(self, prompt, schema=None, **kwargs):
        content = to_openai_content(prompt) # the prompt from deepeval, with the page images inside it

        if not schema: # deepeval sometimes wants plain text back
            return ask_groq(self.name, content)

        keys = ", ".join(schema.model_json_schema()["properties"]) # other times it wants json, so tell the model which keys to send
        content.append({"type": "text", "text": "Reply ONLY with JSON with keys: " + keys})

        for attempt in range(3): # try 3 times to get good json
            text = clean_json(ask_groq(self.name, content))
            try:
                return schema.model_validate(json.loads(text))
            except ValueError:
                print("    judge reply was not valid json, asking again")

        raise RuntimeError("Judge did not return valid JSON 3 times")

    async def a_generate(self, prompt, schema=None, **kwargs): # deepeval needs these to exist
        return self.generate(prompt, schema)

    def load_model(self):
        return self.name

    def get_model_name(self):
        return self.name

    def supports_multimodal(self): # our test cases have page images
        return True


vision = GroqJudge(VISION_MODEL) # two judges: one that can see images, one that only reads text
text_model = GroqJudge(TEXT_MODEL)


# ---------- what we score ----------
def make_metrics():
    image_and_text = [SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT] # each metric is a list of simple steps the judge follows to give a score
    return [
        GEval(name="Accuracy", model=vision, evaluation_params=image_and_text, evaluation_steps=[
            "The input has the page image. The output is the text read from it.",
            "Compare words, names, numbers, emails,symbols, phone numbers and dates with the image.",
            "Lower the score for every misread, garbled or made-up word."]),
        GEval(name="Completeness", model=vision, evaluation_params=image_and_text, evaluation_steps=[
            "The input has the page image. The output is the text read from it.",
            "Lower the score for every heading, sidebar, bullet or line in the image that is missing from the output."]),
        GEval(name="Tables", model=vision, evaluation_params=image_and_text, evaluation_steps=[
            "Tables in the output are JSON between [TABLE] and [/TABLE].",
            "Check every table in the image is there with the right rows, columns and values.",
            "Lower the score for missing or wrong tables, or normal text turned into a table.",
            "If the image has no tables and the output has none, give full score."]),
        GEval(name="Structure", model=text_model, evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT], evaluation_steps=[
            "The output is text read from a document.",
            "Check reading order, headings, clean bullet lists and no unneccesary or garbage characters."]),
    ]


# ---------- page images (the "correct answer" the judge looks at) ----------
def page_images(pdf_path):
    """Qwen takes max 3 images, so pages are stacked into 3 (e.g. 5 pages -> 2+2+1)."""
    pdf = pymupdf.open(pdf_path) # turn every pdf page into a picture
    pages = [Image.open(io.BytesIO(page.get_pixmap(dpi=120).tobytes("png"))) for page in pdf]
    pdf.close()

    per_image = (len(pages) + 2) // 3 # how many pages go in one picture (rounded up)

    images = []
    for start in range(0, len(pages), per_image):
        group = pages[start:start + per_image]

        width = max(page.width for page in group) # put the pages one under the other on a white sheet
        height = sum(page.height for page in group)
        sheet = Image.new("RGB", (width, height), "white")
        y = 0
        for page in group:
            sheet.paste(page, (0, y))
            y += page.height

        buffer = io.BytesIO() # save it as jpeg and wrap it so deepeval knows it is an image
        sheet.save(buffer, "JPEG", quality=80)
        image_b64 = base64.b64encode(buffer.getvalue()).decode()
        images.append(str(MLLMImage(dataBase64=image_b64, mimeType="image/jpeg")))

    return " ".join(images) # deepeval finds the images inside this text


# ---------- scoring ----------
def score_case(case):
    """4 metrics on one test case -> {"scores", "reasons", "avg"} or {"error"}."""
    metrics = make_metrics()
    try:
        for metric in metrics:
            metric.measure(case, _show_indicator=False)
    except Exception as error:
        return {"error": str(error)} # one bad file should not stop the whole run

    scores = {metric.name: metric.score for metric in metrics} # the score and the judge's reason for each metric, plus the average
    reasons = {metric.name: metric.reason for metric in metrics}
    return {"scores": scores, "reasons": reasons, "avg": sum(scores.values()) / len(scores)}


def score_pdf(pdf_path, groq_text, local_text):
    """Score both outputs of one PDF -> {"groq": {...}, "local": {...}} (used by main.py)."""
    images = page_images(pdf_path) # same page images for both, so the scoring is fair
    groq_case = LLMTestCase(input=f"Page image(s): {images}", actual_output=groq_text)
    local_case = LLMTestCase(input=f"Page image(s): {images}", actual_output=local_text)
    return {"groq": score_case(groq_case), "local": score_case(local_case)}
