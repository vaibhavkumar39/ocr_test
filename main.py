
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

import my_code # local OCR (tesseract + rapidocr)
import groq_code # Groq OCR (qwen vision)
import eval as judge # DeepEval (scores both outputs against the page images)

DATA = Path(__file__).resolve().parent / "data"
DIRS = {name: DATA / name for name in ("pdfs", "local_doc", "groq_doc", "images", "results", "runs")}
for folder in DIRS.values():
    folder.mkdir(parents=True, exist_ok=True)

BOTH = [("groq", "Groq"), ("local", "Local")]

st.set_page_config(page_title="OCR compare", layout="wide")
st.title("OCR compare: Local vs Groq")


def read(path):
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""


def scored(r):
    """True when both outputs of a file got scores."""
    return "scores" in r["groq"] and "scores" in r["local"]


def winner(r):
    groq_avg, local_avg = r["groq"]["avg"], r["local"]["avg"]
    if abs(groq_avg - local_avg) < 0.02:
        return "Tie"
    if groq_avg > local_avg:
        return "Groq"
    return "Local"


def show_outputs(name, r, systems, key, height=600):
    """Scores table and the output text of one file. systems = [("groq", "Groq"), ("local", "Local")]"""
    table = {}
    for system, label in systems:
        if "scores" in r[system]:
            table[label] = {**r[system]["scores"], "average": r[system]["avg"]}
    if table:
        st.dataframe(pd.DataFrame(table).round(2), width="stretch")

    for system, label in systems:
        if "error" in r[system]:
            st.error(f"{label} scoring failed: {r[system]['error']}")

    columns = st.columns(len(systems)) # one column per output, side by side
    for column, (system, label) in zip(columns, systems):
        column.markdown(f"**{label} output**")
        column.text_area(label, r.get(f"{system}_text") or "(missing)", height=height,
                         label_visibility="collapsed", key=f"{key}_{name}_{system}")


def run_name(path):
    when = datetime.strptime(path.stem, "%Y-%m-%d_%H-%M-%S")
    count = len(json.loads(read(path))["files"])
    return f"Run {when:%d %b %H:%M} ({count} PDFs)"


# ---------- sidebar: upload + pick what to view ----------
runs = sorted(DIRS["runs"].glob("*.json"), reverse=True) # newest run first
run_label = {path.stem: run_name(path) for path in runs}

if "open_run" in st.session_state: # after processing, show the run that just finished
    st.session_state["view"] = st.session_state.pop("open_run")

with st.sidebar:
    st.header("Add PDFs")
    uploads = st.file_uploader("PDF files", type="pdf", accept_multiple_files=True)
    redo = st.checkbox("Redo steps already done", value=False,
                       help="Off = skip files already extracted/scored (saves credits)")
    run = st.button("Process", type="primary", disabled=not uploads)

    st.header("View results")
    view = st.selectbox("Show", ["all"] + [path.stem for path in runs], key="view",
                        format_func=lambda v: "All saved results" if v == "all" else run_label[v])


# ---------- 1) process new PDFs (and save them as a run) ----------
if run:
    bar = st.progress(0.0)
    snapshot = {}
    for i, up in enumerate(uploads):
        name = Path(up.name).stem
        pdf_path = DIRS["pdfs"] / up.name
        pdf_path.write_bytes(up.getvalue())
        local_txt = DIRS["local_doc"] / f"{name}.txt"
        groq_txt = DIRS["groq_doc"] / f"{name}.txt"
        result_file = DIRS["results"] / f"{name}.json"
        steps = []

        with st.status(up.name, expanded=False) as status:
            if redo or not local_txt.exists(): # step 1: local OCR (skipped if already saved)
                status.update(label=f"{up.name}: local OCR...")
                local_txt.write_text(my_code.extract_text(pdf_path), encoding="utf-8")
                steps.append("Local OCR: done")
            else:
                steps.append("Local OCR: skipped (already saved)")

            if redo or not groq_txt.exists(): # step 2: Groq OCR (skipped if already saved)
                status.update(label=f"{up.name}: Groq OCR...")
                groq_txt.write_text(groq_code.extract_text(pdf_path, DIRS["images"]), encoding="utf-8")
                steps.append("Groq OCR: done")
            else:
                steps.append("Groq OCR: skipped (already saved)")

            if redo or not result_file.exists() or '"error"' in read(result_file): # step 3: scoring (done again if it failed last time)
                status.update(label=f"{up.name}: evaluating...")
                scores = judge.score_pdf(pdf_path, read(groq_txt), read(local_txt))
                result_file.write_text(json.dumps(scores, indent=2, ensure_ascii=False), encoding="utf-8")
                steps.append("Scoring: done")
            else:
                steps.append("Scoring: skipped (already saved)")

            result = json.loads(read(result_file))
            for step in steps:
                st.write(step)
            if scored(result):
                status.update(label=f"{up.name}: done", state="complete")
            else:
                status.update(label=f"{up.name}: scoring failed", state="error")

        snapshot[name] = {"pdf": up.name, "steps": steps, **result,
                          "groq_text": read(groq_txt), "local_text": read(local_txt)}
        bar.progress((i + 1) / len(uploads))

    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") # save this run and reload the page showing it
    run_file = DIRS["runs"] / f"{run_id}.json"
    run_file.write_text(json.dumps({"files": snapshot}, indent=2, ensure_ascii=False), encoding="utf-8")
    st.session_state["open_run"] = run_id
    st.rerun()


# ---------- 2) load what to show ----------
if view == "all":
    results = {} # latest result of every file ever processed
    for path in sorted(DIRS["results"].glob("*.json")):
        results[path.stem] = {**json.loads(read(path)),
                              "groq_text": read(DIRS["groq_doc"] / f"{path.stem}.txt"),
                              "local_text": read(DIRS["local_doc"] / f"{path.stem}.txt")}
else:
    results = json.loads(read(DIRS["runs"] / f"{view}.json"))["files"] # exactly what one upload produced

if not results:
    st.info("Upload PDFs in the sidebar and press Process.")
    st.stop()

st.subheader("All saved results" if view == "all" else run_label[view]) # list of files, like while processing: click one to see its scores and both texts
for name, r in results.items():
    pdf = r.get("pdf", f"{name}.pdf")
    if scored(r):
        label, state = f"{pdf}: done", "complete"
    else:
        label, state = f"{pdf}: scoring failed", "error"
    with st.status(label, state=state, expanded=False):
        if r.get("steps"):
            st.caption(" · ".join(r["steps"]))
        show_outputs(name, r, BOTH, key="list", height=350)

ok = {name: r for name, r in results.items() if scored(r)} # only files where both outputs got scores


# ---------- 3) whole comparison ----------
st.header("Overall")
if ok:
    metrics = list(next(iter(ok.values()))["groq"]["scores"])
    per_metric = {}
    overall = {}
    for system, label in BOTH:
        per_metric[label] = {m: sum(r[system]["scores"][m] for r in ok.values()) / len(ok) for m in metrics}
        overall[system] = sum(r[system]["avg"] for r in ok.values()) / len(ok)
    wins = {"Groq": 0, "Local": 0, "Tie": 0}
    for r in ok.values():
        wins[winner(r)] += 1

    c = st.columns(4)
    c[0].metric("Groq average", f"{overall['groq']:.2f}")
    c[1].metric("Local average", f"{overall['local']:.2f}", f"{overall['local'] - overall['groq']:+.2f} vs Groq")
    c[2].metric("Files compared", len(ok))
    c[3].metric("Wins (Groq / Local / Tie)", f"{wins['Groq']} / {wins['Local']} / {wins['Tie']}")

    left, right = st.columns(2)
    left.subheader("Average per metric")
    left.bar_chart(pd.DataFrame(per_metric), stack=False)
    right.subheader("Score per file")
    rows = [{"file": name, "Groq": r["groq"]["avg"], "Local": r["local"]["avg"], "winner": winner(r)}
            for name, r in ok.items()]
    right.dataframe(pd.DataFrame(rows).round(2), hide_index=True, width="stretch")

failed = [name for name in results if name not in ok]
if failed:
    st.warning("Not scored (judge failed): " + ", ".join(failed))


# ---------- 4) one file: both outputs, or just one ----------
st.header("Per file")
pick, who = st.columns([3, 2])
name = pick.selectbox("File", list(results))
show = who.radio("Show", ["Both", "Groq only", "Local only"], horizontal=True)
r = results[name]
if show == "Groq only":
    systems = [("groq", "Groq")]
elif show == "Local only":
    systems = [("local", "Local")]
else:
    systems = BOTH

with st.expander("Why these scores (judge's reasons)"): # judge's reasons
    for system, label in systems:
        for metric, reason in r[system].get("reasons", {}).items():
            st.markdown(f"**{label} - {metric}:** {reason}")

show_outputs(name, r, systems, key="file")
