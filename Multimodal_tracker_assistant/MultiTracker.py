"""
Multimodal Assistant Tracker - Single-file Streamlit App

Features:
- Accepts text and image inputs (multi-modal)
- Sends inputs to a model via `call_model()` (placeholder)
- Logs interactions to SQLite and saves uploaded images
- Shows history and lets you re-run or export logs

How to use:
1. Install requirements: pip install -r requirements.txt
   requirements.txt should include:
     streamlit
     pillow
     sqlalchemy
     python-multipart

2. Replace `call_model()` with your model integration (OpenAI, local LLM, etc.).
   - For OpenAI: implement API call and handle images (upload or base64) appropriately.

3. Run:
   streamlit run multimodal_tracker_assistant.py

"""

import streamlit as st
from PIL import Image
import os
import io
import base64
import sqlite3
from datetime import datetime
import json
import hashlib

# ----------------------------- Configuration -----------------------------
UPLOAD_DIR = "uploads"
DB_PATH = "interactions.db"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ----------------------------- DB Helpers -----------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS interactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT NOT NULL,
        modality TEXT,
        input_text TEXT,
        image_path TEXT,
        model TEXT,
        response_text TEXT,
        metadata TEXT
    )
    """)
    conn.commit()
    return conn

conn = init_db()

def log_interaction(ts, modality, input_text, image_path, model, response_text, metadata=None):
    c = conn.cursor()
    c.execute(
        "INSERT INTO interactions (ts, modality, input_text, image_path, model, response_text, metadata) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (ts, modality, input_text, image_path, model, response_text, json.dumps(metadata) if metadata else None)
    )
    conn.commit()
    return c.lastrowid

def fetch_history(limit=100):
    c = conn.cursor()
    c.execute("SELECT id, ts, modality, input_text, image_path, model, response_text, metadata FROM interactions ORDER BY id DESC LIMIT ?", (limit,))
    rows = c.fetchall()
    return rows

# ----------------------------- Model stub -----------------------------
def call_model(input_text=None, image_bytes=None, model_name="demo-multimodal-v1"):
    """
    Placeholder for model call. Replace this function with real model integration.
    - input_text: str or None
    - image_bytes: raw bytes of the image or None
    - model_name: string identifier of model

    Should return a dict: {"response_text": str, "metadata": dict}
    """
    # Dummy behavior: echo + image hash
    meta = {"model_used": model_name}
    resp = ""
    if input_text:
        resp += f"[Text understood]: {input_text}\n"
    if image_bytes:
        h = hashlib.sha256(image_bytes).hexdigest()[:12]
        resp += f"[Image received] SHA256:{h}\n"
        meta["image_hash"] = h
    resp += "\n[Note] Replace call_model() with your real model integration."
    return {"response_text": resp, "metadata": meta}

# ----------------------------- Utilities -----------------------------

def save_upload_image(uploaded_file):
    # uploaded_file is a Streamlit UploadedFile
    raw = uploaded_file.read()
    # generate a filename based on content hash and timestamp
    h = hashlib.sha256(raw).hexdigest()[:10]
    ext = os.path.splitext(uploaded_file.name)[1] or ".jpg"
    fname = f"img_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{h}{ext}"
    path = os.path.join(UPLOAD_DIR, fname)
    with open(path, "wb") as f:
        f.write(raw)
    return path, raw

# ----------------------------- Streamlit App -----------------------------
st.set_page_config(page_title="Multimodal Tracker Assistant", layout="wide")
st.title("Multimodal Assistant — Tracker & Logger")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Input")
    model_choice = st.selectbox("Model (placeholder)", ["demo-multimodal-v1", "local-llm-v1", "openai-gpt-4o-multi"], index=0)
    text_in = st.text_area("Text input", height=150)
    uploaded = st.file_uploader("Upload image (optional)", type=["png", "jpg", "jpeg", "gif"])

    if st.button("Send"):
        if not text_in and not uploaded:
            st.warning("Please provide text or upload an image.")
        else:
            image_path = None
            image_bytes = None
            if uploaded:
                image_path, image_bytes = save_upload_image(uploaded)
            # Call model
            with st.spinner("Processing..."):
                result = call_model(input_text=text_in or None, image_bytes=image_bytes, model_name=model_choice)
            response_text = result.get("response_text")
            metadata = result.get("metadata")
            ts = datetime.utcnow().isoformat()
            modality = []
            if text_in:
                modality.append("text")
            if uploaded:
                modality.append("image")
            modality = ",".join(modality)
            log_id = log_interaction(ts, modality, text_in or None, image_path, model_choice, response_text, metadata)
            st.success("Interaction logged — id: %d" % log_id)
            st.subheader("Model response")
            st.code(response_text)

with col2:
    st.header("History & Tools")
    rows = fetch_history(50)
    if not rows:
        st.info("No interactions logged yet.")
    else:
        for r in rows:
            rid, ts, modality, input_text, image_path, model, response_text, meta = r
            with st.expander(f"#{rid} — {ts} — {modality} — {model}"):
                if input_text:
                    st.markdown("**Text:**")
                    st.write(input_text)
                if image_path:
                    if os.path.exists(image_path):
                        st.image(image_path, caption=os.path.basename(image_path), use_column_width=False)
                    else:
                        st.write(f"Image file missing: {image_path}")
                st.markdown("**Response:**")
                st.code(response_text)
                if meta:
                    st.markdown("**Metadata:**")
                    try:
                        st.json(json.loads(meta))
                    except Exception:
                        st.write(meta)
                cols = st.columns([1,1,1])
                if cols[0].button("Re-run", key=f"rerun_{rid}"):
                    # re-run same input through model
                    raw_img = None
                    if image_path and os.path.exists(image_path):
                        with open(image_path, "rb") as f:
                            raw_img = f.read()
                    new_res = call_model(input_text=input_text, image_bytes=raw_img, model_name=model)
                    st.code(new_res.get("response_text"))
                if cols[1].button("Copy response", key=f"copy_{rid}"):
                    st.experimental_set_query_params(last_response=new_res.get("response_text") if 'new_res' in locals() else response_text)
                    st.success("Copied to query params (useful for integration).")
                if cols[2].button("Delete", key=f"del_{rid}"):
                    c = conn.cursor()
                    c.execute("DELETE FROM interactions WHERE id = ?", (rid,))
                    conn.commit()
                    st.experimental_rerun()

st.markdown("---")
st.header("Export / Admin")
colA, colB = st.columns(2)
with colA:
    if st.button("Export CSV"):
        import csv
        rows = fetch_history(10000)
        csv_path = "interactions_export.csv"
        with open(csv_path, "w", newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["id","ts","modality","input_text","image_path","model","response_text","metadata"])
            for r in rows[::-1]:
                writer.writerow(r)
        with open(csv_path, "rb") as f:
            st.download_button("Download CSV", f, file_name=csv_path)
with colB:
    if st.button("Clear all logs"):
        c = conn.cursor()
        c.execute("DELETE FROM interactions")
        conn.commit()
        st.success("All logs cleared")

st.sidebar.header("About")
st.sidebar.write("Multimodal Assistant Tracker - demo. Replace `call_model()` with real model calls. Logs saved to SQLite and uploaded images saved to uploads/.")

# End of file
