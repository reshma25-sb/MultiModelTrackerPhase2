# multimodal_app.py
import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from PIL import Image
import io
import time
import traceback

# Choose defaults (change to True to use OpenAI for the text step)
USE_OPENAI = os.getenv("USE_OPENAI", "false").lower() in ("1", "true", "yes")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# --- Lazy imports for heavier libraries; only import when needed ---
def import_image_captioner():
    from transformers import BlipProcessor, BlipForConditionalGeneration
    return BlipProcessor, BlipForConditionalGeneration

def import_text_model():
    # text model pipeline
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
    return AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

def import_openai_client():
    # minimal OpenAI client using openai>=1.0
    from openai import OpenAI
    return OpenAI

# --- UI ---
st.set_page_config(page_title="Multimodal Assistant", layout="centered")

st.title("Multimodal Assistant — Answer from Image + Text")
st.write(
    "Upload an image, enter a question about the image, and the assistant will answer "
    "by combining a visual description with the text query."
)

col1, col2 = st.columns([1, 2])

with col1:
    uploaded = st.file_uploader("Upload an image (jpg, png, etc.)", type=["png", "jpg", "jpeg"])
    st.markdown("**Model switch**")
    use_openai_checkbox = st.checkbox("Use OpenAI for the text step (sends prompt to OpenAI)", value=USE_OPENAI)
    st.caption("If you check this, set OPENAI_API_KEY in a .env file or environment variable.")

with col2:
    question = st.text_input("Ask a question about the image", value="What is happening in this image?")
    max_tokens = st.slider("Max tokens for answer (if using local model)", 64, 1024, 256)
    run_button = st.button("Run")

# Keep consistent local variable
USE_OPENAI = use_openai_checkbox

# --- helpful function: load PIL image from uploaded bytes ---
def load_pil_from_upload(uploaded_file):
    raw = uploaded_file.read()
    return Image.open(io.BytesIO(raw)).convert("RGB")

# --- Core pipeline functions ---
@st.cache_resource(show_spinner=False)
def load_caption_model():
    try:
        BlipProcessor, BlipForConditionalGeneration = import_image_captioner()
    except Exception as e:
        raise RuntimeError("Please pip install transformers, sentencepiece, torchvision, accelerate, and pillow.") from e
    # model name; change to a smaller model if memory is limited
    model_name = "Salesforce/blip-image-captioning-base"
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name)
    return processor, model

@st.cache_resource(show_spinner=False)
def load_text_model(model_name="google/flan-t5-large"):
    try:
        AutoTokenizer, AutoModelForSeq2SeqLM, pipeline_fn = import_text_model()
    except Exception as e:
        raise RuntimeError("Please pip install transformers accelerate sentencepiece.") from e
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    pipe = pipeline_fn("text2text-generation", model=model, tokenizer=tokenizer, device_map="auto" if "CUDA_VISIBLE_DEVICES" in os.environ else None)
    return pipe

def caption_image(processor, model, pil_image):
    """
    Return a short caption/description for the image using BLIP.
    """
    try:
        inputs = processor(images=pil_image, return_tensors="pt")
        out = model.generate(**inputs, max_new_tokens=64)
        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        # Fall back to a very small safe caption
        return "an image (could not generate a detailed caption due to resource limits)."

def ask_local_text_model(pipe, prompt, max_tokens=256):
    # the transformers pipeline will use max_length or max_new_tokens depending on model; handle simply
    try:
        # For Flan-T5 style models: we provide a prompt -> text generation
        out = pipe(prompt, max_new_tokens=max_tokens, do_sample=False, clean_up_tokenization_spaces=True)[0]
        return out["generated_text"]
    except Exception as e:
        return f"Error running local text model: {e}"

def ask_openai(prompt, max_tokens=256):
    try:
        OpenAI = import_openai_client()
        client = OpenAI(api_key=OPENAI_API_KEY)
        # Using chat completion style: create a short instruction
        # Use client.chat.completions.create if available in your openai lib
        resp = client.chat.completions.create(
            model="gpt-4o-mini",  # change if needed and available
            messages=[
                {"role": "system", "content": "You are a helpful multimodal assistant. Use the image description and the user's question to answer concisely and accurately."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=max_tokens
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"OpenAI request failed: {e}"

# --- Execution flow ---
if run_button:
    if uploaded is None:
        st.error("Please upload an image first.")
    elif question.strip() == "":
        st.error("Please enter a question about the image.")
    else:
        st.info("Processing — this may take a while on CPU. If you have a GPU, it'll be much faster.")
        start = time.time()
        try:
            pil = load_pil_from_upload(uploaded)
            st.image(pil, caption="Uploaded image", use_column_width=True)

            # 1) Run the image captioner (local)
            with st.spinner("Generating image description..."):
                processor, blip_model = load_caption_model()
                caption = caption_image(processor, blip_model, pil)
                st.markdown("**Image description (caption):**")
                st.write(caption)

            # 2) Build combined prompt
            combined_prompt = (
                f"Image description: {caption}\n\n"
                f"User question: {question}\n\n"
                "Answer the user's question based on the image description. If the description lacks details "
                "that are necessary to answer, say you don't know rather than invent facts. Be concise."
            )

            # 3) Run the text model (local or OpenAI)
            with st.spinner("Generating answer..."):
                if USE_OPENAI:
                    if not OPENAI_API_KEY:
                        st.error("OPENAI_API_KEY not set. Set it in .env or environment to use OpenAI mode.")
                    else:
                        answer = ask_openai(combined_prompt, max_tokens=max_tokens)
                else:
                    # load text model (cached)
                    model_choice = os.getenv("LOCAL_TEXT_MODEL", "google/flan-t5-large")
                    st.caption(f"Using local text model: {model_choice}")
                    text_pipe = load_text_model(model_choice)
                    answer = ask_local_text_model(text_pipe, combined_prompt, max_tokens=max_tokens)

                st.markdown("**Answer:**")
                st.write(answer)

            elapsed = time.time() - start
            st.caption(f"Elapsed: {elapsed:.1f}s (approx.)")

        except Exception as e:
            st.error("An unexpected error occurred. See details below.")
            st.text(traceback.format_exc())
