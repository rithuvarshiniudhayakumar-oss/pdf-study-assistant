"""
RAG-based PDF Study Assistant
--------------------------------
A Streamlit app that lets you upload a PDF (lecture notes, textbook chapter, etc.)
and then:
  1. Ask questions answered ONLY from the PDF content (strict RAG)
  2. Get an automatic bullet-point summary
  3. Generate a 5-question multiple-choice quiz with answer key + explanations
  4. Generate revision flashcards (term + definition)

Run with:  streamlit run app.py
"""

import os
import streamlit as st
from PyPDF2 import PdfReader
from google import genai
from dotenv import load_dotenv

# -------------------------------------------------------------------
# 1. SETUP
# -------------------------------------------------------------------

load_dotenv()  # reads a local .env file if present

st.set_page_config(page_title="PDF Study Assistant", page_icon="📚", layout="wide")

MODEL_NAME = "gemini-3.5-flash"  # current fast/cheap Gemini model


def get_client():
    """Create (and cache) the Gemini client using the API key."""
    api_key = os.getenv("GEMINI_API_KEY")

    # Fallback: let the user paste a key directly in the sidebar if no .env is set
    if not api_key:
        api_key = st.session_state.get("manual_api_key", "")

    if not api_key:
        return None

    return genai.Client(api_key=api_key)


# -------------------------------------------------------------------
# 2. PDF EXTRACTION
# -------------------------------------------------------------------

def extract_pdf_text(uploaded_file) -> str:
    """Reads every page of the uploaded PDF and stitches the text together."""
    reader = PdfReader(uploaded_file)
    text_parts = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        text_parts.append(page_text)
    return "\n".join(text_parts)


# -------------------------------------------------------------------
# 3. GEMINI CALL HELPER
# -------------------------------------------------------------------

def query_gemini(system_prompt: str, user_prompt: str, pdf_text: str) -> str:
    """Sends a strict RAG-style prompt to Gemini and returns the text response."""
    client = get_client()
    if client is None:
        return "⚠️ No API key found. Please add your Gemini API key in the sidebar."

    full_prompt = f"""{system_prompt}

--- DOCUMENT CONTENT START ---
{pdf_text}
--- DOCUMENT CONTENT END ---

{user_prompt}
"""

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=full_prompt,
        )
        return response.text
    except Exception as e:
        return f"⚠️ Error calling Gemini API: {e}"


# -------------------------------------------------------------------
# 4. FEATURE-SPECIFIC PROMPTS
# -------------------------------------------------------------------

RAG_SYSTEM_PROMPT = """You are a strict study assistant. Answer the user's question
using ONLY the information contained in the document below. Do not use outside
knowledge. If the answer is not present in the document, respond exactly:
"I couldn't find this information in the uploaded document."
Keep answers clear and concise."""

SUMMARY_SYSTEM_PROMPT = """You are a study assistant. Read the document below and
produce a clear, well-organized summary using bullet points. Group related ideas
under short bold headings if the document covers multiple topics. Focus on key
concepts, definitions, and main takeaways a student would need for exam revision."""

QUIZ_SYSTEM_PROMPT = """You are a study assistant. Based STRICTLY on the document
below, create a 5-question multiple-choice quiz to test understanding of the
material. Follow this exact format for each question:

Q1. <question text>
A. <option>
B. <option>
C. <option>
D. <option>

After all 5 questions, add a section titled "Answer Key" listing the correct
letter for each question followed by a one-sentence explanation, like:
1. B - <short explanation>
2. A - <short explanation>
...

Do not invent facts that aren't in the document."""

FLASHCARD_SYSTEM_PROMPT = """You are a study assistant. Read the document below and
extract the most important terms, concepts, or formulas along with concise
definitions/explanations, suitable for flashcards. Format each one exactly as:

Term: <term>
Definition: <concise definition>

Produce between 8 and 12 flashcards, ordered by importance."""


# -------------------------------------------------------------------
# 5. SIDEBAR — FILE UPLOAD + API KEY
# -------------------------------------------------------------------

with st.sidebar:
    st.header("📄 Upload your PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    st.divider()
    st.header("🔑 API Key")
    if not os.getenv("GEMINI_API_KEY"):
        st.session_state["manual_api_key"] = st.text_input(
            "Gemini API Key", type="password",
            help="Paste your key here if you haven't set it in a .env file."
        )
    else:
        st.success("API key loaded from .env")

    st.divider()
    st.caption("Built with Streamlit + Gemini (google-genai SDK)")

# Extract and cache PDF text so we don't re-parse on every interaction
if uploaded_file is not None:
    if (
        "pdf_text" not in st.session_state
        or st.session_state.get("pdf_name") != uploaded_file.name
    ):
        with st.spinner("Reading PDF..."):
            st.session_state["pdf_text"] = extract_pdf_text(uploaded_file)
            st.session_state["pdf_name"] = uploaded_file.name
        st.sidebar.success(f"Loaded: {uploaded_file.name}")

pdf_text = st.session_state.get("pdf_text", "")

# -------------------------------------------------------------------
# 6. MAIN UI — TITLE + TABS
# -------------------------------------------------------------------

st.title("📚 PDF Study Assistant")
st.caption("Upload a PDF in the sidebar, then use the tabs below.")

tab_qa, tab_summary, tab_quiz, tab_flashcards = st.tabs(
    ["💬 Ask Questions", "📝 Summary", "❓ Practice Quiz", "🗂️ Flashcards"]
)

if not pdf_text:
    st.info("👈 Upload a PDF from the sidebar to get started.")

# ---- TAB 1: ASK QUESTIONS -----------------------------------------
with tab_qa:
    st.subheader("Ask questions about your document")

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    for role, msg in st.session_state["chat_history"]:
        with st.chat_message(role):
            st.markdown(msg)

    user_question = st.chat_input("Type your question here...")

    if user_question:
        if not pdf_text:
            st.warning("Please upload a PDF first.")
        else:
            st.session_state["chat_history"].append(("user", user_question))
            with st.chat_message("user"):
                st.markdown(user_question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer = query_gemini(RAG_SYSTEM_PROMPT, user_question, pdf_text)
                    st.markdown(answer)

            st.session_state["chat_history"].append(("assistant", answer))

# ---- TAB 2: SUMMARY -------------------------------------------------
with tab_summary:
    st.subheader("Document Summary")

    if st.button("Generate Summary", disabled=not pdf_text):
        with st.spinner("Summarizing document..."):
            summary = query_gemini(
                SUMMARY_SYSTEM_PROMPT,
                "Summarize the document as instructed.",
                pdf_text,
            )
            st.session_state["summary"] = summary

    if "summary" in st.session_state:
        st.markdown(st.session_state["summary"])

# ---- TAB 3: QUIZ -----------------------------------------------------
with tab_quiz:
    st.subheader("Practice Quiz (5 Questions)")

    if st.button("Generate Quiz", disabled=not pdf_text):
        with st.spinner("Building quiz..."):
            quiz = query_gemini(
                QUIZ_SYSTEM_PROMPT,
                "Create the 5-question multiple-choice quiz as instructed.",
                pdf_text,
            )
            st.session_state["quiz"] = quiz

    if "quiz" in st.session_state:
        st.markdown(st.session_state["quiz"])

# ---- TAB 4: FLASHCARDS ------------------------------------------------
with tab_flashcards:
    st.subheader("Revision Flashcards")

    if st.button("Generate Flashcards", disabled=not pdf_text):
        with st.spinner("Extracting key terms..."):
            flashcards_raw = query_gemini(
                FLASHCARD_SYSTEM_PROMPT,
                "Create the flashcards as instructed.",
                pdf_text,
            )
            st.session_state["flashcards_raw"] = flashcards_raw

    if "flashcards_raw" in st.session_state:
        raw = st.session_state["flashcards_raw"]
        # Split into individual cards on "Term:" markers
        cards = [c.strip() for c in raw.split("Term:") if c.strip()]

        cols = st.columns(2)
        for i, card in enumerate(cards):
            lines = card.split("Definition:")
            term = lines[0].strip()
            definition = lines[1].strip() if len(lines) > 1 else ""
            with cols[i % 2]:
                with st.container(border=True):
                    st.markdown(f"**{term}**")
                    st.markdown(definition)