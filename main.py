"""Streamlit FAQ assistant for an aesthetic medicine clinic."""

import csv
import json
import os
import threading
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


load_dotenv()

MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
)
QA_FILE = Path(__file__).with_name("QA.csv")
ENCODE_LOCK = threading.Lock()


def get_setting(name: str, default: str = "") -> str:
    """Read a Streamlit secret first, then fall back to the process environment."""
    try:
        value = st.secrets.get(name, os.getenv(name, default))
    except FileNotFoundError:
        value = os.getenv(name, default)
    return str(value or default).strip()


@st.cache_resource
def load_knowledge_base(model_name: str):
    if not QA_FILE.exists():
        raise FileNotFoundError(f"找不到問答資料檔：{QA_FILE}")

    with QA_FILE.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    if not rows or not {"question", "answer"}.issubset(rows[0]):
        raise ValueError("QA.csv 必須包含 question 和 answer 欄位。")

    model = SentenceTransformer(model_name)
    documents = [f"{row['question']} {row['answer']}" for row in rows]
    vectors = model.encode(
        documents,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return rows, model, vectors


def embed_query(model: SentenceTransformer, text: str) -> np.ndarray:
    with ENCODE_LOCK:
        return model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )


def search_supabase(query_vector: np.ndarray):
    """Search Supabase if its URL and anon key are configured."""
    supabase_url = get_setting("SUPABASE_URL").rstrip("/")
    supabase_key = get_setting("SUPABASE_ANON_KEY") or get_setting("SUPABASE_KEY")
    if not supabase_url or not supabase_key:
        return None

    payload = json.dumps(
        {"query_embedding": query_vector.tolist(), "match_count": 1}
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{supabase_url}/rest/v1/rpc/match_qa",
        data=payload,
        headers={
            "apikey": supabase_key,
            "Authorization": f"Bearer {supabase_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        matches = json.load(response)
    return matches[0] if matches else None


def search_local(rows, vectors: np.ndarray, query_vector: np.ndarray):
    scores = vectors @ query_vector
    best_index = int(np.argmax(scores))
    row = rows[best_index]
    return {
        "question": row["question"],
        "answer": row["answer"],
        "similarity": float(scores[best_index]),
    }


def answer_question(message: str, model, rows, vectors) -> str:
    query = (message or "").strip()
    if not query:
        return "請先輸入您想詢問的內容。"

    query_vector = embed_query(model, query)
    try:
        match = search_supabase(query_vector)
    except (urllib.error.URLError, TimeoutError, ValueError) as error:
        print(f"Supabase search failed; using local FAQ: {error}")
        match = None

    if match is None:
        match = search_local(rows, vectors, query_vector)

    similarity = float(match.get("similarity", 0.0))
    min_similarity = float(get_setting("MIN_SIMILARITY", "0.28"))
    if similarity < min_similarity:
        return (
            "目前的問答資料找不到足夠相關的答案。請提供更多細節，或聯絡診所安排醫師諮詢。\n\n"
            "線上客服資訊不能取代醫師診斷；若有急性或嚴重不適，請立即聯絡診所或就醫。"
        )

    answer = match["answer"].strip()
    source_question = match.get("question", "")
    if source_question:
        answer += f"\n\n參考問答：{source_question}"
    return answer


st.set_page_config(page_title="醫美診所線上客服", page_icon="💬", layout="centered")
st.title("醫美診所線上客服")
st.caption("以語意搜尋診所常見問答，即時提供相關資訊。醫療問題請由醫師評估。")

with st.spinner("載入問答資料與語意模型…"):
    try:
        FAQ_ROWS, MODEL, FAQ_VECTORS = load_knowledge_base(MODEL_NAME)
    except Exception as error:
        st.error("客服暫時無法載入問答資料，請稍後再試。")
        st.stop()

with st.sidebar:
    st.subheader("使用說明")
    st.write("輸入療程、預約或術後照護問題，客服會從診所問答資料中尋找最相關的回答。")
    st.info("線上資訊不能取代醫師診斷。若有急性或嚴重不適，請立即聯絡診所或就醫。")
    if st.button("清除對話", use_container_width=True):
        st.session_state.messages = []

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("例如：雷射療程後可以化妝嗎？")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("正在搜尋相關問答…"):
            response = answer_question(prompt, MODEL, FAQ_ROWS, FAQ_VECTORS)
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
