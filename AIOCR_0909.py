# app.py
# -*- coding: utf-8 -*-
"""
📄 請求書OCR×AI（Gemini）連携アプリ（Streamlit 単一ファイル）
機能:
- PDFをドラッグ&ドロップで複数投入
- ネイティブ抽出（PyMuPDF）→ 空ならOCR（pytesseract + pdf2image）でテキスト化
- 指示テキスト + Gemini で JSON 抽出（strict JSON）
- JSONプレビュー & 追記予定の表プレビュー
- 既存Excel（アップロード）へ追記してダウンロード
- APIキーは st.secrets / 環境変数（.env）から取得（ハードコード禁止）

前提インストール（例）:
  pip install streamlit google-generativeai python-dotenv pydantic pdf2image pytesseract pymupdf openpyxl pandas
Windows:
  - Tesseract をインストールし、実行ファイルパスを通す（例: C:\\Program Files\\Tesseract-OCR\\tesseract.exe）
  - poppler を入れる（pdf2image用）。Windows向けバイナリを導入し、PATHに追加
mac:
  - brew install tesseract poppler

起動:  streamlit run app.py
"""

from __future__ import annotations
import io
import json
import os
import hashlib
from typing import List, Dict, Any

import streamlit as st
from pydantic import BaseModel, Field, ValidationError

# PDF抽出
import fitz  # PyMuPDF

# OCR（任意）
try:
    from pdf2image import convert_from_bytes
    import pytesseract
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

# Excel / DataFrame
import pandas as pd
from openpyxl import load_workbook, Workbook
from dotenv import load_dotenv

# Gemini SDK
import google.generativeai as genai

# =========================================================
# セキュアなキー取得
# =========================================================
load_dotenv()  # .env があれば読み込む

def get_gemini_api_key() -> str:
    # 1) Streamlit Cloud / ローカル: .streamlit/secrets.toml に GEMINI_API_KEY を入れる
    # 2) ローカル: .env に GEMINI_API_KEY=... を入れる
    key = None
    try:
        key = st.secrets.get("GEMINI_API_KEY")  # type: ignore[attr-defined]
    except Exception:
        pass
    if not key:
        key = os.getenv("GEMINI_API_KEY")
    if not key:
        st.stop()
        raise RuntimeError("GEMINI_API_KEY が見つかりません。st.secrets か .env に設定してください。")
    return str(key)

# =========================================================
# 期待するJSONスキーマ（Pydantic）
# =========================================================
class LineItem(BaseModel):
    提供品名: str = Field(..., description="商品・サービス名")
    数量: float = Field(..., description="数量（数値）")
    税込単価: float = Field(..., description="税込単価（数値）")
    税込金額: float = Field(..., description="税込金額（数値）")

class InvoiceExtraction(BaseModel):
    請求先: str
    明細: List[LineItem]

# =========================================================
# PDF → テキスト（ネイティブ & OCR フォールバック）
# =========================================================

def extract_text_native(pdf_bytes: bytes) -> str:
    text_chunks: List[str] = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc:
            text_chunks.append(page.get_text("text"))
    return "\n".join(text_chunks).strip()


def extract_text_ocr(pdf_bytes: bytes, tesseract_cmd: str | None = None) -> str:
    if not OCR_AVAILABLE:
        return ""
    if tesseract_cmd:
        # WindowsでPATH未設定の場合などに明示
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    try:
        images = convert_from_bytes(pdf_bytes, fmt="png")
    except Exception as e:
        st.warning(f"OCR用画像変換に失敗: {e}")
        return ""
    out: List[str] = []
    for img in images:
        try:
            out.append(pytesseract.image_to_string(img, lang="jpn+eng"))
        except Exception as e:
            st.warning(f"OCR失敗: {e}")
    return "\n".join(out).strip()


def extract_text_best_effort(pdf_bytes: bytes, tesseract_cmd: str | None = None) -> Dict[str, Any]:
    native_txt = extract_text_native(pdf_bytes)
    used_ocr = False
    if len(native_txt) < 50:  # 空に近い/レイヤ無しPDFっぽい
        ocr_txt = extract_text_ocr(pdf_bytes, tesseract_cmd)
        if len(ocr_txt) > len(native_txt):
            native_txt = ocr_txt
            used_ocr = True
    return {"text": native_txt, "used_ocr": used_ocr}

# =========================================================
# Gemini 呼び出し
# =========================================================
GEMINI_MODEL = "gemini-1.5-flash"

def init_gemini():
    api_key = get_gemini_api_key()
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        generation_config={
            # 可能な限りstrict JSONに寄せる
            "response_mime_type": "application/json",
            # JSON崩れ対策のため、必要なら temperature を下げる
            "temperature": 0.2,
        },
    )
    return model


def build_prompt(user_instruction: str, raw_text: str) -> str:
    # JSON以外返さない指示を強めに
    schema_hint = {
        "type": "object",
        "properties": {
            "請求先": {"type": "string"},
            "明細": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "提供品名": {"type": "string"},
                        "数量": {"type": "number"},
                        "税込単価": {"type": "number"},
                        "税込金額": {"type": "number"},
                    },
                    "required": ["提供品名", "数量", "税込単価", "税込金額"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["請求先", "明細"],
        "additionalProperties": False,
    }
    # SDKのschema指定は環境差があるため、ここはプロンプト制御で担保
    prompt = (
        "以下は請求書テキストです。あなたは厳密にJSONだけを出力します。\n"
        "日本語キーで次スキーマに厳密一致:\n"
        + json.dumps(schema_hint, ensure_ascii=False)
        + "\n禁止: コメント/説明文/コードブロック/前置き/末尾文\n"
        + "ユーザー追加指示: " + user_instruction.strip() + "\n"
        + "--- 請求書テキスト ---\n"
        + raw_text
    )
    return prompt


def call_gemini_to_json(model, instruction: str, raw_text: str) -> InvoiceExtraction | None:
    prompt = build_prompt(instruction, raw_text)
    try:
        res = model.generate_content(prompt)
        txt = res.text if hasattr(res, "text") else str(res)
        # 念のためJSON部分だけ抽出
        start = txt.find("{")
        end = txt.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("JSON形式の出力が得られませんでした")
        json_str = txt[start : end + 1]
        data = json.loads(json_str)
        parsed = InvoiceExtraction(**data)
        return parsed
    except (json.JSONDecodeError, ValidationError, Exception) as e:
        st.error(f"Gemini抽出に失敗: {e}")
        return None

# =========================================================
# Excel 追記
# =========================================================
EXPECTED_HEADERS = ["請求先", "提供品名", "数量", "税込単価", "税込金額"]


def dataframe_from_extractions(items: List[InvoiceExtraction]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for inv in items:
        for li in inv.明細:
            rows.append({
                "請求先": inv.請求先,
                "提供品名": li.提供品名,
                "数量": li.数量,
                "税込単価": li.税込単価,
                "税込金額": li.税込金額,
            })
    df = pd.DataFrame(rows, columns=EXPECTED_HEADERS)
    return df


def append_to_excel_bytes(excel_bytes: bytes | None, df: pd.DataFrame) -> bytes:
    if df.empty:
        return excel_bytes or b""

    if excel_bytes:
        bio_in = io.BytesIO(excel_bytes)
        wb = load_workbook(bio_in)
        ws = wb.active
        # ヘッダ検証
        if ws.max_row == 0 or [c.value for c in ws[1]] != EXPECTED_HEADERS:
            # 先頭行にヘッダを書き直す（既存がズレている場合は上書き注意）
            ws.delete_rows(1, ws.max_row)
            ws.append(EXPECTED_HEADERS)
    else:
        wb = Workbook()
        ws = wb.active
        ws.title = "Sheet1"
        ws.append(EXPECTED_HEADERS)

    # 追記
    for _, r in df.iterrows():
        ws.append([r[c] for c in EXPECTED_HEADERS])

    out = io.BytesIO()
    wb.save(out)
    out.seek(0)
    return out.read()

# =========================================================
# UI
# =========================================================
st.set_page_config(page_title="請求書OCR×AI（Gemini）連携", page_icon="📄", layout="wide")
st.title("📄 請求書OCR×AI（Gemini）連携アプリ")

with st.sidebar:
    st.markdown("### 🔐 APIキー & OCR 設定")
    st.info("GEMINI_API_KEY は st.secrets または .env で設定してください。ハードコード禁止。")
    tesseract_cmd = st.text_input(
        "(任意) Tesseract の実行ファイルパス",
        help="WindowsでPATH未設定のときに指定（例 C:\\\Program Files\\\\Tesseract-OCR\\\\tesseract.exe）",
    )
    st.caption("OCR利用可否: {}".format("OK" if OCR_AVAILABLE else "pdf2image/pytesseract 未導入"))

st.subheader("1) 既存Excel をドラッグ&ドロップ（追記先）")
excel_file = st.file_uploader(
    "左から『請求先, 提供品名, 数量, 税込単価, 税込金額』のヘッダを想定。未指定なら新規作成します。",
    type=["xlsx"],
    accept_multiple_files=False,
    key="excel_uploader",
)

st.subheader("2) 請求書PDF をドラッグ&ドロップ（複数可）")
pdf_files = st.file_uploader(
    "画像系PDFはOCR、テキストPDFはネイティブ抽出を試みます。",
    type=["pdf"],
    accept_multiple_files=True,
    key="pdf_uploader",
)

st.subheader("3) AIへの指示文（抽出ポリシー）")
def_instr = (
    "金額は税込。数量・単価・金額は数値に統一。\n"
    "明細が複数行ならすべて抽出。請求先は御中/株式会社/有限会社など正規表現で正規化。\n"
    "単位や通貨記号は除去し、数値のみ。税込金額=数量×税込単価 で矛盾があれば税込金額を優先。"
)
user_instruction = st.text_area("例: ", value=def_instr, height=120)

colA, colB = st.columns([1,1])
with colA:
    run = st.button("🔎 抽出を実行する", type="primary")
with colB:
    clear = st.button("🧹 画面クリア")

if clear:
    st.experimental_rerun()

extractions: List[InvoiceExtraction] = []
texts_preview: List[Dict[str, Any]] = []

if run:
    if not pdf_files:
        st.warning("PDFが未選択です。")
    else:
        model = init_gemini()
        with st.status("処理中...", expanded=True) as status:
            st.write("PDFテキスト抽出 → Gemini 解析 → プレビュー生成")
            for up in pdf_files:
                pdf_bytes = up.read()
                info = extract_text_best_effort(pdf_bytes, tesseract_cmd or None)
                raw_txt = info["text"]
                texts_preview.append({"filename": up.name, "used_ocr": info["used_ocr"], "text": raw_txt})

                st.write(f"→ {up.name}: テキスト長 {len(raw_txt)} / OCR: {info['used_ocr']}")
                if len(raw_txt) < 10:
                    st.warning(f"{up.name}: テキストが抽出できませんでした。画像/OCR設定を確認してください。")
                    continue

                parsed = call_gemini_to_json(model, user_instruction, raw_txt)
                if parsed:
                    extractions.append(parsed)
                else:
                    st.warning(f"{up.name}: JSON抽出に失敗")

            status.update(label="完了", state="complete")

# ===== プレビュー表示 =====
if texts_preview:
    st.markdown("### 🔍 OCR/ネイティブ抽出テキスト プレビュー")
    for tp in texts_preview:
        fname = str(tp.get("filename", "no_name")).strip()
        base  = f"{fname}|{len(tp.get('text',''))}"
        key   = "txt_" + hashlib.md5(base.encode("utf-8")).hexdigest()

        with st.expander(f"{fname}  (OCR使用: {tp['used_ocr']})", expanded=False):
            st.text_area("抽出テキスト", value=tp.get("text",""), height=220, key=key)

if extractions:
    st.markdown("### 🧩 Gemini JSON 抽出プレビュー")
    for i, ex in enumerate(extractions, 1):
        st.json(json.loads(ex.model_dump_json()))

    st.markdown("### 📋 追記予定データ（左から: 請求先/提供品名/数量/税込単価/税込金額）")
    df = dataframe_from_extractions(extractions)
    st.dataframe(df, use_container_width=True)

    # Excelを作成/更新
    in_bytes = excel_file.read() if excel_file is not None else None
    out_bytes = append_to_excel_bytes(in_bytes, df)

    st.download_button(
        label="⬇ 追記済みExcelをダウンロード",
        data=out_bytes,
        file_name="追記結果.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    # JSONL出力（任意）
    jsonl = "".join([ex.model_dump_json() + "\n" for ex in extractions])
    st.download_button(
        label="⬇ 抽出JSONLをダウンロード",
        data=jsonl.encode("utf-8"),
        file_name="extractions.jsonl",
        mime="application/json",
    )

# フッタ
st.caption(
    "注: このアプリはローカルで処理します。APIキーは st.secrets/.env から読み取り、\n"
    "ソースコードへ直書きしません。アップロードされたファイルはセッション内でのみ使用されます。"
)
