import io
import os
import re
from typing import List, Dict, Optional

import pandas as pd
import streamlit as st
from snowflake.snowpark.context import get_active_session
# ★ Snowparkの関数をインポート
from snowflake.snowpark.functions import col, current_timestamp


# ------------------------- Helpers -------------------------
def _strip_at(stage: str) -> str:
    return stage[1:] if stage.startswith("@") else stage


def refresh_stage_directory(session, stage: str):
    try:
        session.sql(f"ALTER STAGE {_strip_at(stage)} REFRESH").collect()
    except Exception:
        pass


def list_stage_pdfs(session, stage: str, prefix: str) -> pd.DataFrame:
    return session.sql(
        f"SELECT RELATIVE_PATH, SIZE, LAST_MODIFIED FROM DIRECTORY({stage}) "
        f"WHERE RELATIVE_PATH ILIKE '{prefix}%.pdf' ORDER BY RELATIVE_PATH"
    ).to_pandas()


def list_stage_images(session, stage: str, images_prefix: str) -> pd.DataFrame:
    return session.sql(
        f"SELECT RELATIVE_PATH, SIZE, LAST_MODIFIED FROM DIRECTORY({stage}) "
        f"WHERE RELATIVE_PATH ILIKE '{images_prefix}%' ORDER BY RELATIVE_PATH"
    ).to_pandas()


def put_pdf_to_stage(session, stage: str, dest_relpath: str, file_bytes: bytes) -> str:
    stage_path = f"{stage}/{dest_relpath}"
    session.file.put_stream(io.BytesIO(file_bytes), stage_path, auto_compress=False, overwrite=True)
    return stage_path


def parse_with_ai_parse_document(session, stage: str, file_relpath: str, mode: str = "layout"):
    session.sql(
        """
        CREATE TABLE IF NOT EXISTS CURATED.PAGES_AI_PARSE (
          DOC_ID        STRING,
          PAGE_NUM      NUMBER,
          PAGE_CONTENT  STRING,
          PARSE_MODE    STRING,
          LOAD_TS       TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
        )
        """
    ).collect()

    q = f"""
      INSERT INTO CURATED.PAGES_AI_PARSE (DOC_ID, PAGE_NUM, PAGE_CONTENT, PARSE_MODE)
      WITH src AS (
        SELECT '{stage}' AS stage_name, '{file_relpath}' AS file_path
      ),
      res AS (
        SELECT s.stage_name,
               s.file_path,
               AI_PARSE_DOCUMENT(
                 TO_FILE(s.stage_name, s.file_path),
                 OBJECT_CONSTRUCT('mode','{mode}','page_split', true)
               ) AS out
        FROM src s
      )
      SELECT
        SPLIT_PART(file_path, '/', -1)  AS DOC_ID,
        value:index::NUMBER             AS PAGE_NUM,
        value:content::STRING           AS PAGE_CONTENT,
        '{mode}'                        AS PARSE_MODE
      FROM res,
      LATERAL FLATTEN(input => out:pages);
    """
    session.sql(q).collect()


def fetch_pages_for_doc(session, doc_id: str, parse_mode: str) -> List[Dict]:
    df = session.sql(
        """
        SELECT PAGE_NUM, PAGE_CONTENT
        FROM CURATED.PAGES_AI_PARSE
        WHERE DOC_ID = ? AND PARSE_MODE = ?
        ORDER BY PAGE_NUM
        """,
        params=[doc_id, parse_mode],
    ).to_pandas()
    return df.to_dict("records")


def get_image_bytes(session, stage: str, relpath: str) -> Optional[bytes]:
    try:
        with session.file.get_stream(f"{stage}/{relpath}") as f:
            return f.read()
    except Exception:
        return None


def parse_image_basename(name: str) -> Optional[Dict[str, Optional[str]]]:
    """Parse image basename like 'doc-name-page_1.png' or 'doc-name-page-1.png'.
    Returns {'base': <doc_base>, 'page': <int_str>} or None if not matched.
    """
    m = re.match(r"^(?P<base>.+)-page[_-](?P<page>\d+)\.png$", name)
    if not m:
        return None
    return {"base": m.group("base"), "page": m.group("page")}


# ------------------------- App -------------------------
st.set_page_config(page_title="IR RAG Helper (SPCS)", layout="wide")
st.title("IR Report Ingestion & Preview (SPCS)")

session = get_active_session()

# Sidebar
st.sidebar.header("Stage & Settings")
stage = st.sidebar.text_input("Target stage", value="@IRINFO_RAG.RAW.IR_STAGE")
pdf_prefix = st.sidebar.text_input("PDF prefix", value="file2/")
images_prefix = st.sidebar.text_input("Images prefix", value="output_image2/")
parse_mode = st.sidebar.selectbox("AI_PARSE_DOCUMENT mode", ["layout", "OCR"], index=0)

st.sidebar.markdown("---")
if st.sidebar.button("Refresh app (rerun)"):
    try:
        st.experimental_rerun()
    except Exception:
        pass

# Validate prefixes
def _overlap(a: str, b: str) -> bool:
    a, b = a.strip(), b.strip()
    return a == b or a.startswith(b) or b.startswith(a)

if _overlap(pdf_prefix, images_prefix):
    st.sidebar.error("PDF prefix と Images prefix は重複しない値にしてください。")
    st.stop()

# 追加チェック: 先頭フォルダ名が同一なら停止（Target stage 内での運用衝突回避）
def _first_segment(p: str) -> str:
    p = (p or "").strip()
    p = p[1:] if p.startswith("/") else p
    return p.strip("/").split("/")[0] if p else ""

pdf_root = _first_segment(pdf_prefix).lower()
img_root = _first_segment(images_prefix).lower()
if pdf_root and img_root and pdf_root == img_root:
    st.sidebar.error("Target stage 内で PDF prefix と Images prefix の先頭フォルダ名が同一です。別名にしてください。")
    st.stop()


# 1. Upload PDF --------------------------------------------------------------
st.header("1. Upload PDF to Stage")
uploaded = st.file_uploader("Upload PDF", type=["pdf"])
if uploaded is not None:
    dest_rel = f"{pdf_prefix}{uploaded.name}"
    try:
        path = put_pdf_to_stage(session, stage, dest_rel, uploaded.getvalue())
        st.success(f"Uploaded to {path}")
        refresh_stage_directory(session, stage)
        try:
            st.experimental_rerun()
        except Exception:
            pass
    except Exception as e:
        st.error(f"Upload failed: {e}")

pdfs_df = list_stage_pdfs(session, stage, pdf_prefix)
st.dataframe(pdfs_df, use_container_width=True)


# 2. Extract with AI_PARSE_DOCUMENT -----------------------------------------
st.header("2. Extract Pages (AI_PARSE_DOCUMENT)")
target_pdf = st.selectbox(
    "Select PDF to parse",
    [r for r in pdfs_df["RELATIVE_PATH"].tolist()] if not pdfs_df.empty else [],
)

col_a, col_b = st.columns([1, 1])
with col_a:
    if st.button("Run AI_PARSE_DOCUMENT") and target_pdf:
        try:
            parse_with_ai_parse_document(session, stage, target_pdf, parse_mode)
            doc_basename = os.path.basename(target_pdf)
            cnt_df = session.sql(
                """
                SELECT COUNT(*) AS C
                FROM CURATED.PAGES_AI_PARSE
                WHERE DOC_ID = ? AND PARSE_MODE = ?
                """,
                params=[doc_basename, parse_mode],
            ).to_pandas()
            total = int(cnt_df.iloc[0]["C"]) if not cnt_df.empty else 0
            st.success(f"Parsed rows for {doc_basename} [{parse_mode}] = {total}")
            sample_df = session.sql(
                """
                SELECT PAGE_NUM, LEFT(PAGE_CONTENT, 200) AS SNIPPET
                FROM CURATED.PAGES_AI_PARSE
                WHERE DOC_ID = ? AND PARSE_MODE = ?
                ORDER BY PAGE_NUM
                LIMIT 5
                """,
                params=[doc_basename, parse_mode],
            ).to_pandas()
            if not sample_df.empty:
                st.dataframe(sample_df, use_container_width=True)
        except Exception as e:
            st.error(f"Parse failed: {e}")

with col_b:
    st.info("画像変換は SPCS Service で行います（次のセクションを使用）。")


# 2b. View Parsed Pages ------------------------------------------------------
st.subheader("2b. View Parsed Pages")
try:
    docs_modes = session.sql(
        "SELECT DOC_ID, PARSE_MODE, COUNT(*) AS PAGES, MAX(LOAD_TS) AS LAST_LOAD FROM CURATED.PAGES_AI_PARSE GROUP BY DOC_ID, PARSE_MODE ORDER BY LAST_LOAD DESC, DOC_ID, PARSE_MODE"
    ).to_pandas()
except Exception:
    docs_modes = None

if docs_modes is None or docs_modes.empty:
    st.info("No parsed pages yet. Run AI_PARSE_DOCUMENT above.")
else:
    doc_id_sel = st.selectbox("DOC_ID", sorted(docs_modes["DOC_ID"].unique().tolist()))
    modes_for_doc = docs_modes[docs_modes["DOC_ID"] == doc_id_sel]["PARSE_MODE"].unique().tolist()
    mode_sel = st.selectbox("Parse mode", modes_for_doc, index=0)
    dfp = session.sql(
        """
        SELECT PAGE_NUM, PAGE_CONTENT
        FROM CURATED.PAGES_AI_PARSE
        WHERE DOC_ID = ? AND PARSE_MODE = ?
        ORDER BY PAGE_NUM
        """,
        params=[doc_id_sel, mode_sel],
    ).to_pandas()
    if dfp.empty:
        st.warning("No rows for the selected DOC_ID and mode.")
    else:
        max_page = int(dfp["PAGE_NUM"].max()) + 1
        page_disp = st.slider("Page (1-based)", min_value=1, max_value=max_page, value=1, key=f"pv_{doc_id_sel}_{mode_sel}")
        page_idx = page_disp - 1
        rec = dfp[dfp["PAGE_NUM"] == page_idx]
        text_val = rec.iloc[0]["PAGE_CONTENT"] if not rec.empty else ""
        st.text_area("Parsed text", text_val or "", height=260)


# 3. SPCS Service controls ---------------------------------------------------
st.header("3. Convert PDFs to Images (SPCS Service)")
st.write("初回のサービス実行前に、サイドバーからアプリをRerunさせてください")

# Helpers: robust start/stop functions for SPCS Service
def _try_sql(stmt: str, params: Optional[list] = None) -> Optional[str]:
    try:
        session.sql(stmt, params=params or []).collect()
        return None
    except Exception as e:
        return str(e)

def start_service(name: str) -> str:
    errs = []
    for stmt in [
        f"ALTER SERVICE {name} SET MIN_INSTANCES = 1",
        f"ALTER SERVICE {name} RESUME",
        "SELECT SYSTEM$START_SERVICE(?)",
    ]:
        err = _try_sql(stmt, params=[name] if "?" in stmt else None)
        if not err:
            return "Service started"
        errs.append(err)
    return "; ".join(errs)

def stop_service(name: str) -> str:
    errs = []
    for stmt in [
        f"ALTER SERVICE {name} SUSPEND",
        f"ALTER SERVICE {name} SET MAX_INSTANCES = 0",
        f"ALTER SERVICE {name} SET MIN_INSTANCES = 0",
        "SELECT SYSTEM$STOP_SERVICE(?)",
    ]:
        err = _try_sql(stmt, params=[name] if "?" in stmt else None)
        if not err:
            return "Service stopped"
        errs.append(err)
    return "; ".join(errs)

# Persist service name to session for easier replacement
svc_name = st.text_input(
    "Service name",
    value=st.session_state.get("svc_name", "IRINFO_RAG.SIS_APP.PDF_TO_IMAGE_SVC"),
    key="svc_name",
)
svc_name_val = st.session_state.get("svc_name", "IRINFO_RAG.SIS_APP.PDF_TO_IMAGE_SVC")
col_s1, col_s2, col_s3 = st.columns(3)
with col_s1:
    if st.button("Start Service"):
        try:
            session.sql(f"ALTER SERVICE {svc_name_val} RESUME").collect()
            st.success("Service started")
        except Exception as e:
            st.error(f"Start failed: {e}")
with col_s2:
    if st.button("Stop Service"):
        try:
            session.sql(f"ALTER SERVICE {svc_name_val} SUSPEND").collect()
            st.success("Service stopped")
        except Exception as e:
            st.error(f"Stop failed: {e}")
with col_s3:
    if st.button("Refresh + List Outputs"):
        try:
            refresh_stage_directory(session, stage)
            imgs = list_stage_images(session, stage, images_prefix)
            st.dataframe(imgs, use_container_width=True)
        except Exception as e:
            st.error(f"List failed: {e}")

## 3b. Conversion Queue was removed per request

with st.expander("Service Status & Logs"):
    c1, c2= st.columns([1, 1])
    with c1:
        if st.button("Service Status"):
            try:
                service_status = session.sql(
                    f"DESCRIBE SERVICE {svc_name_val}"
                ).to_pandas()
                st.dataframe(service_status)
            except Exception as e:
                st.error(f"Status failed: {e}")
    with c2:
        if st.button("Get Status"):
            try:
                # Returns VARIANT (JSON)
                status = session.sql(
                    "SELECT SYSTEM$GET_SERVICE_STATUS(?) AS S", params=[svc_name_val]
                ).to_pandas()
                st.json(status.iloc[0]["S"] if not status.empty else {})
            except Exception as e:
                st.error(f"Status failed: {e}")

# 4. Preview Images + Parsed Text -------------------------------------------
st.header("4. Preview Images + Parsed Text")

# Session-state containers for evaluation history and last selection
if "eval_all_results" not in st.session_state:
    # key => { 'df': pd.DataFrame, 'page_map': Dict[int, str] }
    st.session_state["eval_all_results"] = {}
if "eval_all_last_key" not in st.session_state:
    st.session_state["eval_all_last_key"] = None

# Optional: show previously computed batch results (hidden by default)
with st.expander("Batch Results (history)"):
    results_dict = st.session_state.get("eval_all_results", {})
    if results_dict:
        # Build labels like "doc.pdf / layout / claude-3-5-sonnet"
        keys = list(results_dict.keys())
        def _label_for(k: str) -> str:
            try:
                d, m, mdl = k.split("|")
                return f"{d} / {m} / {mdl}"
            except Exception:
                return k
        labels = [_label_for(k) for k in keys]
        # Default to last run if available
        default_idx = 0
        if st.session_state.get("eval_all_last_key") in keys:
            default_idx = keys.index(st.session_state["eval_all_last_key"])
        sel_idx = st.selectbox("Saved batches", list(range(len(keys))), format_func=lambda i: labels[i], index=default_idx if keys else 0)
        sel_key = keys[sel_idx]
        st.dataframe(results_dict[sel_key]["df"], use_container_width=True)
    else:
        st.caption("No batch results yet. Run 'Evaluate All Pages' to populate.")

imgs_df = list_stage_images(session, stage, images_prefix)

# Build mapping: doc_base -> list of (relpath, page_num)
doc_map: Dict[str, List[Dict[str, str]]] = {}
for rel in imgs_df["RELATIVE_PATH"].tolist() if not imgs_df.empty else []:
    base = os.path.basename(rel)
    parsed = parse_image_basename(base)
    if not parsed:
        continue
    doc_map.setdefault(parsed["base"], []).append({"rel": rel, "page": parsed["page"]})

doc_options = sorted(doc_map.keys())
sel_doc = st.selectbox("Doc (by image prefix)", doc_options)

if sel_doc:
    items = doc_map.get(sel_doc, [])
    pages = sorted([int(it["page"]) for it in items])
    if pages:
        # Stable key to ensure slider state ties to the selected doc
        page_num_disp = st.slider(
            "Page (1-based in filename)",
            min_value=min(pages), max_value=max(pages), value=min(pages), key=f"img_page_{sel_doc}"
        )
        # Find relpath for selected page (supports both -page_ and -page-)
        current_rel = None
        for it in items:
            if int(it["page"]) == page_num_disp:
                current_rel = it["rel"]
                break

        expected_0idx = page_num_disp - 1
        doc_id = f"{sel_doc}.pdf"
        exp_rows = fetch_pages_for_doc(session, doc_id, parse_mode)
        exp_text = next((r["PAGE_CONTENT"] for r in exp_rows if int(r["PAGE_NUM"]) == expected_0idx), "")

        img_col, txt_col, eval_col = st.columns([1, 1, 1])
        with img_col:
            if current_rel:
                img_bytes = get_image_bytes(session, stage, current_rel)
                if img_bytes:
                    st.image(img_bytes, caption=current_rel)
                else:
                    st.warning("Unable to fetch image bytes from stage")
            else:
                st.warning("Selected page image not found for this doc.")
        with txt_col:
            tab_view, tab_edit = st.tabs(["tab_view", "tab_edit"])
            # View-only tab
            with tab_view:
                st.text_area(
                    "Expected (from AI_PARSE_DOCUMENT)",
                    exp_text or "",
                    height=650,
                    placeholder="AI_PARSE_DOCUMENTで抽出されたテキストが表示されます（未取得の場合は空）。",
                    key=f"exp_view_{sel_doc}_{page_num_disp}"
                )
            # Editable tab (initialized with the expected text)
            with tab_edit:
                edit_key = f"exp_edit_{sel_doc}_{page_num_disp}"
                if edit_key not in st.session_state:
                    st.session_state[edit_key] = exp_text or ""
                st.text_area(
                    "Editable Copy (based on Expected)",
                    st.session_state.get(edit_key, exp_text or ""),
                    height=650,
                    key=edit_key,
                )

        # Store selection for evaluation
        st.session_state["eval_relpath"] = current_rel
        st.session_state["eval_doc"] = doc_id
        st.session_state["eval_expected_text"] = exp_text
        st.session_state["eval_page"] = page_num_disp
        st.session_state["eval_stage"] = stage

        # 'eval_col'コンテナ内のUIとロジックをここに記述します
        with eval_col:
            # # 必要なライブラリや関数をインポート
            # import re
            # from snowflake.snowpark.functions import current_timestamp
            # from typing import Dict, Any
        
            # --------------------------------------------------------------------------
            # STEP 1: 評価実行のロジックを関数として定義
            # --------------------------------------------------------------------------
            def run_full_evaluation(
                session: "Session",
                doc_id: str,
                sel_doc: str,
                parse_mode: str,
                model: str,
                stage: str,
                images_prefix: str,
                items: list,
            ) -> Dict[int, str]:
                """
                ドキュメント全体のLLM評価を実行し、結果をデータベースに保存し、
                ページ番号をキーとする結果の辞書(page_map)を返す。
                """
                st.info(f"'{sel_doc}' の全ページ評価を開始します (モデル: {model})。")
                page_map: Dict[int, str] = {}
                try:
                    base_pat = re.escape(images_prefix) + re.escape(sel_doc) + r"-page[_-][0-9]+\.png$"
                    sql_tpl = """
                        WITH IMGS AS (
                          SELECT RELATIVE_PATH,
                                 TRY_TO_NUMBER(REGEXP_SUBSTR(RELATIVE_PATH, 'page[_-]([0-9]+)\\\\.png', 1, 1, 'e', 1)) AS PAGE_NO_1BASED
                          FROM DIRECTORY(__STAGE__)
                          WHERE REGEXP_LIKE(RELATIVE_PATH, ?)
                        ),
                        PAGES AS (
                          SELECT RELATIVE_PATH,
                                 (PAGE_NO_1BASED - 1) AS PAGE_IDX,
                                 CONCAT(?, '.pdf') AS DOC_ID
                          FROM IMGS
                          WHERE PAGE_NO_1BASED IS NOT NULL
                        ),
                        EXP AS (
                          SELECT p.RELATIVE_PATH,
                                 REPLACE(REPLACE(t.PAGE_CONTENT, '{', '{{'), '}', '}}') AS expected_text
                          FROM PAGES p
                          JOIN CURATED.PAGES_AI_PARSE t
                            ON t.DOC_ID = p.DOC_ID AND t.PAGE_NUM = p.PAGE_IDX AND t.PARSE_MODE = ?
                        )
                        SELECT e.RELATIVE_PATH,
                               SNOWFLAKE.CORTEX.AI_COMPLETE(
                                 ?,
                                 PROMPT(
                                   'あなたはRAG用の情報をチェックする責任者です。画像{0}は正として、テキスト{1}が正しいか確認してください。\\n- evaluate_score は0〜1（1が完全一致）。なお、数値のズレや図やグラフの記載漏れについては厳しく減点すること。\\n- differences は相違点を箇条書きで日本語の短文にする。\\n- extract 画像からテキストをそのまま抽出する。グラフや画像がある場合、その説明を記載する。この説明はRAGで利用するため検索しやすい記述にすること。アウトプットはMarkdown形式。',
                                   TO_FILE(?, e.RELATIVE_PATH),
                                   e.expected_text
                                 )
                               ) AS RESULT
                        FROM EXP e
                        ORDER BY e.RELATIVE_PATH
                    """
                    sql_stmt = sql_tpl.replace("__STAGE__", stage)
                    df_res_all = session.sql(
                        sql_stmt,
                        params=[base_pat, sel_doc, parse_mode, model, stage],
                    ).to_pandas()
        
                    # 結果を page_map (辞書) に変換
                    for _, row in df_res_all.iterrows():
                        rel = str(row["RELATIVE_PATH"])
                        m = re.search(r"page[_-](\d+)\.png$", rel)
                        if m:
                            page_num = int(m.group(1))
                            page_map[page_num] = str(row["RESULT"]) if row["RESULT"] is not None else ""
        
                    # セッションステートに結果を保存
                    batch_key = f"{doc_id}|{parse_mode}|{model}"
                    if "eval_all_results" not in st.session_state:
                        st.session_state["eval_all_results"] = {}
                    st.session_state["eval_all_results"][batch_key] = {
                        "df": df_res_all,
                        "page_map": page_map,
                    }
                    st.session_state["eval_all_last_key"] = batch_key
        
                    # 評価ログをCURATED.PAGE_EVAL_LOGテーブルに保存
                    rows_to_save = []
                    for p, txt in page_map.items():
                        img_path = next((it["rel"] for it in items if int(it["page"]) == p), f"{images_prefix}{sel_doc}-page_{p}.png")
                        rows_to_save.append({
                            "DOC_ID": doc_id, "PAGE_NUM": p, "MODEL": model,
                            "IMAGE_PATH": img_path, "RESULT": txt,
                        })
                    
                    if rows_to_save:
                        session.sql("CREATE SCHEMA IF NOT EXISTS CURATED").collect()
                        session.sql("""
                            CREATE TABLE IF NOT EXISTS CURATED.PAGE_EVAL_LOG (
                              DOC_ID STRING, PAGE_NUM NUMBER, MODEL STRING,
                              IMAGE_PATH STRING, RESULT STRING,
                              LOAD_TS TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
                            )
                        """).collect()
                        
                        new_evals_df = session.create_dataframe(rows_to_save)
                        # LOAD_TS列を追加して保存
                        new_evals_df.with_column("LOAD_TS", current_timestamp()).write.save_as_table(
                            "CURATED.PAGE_EVAL_LOG", mode="append"
                        )
                        st.success(f"評価結果 {len(rows_to_save)} 件を CURATED.PAGE_EVAL_LOG に保存しました。")
        
                except Exception as e:
                    st.error(f"バッチ評価中にエラーが発生しました:")
                    st.exception(e)
                
                return page_map

            # （トークン数カウント機能は要件により削除しました）

            # --------------------------------------------------------------------------
            # STEP 2: 表示ロジック
            # --------------------------------------------------------------------------
            model = st.selectbox("Model", ["claude-4-sonnet", "pixtral-large"], index=0, key="eval_model")
            run_all = st.button("Evaluate All Pages (Doc)", key="eval_all")
            auto_eval = st.checkbox(
                "Auto-run full evaluation if no saved result",
                value=False,
                key=f"auto_eval_{doc_id}_{model}"
            )
            # Count Tokens 機能は削除
        
            # --- メインの表示ロジック ---
            cur_eval_txt = None
            batch_key = f"{doc_id}|{parse_mode}|{model}"
        
            # 1. まずセッションステート（今回の実行結果）を確認
            if batch_key in st.session_state.get("eval_all_results", {}):
                page_map = st.session_state["eval_all_results"][batch_key]["page_map"]
                cur_eval_txt = page_map.get(page_num_disp)
        
            # 2. セッションになければ、データベース（過去の実行結果）を確認
            if cur_eval_txt is None:
                try:
                    res_df = session.sql(
                        """
                        SELECT RESULT FROM CURATED.PAGE_EVAL_LOG
                        WHERE DOC_ID = ? AND PAGE_NUM = ? AND MODEL = ?
                        ORDER BY LOAD_TS DESC LIMIT 1
                        """,
                        params=[doc_id, page_num_disp, model]
                    ).to_pandas()
                    if not res_df.empty:
                        cur_eval_txt = res_df.iloc[0]["RESULT"]
                except Exception:
                    # テーブルが存在しない場合などは無視して次に進む
                    pass
        
            # 3. ボタンが押された場合は、強制的に再評価を実行
            if run_all:
                with st.spinner("ドキュメント全体の評価を再実行します..."):
                    new_page_map = run_full_evaluation(
                        session, doc_id, sel_doc, parse_mode, model, stage, images_prefix, items
                    )
                    cur_eval_txt = new_page_map.get(page_num_disp, "") # 表示用にテキストを更新
            # （トークン数カウント機能は削除）
            
            # 4. ここまで来ても結果がない場合（初回表示など）、自動で評価を実行
            elif cur_eval_txt is None and auto_eval:
                with st.spinner(f"ページ {page_num_disp} の既存評価データがありません。ドキュメント全体の評価を実行します..."):
                    new_page_map = run_full_evaluation(
                        session, doc_id, sel_doc, parse_mode, model, stage, images_prefix, items
                    )
                    cur_eval_txt = new_page_map.get(page_num_disp, "")
            elif cur_eval_txt is None and not auto_eval:
                st.info("保存済みの評価が見つかりません。'Evaluate All Pages (Doc)' を押すか、Auto-run を有効にしてください。")
        
            # 5. 最終的な結果を表示エリアに描画
            st.text_area(
                "LLM Output (This Page)",
                cur_eval_txt or "",  # Noneの場合は空文字にする
                height=500,
                placeholder="このページの評価データはまだありません。"
            )
            st.caption("評価スコアは 0〜1（1 が完全一致＝良いスコア）")

st.markdown("---") # 区切り線

# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
# ★ 修正箇所：セクション5を新設
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
st.header("5. Curate Parsed Text")

# Build doc/mode options from existing parses
try:
    docs_modes2 = session.sql(
        "SELECT DOC_ID, PARSE_MODE, COUNT(*) AS PAGES, MAX(LOAD_TS) AS LAST_LOAD "
        "FROM CURATED.PAGES_AI_PARSE GROUP BY DOC_ID, PARSE_MODE "
        "ORDER BY LAST_LOAD DESC, DOC_ID, PARSE_MODE"
    ).to_pandas()
except Exception:
    docs_modes2 = None

if docs_modes2 is None or docs_modes2.empty:
    st.info("AI_PARSE_DOCUMENTの結果がありません。上のセクションで抽出してください。")
else:
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        edit_doc = st.selectbox("DOC_ID (編集対象)", sorted(docs_modes2["DOC_ID"].unique().tolist()))
    with c2:
        modes_for_doc2 = docs_modes2[docs_modes2["DOC_ID"] == edit_doc]["PARSE_MODE"].unique().tolist()
        edit_mode = st.selectbox("Parse mode (編集対象)", modes_for_doc2, index=0)
    with c3:
        # モデルは保存済みログから候補を取得
        try:
            mdl_df = session.sql(
                "SELECT DISTINCT MODEL FROM CURATED.PAGE_EVAL_LOG WHERE DOC_ID = ? ORDER BY MODEL",
                params=[edit_doc],
            ).to_pandas()
            model_opts = mdl_df["MODEL"].dropna().astype(str).tolist() if not mdl_df.empty else []
        except Exception:
            model_opts = []
        edit_model = st.selectbox("LLM Model (ログ参照)", model_opts or ["<none>"], index=0)

    # 編集用データをロード
    if st.button("Load for editing", key=f"load_edit_{edit_doc}_{edit_mode}"):
        try:
            base_df = session.sql(
                """
                SELECT DOC_ID, PAGE_NUM, PARSE_MODE, PAGE_CONTENT
                FROM CURATED.PAGES_AI_PARSE
                WHERE DOC_ID = ? AND PARSE_MODE = ?
                ORDER BY PAGE_NUM
                """,
                params=[edit_doc, edit_mode],
            ).to_pandas()

            # LLM_OUTPUT をログから取得（ページ単位で最新）
            llm_map = {}
            if edit_model and edit_model != "<none>":
                try:
                    df_llm = session.sql(
                        """
                        SELECT PAGE_NUM, RESULT
                        FROM (
                          SELECT PAGE_NUM, RESULT,
                                 ROW_NUMBER() OVER (PARTITION BY PAGE_NUM ORDER BY LOAD_TS DESC) AS RN
                          FROM CURATED.PAGE_EVAL_LOG
                          WHERE DOC_ID = ? AND MODEL = ?
                        )
                        WHERE RN = 1
                        ORDER BY PAGE_NUM
                        """,
                        params=[edit_doc, edit_model],
                    ).to_pandas()
                    for _, r in df_llm.iterrows():
                        try:
                            llm_map[int(r["PAGE_NUM"])] = str(r["RESULT"]) if r["RESULT"] is not None else ""
                        except Exception:
                            pass
                except Exception:
                    pass

            base_df["LLM_OUTPUT"] = base_df["PAGE_NUM"].apply(lambda i: llm_map.get(int(i) + 1, ""))
            base_df["CURATED_TEXT"] = base_df["PAGE_CONTENT"]
            st.session_state["curate_df_key"] = f"curate_{edit_doc}_{edit_mode}_{edit_model}"
            st.session_state[st.session_state["curate_df_key"]] = base_df
            st.success("編集用データを読み込みました。")
        except Exception as e:
            st.error(f"読み込みに失敗: {e}")

    df_cur = st.session_state.get(st.session_state.get("curate_df_key", ""))
    if isinstance(df_cur, pd.DataFrame) and not df_cur.empty:
        st.caption("以下の3カラムを横並びで編集できます（PAGE_CONTENTは参照のみ）")
        edited = st.data_editor(
            df_cur,
            use_container_width=True,
            key=f"editor_{st.session_state['curate_df_key']}",
            column_config={
                "DOC_ID": st.column_config.TextColumn("DOC_ID", disabled=True),
                "PAGE_NUM": st.column_config.NumberColumn("PAGE_NUM", disabled=True),
                "PARSE_MODE": st.column_config.TextColumn("PARSE_MODE", disabled=True),
                "PAGE_CONTENT": st.column_config.TextColumn("1. AI_PARSE_DOCUMENT", disabled=True),
                "LLM_OUTPUT": st.column_config.TextColumn("2. LLM OUTPUT (編集可)"),
                "CURATED_TEXT": st.column_config.TextColumn("3. CURATED (編集可)"),
            },
            height=420,
        )

        # Always append to CURATED.PAGES_AI_PARSE_CURATED
        if st.button("Save (append to CURATED.PAGES_AI_PARSE_CURATED)", key=f"save_curated_{st.session_state['curate_df_key']}"):
            try:
                session.sql(
                    """
                    CREATE TABLE IF NOT EXISTS CURATED.PAGES_AI_PARSE_CURATED (
                      DOC_ID        STRING,
                      PAGE_NUM      NUMBER,
                      PARSE_MODE    STRING,
                      PAGE_CONTENT  STRING,
                      LLM_OUTPUT    STRING,
                      CURATED_TEXT  STRING,
                      LOAD_TS       TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
                    )
                    """
                ).collect()
                out_df = edited.copy()
                out_df["PAGE_NUM"] = pd.to_numeric(out_df["PAGE_NUM"], errors="coerce").fillna(0).astype(int)
                (
                    session.create_dataframe(
                        out_df[[
                            "DOC_ID", "PAGE_NUM", "PARSE_MODE", "PAGE_CONTENT", "LLM_OUTPUT", "CURATED_TEXT"
                        ]]
                    )
                    .with_column("LOAD_TS", current_timestamp())
                    .write
                    .save_as_table("CURATED.PAGES_AI_PARSE_CURATED", mode="append")
                )
                st.success(f"Saved {len(out_df)} rows to CURATED.PAGES_AI_PARSE_CURATED (append)")
            except Exception as e:
                st.error(f"保存に失敗: {e}")

st.header("6. View Evaluation Results")

# --- セッション内で最後に実行したバッチ評価の結果を表示 ---
with st.expander("Latest Batch Evaluation Result (from this session)"):
    if st.session_state.get("eval_all_last_key"):
        last_key = st.session_state["eval_all_last_key"]
        if last_key in st.session_state.get("eval_all_results", {}):
            # df_res_all を表示
            st.dataframe(st.session_state["eval_all_results"][last_key]["df"], use_container_width=True)
        else:
            st.info("No batch evaluation has been run in this session yet.")
    else:
        st.info("No batch evaluation has been run in this session yet.")

# --- Snowflakeに保存された評価ログを表示 ---
st.subheader("Saved Evaluation Log (from CURATED.PAGE_EVAL_LOG)")
if sel_doc:
    doc_id_to_filter = f"{sel_doc}.pdf"
    st.caption(f"Displaying logs for: **{doc_id_to_filter}**")
    try:
        log_df = session.table("CURATED.PAGE_EVAL_LOG").filter(col("DOC_ID") == doc_id_to_filter).to_pandas()
        if not log_df.empty:
            # related_pathが一致するものだけ、というご要望は DOC_ID でのフィルタリングで実現
            st.dataframe(log_df, use_container_width=True)
        else:
            st.info(f"No saved logs found for '{doc_id_to_filter}' in CURATED.PAGE_EVAL_LOG.")
    except Exception as e:
        st.error("Failed to fetch logs from CURATED.PAGE_EVAL_LOG. Does the table exist?")
        # st.exception(e) # デバッグ用に有効化しても良い
else:
    st.info("Select a document in Section 4 to see its saved evaluation logs.")
