# sfc-gh-ryoshida-streamlit_extract_data_for_rag
Snowflake 上で IR レポート取り込み/プレビュー用の Streamlit アプリを再現するためのファイルが入っています。

- `streamlit_app.py`: Streamlit in Snowflakeのアプリ
- `setup.sql`: Snowflake 上のデータベース/スキーマ/ステージ/テーブルの初期セットアップ用SQL

## 前提条件
- Snowflake アカウント（Snowsight が利用可能）
- 利用ロールが DB/スキーマ/テーブル/ステージの作成権限を持つこと

## 1. 初期セットアップ
1) Snowsight でワークシートを開き、実行ロール/ウェアハウスを選択します。
2) `base/setup.sql` の内容を実行します。
   - DB: `IRINFO_RAG`
   - スキーマ: `RAW`, `CURATED`
   - ステージ: `IRINFO_RAG.RAW.IR_STAGE`（ディレクトリリスティング有効）
   - テーブル: `CURATED.PAGES_AI_PARSE`, `CURATED.PAGE_EVAL_LOG`, `CURATED.PAGES_AI_PARSE_CURATED`

必要に応じて、コメントアウトしている GRANT 文を環境のロール名に置換して実行してください。

## 2. SPCS での画像自動変換実施する
このリポジトリには、SPCS 用の最小テンプレートが `spcs/` 配下にあります。
- 参照ファイル
  - `spcs/README.md`: 全体像と手順
  - `spcs/setup.sql`: Compute Pool / Image Repository / Service の作成SQL（サービス名はアプリ既定と一致: `IRINFO_RAG.SIS_APP.PDF_TO_IMAGE_SVC`）
  - `spcs/job_spec.yaml`: 汎用の SPec（自前のイメージURL/ステージに置換して利用）
手順の要点（概要）
1) Snowsight で `spcs/setup.sql` を実行
   - Compute Pool: `IRINFO_POOL`
   - Image Repository: `IRINFO_RAG.RAW.PDF_IMG_REPO`
   - Service: `IRINFO_RAG.SIS_APP.PDF_TO_IMAGE_SVC`（spec 内 `image:` が上記 push 済みのURLであること）
   - ステージは `@IRINFO_RAG.RAW.IR_STAGE` を `/stage` にマウント
2) コンテナイメージを用意して Snowflake Image Repository に push
  ローカルで `Dockerfile` を用意（例）:
    ```Dockerfile
    FROM alpine:3.19
    RUN apk add --no-cache poppler-utils bash coreutils
    WORKDIR /work
    CMD ["/bin/sh", "-lc", "echo container ready"]
    ```

  linux/amd64 イメージを作成して push:
  
    ```
    # 1) レジストリにログイン
    docker login <org_name>-<account_name>.registry.snowflakecomputing.com -u <user_name>

    # 2) buildx のセットアップ
    docker buildx create --use 2>/dev/null || docker buildx use default
    docker buildx inspect --bootstrap

    # 3) linux/amd64 でビルドしてそのまま push（ビルドコンテキストは spcs）
    docker buildx build \
      --platform linux/amd64 \
      -t <repository_urlの出力>/pdf2img:latest \
      --push \
      .
    ```
3) サービスを作成する
   -  `spcs/setup.sql`の `image:` を上記に合わせて更新してください。
4) サービス起動と起動状況の確認
   - 起動: `ALTER SERVICE IRINFO_RAG.SIS_APP.PDF_TO_IMAGE_SVC RESUME;`
   - 起動状況を確認: `ESCRIBE SERVICE IRINFO_RAG.SIS_APP.PDF_TO_IMAGE_SVC;`


## 4. Streamlit アプリの作成
Snowsight の Streamlit から新規アプリを作成し、`base/streamlit_app.py` の内容を貼り付けて保存します。

- アプリ内の既定設定
  - ステージ: `@IRINFO_RAG.RAW.IR_STAGE`
  - PDF プレフィックス: `file2/`
  - 画像プレフィックス: `output_image2/`
  - AI_PARSE_DOCUMENT モード: `layout`（`OCR` も可）

必要に応じてサイドバーから変更できます。

## 5. 使い方の概略
- セクション1: PDF を `IR_STAGE` にアップロード
- セクション2: `AI_PARSE_DOCUMENT` でページごとにテキスト抽出
- セクション2b: 抽出済みページの参照
- セクション3: PDF2ImangeのSPCS サービスの開始/停止、ステージの出力一覧
- セクション4: 画像＋Expectedテキストのプレビュー
  - テキスト欄は `tab_view`（表示のみ）と `tab_edit`（初期値は Expected の複製）
- セクション5: 抽出結果・LLM出力・CURATED の3列を編集し、`CURATED.PAGES_AI_PARSE_CURATED` に保存
- セクション6: 評価ログの確認（チェック用）（`CURATED.PAGE_EVAL_LOG`）


