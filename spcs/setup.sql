-- Snowpark Container Services (SPCS) minimal setup for PDF→Image Job
-- Adjust identifiers to your environment.
-- このスクリプトは README の手順に沿って、イメージ push 後に実行してください。

-- 1) Compute pool (create if needed)
CREATE COMPUTE POOL IF NOT EXISTS IRINFO_POOL
  MIN_NODES = 1
  MAX_NODES = 1
  INSTANCE_FAMILY = CPU_X64_XS;

-- Verify
SHOW COMPUTE POOLS LIKE 'IRINFO_POOL';

-- 2) Image repository (stores container images)
CREATE IMAGE REPOSITORY IF NOT EXISTS IRINFO_RAG.RAW.PDF_IMG_REPO;

-- Verify
SHOW IMAGE REPOSITORIES LIKE 'PDF_IMG_REPO';
-- repository_urlをメモする
-- IDからDocker ImageをImage repositoryにPushする





-- Use SERVICE (JOB may be unavailable depending on account)
-- 既存のサービスがある場合は先に削除（REPLACEが使えないアカウント向け）
DROP SERVICE IF EXISTS IRINFO_RAG.SIS_APP.PDF_TO_IMAGE_SVC;

CREATE SERVICE IRINFO_RAG.SIS_APP.PDF_TO_IMAGE_SVC
  IN COMPUTE POOL IRINFO_POOL
  FROM SPECIFICATION $$
spec:
  containers:
    - name: pdf2img
      image: sfseapac-fsi-japan.registry.snowflakecomputing.com/irinfo_rag/raw/pdf_img_repo/pdf2img:latest
      command:
        - /bin/sh
        - -lc
        - >
          set -e;
          STAGE_MOUNT="${STAGE_MOUNT:-/stage}";
          IN_PREFIX="${PDF_PREFIX:-file2/}";
          OUT_PREFIX="${IMAGES_PREFIX:-output_image2/}";
          DPI="${DPI:-200}";
          mkdir -p "$STAGE_MOUNT/$OUT_PREFIX";
          for pdf in $(find "$STAGE_MOUNT/$IN_PREFIX" -type f -name '*.pdf' | sort); do
            base=$(basename "${pdf%.pdf}");
            pdftoppm -png -r "$DPI" "$pdf" "$STAGE_MOUNT/$OUT_PREFIX${base}-page";
            for p in "$STAGE_MOUNT/$OUT_PREFIX${base}"-page-*.png; do
              [ -e "$p" ] || continue;
              mv "$p" "${p/-page-/-page_}";
            done;
          done;
          sleep "${SVC_IDLE_AFTER_RUN:-30}";
      volumeMounts:
        - name: stage-volume
          mountPath: /stage
      resources:
        requests:
          cpu: 1
          memory: 2Gi
      env:
        STAGE_MOUNT: /stage
        PDF_PREFIX: file2/
        IMAGES_PREFIX: output_image2/
        DPI: "200"
        SVC_IDLE_AFTER_RUN: "30"
  volumes:
    - name: stage-volume
      source: "@IRINFO_RAG.RAW.IR_STAGE"
$$;

-- Verify service
SHOW SERVICES LIKE 'PDF_TO_IMAGE_SVC';
DESCRIBE SERVICE IRINFO_RAG.SIS_APP.PDF_TO_IMAGE_SVC;

-- Start service (runs conversion and waits briefly)
ALTER SERVICE IRINFO_RAG.SIS_APP.PDF_TO_IMAGE_SVC RESUME;

-- Optional stop
-- ALTER SERVICE IRINFO_RAG.SIS_APP.PDF_TO_IMAGE_SVC SUSPEND;
