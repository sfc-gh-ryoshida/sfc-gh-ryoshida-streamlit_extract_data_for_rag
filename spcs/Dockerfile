# Minimal container for PDF → PNG conversion using Poppler
# Build locally and push to your Snowflake Image Repository.

FROM alpine:3.19

# Install poppler-utils (pdftoppm), bash, coreutils for convenience
RUN apk add --no-cache poppler-utils bash coreutils

WORKDIR /work

# Default command just for sanity check
CMD ["/bin/sh", "-lc", "echo container ready; pdftoppm -v || true"]

