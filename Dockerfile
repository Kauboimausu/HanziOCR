# Phase 1: UV for project dependencies
FROM ghcr.io/astral-sh/uv:python3.14-trixie-slim AS uv
WORKDIR /HanziOCR

ENV UV_NO_DEV=1
ENV UV_COMPILE_BYTECODE=1

# Copy dependencies file
COPY pyproject.toml uv.lock ./

# Install only runtime dependencies, without installing the project
RUN uv sync --locked --no-install-project

# We copy and install the source code 
COPY . .
RUN uv sync --locked


FROM python:3.14-slim-trixie AS python
WORKDIR /HanziOCR

COPY --from=uv  /HanziOCR/.venv /HanziOCR/.venv
COPY . .

ENV PATH="/HanziOCR/.venv/bin:$PATH"

CMD ["uvicorn", "hanzi_ocr.api:app", "--host", "0.0.0.0", "--port", "8080"]
EXPOSE 8080
