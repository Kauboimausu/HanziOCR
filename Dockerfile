# Phase 1: UV for project dependencies
FROM ghcr.io/astral-sh/uv:python3.14-dhi AS uv
WORKDIR /HanziOCR

# Copy dependencies file
COPY pyproject.toml uv.lock ./

# Omit development libraries
RUN uv sync --frozen --no-dev --no-install-project

FROM python:3.14 AS python
WORKDIR /HanziOCR

# We copy the source code
COPY --from=uv  /hanziOCR/.venv /hanziOCR/.venv
COPY .dockerignore .
COPY . .

ENV PATH="/HanziOCR/.venv/bin:$PATH"

CMD ["python3", "main.py"]
EXPOSE 8080
