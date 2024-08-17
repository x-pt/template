# Base image for building the virtual environment
FROM python:{{cookiecutter.python_version}}-bullseye AS builder

ENV PATH="/root/.cargo/bin:$PATH" \
    PIP_INDEX_URL="https://mirrors.cernet.edu.cn/pypi/web/simple"

RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    uv tool install hatch && \
    uv tool install ruff

WORKDIR /app

COPY pyproject.toml .

RUN uv venv venv && \
    . venv/bin/activate && \
    uv pip install .[dev]

# Final image for running the application
FROM python:{{cookiecutter.python_version}}-slim-bullseye

LABEL author="{{cookiecutter.full_name}}"

ENV PATH="/app/venv/bin:/root/.local/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY --from=builder /app/venv /app/venv
COPY --from=builder /root/.local /root/.local
COPY . .

RUN hatch build && hatch test

CMD ["python", "src/app.py"]
