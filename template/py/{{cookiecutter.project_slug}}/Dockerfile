# Base image for building the virtual environment
FROM python:{{cookiecutter.python_version}}-bullseye AS builder

ENV PATH="/root/.cargo/bin:$PATH" \
    PIP_INDEX_URL="https://mirrors.cernet.edu.cn/pypi/web/simple"

# Install uv and tools
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    uv tool install hatch && \
    uv tool install ruff

WORKDIR /app

COPY pyproject.toml .

# Create and install dependencies in the virtual environment
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

# Copy the virtual environment and application code
COPY --from=builder /app/venv /app/venv
COPY --from=builder /root/.local /root/.local
COPY . .

# Ensure the application builds and tests are executed correctly
RUN hatch build && hatch test

HEALTHCHECK --start-period=30s CMD python -c "import requests; requests.get('http://localhost:8000', timeout=2)"

CMD ["python", "src/app.py"]
