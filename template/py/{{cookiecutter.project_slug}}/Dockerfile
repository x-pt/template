# Base image for building the virtual environment
FROM python:{{cookiecutter.python_version}}-bookworm AS builder

ENV PATH="/root/.cargo/bin:$PATH" \
    UV_INDEX_URL="https://mirrors.cernet.edu.cn/pypi/web/simple" \
    PIP_INDEX_URL="https://mirrors.cernet.edu.cn/pypi/web/simple"

# Install uv and tools
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /app

COPY pyproject.toml .

# Create and install dependencies in the virtual environment
RUN uv sync

# Separate stage for validation (build and test)
FROM builder AS validator

WORKDIR /app
COPY . .

# Run build and test as part of the validation
RUN make build && make test

# Final image for running the application
FROM python:{{cookiecutter.python_version}}-slim-bookworm

LABEL author="{{cookiecutter.full_name}}"

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Copy the virtual environment and application code
COPY --from=builder /app/.venv /app/.venv
COPY src ./src

HEALTHCHECK --start-period=30s CMD python -c "import requests; requests.get('http://localhost:8000', timeout=2)"

CMD ["python", "src/{{cookiecutter.package_name}}/app.py"]
