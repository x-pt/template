FROM python:{{cookiecutter.python_version}}-bullseye AS venv

ENV PATH="/root/.cargo/bin/:$PATH"
ENV PIP_INDEX_URL="https://mirrors.cernet.edu.cn/pypi/web/simple"
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    uv tool install hatch && \
    uv tool install ruff

WORKDIR /app

COPY pyproject.toml .

RUN uv venv venv && \
    . venv/bin/activate && \
    uv pip install .[dev]


FROM python:{{cookiecutter.python_version}}-slim-bullseye

LABEL author="{{cookiecutter.full_name}}"

COPY --from=venv /app/venv /app/venv
COPY --from=venv /root/.local/bin/hatch /root/.local/bin/hatch
COPY --from=venv /root/.local/bin/ruff /root/.local/bin/ruff
ENV PATH="/app/venv/bin:$PATH"
ENV PATH="/root/.local/bin/:$PATH"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# if something you want ignore it, add it to .dockerignore
COPY . .

RUN hatch build && hatch test

# CMD . .venv/bin/activate && python src/app.py
CMD python src/app.py
