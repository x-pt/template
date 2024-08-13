FROM python:{{cookiecutter.python_version}} AS venv

LABEL author="{{cookiecutter.full_name}}"

# set environment variables
# NO Cache
ENV PIP_NO_CACHE_DIR 1
ENV PIP_INDEX_URL https://pypi.tuna.tsinghua.edu.cn/simple

WORKDIR /app

COPY requirement.txt .

RUN python -m venv venv

# TODO using pex or not?
RUN . venv/bin/activate && \
    python -m ensurepip -U && \
    pip install -r requirement.txt


FROM python:{{cookiecutter.python_version}}-slim

LABEL author="{{cookiecutter.full_name}}"

COPY --from=venv /app/venv /app/venv
ENV PATH /app/venv/bin:$PATH

# set environment variables
# Prevents Python from writing pyc files to disc (equivalent to python -B option)
# https://docs.python.org/3/using/cmdline.html#cmdoption-B
ENV PYTHONDONTWRITEBYTECODE 1
# Prevents Python from buffering stdout and stderr (equivalent to python -u option)
# https://docs.python.org/3/using/cmdline.html#cmdoption-u
ENV PYTHONUNBUFFERED 1

WORKDIR /app

# if something you want ignore it, add it to .dockerignore
COPY . .

HEALTHCHECK --start-period=30s CMD python -c "import requests; requests.get('http://localhost:8000', timeout=2)"

# CMD . .venv/bin/activate && python src/app.py
CMD python src/app.py
