FROM python:3.8-slim as builder

ARG PYTHON_ENV=production
ENV PYTHON_ENV=${PYTHON_ENV}

WORKDIR /app
COPY Pipfile* /app/

# RUN apt update && apt install gcc -y
RUN pip install --no-cache-dir pipenv
RUN python3 -m venv /app/.venv
RUN pipenv install


FROM python:3.8-slim

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
RUN rm /app/.venv/pyvenv.cfg

COPY . /app

# Extends python modules search path to include ones from the builder.
ENV PYTHONPATH="$PYTHONPATH:/app/.venv/lib/python3.8/site-packages"
# Extends search path to include runnable python utilities from the builder.
ENV PATH="$PATH:/app/.venv/bin"
ENV PYTHONUNBUFFERED TRUE