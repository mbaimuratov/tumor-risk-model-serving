FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /mlflow

RUN python -m pip install --upgrade pip \
    && python -m pip install mlflow==3.14.0 psycopg2-binary==2.9.11

EXPOSE 5000
