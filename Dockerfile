FROM python:3.11-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY ".python-version" "pyproject.toml" "uv.lock" ./
RUN uv sync --locked

RUN mkdir -p models data
COPY predict.py train.py ./
COPY models/heart_model.bin models/
COPY data/heart.csv data/

EXPOSE 8000

# ENTRYPOINT ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "8000"]
ENTRYPOINT ["uv", "run", "uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "8000"]
