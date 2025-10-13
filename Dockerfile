FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

WORKDIR /app

# System deps: tini + build toolchain for llama_cpp_python + OpenBLAS runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl tini \
      build-essential cmake \
      libopenblas0 \
  && rm -rf /var/lib/apt/lists/*

# Python deps first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY . .

# Make sure Python can import "app.*"
ENV PYTHONPATH=/app

EXPOSE 8000

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","8000", "--proxy-headers","--forwarded-allow-ips","*", "--timeout-keep-alive","120"]
