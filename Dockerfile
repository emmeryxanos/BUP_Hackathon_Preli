FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

EXPOSE 8000

ENV HOST=0.0.0.0 \
    PORT=8000

# No secrets are baked into the image. ANTHROPIC_API_KEY and any other
# configuration must be supplied at `docker run` time via -e / --env-file.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
