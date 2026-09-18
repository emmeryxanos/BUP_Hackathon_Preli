FROM python:3.11-slim

WORKDIR /app

# No external solver binary needed: the optimizer uses SciPy's bundled HiGHS
# solver (scipy.optimize.linprog(method="highs")), a pure wheel dependency.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

EXPOSE 8000

# No secrets are baked into the image. OPENAI_API_KEY and any other
# configuration must be supplied at `docker run` time via -e / --env-file.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
