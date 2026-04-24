# ── Base image ───────────────────────────────────────────────────────────────
# Match the project's pyenv Python version (see .python-version).
FROM python:3.10.6-slim

WORKDIR /app

# ── System dependencies ──────────────────────────────────────────────────────
# Build tools are needed for some scientific Python wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ──────────────────────────────────────────────────────
# Install requirements first so Docker can cache this layer when only
# application code changes.
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Application code ─────────────────────────────────────────────────────────
# Copy the package source and install it in editable mode.
COPY src ./src
COPY setup.py pyproject.toml ./
RUN pip install --no-cache-dir -e .

COPY api ./api

# ── Runtime data directories ─────────────────────────────────────────────────
# These are mounted or populated at runtime (not baked into the image).
RUN mkdir -p /app/data/raw /app/data/processed /app/models /app/submissions

# ── Default command ──────────────────────────────────────────────────────────
# Run the preprocessing pipeline. Override with e.g.:
#   docker run sentinel python -m sentinel.main train
CMD ["uvicorn", "api.fast:app", "--host", "0.0.0.0", "--port", "8000"]
