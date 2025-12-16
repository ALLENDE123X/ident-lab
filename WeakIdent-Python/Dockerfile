# WeakIdent PDE-Selector
# Reproducible Docker environment with pinned scientific Python stack

FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements-docker.txt .

# Install Python dependencies with pinned versions
RUN pip install --no-cache-dir -r requirements-docker.txt

# Copy pyproject.toml for editable install
COPY pyproject.toml .
COPY src/ src/
COPY scripts/ scripts/
COPY tests/ tests/
COPY config/ config/

# Install the package in editable mode
RUN pip install -e .

# Default command: run tests
CMD ["pytest", "tests/", "-v", "--tb=short"]
