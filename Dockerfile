FROM python:3.12-slim

LABEL maintainer="Tyler J. Newton" \
      description="AquaContam reproducibility container"

# System dependencies for geospatial libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    libspatialindex-dev \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (for layer caching)
COPY pyproject.toml setup.cfg* setup.py* ./
COPY src/ src/
RUN pip install --no-cache-dir -e ".[test,boost,interpret,hpo,paper,gate]"

# Copy remaining project files
COPY . .

# Default: run the full reproducibility pipeline
ENTRYPOINT ["python", "scripts/reproduce.py"]
CMD ["--all", "--strict"]
