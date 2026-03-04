FROM python:3.11-slim

WORKDIR /app

# System dependencies for pdfplumber, pymupdf
RUN apt-get update && apt-get install -y \
    libmupdf-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install uv for faster dependency installation
RUN pip install uv

# Copy dependency manifest first (layer caching)
COPY pyproject.toml ./

# Install Python dependencies
RUN uv pip install --system -e "." 

# Copy source code
COPY src/ ./src/
COPY rubric/ ./rubric/
COPY data/ ./data/
COPY DOMAIN_NOTES.md ./

# Create refinery artifact directories
RUN mkdir -p .refinery/profiles .refinery/pageindex

# Environment
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default: run the CLI help
ENTRYPOINT ["python", "-m", "src.main"]
CMD ["--help"]
