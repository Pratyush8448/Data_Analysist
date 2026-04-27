# -------------------------
# Base image
# -------------------------
FROM python:3.12-slim

# -------------------------
# Environment settings
# -------------------------
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# -------------------------
# Work directory
# -------------------------
WORKDIR /app

# -------------------------
# System dependencies
# -------------------------
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# -------------------------
# Install Python dependencies
# -------------------------
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# -------------------------
# Copy application
# -------------------------
COPY . .

# -------------------------
# Ensure entrypoint is executable
# -------------------------
RUN chmod +x entrypoint.sh

# -------------------------
# Expose port (informational)
# -------------------------
EXPOSE 8000

# -------------------------
# Start app
# -------------------------
CMD ["./entrypoint.sh"]
