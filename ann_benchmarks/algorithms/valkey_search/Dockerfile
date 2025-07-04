FROM python:3.9-slim

# Install Redis client for Python
RUN pip install redis

# Set working directory
WORKDIR /app

# Copy the algorithm module
COPY module.py /app/

# Default command (can be overridden for local testing)
CMD ["python", "-c", "print('Valkey-Search NGT algorithm ready')"] 