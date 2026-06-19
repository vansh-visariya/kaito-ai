# Use a lightweight python 3.13 image
FROM python:3.13-slim

# Set the working directory
WORKDIR /app

# Install uv package manager
RUN pip install uv

# Copy the project files
COPY . .

# Install dependencies using uv
RUN uv sync --frozen

# Ensure data directories exist and are accessible
ENV DATA_DIR=/data
RUN mkdir -p /data/database /data/uploads /data/chroma_langchain_db

# Expose the API port
EXPOSE 8000

# Start the Uvicorn server bound to 0.0.0.0
CMD ["uv", "run", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
