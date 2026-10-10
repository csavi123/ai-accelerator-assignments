# One image for both the backend and the frontend.
# docker-compose.yml decides which one each container runs.

# Start from a small official Python image.
FROM python:3.12-slim

WORKDIR /app

# Install the Python packages first, so Docker can reuse this step
# when only our code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy our code into the image.
COPY . .

# By default, run the backend.
CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8000"]
