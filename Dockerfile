FROM python:3.8-slim
WORKDIR /app
COPY requirements.txt /app
RUN apt-get update && apt-get install -y bash \
    && /bin/bash -c "pip install --no-cache-dir -r requirements.txt"
COPY . /app
EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]