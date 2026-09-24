FROM python:3.10

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir --default-timeout=300 --retries 10 -r requirements.txt


EXPOSE 8000

CMD ["sh", "-c", "python pipeline/training_pipeline.py && uvicorn main:app --host 0.0.0.0 --port 8000"]