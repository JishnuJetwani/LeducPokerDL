FROM python:3.11-slim

WORKDIR /app

COPY leduc-bot-service/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY leduc-bot-service /app/service
COPY leduc-bot-training /app/training

ENV PYTHONPATH=/app/training
ENV LEDUC_TRAINING_DIR=/app/training
ENV LEDUC_MODEL_PATH=/app/training/policy_net_leduc.pt

WORKDIR /app/service

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
