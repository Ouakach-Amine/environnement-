FROM python:3.10

WORKDIR /app

COPY . .

RUN pip install flask kafka-python pymongo

CMD ["python", "app.py"]
