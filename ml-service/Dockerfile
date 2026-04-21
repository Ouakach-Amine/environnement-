FROM python:3.10

WORKDIR /app

COPY . .

RUN pip install scikit-learn numpy kafka-python pymongo

CMD ["python", "model.py"]
