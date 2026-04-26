FROM python:3.10

WORKDIR /app

COPY . .

RUN pip install pandas scikit-learn numpy kafka-python pymongo

CMD ["python", "model.py"]
