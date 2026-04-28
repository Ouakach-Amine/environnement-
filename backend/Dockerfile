FROM python:3.10

WORKDIR /app

COPY . .

RUN pip install flask kafka-python pymongo flask-socketio eventlet

CMD ["python", "app.py"]
