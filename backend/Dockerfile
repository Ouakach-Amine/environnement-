FROM python:3.10

WORKDIR /app

COPY . .

RUN pip install flask kafka-python pymongo flask-socketio eventlet flask-cors

CMD ["python", "app.py"]
