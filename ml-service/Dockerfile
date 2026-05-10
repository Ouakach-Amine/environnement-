FROM python:3.10-slim

WORKDIR /app

# Install system deps (slim needs these for scikit-learn)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --no-cache-dir \
    pandas \
    scikit-learn \
    numpy \
    kafka-python \
    pymongo

# ── CMD explanation ──────────────────────────────────────────────
# 1. Sleep 8 s so MongoDB & Kafka are up before we connect
# 2. Run model.py in background (infinite loop, writes ML results)
# 3. Run fuzzy_ahp.py in a separate infinite shell loop
#    "|| true" prevents the outer while-loop from exiting on error
# The "wait" at the end keeps the container alive even if a child exits
CMD ["bash", "-c", "\
  echo '⏳ Waiting 8s for services...' && sleep 8 && \
  echo '🚀 Starting model.py...' && \
  python -u model.py & \
  MODEL_PID=$! && \
  echo '🚀 Starting fuzzy_ahp loop...' && \
  while true; do \
    python -u fuzzy_ahp.py || echo '⚠ fuzzy_ahp.py exited with error, retrying in 15s'; \
    sleep 15; \
  done & \
  FUZZY_PID=$! && \
  echo \"PIDs: model=$MODEL_PID fuzzy=$FUZZY_PID\" && \
  wait $MODEL_PID \
"]
