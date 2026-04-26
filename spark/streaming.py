from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import StructType, IntegerType , DoubleType
from pymongo import MongoClient

spark = SparkSession.builder \
    .appName("KafkaToMongo") \
    .getOrCreate()

# Lire Kafka
df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "kafka:9092") \
    .option("subscribe", "game_topic") \
    .option("startingOffsets", "earliest") \
    .load()

# Convertir
df_string = df.selectExpr("CAST(value AS STRING)")

# Schema
schema = StructType() \
    .add("player_id", IntegerType()) \
    .add("score", IntegerType()) \
    .add("time", IntegerType()) \
    .add("clicks", IntegerType()) \
    .add("moves", IntegerType()) \
    .add("errors", IntegerType()) \
    .add("response_time", DoubleType()) \
    .add("level", IntegerType()) \
    .add("success", IntegerType()) \
    .add("repetition", IntegerType())

# Parser JSON
df_parsed = df_string.select(
    from_json(col("value"), schema).alias("data")
).select("data.*")

# Fonction pour écrire dans MongoDB
def write_to_mongo(batch_df, batch_id):
    print(f"🚀 Writing batch {batch_id} to MongoDB")

    batch_df.show(truncate=False)

    data = batch_df.collect()

    if len(data) == 0:
        return

    client = MongoClient("mongodb://mongodb:27017/")
    db = client["game_db"]
    collection = db["players"]

    docs = [row.asDict() for row in data]

    collection.insert_many(docs)

    print(f"✅ Inserted {len(docs)} records")

# Stream vers MongoDB
query = df_parsed.writeStream \
    .outputMode("append") \
    .foreachBatch(write_to_mongo) \
    .start()

query.awaitTermination()
