from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import StructType, IntegerType

spark = SparkSession.builder \
    .appName("KafkaStreaming") \
    .getOrCreate()

# 1. Lire depuis Kafka
df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "kafka:9092") \
    .option("subscribe", "game_topic") \
    .option("startingOffsets", "earliest") \
    .load()

# 2. Convertir value → string
df_string = df.selectExpr("CAST(value AS STRING)")

# 3. Définir schema JSON
schema = StructType() \
    .add("player_id", IntegerType()) \
    .add("score", IntegerType()) \
    .add("time", IntegerType())

# 4. Parser JSON
df_parsed = df_string.select(
    from_json(col("value"), schema).alias("data")
).select("data.*")

# 5. Affichage FINAL (IMPORTANT)
query = df_parsed.writeStream \
    .outputMode("append") \
    .format("console") \
    .start()

query.awaitTermination()
