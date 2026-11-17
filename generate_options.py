
import os
import json

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, xxhash64, pmod, lit

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "Electric_Vehicle_Population_Data.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "options.json")

spark = SparkSession.builder \
    .appName("EV Dropdown Options") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

df = spark.read.csv(DATASET_PATH, header=True, inferSchema=True)

df = df.filter(col("Electric Range") > 0)

df = df.select(
    "Model Year",
    "Make",
    "Model",
    "Electric Vehicle Type",
    "Electric Range"
)

df = df.withColumn(
    "_bucket",
    pmod(xxhash64("Make", "Model"), lit(5))
)

train_data = df.filter(col("_bucket") != 0).drop("_bucket")

rows = train_data.select("Make", "Model").distinct().collect()

make_models = {}

for row in rows:
    make_models.setdefault(row["Make"], []).append(row["Model"])

for make in make_models:
    make_models[make] = sorted(make_models[make])

makes = sorted(make_models.keys())

vehicle_types = sorted([
    row["Electric Vehicle Type"]
    for row in train_data.select("Electric Vehicle Type").distinct().collect()
])

min_year = int(train_data.agg({"Model Year": "min"}).first()[0])
max_year = int(train_data.agg({"Model Year": "max"}).first()[0])

options = {
    "makes": makes,
    "make_models": make_models,
    "vehicle_types": vehicle_types,
    "min_year": min_year,
    "max_year": max_year
}

with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
    json.dump(options, file, indent=4)

print("Dropdown options saved successfully!")
print("Manufacturers:", len(makes))
print("Vehicle types:", len(vehicle_types))

spark.stop()
