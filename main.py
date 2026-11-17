from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum, when
from pyspark.sql.functions import avg, min, max, stddev, col, count,countDistinct
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.regression import GBTRegressor
from pyspark.sql.functions import xxhash64, pmod, lit
# import os
# os.environ["HADOOP_HOME"] = r"E:\Spark BigData\spark-4.3.0-bin-hadoop3"
# os.environ["hadoop.home.dir"] = r"E:\Spark BigData\spark-4.3.0-bin-hadoop3"

# Initialize Spark Session
spark = SparkSession.builder.getOrCreate()

# Load Dataset
df = spark.read.csv(
    "C:\\Users\\rajpr\\OneDrive\\Desktop\\EV range\\Electric_Vehicle_Population_Data.csv",
    header=True,
    inferSchema=True
)

# Display Dataset
df.show(5, truncate=False)

# Total number of records
print("Total Rows:", df.count())

# Total number of columns
print("Total Columns:", len(df.columns))

# Display column names
print(df.columns)

# Display schema
df.printSchema()

null_counts = df.select([
    sum(when(col(c).isNull(), 1).otherwise(0)).alias(c)
    for c in df.columns
])

null_counts.show(truncate=False)

df.select("Electric Range").describe().show()

print(
    "Zero Electric Range:",
    df.filter(col("Electric Range") == 0).count()
)

df.groupBy("Electric Vehicle Type").count().orderBy(col("count").desc()).show(truncate=False)

df.groupBy("Model Year").count().orderBy("Model Year").show(50)

print("Zero range EV are not actually pure EV!!")

# Remove records having zero electric range
df_clean = df.filter(col("Electric Range") > 0)

# Display remaining records
df_clean.show(5, truncate=False)

# Check total records
print("Original Records:", df.count())
print("Records After Removing Zero Range:", df_clean.count())

# Verify zero values
print(
    "Zero Range Records:",
    df_clean.filter(col("Electric Range") == 0).count()
)

df.filter(col("Electric Range") == 0) \
    .groupBy("Electric Vehicle Type") \
    .count() \
    .show(truncate=False)

df.groupBy("Electric Vehicle Type") \
    .agg(
        sum(when(col("Electric Range") == 0, 1).otherwise(0)).alias("Zero_Range"),
        sum(when(col("Electric Range") > 0, 1).otherwise(0)).alias("Non_Zero_Range")
    ) \
    .show(truncate=False)

df_clean.groupBy("Electric Vehicle Type") \
    .agg(
        avg("Electric Range").alias("Average Range"),
        min("Electric Range").alias("Minimum Range"),
        max("Electric Range").alias("Maximum Range"),
        stddev("Electric Range").alias("Standard Deviation")
    ) \
    .show(truncate=False)

df_clean.groupBy("Make") \
    .agg(
        avg("Electric Range").alias("Average Range"),
        count("*").alias("Total Vehicles")
    ) \
    .orderBy(col("Average Range").desc()) \
    .show(20, truncate=False)

df_clean.groupBy("Make", "Model") \
    .count() \
    .orderBy(col("count").desc()) \
    .show(20, truncate=False)

df_clean.select(
    countDistinct("Make").alias("Unique_Makes"),
    countDistinct("Model").alias("Unique_Models")
).show()

df_clean.select("Make", "Model") \
    .distinct() \
    .count()

selected_columns = [
    "Model Year",
    "Make",
    "Model",
    "Electric Vehicle Type",
    "Electric Range"
]

df_clean.select([
    sum(when(col(c).isNull(), 1).otherwise(0)).alias(c)
    for c in selected_columns
]).show()

feature_columns = [
    "Model Year",
    "Make",
    "Model",
    "Electric Vehicle Type"
]

target_column = "Electric Range"

df_model = df_clean.select(
    *feature_columns,
    target_column
)

df_model.show(10, truncate=False)

print("Total Rows:", df_model.count())
print("Total Columns:", len(df_model.columns))


# Group-based train-test split
df_grouped = df_model.withColumn(
    "_bucket",
    pmod(xxhash64("Make", "Model"), lit(5))
)

train_data = df_grouped.filter(
    col("_bucket") != 0
).drop("_bucket")

test_data = df_grouped.filter(
    col("_bucket") == 0
).drop("_bucket")

print("Training Rows:", train_data.count())
print("Testing Rows:", test_data.count())


make_indexer = StringIndexer(
    inputCol="Make",
    outputCol="Make_Index",
    handleInvalid="keep"
)

model_indexer = StringIndexer(
    inputCol="Model",
    outputCol="Model_Index",
    handleInvalid="keep"
)

type_indexer = StringIndexer(
    inputCol="Electric Vehicle Type",
    outputCol="Vehicle_Type_Index",
    handleInvalid="keep"
)

encoder = OneHotEncoder(
    inputCols=["Make_Index", "Model_Index", "Vehicle_Type_Index"],
    outputCols=["Make_Vector", "Model_Vector", "Vehicle_Type_Vector"],
    handleInvalid="keep"
)

pipeline = Pipeline(stages=[
    make_indexer,
    model_indexer,
    type_indexer,
    encoder
])

pipeline_model = pipeline.fit(train_data)

train_encoded = pipeline_model.transform(train_data)
test_encoded = pipeline_model.transform(test_data)

train_encoded.select(
    "Make",
    "Make_Index",
    "Make_Vector",
    "Model",
    "Model_Index",
    "Model_Vector",
    "Vehicle_Type_Index",
    "Vehicle_Type_Vector"
).show(5, truncate=False)

assembler = VectorAssembler(
    inputCols=[
        "Model Year",
        "Make_Vector",
        "Model_Vector",
        "Vehicle_Type_Vector"
    ],
    outputCol="features"
)

train_final = assembler.transform(train_encoded)
test_final = assembler.transform(test_encoded)

train_final.select(
    "Model Year",
    "Make",
    "Model",
    "Electric Vehicle Type",
    "features",
    "Electric Range"
).show(5, truncate=False)


# Regression Evaluators
evaluator_r2 = RegressionEvaluator(
    labelCol="Electric Range",
    predictionCol="prediction",
    metricName="r2"
)

evaluator_rmse = RegressionEvaluator(
    labelCol="Electric Range",
    predictionCol="prediction",
    metricName="rmse"
)

evaluator_mae = RegressionEvaluator(
    labelCol="Electric Range",
    predictionCol="prediction",
    metricName="mae"
)


# Linear Regression
lr = LinearRegression(
    featuresCol="features",
    labelCol="Electric Range",
    predictionCol="prediction"
)

lr_model = lr.fit(train_final)

lr_predictions = lr_model.transform(test_final)

lr_predictions.select(
    "Electric Range",
    "prediction"
).show(10, truncate=False)

lr_r2 = evaluator_r2.evaluate(lr_predictions)
lr_rmse = evaluator_rmse.evaluate(lr_predictions)
lr_mae = evaluator_mae.evaluate(lr_predictions)

print("Linear Regression Results")
print("R2 Score:", lr_r2)
print("RMSE:", lr_rmse)
print("MAE:", lr_mae)


# Decision Tree Regression

dt = DecisionTreeRegressor(
    featuresCol="features",
    labelCol="Electric Range",
    maxDepth=5
)

dt_model = dt.fit(train_final)
dt_predictions = dt_model.transform(test_final)

dt_predictions.select(
    "Electric Range",
    "prediction"
).show(10, truncate=False)

dt_r2 = evaluator_r2.evaluate(dt_predictions)
dt_rmse = evaluator_rmse.evaluate(dt_predictions)
dt_mae = evaluator_mae.evaluate(dt_predictions)

print("Decision Tree Regression Results")
print("R2 Score:", dt_r2)
print("RMSE:", dt_rmse)
print("MAE:", dt_mae)


# Random Forest Regression

rf = RandomForestRegressor(
    featuresCol="features",
    labelCol="Electric Range",
    numTrees=100,
    maxDepth=5,
    seed=42
)

rf_model = rf.fit(train_final)

rf_predictions = rf_model.transform(test_final)

rf_predictions.select(
    "Electric Range",
    "prediction"
).show(10, truncate=False)

rf_r2 = evaluator_r2.evaluate(rf_predictions)
rf_rmse = evaluator_rmse.evaluate(rf_predictions)
rf_mae = evaluator_mae.evaluate(rf_predictions)

print("Random Forest Regression Results")
print("R2 Score:", rf_r2)
print("RMSE:", rf_rmse)
print("MAE:", rf_mae)


# Gradient Boosted Tree Regression

gbt = GBTRegressor(
    featuresCol="features",
    labelCol="Electric Range",
    maxIter=50,
    maxDepth=5,
    seed=42
)

gbt_model = gbt.fit(train_final)
gbt_predictions = gbt_model.transform(test_final)

gbt_predictions.select(
    "Electric Range",
    "prediction"
).show(10, truncate=False)

gbt_r2 = evaluator_r2.evaluate(gbt_predictions)
gbt_rmse = evaluator_rmse.evaluate(gbt_predictions)
gbt_mae = evaluator_mae.evaluate(gbt_predictions)

print("Gradient Boosted Tree Regression Results")
print("R2 Score:", gbt_r2)
print("RMSE:", gbt_rmse)
print("MAE:", gbt_mae)


# Group-based split verification
train_groups = train_data.select("Make", "Model").distinct()
test_groups = test_data.select("Make", "Model").distinct()

common_groups = train_groups.join(
    test_groups,
    ["Make", "Model"],
    "inner"
)

print("Common Make-Model Groups:", common_groups.count())
# Saving the Complete Decision Tree Pipeline

# final_model = PipelineModel(
#     stages=pipeline_model.stages + [assembler, dt_model]
# )

# model_path = r"C:\Users\rajpr\OneDrive\Desktop\EV range\saved_model"

# final_model.write().overwrite().save(model_path)

# print("Decision Tree Model Saved Successfully!")