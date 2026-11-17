
import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, xxhash64, pmod, lit

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.regression import DecisionTreeRegressor


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def train_model():

    spark = SparkSession.builder \
        .appName("EV Range Prediction") \
        .master("local[*]") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    # Load Dataset

    dataset_path = os.path.join(
        BASE_DIR,
        "Electric_Vehicle_Population_Data.csv"
    )

    df = spark.read.csv(
        dataset_path,
        header=True,
        inferSchema=True
    )

    # Data Cleaning

    df_clean = df.filter(col("Electric Range") > 0)

    df_model = df_clean.select(
        "Model Year",
        "Make",
        "Model",
        "Electric Vehicle Type",
        "Electric Range"
    )

    # Group Based Train Test Split

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

    # Preprocessing

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
        inputCols=[
            "Make_Index",
            "Model_Index",
            "Vehicle_Type_Index"
        ],
        outputCols=[
            "Make_Vector",
            "Model_Vector",
            "Vehicle_Type_Vector"
        ],
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

    # Feature Assembly

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

    # Decision Tree Model

    dt = DecisionTreeRegressor(
        labelCol="Electric Range",
        featuresCol="features",
        maxDepth=5
    )

    dt_model = dt.fit(train_final)

    # Dropdown Values

    make_model_rows = train_data.select(
        "Make", "Model"
    ).distinct().collect()

    make_models = {}

    for row in make_model_rows:
        make_models.setdefault(
            row["Make"], []
        ).append(row["Model"])

    makes = sorted(make_models.keys())

    vehicle_types = [
        row["Electric Vehicle Type"]
        for row in train_data.select(
            "Electric Vehicle Type"
        ).distinct().collect()
    ]

    vehicle_types = sorted(vehicle_types)

    min_year = int(
        train_data.agg({"Model Year": "min"}).first()[0]
    )

    max_year = int(
        train_data.agg({"Model Year": "max"}).first()[0]
    )

    return (
        spark,
        pipeline_model,
        assembler,
        dt_model,
        make_models,
        makes,
        vehicle_types,
        min_year,
        max_year
    )


# Prediction Function

def predict_range(
    spark,
    pipeline_model,
    assembler,
    dt_model,
    model_year,
    make,
    model,
    vehicle_type
):

    input_data = spark.createDataFrame(
        [(
            int(model_year),
            make,
            model,
            vehicle_type
        )],
        [
            "Model Year",
            "Make",
            "Model",
            "Electric Vehicle Type"
        ]
    )

    input_encoded = pipeline_model.transform(input_data)

    input_final = assembler.transform(input_encoded)

    prediction = dt_model.transform(input_final)

    predicted_range = prediction.select(
        "prediction"
    ).first()[0]

    return float(predicted_range)
