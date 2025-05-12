from pyspark.sql import SparkSession
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import onnxmltools
from onnxconverter_common import FloatTensorType, Int64TensorType


def train(dataset):
    spark = (
        SparkSession.builder
        .master("local[*]")
        .appName("RF model training")
        .getOrCreate()
    )

    df = spark.read.csv(dataset, header=False, inferSchema=True)

    # Data labeling
    num_features = 140
    feature_cols = [f"V{i}" for i in range(num_features)]
    all_cols = feature_cols + ["label"]
    df = df.toDF(*all_cols)

    # Split the data into training and test sets
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

    label_indexer = StringIndexer(
        inputCol="label",
        outputCol="label_index",
    )

    # Combine features into a single vector column
    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol='features',
    )

    # Scale features
    standardizer = StandardScaler(
        inputCol="features",
        outputCol="scaled_features"
    )

    rf_classifier = RandomForestClassifier(
        featuresCol="features",
        labelCol="label_index",
        numTrees=100,
        maxDepth=5,
        seed=42
    )

    pipeline = Pipeline(stages=[
        label_indexer,
        assembler,
        standardizer,
        rf_classifier
    ])

    model = pipeline.fit(train_df)

    # Save the model
    model.write().overwrite().save("../model/rf_model")

    # Test the trained model
    pipeline_model = PipelineModel.load("../model/rf_model")
    pred = pipeline_model.transform(test_df)

    # Evaluate the model
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label_index", predictionCol="prediction", metricName="accuracy")

    acc = evaluator.evaluate(pred)
    print(f"Test Accuracy = {acc:.4f}")

    # Confusion Matrix
    pred.groupBy("label_index", "prediction").count().show()

    rf_model = PipelineModel.load("../model/rf_model").stages[-1]
    # Convert to ONNX
    initial_types = [
        ("features", FloatTensorType([None, num_features]))
    ]
    onnx_model = onnxmltools.convert_sparkml(rf_model, initial_types=initial_types, spark_session=spark)

    # Save the ONNX model
    with open("../model/rf_model.onnx", "wb") as f:
        f.write(onnx_model.SerializeToString())

if __name__ == "__main__":
    train("../data/ecg.csv")
