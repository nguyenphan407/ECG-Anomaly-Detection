from onnxconverter_common import FloatTensorType
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType
from pyspark.sql.functions import col, from_json
from pyspark.ml.feature import VectorAssembler, StringIndexer, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline, PipelineModel
import onnxmltools

class Streaming:
    def __init__(self):
        self.spark = (
            SparkSession.builder
            .appName("Streaming retrain model")
            .master("local[*]")
            .config("spark.jars.packages",
                    "org.apache.kafka:kafka-clients:3.3.0,"
                    "org.apache.spark:spark-sql-kafka-0-10_2.12:3.3.0")
            .getOrCreate()
        )

        # Feature & label names
        self.feature_cols = [str(i) for i in range(140)]
        self.label_col = "prediction"
        self.all_cols = self.feature_cols + [self.label_col]

        # JSON schema: 140 doubles + prediction
        self.schema = StructType(
            [StructField(c, DoubleType()) for c in self.feature_cols] +
            [StructField(self.label_col, DoubleType())]
        )

        # I/O paths
        self.history_path = "../data/ecg.csv"    # nơi lưu history (Parquet hoặc CSV)
        self.model_path   = "../model/"   # nơi lưu Spark pipeline

        # Build ML pipeline
        indexer = StringIndexer(inputCol=self.label_col, outputCol="label_index")
        assembler = VectorAssembler(inputCols=self.feature_cols, outputCol="features")
        scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
        rf = RandomForestClassifier(labelCol="label_index", featuresCol="scaled_features", predictionCol="rf_prediction",
                                    numTrees=100, maxDepth=5, seed=42)
        self.pipeline = Pipeline(stages=[indexer, assembler, scaler, rf])

    def cast_to_double(self, df):
        """Cast tất cả các cột feature + label về DoubleType."""
        return df.select(
            *[col(c).cast("double").alias(c) for c in self.all_cols]
        )

    def read(self):
        raw = (self.spark.readStream
               .format("kafka")
               .option("kafka.bootstrap.servers", "localhost:9092")
               .option("subscribe", "output-topic")
               .option("startingOffsets", "latest")
               .load()
               .select(from_json(col("value").cast("string"), self.schema).alias("data"))
               .select("data.*")
        )

        # Cast ngay khi đọc
        return self.cast_to_double(raw)

    def retrain_on_batch(self, batch_df, batch_id):
        # 1) Skip nếu rỗng
        if batch_df.rdd.isEmpty():
            print(f"[batch {batch_id}] empty, skip")
            return

        # 2) Đọc history, cast và union
        try:
            history_df = self.spark.read.csv(self.history_path, header=False, inferSchema=True)
            history_df = self.cast_to_double(history_df)
            train_df = history_df.union(batch_df)
            print(f"[batch {batch_id}] history count = {history_df.count()}")
        except Exception:
            print(f"[batch {batch_id}] no history found, train on batch only")
            train_df = batch_df

        # 3) Fit pipeline
        model = self.pipeline.fit(train_df)

        # 4) Save Spark pipeline
        model.write().overwrite().save(self.model_path)

        # # 5) Append batch into history
        # batch_df.write.mode("append").parquet(self.history_path)
        # print(f"[batch {batch_id}] appended {batch_df.count()} rows to history")

        # --- Export last RF stage to ONNX ---
        rf_model = model.stages[-1]
        initial_types = [
            ("scaled_features", FloatTensorType([None, 140]))
        ]
        onnx_model = onnxmltools.convert_sparkml(
            rf_model,
            initial_types=initial_types,
            spark_session=self.spark
        )
        with open(f"{self.model_path}/rf_model_{batch_id}.onnx", "wb") as f:
            f.write(onnx_model.SerializeToString())
        print(f"[batch {batch_id}] exported ONNX to {self.model_path}/rf_model_{batch_id}.onnx")

    def write(self, df):
        (df.writeStream
           .foreachBatch(self.retrain_on_batch)
           .outputMode("append")
           .trigger(processingTime="1 minute")
           .start()
           .awaitTermination())

if __name__ == "__main__":
    job = Streaming()
    df  = job.read()
    job.write(df)
