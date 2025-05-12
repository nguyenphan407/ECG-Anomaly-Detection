# ECG Anomaly Detection Project

## Prerequisites

* **Docker & Docker Compose**
* **Java 11+** (JDK) for Kafka & Flink
* **Python 3.8+** on the host machine to run Spark scripts

## Project Directory Structure

```
project-root/
├── docker-compose.yml      # Kafka + Zookeeper + Flink (JobManager & TaskManager)
├── docker/                 # Dockerfile for custom Flink image
│   └── Dockerfile          # Installs PyFlink, onnxruntime
├── models/                 # Contains ONNX model (mounted into Flink)
├── data/                   # Source Parquet & CSV history
├── spark/                  # Spark batch & streaming scripts
│   ├── train.py            # Batch training and ONNX export
│   └── streaming.py        # Structured Streaming model retraining
└── flink/                  # PyFlink job
    ├── streaming.py        # Reads Kafka + ONNX inference
    ├── checkpoints/        # Checkpoint files
    └── usrlib/             # Python libs for Flink
        └── jobs/           # Flink job modules
```

## 1. Docker Compose Setup

**Start the cluster**

   ```bash
   docker compose up
   docker compose ps
   ```

## 2. Batch Training & ONNX Export

1. **Run the training script**

   ```bash
   spark-submit train.py
   ```

2. **Output**

   * Spark pipeline model saved in `models/`
   * ONNX file for the RandomForest stage at `models/rf_model.onnx`

## 3. Deploy PyFlink Detection

1. **Submit PyFlink job**

   ```bash
   flink run \
      --python /opt/flink/usrlib/jobs/streaming.py \
      --python-files /opt/flink/usrlib/jobs/ \
      --pyModule jobs.kafka_processor \
      --pythonExec python3
   ```

2. **Producer** sends ECG JSON to the `raw` topic

3. **PyFlink** reads, performs ONNX inference, and sends anomalies to the `ecg_anomaly` topic

## 4. Run Spark Structured Streaming Retraining

1. **Adjust schema** in `spark/streaming.py` to match JSON from `ecg_anomaly` (DoubleType)

2. **Run the script**

   ```bash
   spark-submit spark/streaming.py
   ```

3. **Verify**

   * Checkpoints at `data/checkpoints/`
   * New model in `models/` (Spark + ONNX versioned by batch)
