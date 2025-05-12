import os
import json
import logging
import sys

from pyflink.common import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.connectors.kafka import FlinkKafkaConsumer, FlinkKafkaProducer
from pyflink.common.serialization import SimpleStringSchema
from pyflink.datastream.functions import MapFunction, KeySelector, RuntimeContext
import onnxruntime as rt

import numpy as np

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Các thông số cấu hình
INPUT_TOPIC = "raw"
OUTPUT_TOPIC = "output-topic"
BOOTSTRAP_SERVERS = "broker:29092"  # Tên service trong docker-compose
GROUP_ID = "flink-processing-group"
ONNX_MODEL_PATH   = "/opt/flink/lib/rf_model.onnx"

def get_jar_path():
    """
    Trả về đường dẫn đến các JAR files
    """
    return 'file:///opt/flink/usrlib'


class ECGAnomalyDetector(MapFunction):

    def open(self, runtime_ctx: RuntimeContext):
        logger.info("Loading ONNX model from %s", ONNX_MODEL_PATH)
        self.session = rt.InferenceSession(ONNX_MODEL_PATH)
        self.input_name = self.session.get_inputs()[0].name

    def map(self, value: str) -> str:
        try:
            data = json.loads(value)
            features = [float(data.get(f"f{i}", 0.0)) for i in range(140)]
            arr = np.array(features, dtype=np.float32).reshape(1, -1)

            pred = self.session.run(None, {self.input_name: arr})
            # Ép về float để JSON ghi ra số thực
            pred_label = float(pred[0][0])
            data["prediction"] = pred_label

            # Nếu bạn muốn đánh dấu anomaly cũng dưới dạng số (0.0 / 1.0)
            if "label" in data:
                is_anom = 1.0 if pred_label != float(data["label"]) else 0.0
                data["anomaly"] = is_anom

            return json.dumps(data)

        except Exception as e:
            logger.error("Error in detection: %s, input=%s", e, value)
            return json.dumps({
                "error": str(e),
                "original": value
            })

def main():
    # Create Environment
    env = StreamExecutionEnvironment.get_execution_environment()

    # Configuration checkpointing
    env.enable_checkpointing(60000)  # 60 giây

    # Kafka consumer properties
    props = {
        'bootstrap.servers': BOOTSTRAP_SERVERS,
        'group.id': GROUP_ID,
        'auto.offset.reset': 'latest'
    }

    # Create Kafka source
    source = FlinkKafkaConsumer(
        topics=INPUT_TOPIC,
        deserialization_schema=SimpleStringSchema(),
        properties=props
    )

    # Kafka sink properties
    sink_props = {
        'bootstrap.servers': BOOTSTRAP_SERVERS,
        'transaction.timeout.ms': '1000'
    }

    # Create Kafka sink
    sink = FlinkKafkaProducer(
        topic=OUTPUT_TOPIC,
        serialization_schema=SimpleStringSchema(),
        producer_config=sink_props
    )

    # Create Pipeline model
    stream = (
        env.add_source(source)
        .map(ECGAnomalyDetector(), output_type=Types.STRING())
        .add_sink(sink)
    )

    # Submit job
    job_name = "Kafka Processing Job"
    logger.info(f"Executing job: {job_name}")
    env.execute(job_name)

if __name__ == "__main__":
    main()
