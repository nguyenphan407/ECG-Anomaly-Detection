import json
import time
import pandas as pd

from kafka import KafkaProducer, KafkaAdminClient
from kafka.admin import NewTopic

def create_topic(admin, topic_name):
    # Create topic if not exists
    try:
        # Create Kafka topic
        topic = NewTopic(name=topic_name, num_partitions=1,
                         replication_factor=1)
        admin.create_topics([topic])
        print(f"A new topic {topic_name} has been created!")
    except:
        print(f"Topic {topic_name} already exists. Skipping creation!")
        pass


def create_streams(topic_name: str, servers):
    producer = None
    admin = None
    for _ in range(10):
        try:
            producer = KafkaProducer(bootstrap_servers=servers)
            admin = KafkaAdminClient(bootstrap_servers=servers)
            print("SUCCESS: instantiated Kafka admin and producer")
            break
        except Exception as e:
            print(
                f"Trying to instantiate admin and producer with bootstrap servers {servers} with error {e}"
            )
            time.sleep(10)
            pass

    df = pd.read_csv(f"data/ecg.csv", header=None)
    df = df.iloc[:, :-1]
    records = df.to_dict(orient="records")

    for record in records:
        producer.send(
            topic_name,
            json.dumps(record).encode("utf-8")
        )
        print(record)
        time.sleep(5)

if __name__ == '__main__':
    create_streams(topic_name="raw", servers="localhost:9092", )