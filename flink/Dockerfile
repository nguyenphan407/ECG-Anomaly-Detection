# Dockerfile
FROM apache/flink:1.19.0

USER root

# Install Python3-pip and then PyFlink *globally*
RUN apt-get update && \
    apt-get install -y python3-pip && \
    pip3 install --no-cache-dir apache-flink onnxruntime

RUN curl -O https://repo1.maven.org/maven2/org/apache/flink/flink-connector-kafka/3.3.0-1.19/flink-connector-kafka-3.3.0-1.19.jar
RUN curl -O https://repo1.maven.org/maven2/org/apache/kafka/kafka-clients/3.3.0/kafka-clients-3.3.0.jar
RUN mv flink-connector-kafka-3.3.0-1.19.jar /opt/flink/lib/
RUN mv kafka-clients-3.3.0.jar /opt/flink/lib/

# (Optional) create a /usr/bin/python symlink if needed
RUN ln -sf /usr/bin/python3 /usr/bin/python

USER flink
