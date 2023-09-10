
Kinesis Streams is a service for real-time processing of streaming big data. It’s typically
used for quick log and data intake, metrics, analytics, and reporting. It’s different
from SQS in that Amazon recommends that Kinesis Streams be used primarily for
streaming big data, whereas SQS is used as a reliable hosted queue, especially if more
fine-grained control over messages, such as visibility timeouts or individual delays, is
required.
In Kinesis Streams, shards specify the throughput capacity of a stream. The num-
ber of shards needs to be stipulated when the stream is created, but resharding is pos-
sible if throughput needs to be increased or reduced. In comparison, SQS makes
scaling much more transparent. Lambda can integrate with Kinesis to read batches of
records from a stream as soon as they’re detected.

Amazon Kinesis Streams is a real-time data streaming service provided by Amazon Web Services (AWS). It's designed for the processing of large volumes of streaming data in real-time. Here are the key features and concepts associated with Kinesis Streams:

Real-Time Data Processing:

    Kinesis Streams is used for the ingestion and processing of streaming data, which includes log and data intake, metrics collection, analytics, and reporting. It enables applications to process data in real time as it becomes available.

Differentiation from SQS:

    Kinesis Streams is distinct from Amazon Simple Queue Service (SQS) in its recommended use cases. While both services can be used for messaging and queuing, Kinesis Streams is specifically designed for handling large-scale, real-time streaming data, whereas SQS is more suited for message queuing scenarios with features like visibility timeouts and individual message delays.

Shards for Throughput Capacity:

    In Kinesis Streams, the throughput capacity of a stream is determined by the number of shards allocated to it. When you create a stream, you specify the number of shards, and this number affects the maximum rate at which data can be ingested and processed. You can adjust the number of shards through a process called "resharding" if your throughput needs change.

Lambda Integration:

    AWS Lambda can seamlessly integrate with Kinesis Streams. Lambda functions can be configured to read batches of records from a Kinesis stream as soon as new data becomes available. This integration enables serverless processing of streaming data.

Transparent Scaling:

    Unlike Kinesis Streams, which requires you to manage the number of shards, SQS offers more transparent scaling. SQS automatically scales to accommodate the volume of messages in the queue, making it a simpler choice for certain queuing scenarios.

In summary, Amazon Kinesis Streams is a powerful service for real-time processing of streaming data, often used for applications that require rapid ingestion, analysis, and response to data as it arrives. It is particularly well-suited for use cases involving big data analytics, real-time monitoring, and event-driven architectures.