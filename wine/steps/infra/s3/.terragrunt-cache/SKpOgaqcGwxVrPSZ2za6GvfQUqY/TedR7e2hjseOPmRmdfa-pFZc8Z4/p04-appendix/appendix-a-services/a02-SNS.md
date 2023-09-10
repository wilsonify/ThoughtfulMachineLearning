Amazon Simple Notification Service ( SNS ) is a scalable pub-sub service designed to
deliver messages. Producers or publishers create and send messages to a topic. Sub-
scribers or consumers subscribe to a topic and receive messages over one of the sup-
ported protocols. SNS stores messages across multiple servers and data centers for
redundancy and guarantees at-least-once delivery. At-least-once delivery stipulates that
a message will be delivered at least once to a subscriber but, on rare occasions due to
the distributed nature of SNS , it may be delivered multiple times.
In cases when a message can’t be delivered by SNS to HTTP endpoints, it can be
configured to retry deliveries at a later time. SNS can also retry failed deliveries to
Lambda in cases where throttling is applied. SNS supports message payloads of up to
256 KB .


Amazon Simple Notification Service (SNS) is a highly scalable publish-subscribe service designed for sending messages in a flexible and reliable manner. Here's a breakdown of its key features and functionalities:

Publish-Subscribe Model:

    SNS operates on a publish-subscribe model. Producers or publishers create and send messages to a specific topic. Subscribers or consumers subscribe to these topics to receive messages that interest them. This decoupled architecture allows for flexible communication between components in distributed systems.

Message Redundancy and Delivery Guarantees:

    SNS ensures the reliability of message delivery by storing messages across multiple servers and data centers. It guarantees "at-least-once" delivery, meaning that a message will be delivered to a subscriber at least once. However, due to the distributed nature of SNS, it's possible for a message to be delivered multiple times, though this is rare.

Retry Mechanisms:

    SNS provides robust retry mechanisms to handle various delivery scenarios:
        Retry to HTTP Endpoints: If SNS encounters an issue delivering a message to an HTTP endpoint (e.g., a web service), it can be configured to retry the delivery at a later time. This ensures that transient issues do not result in message loss.
        Retry to Lambda: In cases where AWS Lambda functions are subscribers and throttling occurs, SNS can retry failed deliveries to Lambda. This is particularly useful for handling varying workloads without losing messages.

Payload Size:

    SNS supports message payloads of up to 256 KB, allowing you to send substantial amounts of data within your messages. This flexibility is useful for transmitting various types of information, including JSON data, text, or binary content.

In summary, Amazon SNS is a versatile and scalable messaging service that facilitates communication between different components in distributed systems. It ensures message reliability through redundancy, supports retries for failed deliveries, and accommodates a wide range of message payload sizes, making it a valuable tool for building resilient and responsive applications.