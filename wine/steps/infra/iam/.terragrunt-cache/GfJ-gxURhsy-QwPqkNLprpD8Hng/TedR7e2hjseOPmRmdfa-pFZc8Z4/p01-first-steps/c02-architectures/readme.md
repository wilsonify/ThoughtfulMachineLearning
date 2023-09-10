# Chapter Overview: Use Cases, Architectures, and Patterns

This chapter focused on understanding critical aspects of use cases, 
architectures, and patterns in the context of building systems. 
It introduced various architectures and patterns that will be explored further in the book.

## Architectures Covered

The architectures discussed in this chapter include:

- Compute as back end
- Compute as glue
- Legacy API wrapper
- Hybrid
- GraphQL
- Real-time processing

## Patterns Explored

Patterns highlighted in this chapter comprise:

- Command pattern
- Messaging pattern
- Priority queue pattern
- Fan-out pattern
- Pipes and filters pattern

Architectures:

    Compute as Back End: In this architecture, cloud services (e.g., AWS Lambda) act as the core engine of your application, handling the computing work without the need to manage physical servers.

    Compute as Glue: This architecture uses computing services to seamlessly connect various parts of your application or systems, much like glue binds different objects together.

    Legacy API Wrapper: This involves creating a modern layer around older systems with outdated interfaces (APIs) to make them compatible with newer software.

    Hybrid: The hybrid architecture combines both cloud-based and on-premises solutions, offering flexibility and scalability by having parts of your system in the cloud and part on local servers.

    GraphQL: GraphQL is a query language for APIs, allowing clients to request specific data, similar to customizing your order at a restaurant rather than choosing from a fixed menu.

    Real-time Processing: This architecture deals with processing data as it arrives in real-time, such as updating a live scoreboard during a sports game.

Patterns:

    Command Pattern: This pattern encapsulates a request as an object, akin to placing an order in a restaurant menu where you provide your request, and the kitchen processes it.

    Messaging Pattern: Similar to sending text messages, it enables different parts of a system to communicate by passing messages, just like texting friends to coordinate plans.

    Priority Queue Pattern: Imagine a to-do list where you prioritize tasks by importance. This pattern organizes tasks or messages to ensure important ones are handled first.

    Fan-Out Pattern: Think of it as sending one message that gets distributed to multiple recipients, much like posting a message on social media that many people can see and respond to.

    Pipes and Filters Pattern: This is similar to a production line in a factory, where data flows through a series of processing steps (filters) before reaching the final output (the end of the pipe).

These concepts are fundamental in modern software development, offering diverse approaches to structuring applications and managing data effectively.