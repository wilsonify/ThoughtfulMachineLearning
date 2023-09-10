Simple Email Service ( SES ) is a service designed to send and receive email. SES han-
dles email-receiving operations such as scanning for spam and viruses and rejection of
email from untrusted sources. Incoming email can be delivered to an S3 bucket, or
used to invoke a Lambda notification or create an SNS notification. These actions can
be configured as part of the receipt rule, which tells SES what to do with the email
once it arrives.
Sending emails with SES is straightforward but there are limits, which are in place
to regulate the rate and the number of messages being sent out. SES will automatically
increase the quota as long as high-quality email, and not spam, is being sent.


Amazon Simple Email Service (SES) is a comprehensive email management service provided by Amazon Web Services (AWS). It encompasses both sending and receiving email. Here's an in-depth look at its key features and functionalities:

Sending and Receiving Email:

    SES is designed for both sending and receiving email. It provides a unified solution for managing email communication.

Email Receiving Operations:

    SES handles various email-receiving operations, including:
        Scanning for Spam and Viruses: SES scans incoming email messages for spam and viruses to help maintain email security.
        Rejection of Email from Untrusted Sources: SES can reject email from untrusted or suspicious sources, contributing to your email security.

Incoming Email Handling:

    Incoming email can be processed in different ways, including:
        Delivery to S3 Bucket: SES can deliver incoming email messages to an Amazon S3 bucket, allowing you to store and manage email content.
        Invocation of Lambda Functions: SES can trigger AWS Lambda functions upon the receipt of specific email messages, enabling you to create custom actions based on incoming emails.
        Creation of SNS Notifications: SES can also generate Simple Notification Service (SNS) notifications to notify you or other AWS services about incoming emails.

Receipt Rules:

    SES provides a mechanism called "receipt rules" that lets you define how incoming emails should be handled once they arrive. These rules specify actions to be taken, such as delivering to S3, invoking Lambda functions, or creating SNS notifications.

Sending Email:

    SES simplifies sending email messages, making it straightforward for applications and services to send transactional or marketing emails.

Sending Limits and Quotas:

    SES imposes limits on the rate and number of email messages you can send. These limits are in place to prevent misuse and ensure responsible email sending. However, SES automatically increases these quotas as long as you maintain a record of sending high-quality email (i.e., not spam) and adhere to AWS's acceptable use policies.

In summary, Amazon SES is a versatile email service that handles both sending and receiving email. It offers features for email security, flexible handling of incoming emails, and smooth integration with other AWS services like S3, Lambda, and SNS. While there are sending limits, SES can increase quotas as you demonstrate responsible email sending practices.