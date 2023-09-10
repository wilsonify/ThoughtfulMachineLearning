In simple web and mobile applications, the authentication and authorization process typically works as follows:

    User Login: A user enters their username and password in a mobile application or a website.

    Credential Validation: The application sends the user's credentials (username and password) to the server.

    Validation on the Server: The server looks up the user in a database and validates the submitted password. This validation ensures that the provided credentials match those stored in the server's database.

    Successful Validation: If the validation succeeds, the server generates a security token or a cookie, which is then sent back to the client (the mobile app or web browser). This token or cookie may contain additional information about the user, known as "claims." Claims can include the user's unique identifier, role, email address, or other relevant information.

    Authentication Failure: If the validation fails (i.e., the provided credentials are incorrect), the user is notified, and they are prompted to reenter their credentials.

    Subsequent Requests: For subsequent requests to the server, the client includes the security token or cookie received earlier in the request headers.

    Authorization: The server inspects the security token or cookie to verify the user's identity. It may also check for embedded claims within the token to determine the user's role or other relevant information. Alternatively, the server can look up the user's role in a database to determine whether the user is authorized to perform a specific action.

This authentication and authorization process ensures that users are securely identified and allowed access to specific resources or actions based on their roles and permissions. It helps protect sensitive data and functionalities within the application while providing a seamless user experience.