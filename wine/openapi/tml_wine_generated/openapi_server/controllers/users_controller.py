from openapi_server.controllers.security_controller import fake_users_db


def read_users_me_users_me_get(user, token_info):  # noqa: E501
    """
    Read Users Me

    :rtype: Union[User, Tuple[User, int], Tuple[User, int, Dict[str, str]]
    """
    user_dict = fake_users_db[user]
    response = dict(
        full_name=user_dict["full_name"],
        disabled=user_dict["disabled"],
        email=user_dict["email"],
        username=user_dict["username"]
    )
    print(f"token_info: {token_info}")
    return response
