from openapi_server.controllers.security_controller import fake_users_db


def read_items_items_get(user, token_info):  # noqa: E501
    """Read Items

     # noqa: E501


    :rtype: Union[object, Tuple[object, int], Tuple[object, int, Dict[str, str]]
    """

    return token_info


def read_own_items_users_me_items_get(user, token_info):  # noqa: E501
    """Read Own Items

     # noqa: E501


    :rtype: Union[object, Tuple[object, int], Tuple[object, int, Dict[str, str]]
    """

    return token_info


def read_users_me_users_me_get(user, token_info):  # noqa: E501
    """Read Users Me

     # noqa: E501


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
