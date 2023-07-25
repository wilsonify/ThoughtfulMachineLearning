# coding: utf-8

from __future__ import absolute_import
import unittest

from flask import json
from six import BytesIO

from openapi_server.models.http_validation_error import HTTPValidationError  # noqa: E501
from openapi_server.models.predict_ws_input import PredictWsInput  # noqa: E501
from openapi_server.models.predict_ws_output import PredictWsOutput  # noqa: E501
from openapi_server.models.token import Token  # noqa: E501
from openapi_server.models.user import User  # noqa: E501
from openapi_server.test import BaseTestCase


class TestDefaultController(BaseTestCase):
    """DefaultController integration test stubs"""

    @unittest.skip("application/x-www-form-urlencoded not supported by Connexion")
    def test_login_for_access_token_token_post(self):
        """Test case for login_for_access_token_token_post

        Login For Access Token
        """
        headers = { 
            'Accept': 'application/json',
            'Content-Type': 'application/x-www-form-urlencoded',
        }
        data = dict(grant_type=None,
                    username=None,
                    password=None,
                    scope=None,
                    client_id=None,
                    client_secret=None)
        response = self.client.open(
            '/token',
            method='POST',
            headers=headers,
            data=data,
            content_type='application/x-www-form-urlencoded')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_predict_predict_post(self):
        """Test case for predict_predict_post

        Predict
        """
        predict_ws_input = openapi_server.PredictWsInput()
        headers = { 
            'Accept': 'application/json',
            'Content-Type': 'application/json',
        }
        response = self.client.open(
            '/predict',
            method='POST',
            headers=headers,
            data=json.dumps(predict_ws_input),
            content_type='application/json')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_read_items_items_get(self):
        """Test case for read_items_items_get

        Read Items
        """
        headers = { 
            'Accept': 'application/json',
            'Authorization': 'Bearer special-key',
        }
        response = self.client.open(
            '/items/',
            method='GET',
            headers=headers)
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_read_own_items_users_me_items_get(self):
        """Test case for read_own_items_users_me_items_get

        Read Own Items
        """
        headers = { 
            'Accept': 'application/json',
            'Authorization': 'Bearer special-key',
        }
        response = self.client.open(
            '/users/me/items/',
            method='GET',
            headers=headers)
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_read_users_me_users_me_get(self):
        """Test case for read_users_me_users_me_get

        Read Users Me
        """
        headers = { 
            'Accept': 'application/json',
            'Authorization': 'Bearer special-key',
        }
        response = self.client.open(
            '/users/me/',
            method='GET',
            headers=headers)
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))


if __name__ == '__main__':
    unittest.main()
