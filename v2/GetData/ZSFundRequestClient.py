"""
Created on Thu Oct 15 14:54:23 2020

@author: Zhoux26
"""

import json, base64, requests
from requests_negotiate_sspi import HttpNegotiateAuth
from datetime import datetime


class RequestZSFundDateServer(object):

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = object.__new__(cls)
        return cls._instance

    def __init__(self):
        # 固定获取token所需提供给服务器的信息
        self.client_id = 'client'
        self.client_secret = '642EE07F-A498-47E0-84B7-CAE77EEF3689'
        self.token_url = 'https://ids.zsfund.com/connect/token'

        self.data = {'grant_type': 'password',
                     'username': 'infoadmin',
                     'password': "ser2017sql,.",
                     'scope': 'webapi'}

        if not hasattr(self, 'inited'):
            self.inited = False

        if not self.inited:
            self.inited = True
            self.token = self.get_token()[0]
            self.start_time = self.get_token()[1]

    def get_token(self):

        headers = {'Authorization': 'Basic {}'.format(base64.b64encode(
            bytes('{}:{}'.format(self.client_id, self.client_secret).encode('utf-8'))).decode('utf-8'))}
        try:
            # 访问服务器发出获取token请求并记录访问时服务器本地时间
            response = requests.post(self.token_url, headers=headers, data=self.data, verify=True,
                                     allow_redirects=False)
            access_token = json.loads(response.text)["access_token"]
            # 若获取到token 记录服务器本地时间
            if response.text is not None:
                retrieve_time = datetime.strptime(response.headers["Date"], "%a, %d %b %Y %H:%M:%S GMT")

            self.token = access_token
            self.start_time = retrieve_time

            return access_token, retrieve_time

        except:
            return access_token, 'Retrieve access_token failed!'

    # token: eyJhbGciOiJSUzI1NiIsImtpZCI6IjkyNjNCQTAyNEIzMDQ3OEI4Q0ZEQjc5OT...

    def if_token_refresh(self, startTime, Token):

        import datetime
        # 判断当前时刻与上一次获取token的时间是否逾期， token时效为3600s
        if (datetime.datetime.now() - startTime).seconds <= 28800 + 3500:  # if token expires in 3500 seconds
            token = Token
        # 若逾期， 获取新的token
        else:
            token = self.get_token()[0]

        return token

    def request_client(self, url, params, method):

        Token = self.if_token_refresh(self.start_time, self.token)
        # 判断token有效性之后进行get/post请求
        if method == 'get':

            if Token is not None:
                response = requests.get(url, params=params,
                                        headers={'Content-Type': 'application/json',
                                                 "Authorization": "Bearer " + Token})

            # 若token没取到，尝试以Windows身份认证进行访问服务器
            else:
                try:
                    response = requests.get(url, params=params, headers={'Content-Type': 'application/json'},
                                            auth=HttpNegotiateAuth())
                except Exception as e:
                    print(str(e))

        elif method == 'post':

            if Token is not None:
                response = requests.post(url, data=json.dumps(params),
                                         headers={'Content-Type': 'application/json',
                                                  "Authorization": "Bearer " + Token})

            else:
                try:
                    response = requests.post(url, data=json.dumps(params), headers={'Content-Type': 'application/json'},
                                             auth=HttpNegotiateAuth())

                except Exception as e:
                    print(str(e))

        return json.loads(response.text)


# -----------------------------------------------------------------------------------------------
# Initial Settings
# -----------------------------------------------------------------------------------------------
# obj = RequestZSFundDateServer()
# token = obj.get_token()[0]
# start_time = obj.get_token()[1]
# -----------------------------------------------------------------------------------------------

