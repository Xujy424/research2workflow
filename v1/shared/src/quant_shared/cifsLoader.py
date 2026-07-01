# -*- coding: UTF-8 -*-

import os
import zipfile
from configparser import ConfigParser
from io import BytesIO

import pandas as pd
from smb.SMBConnection import SMBConnection    # pysmb

import polars as pl

"""
基于CIFS协议的数据载入类
"""


class CIFSLoader(object):

    def __init__(self, username, password):
        self._server_ip = 'win-L2'  # '10.110.0.50'  # 服务器的网络名称或域名
        self._server_port = 445
        self._raw_share_name = 'L2Data'  # 共享名称

        self._username = username
        self._password = password

    @classmethod
    def get_user_info(cls, user_name='default'):
        cp = ConfigParser()
        cfg_path = os.path.join(os.path.dirname(__file__), r'userInfo.cfg')
        cp.read(cfg_path, 'utf-8')
        ret = {
            'username': cp.get(user_name, 'user'),
            'password': cp.get(user_name, 'pwd')
        }
        return ret

    def get_conn(self):
        # 建立SMB连接
        conn = SMBConnection(
            username=self._username,
            password=self._password,
            my_name='',
            remote_name=self._server_ip,
            is_direct_tcp=True,
            # use_ntlm_v2=True,
        )
        return conn

    def close(self):
        pass

    def get_filenames(self, path='', share_name=None):

        if share_name is None:
            share_name = self._raw_share_name

        conn = None

        try:
            conn = self.get_conn()

            if conn.connect(self._server_ip, self._server_port):

                # 列出共享中的文件
                files = conn.listPath(share_name, path)
                files_name = sorted([f.filename for f in files])
                # files_name = sorted([f.filename for f in files if len(f.filename) == 8])

                # 连接被关闭
                conn.close()
                return files_name

        except Exception as e:
            print(f"发生错误: {e}")

        finally:
            # 确保连接被关闭
            if conn is not None:
                conn.close()

    def get_data_csv(self, share_name, loc_path):
        conn = None

        try:
            conn = self.get_conn()

            if conn.connect(self._server_ip, self._server_port):
                #
                stream_data = BytesIO()
                conn.retrieveFile(share_name, loc_path, stream_data, show_progress=False)
                print(f"✓ 已下载: {os.path.basename(loc_path)}")
                stream_data.seek(0)

                # 连接被关闭
                conn.close()

                # 创建一个zip文件对象
                with zipfile.ZipFile(stream_data, 'r') as z:
                    # 假设zip文件中的唯一文件是一个csv文件
                    with z.open(z.namelist()[0]) as f:
                        # 直接使用 pandas 读取 csv 文件内容到 DataFrame
                        df = pd.read_csv(f, index_col=False)

                return df

        except Exception as e:
            print(f"发生错误: {e}")

        finally:
            # 确保连接被关闭
            if conn is not None:
                conn.close()

    def get_data_pq(self, share_name, loc_path):
        conn = None

        try:
            conn = self.get_conn()

            if conn.connect(self._server_ip, self._server_port):
                #
                stream_data = BytesIO()
                conn.retrieveFile(share_name, loc_path, stream_data, show_progress=False)
                print(f"✓ 已下载: {os.path.basename(loc_path)}")

                # 连接被关闭
                conn.close()

                df = pl.read_parquet(stream_data)
                return df

        except Exception as e:
            print(f"发生错误: {e}")

        finally:
            # 确保连接被关闭
            if conn is not None:
                conn.close()

    # -----------------------
    def save_df_to_cifs(self, df, output_path, share_name='量化中间数据'):
        """

        Args:
            df:
            output_path:
            share_name:

        Returns:

        """
        cache_name = f'{share_name}_tmp.parquet.gzip'
        cache_path = os.path.join(os.path.dirname(__file__), cache_name)

        conn = None
        try:
            conn = self.get_conn()

            if conn.connect(self._server_ip, self._server_port):
                # 保存本地缓存文件
                df.to_parquet(cache_path, compression="gzip")
                # 存到cifs共享目录
                with open(cache_path, 'rb') as cache_file:
                    conn.storeFile(share_name, output_path,
                                   file_obj=cache_file,
                                   timeout=10000,
                                   show_progress=True,
                                   )

        except Exception as e:
            print(f"发生错误: {e}")

        finally:
            # 删除本地缓存
            os.remove(cache_path)

            # 确保连接被关闭
            if conn is not None:
                conn.close()



if __name__ == '__main__':

    cifs = CIFSLoader('xujiayi','ZSfund.com@202601')
    files_name = cifs.get_filenames('level2/逐笔成交数据','量化中间数据')
    for file in files_name[1460:1946]:  # 490:959    1460:1946  958:1460
        if file in ['.','..']:
            continue
        df = cifs.get_data_pq('量化中间数据',os.path.join('level2/逐笔成交数据',file))
        #data_dir = os.path.expanduser("~/data/量化中间数据/l2/逐笔委托数据")
        data_dir = os.path.expanduser("/data/xujiayi/cj/")
        df.write_parquet(os.path.join(data_dir,file), compression="gzip")
        print('add one')

    # cifs = CIFSLoader('xujiayi', 'ZSfund.com@202601')
    # files_name = cifs.get_filenames('level2/逐笔成交数据', '量化中间数据')
    # file = files_name[1946]  # 958:1460
    # df = cifs.get_data_pq('量化中间数据', os.path.join('level2/逐笔成交数据', file))
    # # data_dir = os.path.expanduser("~/data/量化中间数据/l2/逐笔成交数据")
    # data_dir = os.path.expanduser("/data/xujiayi/cj/")
    # df.write_parquet(os.path.join(data_dir, file), compression="gzip")
    # print('add one')



