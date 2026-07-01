# -*- coding: utf-8 -*-

import os
import zipfile
from configparser import ConfigParser
from io import BytesIO

import polars as pl
from smb.SMBConnection import SMBConnection  # pysmb


class CIFSLoader(object):
    """基于 CIFS/SMB 协议的数据加载类。"""

    def __init__(self, username, password):
        self._server_ip = "slowdata"  # 服务器的网络名称或域名
        self._server_port = 445
        self._raw_share_name = "backup"  # 默认共享名称
        self._username = username
        self._password = password

    @classmethod
    def get_user_info(cls, user_name="default"):
        cp = ConfigParser()
        cfg_path = os.path.join(os.path.dirname(__file__), "userInfo.cfg")
        cp.read(cfg_path, "utf-8")
        return {
            "username": cp.get(user_name, "user"),
            "password": cp.get(user_name, "pwd"),
        }

    def get_conn(self):
        return SMBConnection(
            username=self._username,
            password=self._password,
            my_name="",
            remote_name=self._server_ip,
            is_direct_tcp=True,
            # use_ntlm_v2=True,
        )

    def close(self):
        pass

    def get_filenames(self, path="", share_name=None):
        if share_name is None:
            share_name = self._raw_share_name

        conn = None
        try:
            conn = self.get_conn()
            if conn.connect(self._server_ip, self._server_port):
                files = conn.listPath(share_name, path)
                return sorted([f.filename for f in files])
        except Exception as e:
            print(f"发生错误: {e}")
        finally:
            if conn is not None:
                conn.close()

    def get_data_csv(self, share_name, loc_path):
        conn = None
        try:
            conn = self.get_conn()
            if conn.connect(self._server_ip, self._server_port):
                stream_data = BytesIO()
                conn.retrieveFile(share_name, loc_path, stream_data, show_progress=False)
                stream_data.seek(0)

                with zipfile.ZipFile(stream_data, "r") as z:
                    members = [
                        info
                        for info in z.infolist()
                        if not info.is_dir() and info.file_size > 0
                    ]
                    if not members:
                        raise ValueError(f"zip 内没有非空文件: {loc_path}")
                    with z.open(members[0]) as f:
                        return pl.read_csv(f, truncate_ragged_lines=True)
        except Exception as e:
            print(f"发生错误: {e}")
        finally:
            if conn is not None:
                conn.close()

    def get_data_pq(self, share_name, loc_path):
        conn = None
        try:
            conn = self.get_conn()
            if conn.connect(self._server_ip, self._server_port):
                stream_data = BytesIO()
                conn.retrieveFile(share_name, loc_path, stream_data, show_progress=False)
                stream_data.seek(0)
                return pl.read_parquet(stream_data)
        except Exception as e:
            print(f"发生错误: {e}")
        finally:
            if conn is not None:
                conn.close()

    def save_df_to_cifs(self, df, output_path, share_name="量化中间数据"):
        cache_name = f"{share_name}_tmp.parquet.gzip"
        cache_path = os.path.join(os.path.dirname(__file__), cache_name)
        conn = None
        try:
            conn = self.get_conn()
            if conn.connect(self._server_ip, self._server_port):
                df.to_parquet(cache_path, compression="gzip")
                with open(cache_path, "rb") as cache_file:
                    conn.storeFile(
                        share_name,
                        output_path,
                        file_obj=cache_file,
                        timeout=10000,
                        show_progress=True,
                    )
        except Exception as e:
            print(f"发生错误: {e}")
        finally:
            if os.path.exists(cache_path):
                os.remove(cache_path)
            if conn is not None:
                conn.close()


if __name__ == "__main__":
    cifs = CIFSLoader("xujiayi", "ZSfund.com@202601")
    files_name = cifs.get_filenames("level2/逐笔成交数据", "量化中间数据")
    for file in files_name[1460:1946]:  # 490:959 1460:1946 958:1460
        if file in [".", ".."]:
            continue
        df = cifs.get_data_pq("量化中间数据", os.path.join("level2/逐笔成交数据", file))
        data_dir = os.path.expanduser("/data/xujiayi/cj/")
        df.write_parquet(os.path.join(data_dir, file), compression="gzip")
        print("add one")