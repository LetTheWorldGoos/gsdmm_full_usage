# -*- coding:utf-8 -*

import configparser
import os
import pymysql
import pandas as pd


class HandleDB(object):
    """data pool to MySQL"""

    def __init__(self):
        cf = self.load_parameters()
        self.host = cf.get("LocalDatabase", "host")
        self.uname = cf.get("LocalDatabase", "uname")
        self.pwd = cf.get("LocalDatabase", "pwd")
        self.port = int(cf.get("LocalDatabase", "port"))
        self.dbname = cf.get("LocalDatabase", "dbname")

    @staticmethod
    def load_parameters():
        dir = os.path.dirname(os.path.abspath(__file__))
        cfg_file = dir + '/config.ini'
        print(cfg_file)
        cfg = configparser.ConfigParser()
        cfg.read(cfg_file, encoding='utf-8')
        return cfg

    def handleMysql(self):
        localdb = pymysql.connect(self.host, self.uname, self.pwd, self.dbname, self.port)
        return localdb

    def handleMysql_szf(self):
        szfdb = pymysql.connect(self.host_szf, self.uname_szf, self.pwd_szf, self.dbname_szf, self.port_szf)
        return szfdb

    def get_df_use_sql(self, sql_qr):
        """pandas api only"""
        return pd.read_sql(sql_qr, self.db)


class Tokenizer(object):
    """
    tokenizer：jieba for chinese
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer