import pymysql
pymysql.install_as_MySQLdb()
# Django 3+ expects mysqlclient >= 2.2.1; PyMySQL reports lower, so patch it
import sys
if "MySQLdb" in sys.modules:
    sys.modules["MySQLdb"].version_info = (2, 2, 1, "final", 0)