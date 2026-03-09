import os
import json

import psycopg2
from psycopg2 import sql
from psycopg2.extras import RealDictCursor


class PokiDatabase:
    def __init__(self):
        self.host = os.getenv("DB_HOST", "localhost")
        self.port = int(os.getenv("DB_PORT", "5432"))
        self.dbname = os.getenv("DB_NAME", "mydatabase")
        self.user = os.getenv("DB_USER", "barry")
        self.password = os.getenv("DB_PASSWORD", "password")

    def get_connection(self):
        return psycopg2.connect(
            host=self.host,
            port=self.port,
            dbname=self.dbname,
            user=self.user,
            password=self.password,
        )

    def get_poki_nth_row(self, n, order_by="id"):
        if n < 1:
            raise ValueError("n must be >= 1")

        offset = n - 1
        query = sql.SQL(
            "select * from public.poki order by {} offset %s limit 1"
        ).format(sql.Identifier(order_by))

        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, (offset,))
                return cur.fetchone()

    def get_poki_nth_data_dict(self, n, order_by="id"):
        row = self.get_poki_nth_row(n, order_by=order_by)
        if row is None:
            return None

        data = row.get("data")
        if isinstance(data, dict):
            return data

        if isinstance(data, (bytes, bytearray)):
            data = data.decode("utf-8")

        if isinstance(data, str):
            parsed = json.loads(data)
            if not isinstance(parsed, dict):
                raise TypeError("data column JSON is not a dict/object")
            return parsed

        raise TypeError(f"Unsupported data column type: {type(data).__name__}")

    def get_poki_nth_board_and_data(self, n, order_by="id"):
        row = self.get_poki_nth_row(n, order_by=order_by)
        if row is None:
            return None, None, None, None, None, None

        board = row.get("board")
        data = row.get("data")
        stack = row.get("stack")
        pos = row.get("pos")
        pot = row.get("pot")
        line = row.get("line")

        if isinstance(data, dict):
            pass
        elif isinstance(data, (bytes, bytearray)):
            data = data.decode("utf-8")
            data = json.loads(data)
        elif isinstance(data, str):
            data = json.loads(data)
        else:
            raise TypeError(f"Unsupported data column type: {type(data).__name__}")

        return board, data, stack, pos, pot, line
