import pymysql
import pymysql.cursors

class Database:

    def __init__(self, config):
        
        # Enter your database information here
        if config == 'Config Name': 
            self.host = ''
            self.username = ''
            self.password = ''
            self.port = 3307
            self.dbname = ''
            self.conn = None
            
        
    def open_connection(self):
        try:
            if self.conn is None:
                self.conn = pymysql.connect(host=self.host,
                                            port=self.port,
                                            user=self.username,
                                            passwd=self.password,
                                            db=self.dbname,
                                            connect_timeout=5)
        except pymysql.MySQLError as e:
            print(e) 
            return False 
        
        print('Database connected')
        return True
    
    def run_query(self, query):
        if not self.conn: self.open_connection()
        try:
            with self.conn.cursor() as cur:
                records = []
                cur.execute(query)
                result = cur.fetchall()
                for row in result:
                    records.append(row)
                cur.close()
                return records
        except pymysql.MySQLError as e:
            print(e)
            
    def run(self, query):
        return self.run_query(query)
            
    def get_list(self, query): 
        l = self.run(query)
        return [x[0] for x in l]

    def close_connection(self):
        if self.conn:
            self.conn.close()
            self.conn = None
            print('Database connection closed.')