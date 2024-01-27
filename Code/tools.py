from sshtunnel import SSHTunnelForwarder
import configparser


def parse_config_db(path_to_config):
    config = configparser.ConfigParser()
    config.read(path_to_config)

    config_dict = {}

    config_dict['SSH_SERV'] = config['DATABASE']['ssh_server']
    config_dict['SSH_USER'] = config['DATABASE']['ssh_user']
    config_dict['SSH_PASS'] = config['DATABASE']['ssh_pass']

    config_dict['MYSQL_HOST'] = config['DATABASE']['mysql_host']
    config_dict['MYSQL_USER'] = config['DATABASE']['mysql_user']
    config_dict['MYSQL_DB_AWAREDX'] = config['DATABASE']['mysql_db_awaredx']
    config_dict['MYSQL_DB_OPENFDA'] = config['DATABASE']['mysql_db_openfda']
    config_dict['MYSQL_PASS'] = config['DATABASE']['mysql_pass']

    config_dict['USE_SSH_TUNNEL'] = True

    return config_dict
    

def stablish_server(config):

    SERVER = SSHTunnelForwarder((config.SSH_SERV, 22),
                            ssh_password=config.SSH_PASS,
                            ssh_username=config.SSH_USER,
                            remote_bind_address=('127.0.0.1', 3306))
    SERVER.start()
    return SERVER