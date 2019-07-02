import os
from configparser import ConfigParser


def main():
    config = ConfigParser()
    config.read('config.ini')
    
    for item in list(Config.items('Section')):
        os.makedirs(item, exist_ok=True)

if __name__ == "__main__":
    main()
