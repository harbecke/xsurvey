import os
from configparser import ConfigParser


def main():
    config = ConfigParser()
    config.read('config.ini')
    
    for _, item in list(config.items('DATA FOLDER')):
        os.makedirs(item, exist_ok=True)

if __name__ == "__main__":
    main()
