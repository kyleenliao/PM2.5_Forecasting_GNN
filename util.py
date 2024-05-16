import yaml
import sys
import os
import numpy as np

proj_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(proj_dir)
conf_fp = os.path.join(proj_dir, 'config.yaml')
#conf_fp = os.path.join(proj_dir, 'config_ambient.yaml')
#conf_fp = os.path.join(proj_dir, 'config_exp1.yaml')

with open(conf_fp) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

file_dir = config['filepath']['GPU-Server']

def main():
    pass

if __name__ == '__main__':
    main()
