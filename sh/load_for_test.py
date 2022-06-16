import sys, yaml, os, argparse

os.chdir(sys.path[0])
sys.path.append('../')
os.getcwd()

from utils.save_load_attack import *

def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--attack_result_file_path', type=str)
    return parser

parser = (add_args(argparse.ArgumentParser(description=sys.argv[0])))
args = parser.parse_args()

a = load_attack_result(f'{args.attack_result_file_path}') # just load, record the log

