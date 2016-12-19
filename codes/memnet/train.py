from state import *
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="exp2")
    return parser.parse_args()

def main():
    args = parse_args()
    exp_fn = eval(args.exp)
    exp_fn()

if __name__ == "__main__":
    main()
