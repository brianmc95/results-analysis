#!/usr/bin/python3

import argparse
import subprocess
import os
import datetime


def extract_results(args):
    orig_loc = os.getcwd()
    os.chdir(args.results_folder)
    command = "scavetool x"

    if args.scalars:
        command += " *.sca"

    if args.vectors:
        command += " *.vec"

    command += " -o "
    command += args.output_folder

    print(command)

    subprocess.run(command, shell=True)

    os.chdir(orig_loc)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():

    parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

    raw_data_folder = os.path.join(parent_dir, "data/raw_data/{}-output.csv".format(datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")))

    parser = argparse.ArgumentParser(description='Retrieve results from simulation and store to raw_data')
    parser.add_argument("-r", "--results-folder", help="Results folder")
    parser.add_argument("-o", "--output-folder", help="Folder to place raw data in", default=raw_data_folder)
    parser.add_argument("-s", "--scalars", help="Retrieve scalar results", type=str2bool, default=True)
    parser.add_argument("-v", "--vectors", help="Retrieve vector results", type=str2bool, default=True)

    args = parser.parse_args()

    extract_results(args)


if __name__ == "__main__":
    main()
