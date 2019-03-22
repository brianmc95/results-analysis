#!/usr/bin/python3

import argparse
import subprocess
import os


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


def main():

    parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

    raw_data_folder = os.path.join(parent_dir, "data/raw_data/output.csv")

    parser = argparse.ArgumentParser(description='Retrieve results from simulation and store to raw_data')
    parser.add_argument("-r", "--results-folder", help="Results folder")
    parser.add_argument("-o", "--output-folder", help="Folder to place raw data in", default=raw_data_folder)
    parser.add_argument("-s", "--scalars", help="Retrieve scalar results", default=True)
    parser.add_argument("-v", "--vectors", help="Retrieve vector results", default=True)

    args = parser.parse_args()

    extract_results(args)


if __name__ == "__main__":
    main()
