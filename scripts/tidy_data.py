#!/usr/bin/python3

import argparse
import pandas as pd
import numpy as np
import os
import datetime
import json


def parse_if_number(s):
    try:
        return float(s)
    except:
        return True if s == "true" else False if s == "false" else s if s else None


def parse_ndarray(s):
    return np.fromstring(s, sep=' ') if s else None


def parse_vectime_vecvalue(df):

    new_df = pd.DataFrame(columns=["Node", "Name", "Time", "Value"])

    names = []
    nodes = []

    count = 1

    print("Parsing vector file")

    vectimes = np.array([])
    vecvalues = np.array([])

    df_len = len(df.index)

    for index, row in df.iterrows():
        names.append([row[2] for _ in range(len(row.vectime))])
        nodes.append([row.node for _ in range(len(row.vectime))])
        vectimes = np.concatenate((vectimes, row.vectime))
        vecvalues = np.concatenate((vecvalues, row.vecvalue))

        print("Processed row: {} of {}".format(count, df_len))
        count += 1

    new_df = new_df.append({"Node": nodes, "Name": names, "Time": vectimes, "Value": vecvalues}, ignore_index=True)

    return new_df


def tidy_data(args):
    raw_df = pd.read_csv(args.raw_results, converters={
        "attrvalue": parse_if_number,
        "binedges" : parse_ndarray,
        "binvalues": parse_ndarray,
        "vectime"  : parse_ndarray,
        "vecvalue" : parse_ndarray})

    print("Loaded csv into DataFrame")

    # It's likely this will change depending on the run/system
    # Might be worth investigating some form of alternative
    broken_module = raw_df['module'].str.split('.', 3, expand=True)

    raw_df["network"] = broken_module[0]
    raw_df["node"] = broken_module[1]
    raw_df["interface"] = broken_module[2]
    raw_df["layer"] = broken_module[3]

    raw_df = raw_df.drop("module", axis=1)

    # Remove junk from common node names
    raw_df.node = raw_df.node.str.replace("node", "")
    raw_df.node = raw_df.node.str.replace("[", "")
    raw_df.node = raw_df.node.str.replace("]", "")
    raw_df.node = raw_df.node.str.replace("car", "")

    # This will always remain the same for all runs.
    broken_run = raw_df['run'].str.split('-', 4, expand=True)

    raw_df["scenario"] = broken_run[0]
    raw_df["run"] = broken_run[1]
    raw_df["date"] = broken_run[2]
    raw_df["time"] = broken_run[3]
    raw_df["processId"] = broken_run[4]

    runattr_df = raw_df[raw_df["type"] == "runattr"]
    runattr_df = runattr_df.dropna(axis=1, how="all")

    itervar_df = raw_df[raw_df["type"] == "itervar"]
    itervar_df = itervar_df.dropna(axis=1, how="all")

    param_df = raw_df[raw_df["type"] == "param"]
    param_df = param_df.dropna(axis=1, how="all")

    attr_df = raw_df[raw_df["type"] == "attr"]
    attr_df = attr_df.dropna(axis=1, how="all")

    if args.stats:
        with open(args.stats) as json_file:
            data = json.load(json_file)
            raw_df = raw_df[(raw_df["name"].isin(data["filtered_vectors"])) | (raw_df["name"].isin(data["filtered_scalars"]))]

    scalar_df = raw_df[raw_df["type"] == "scalar"]
    scalar_df = scalar_df.dropna(axis=1, how="all")

    vector_df = raw_df[raw_df["type"] == "vector"]
    vector_df = vector_df.dropna(axis=1, how="all")

    vector_df = parse_vectime_vecvalue(vector_df)

    now = datetime.datetime.now()
    if args.name != now:
        directory = "{}/{}-{}".format(args.tidied_results, now.strftime("%Y-%m-%d_%H:%M"), args.name)
    else:
        directory = "{}/{}".format(args.tidied_results, now.strftime("%Y-%m-%d_%H:%M"))

    os.mkdir(directory)

    runattr_df.to_csv("{}/{}".format(directory, "runattr.csv"), index=False)
    itervar_df.to_csv("{}/{}".format(directory, "itervar.csv"), index=False)
    param_df.to_csv("{}/{}".format(directory, "params.csv"), index=False)
    attr_df.to_csv("{}/{}".format(directory, "attr.csv"), index=False)
    vector_df.to_csv("{}/{}".format(directory, "vector.csv"), index=False)
    scalar_df.to_csv("{}/{}".format(directory, "scalar.csv"), index=False)


def main():

    parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

    now = datetime.datetime.now()

    processed_data_folder = os.path.join(parent_dir, "data/processed_data/output.csv")

    parser = argparse.ArgumentParser(description='Retrieve results from simulation and store to raw_data')
    parser.add_argument("-r", "--raw-results", help="Raw results file")
    parser.add_argument("-o", "--tidied-results", help="File to save results to", default=processed_data_folder)
    parser.add_argument("-n", "--name", help="Name of simulation results file", default=now.strftime("%Y-%m-%d_%H:%M"))
    parser.add_argument("-s", "--stats", help="Json file describing stats we are interested in")
    args = parser.parse_args()

    tidy_data(args)


if __name__ == "__main__":
    main()