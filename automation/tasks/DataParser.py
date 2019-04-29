import os
import subprocess
import pandas as pd
import numpy as np
import json


class DataParser:

    def __init__(self, output_path, results_folder, scalars=True, vectors=True, stats=None, all_types=False, tidied_results=None):
        self.output_path = output_path
        self.results_folder = results_folder
        self.scalars = scalars
        self.vectors = vectors
        self.stats = stats
        self.all_types = all_types
        self.tidied_results = tidied_results

    @staticmethod
    def parse_if_number(s):
        try:
            return float(s)
        except:
            return True if s == "true" else False if s == "false" else s if s else None

    @staticmethod
    def parse_ndarray(s):
        return np.fromstring(s, sep=' ') if s else None

    @staticmethod
    def parse_vectime_vecvalue(df):

        print("Parsing Vector file")
        rows_we_want = df.drop(
            ["run", "type", "network", "interface", "layer", "scenario", "date", "time", "processId"], axis=1)
        rows_we_want = rows_we_want.reset_index(drop=True)

        lst_col = "vectime"
        vectime_split = pd.DataFrame({
            col: np.repeat(rows_we_want[col].values, rows_we_want[lst_col].str.len())
            for col in rows_we_want.columns.difference([lst_col])}).assign(
            **{lst_col: np.concatenate(rows_we_want[lst_col].values)})[rows_we_want.columns.tolist()]

        lst_col = "vecvalue"
        vecvalue_split = pd.DataFrame({
            col: np.repeat(rows_we_want[col].values, rows_we_want[lst_col].str.len())
            for col in rows_we_want.columns.difference([lst_col])}).assign(
            **{lst_col: np.concatenate(rows_we_want[lst_col].values)})[rows_we_want.columns.tolist()]

        vecvalue_split["vectime"] = vectime_split["vectime"]

        print("Vector file parsed")

        return vecvalue_split

    def extract_raw_data(self):
        orig_loc = os.getcwd()

        filename = os.path.basename(self.output_path)
        dirpath = os.path.dirname(self.output_path)
        if not os.path.isdir(dirpath):
            os.mkdir(dirpath)

        output_path = os.path.abspath(dirpath)
        output_file = os.path.join(output_path, filename)

        os.chdir(self.results_folder)
        command = "scavetool x"

        if self.scalars:
            command += " *.sca"

        if self.vectors:
            command += " *.vec"

        command += " -o "
        command += output_file

        print(command)

        subprocess.run(command, shell=True)

        os.chdir(orig_loc)

    def tidy_data(self):
        raw_df = pd.read_csv(self.output_path, converters={
            "attrvalue": self.parse_if_number,
            "binedges" : self.parse_ndarray,
            "binvalues": self.parse_ndarray,
            "vectime"  : self.parse_ndarray,
            "vecvalue" : self.parse_ndarray})

        print("Loaded csv into DataFrame")

        # It's likely this will change depending on the run/system
        # Might be worth investigating some form of alternative
        # TODO: Sort out some better means of fixing this.
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

        if self.all_types:
            runattr_df = raw_df[raw_df["type"] == "runattr"]
            runattr_df = runattr_df.dropna(axis=1, how="all")

            itervar_df = raw_df[raw_df["type"] == "itervar"]
            itervar_df = itervar_df.dropna(axis=1, how="all")

            param_df = raw_df[raw_df["type"] == "param"]
            param_df = param_df.dropna(axis=1, how="all")

            attr_df = raw_df[raw_df["type"] == "attr"]
            attr_df = attr_df.dropna(axis=1, how="all")

        if self.stats:
            with open(self.stats) as json_file:
                data = json.load(json_file)
                raw_df = raw_df[(raw_df["name"].isin(data["filtered_vectors"])) | (
                    raw_df["name"].isin(data["filtered_scalars"]))]

        scalar_df = raw_df[raw_df["type"] == "scalar"]
        scalar_df = scalar_df.dropna(axis=1, how="all")

        vector_df = raw_df[raw_df["type"] == "vector"]
        vector_df = vector_df.dropna(axis=1, how="all")

        vector_df = self.parse_vectime_vecvalue(vector_df)

        if self.tidied_results:

            os.mkdir(self.tidied_results)

            print("Saving processed data into {}".format(self.tidied_results))

            vector_df.to_csv("{}/{}".format(self.tidied_results, "vector.csv"), index=False)
            scalar_df.to_csv("{}/{}".format(self.tidied_results, "scalar.csv"), index=False)

            if self.all_types:
                runattr_df.to_csv("{}/{}".format(self.tidied_results, "runattr.csv"), index=False)
                itervar_df.to_csv("{}/{}".format(self.tidied_results, "itervar.csv"), index=False)
                param_df.to_csv("{}/{}".format(self.tidied_results, "params.csv"), index=False)
                attr_df.to_csv("{}/{}".format(self.tidied_results, "attr.csv"), index=False)

        else:
            if self.all_types:
                return vector_df, scalar_df, runattr_df, itervar_df, param_df, attr_df
            return vector_df, scalar_df
