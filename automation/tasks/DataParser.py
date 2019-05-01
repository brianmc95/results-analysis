import os
import multiprocessing
from subprocess import Popen, PIPE, STDOUT
import pandas as pd
import numpy as np
import json
import logging
import re


class DataParser:

    def __init__(self, config, experiment_type, scalars=True, vectors=True, stats=None, all_types=False, tidied_results=None):
        self.config = config
        self.experiment_type = experiment_type
        self.scalars = scalars
        self.vectors = vectors
        self.stats = stats
        self.all_types = all_types
        self.tidied_results = tidied_results
        self.processors = multiprocessing.cpu_count()
        self.logger = logging.getLogger("multi-process")

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

    def log_subprocess_output(self, pipe):
        for line in iter(pipe.readline, b''):  # b'\n'-separated lines
            self.logger.debug('Subprocess Line: %r', line)

    def extract_raw_data(self, result_dirs):
        orig_loc = os.getcwd()
        self.logger.debug("Original path is: {}".format(orig_loc))

        for result_dir in result_dirs:

            if "automation" in orig_loc:
                raw_results_dir = "../data/raw_data/{}/{}".format(self.experiment_type, result_dir)
                omnet_results_dir = "../data/omnet/{}/{}".format(self.experiment_type, result_dir)
            else:
                raw_results_dir = "{}/data/raw_data/{}/{}".format(orig_loc, self.experiment_type, result_dir)
                omnet_results_dir = "{}/data/omnet/{}/{}".format(orig_loc, self.experiment_type, result_dir)

            os.makedirs(raw_results_dir)

            os.chdir(omnet_results_dir)
            self.logger.debug("Moved into {}".format(omnet_results_dir))

            file_names = os.listdir(omnet_results_dir)
            self.logger.debug(file_names)

            # TODO: Improve this it's a bit silly
            pattern = r"\d+"
            run_numbers = []
            for name in file_names:
                run_num = (re.findall(pattern, name))
                if len(run_num) <= 0:
                    continue
                run_num = run_num[0]
                if run_num not in run_numbers:
                    run_numbers.append(run_num)
            run_numbers.sort()

            self.logger.info("Run numbers are the following: {}".format(run_numbers))

            num_processes = self.config["parallel_processes"]
            if num_processes > self.processors:
                self.logger.warn("Too many processes, going to revert to total - 1")
                num_processes = self.processors - 1

            self.logger.debug("Number of files to parse : {}".format(len(run_numbers)))
            number_of_batches = len(run_numbers) // num_processes
            if number_of_batches == 0:
                number_of_batches = 1

            i = 0
            while i < len(run_numbers):
                if len(run_numbers) < num_processes:
                    num_processes = len(run_numbers)
                self.logger.info(
                    "Starting up processes, batch {}/{}".format((i // num_processes) + 1, number_of_batches))
                pool = multiprocessing.Pool(processes=num_processes)

                pool.map(self.scavefiles, run_numbers[i:i+num_processes])

                self.logger.info("Batch {}/{} complete".format((i // num_processes) + 1, number_of_batches))

                i += num_processes

            for file_name in os.listdir(omnet_results_dir):
                os.rename("{}/{}".format(omnet_results_dir, file_name), "{}/{}".format(raw_results_dir, file_name))

            os.chdir(orig_loc)
            self.logger.debug("returned to original directory: {}".format(orig_loc))

    def scavefiles(self, run_number):

        run_command = ["scavetool", "x", "run-{}.sca".format(run_number), "run-{}.vec".format(run_number), "-o",
                       "run-{}.csv".format(run_number)]
        self.logger.info(run_command)

        process = Popen(run_command, stdout=PIPE, stderr=STDOUT)
        with process.stdout:
            self.log_subprocess_output(process.stdout)
        exitcode = process.wait()  # 0 means success

        if exitcode != 0:
            self.logger.warn(
                "Scavetool exitied with {} code, {}/run-{}.csv may not have been created".format(exitcode, os.getcwd(),
                                                                                                 run_number))

    def tidy_data(self, raw_csv):
        raw_df = pd.read_csv(raw_csv, converters={
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
