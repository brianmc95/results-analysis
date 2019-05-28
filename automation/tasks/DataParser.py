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
        self.results = self.config["results"]
        self.logger = logging.getLogger("DataParser")

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
    def create_bins(lower_bound, width, quantity):
        """ create_bins returns an equal-width (distance) partitioning.
            It returns an ascending list of tuples, representing the intervals.
            A tuple bins[i], i.e. (bins[i][0], bins[i][1])  with i > 0
            and i < quantity, satisfies the following conditions:
                (1) bins[i][0] + width == bins[i][1]
                (2) bins[i-1][0] + width == bins[i][0] and
                    bins[i-1][1] + width == bins[i][1]
        """
        bins = []
        for low in range(lower_bound, lower_bound + quantity * width + 1, width):
            bins.append((low, low + width))
        return bins

    def parse_vectime_vecvalue(self, df):

        self.logger.info("Parsing Vector file")
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

        self.logger.info("Vector file parsed")

        return vecvalue_split

    def log_subprocess_output(self, pipe):
        for line in iter(pipe.readline, b''):  # b'\n'-separated lines
            self.logger.debug('Subprocess Line: %r', line)

    def extract_raw_data(self, result_dirs):
        orig_loc = os.getcwd()
        self.logger.debug("Original path is: {}".format(orig_loc))

        results_folders = []

        for result_dir in result_dirs:

            folder_name = os.path.basename(result_dir)
            raw_results_dir = "{}/data/raw_data/{}/{}".format(orig_loc, self.experiment_type, folder_name)

            os.makedirs(raw_results_dir)

            os.chdir(result_dir)
            self.logger.debug("Moved into {}".format(result_dir))

            file_names = os.listdir(result_dir)
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

            for file_name in os.listdir(result_dir):
                if ".csv" not in file_name:
                    continue
                self.logger.info("Moving file {}/{} to {}/{}".format(result_dir, file_name,
                                                                     raw_results_dir, file_name))
                os.rename("{}/{}".format(result_dir, file_name), "{}/{}".format(raw_results_dir, file_name))

            os.chdir(orig_loc)
            self.logger.debug("returned to original directory: {}".format(orig_loc))

            results_folders.append(raw_results_dir)

        return results_folders

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

    def bin_fields(self, df, fields, bin_width=10, bin_quantity=49):
        """
        Bins multiple dfs into a single dictionary that can be used as an average for multiple fields across multiple
        runs
        :param dfs: list of dataframes to bin
        :param fields: fields to be binned.
        :param bin_width: width of each bin
        :param bin_quantity: total number of bins
        :return:
        """
        bins = self.create_bins(lower_bound=0, width=bin_width, quantity=bin_quantity)
        distances = []
        overall_fields = {}
        for interval in bins:
            upper_b = interval[1]
            distances.append(upper_b)

        for field in fields:
            self.logger.debug("{} being binned".format(field))
            overall_fields[field] = []

        overall_fields["distance"] = distances

        for i in range(len(bins)):
            lower_b = bins[i][0]
            upper_b = bins[i][1]
            fields_temp = df[(df["distance"] >= lower_b) & (df["distance"] < upper_b)]
            for field in fields:
                if i < len(overall_fields[field]):
                    overall_fields[field][i] = (fields_temp[field].mean() + overall_fields[field][i]) / 2
                else:
                    overall_fields[field].append(fields_temp[field].mean())

        return overall_fields

    def pdr_calc(self, df):
        """
        Calculates Packet Delivery Ratio for a DataFrame based on fields provided in self.fields JSON file.
        :param df: DataFrame which we will calculate df from.
        :return: new_df which included pdr in it.
        """

        new_df = pd.DataFrame()

        decoded = df[df["name"] == self.results["decoded"]]
        decoded = decoded.reset_index(drop=True)
        dist = df[df["name"] == self.results["distance"]]
        dist = dist.reset_index(drop=True)

        new_df["time"] = dist["vectime"]
        new_df["distance"] = dist["vecvalue"]
        new_df["decoded"] = decoded["vecvalue"]
        new_df["node"] = dist["node"]

        for i in range(len(self.results["fails"])):
            self.logger.info("Field being binned: {}".format(self.results["fails"][i]))
            fail_df = df[df["name"] == self.results["fails"][i]]
            fail_df = fail_df.reset_index(drop=True)
            fail_df = fail_df.fillna(0)
            if fail_df.empty:
                self.logger.error("{} does not appear in the overall dataframe".format(self.results["fails"][i]))
                raise Exception("{} does not appear in the overall dataframe".format(self.results["fails"][i]))
            new_df[self.results["fails"][i]] = fail_df["vecvalue"]
            if "total_fails" in new_df:
                new_df["total_fails"] += fail_df["vecvalue"]
            else:
                new_df["total_fails"] = fail_df["vecvalue"]

        self.logger.debug("Calculating pdr for graph")
        new_df["pdr"] = ((new_df["decoded"]) / (new_df["decoded"] + new_df["total_fails"])) * 100

        return new_df

    def tidy_data(self, raw_csv):
        raw_df = pd.read_csv(raw_csv, converters={
            "attrvalue": self.parse_if_number,
            "binedges" : self.parse_ndarray,
            "binvalues": self.parse_ndarray,
            "vectime"  : self.parse_ndarray,
            "vecvalue" : self.parse_ndarray})

        self.logger.info("Loaded {} as a DataFrame".format(raw_csv))

        # It's likely this will change depending on the run/system
        # Might be worth investigating some form of alternative
        # TODO: Sort out some better means of fixing this.
        broken_module = raw_df['module'].str.split('.', 3, expand=True)
        #
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

        if "filtered_vectors" in self.results and "filtered_scalars" in self.results:
            raw_df = raw_df[(raw_df["name"].isin(self.results["filtered_vectors"])) |
                            (raw_df["name"].isin(self.results["filtered_scalars"]))]

        scalar_df = raw_df[raw_df["type"] == "scalar"]
        scalar_df = scalar_df.dropna(axis=1, how="all")

        vector_df = raw_df[raw_df["type"] == "vector"]
        vector_df = vector_df.dropna(axis=1, how="all")

        vector_df = self.parse_vectime_vecvalue(vector_df)

        if self.tidied_results:

            os.mkdir(self.tidied_results)

            self.logger.info("Saving processed data into {}".format(self.tidied_results))

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

    def parse_data(self, results_dirs, now):
        combined_results = {}

        configs = []
        for config in self.config["config_names"]:
            config_data = self.config["config_names"][config]
            if config_data["repeat"] != 0:
                if len(config_data["naming"]) > 0:
                    for name in config_data["naming"]:
                        configs.append(name)
                else:
                    configs.append(config)

        for result_dir, config_name in zip(results_dirs, configs):

            folder_name = os.path.basename(result_dir)

            self.logger.info("Dealing with config: {} of result folder: {}".format(config_name, folder_name))
            combined_results[config_name] = {}

            orig_loc = os.getcwd()

            self.logger.debug("Moving to results dir: {}".format(result_dir))
            os.chdir(result_dir)

            runs = os.listdir(result_dir)

            num_processes = self.config["parallel_processes"]
            if num_processes > multiprocessing.cpu_count():
                self.logger.warn("Too many processes, going to revert to total - 1")
                num_processes = multiprocessing.cpu_count() - 1

            self.logger.debug("Number of files to parse : {}".format(len(runs)))
            number_of_batches = len(runs) // num_processes
            if number_of_batches == 0:
                number_of_batches = 1

            i = 0
            while i < len(runs):
                if len(runs) < num_processes:
                    num_processes = len(runs)
                self.logger.info(
                    "Starting up processes, batch {}/{}".format((i // num_processes) + 1, number_of_batches))
                pool = multiprocessing.Pool(processes=num_processes)

                multiple_results = pool.map(self.filter_data, runs[i:i + num_processes])

                combined_results[config_name] = self.combine_results(combined_results[config_name], multiple_results)

                self.logger.info("Batch {}/{} complete".format((i // num_processes) + 1, number_of_batches))

                i += num_processes

            self.logger.debug("Moving back to original location: {}".format(orig_loc))
            os.chdir(orig_loc)

        processed_file = "{}/data/processed_data/{}-{}.json".format(os.getcwd(), self.experiment_type, now)
        self.logger.info("Writing processed data to {}".format(processed_file))
        with open(processed_file, "w") as json_output:
            json.dump(combined_results, json_output)

        return processed_file

    def combine_results(self, combined, results):
        for result in results:
            for field in result:
                if field in combined:
                    for i in range(len(result[field])):
                        combined[field][i] = (combined[field][i] + result[field][i]) / 2
                else:
                    combined[field] = result[field]
        return results

    def filter_data(self, raw_data_file):
        vector_df, scalar_df = self.tidy_data(raw_data_file)
        del scalar_df

        self.logger.info("Completed tidying of dataframes")

        graphs = self.config["results"]["graphs"]
        self.logger.info("The data for the following graphs must be prepared {}".format(graphs))

        fields = []
        if "pdr-dist" in graphs:
            self.logger.info("Calculating pdr for pdr graph")
            vector_df = self.pdr_calc(vector_df)
            fields.append("pdr")
        if "error-dist" in graphs:
            for fail in self.config["results"]["fails"]:
                fields.append(fail)
            fields.append("decoded")

        self.logger.info("Binning all the necessary information for the graphs")
        binned_results = self.bin_fields(vector_df, fields)

        self.logger.info("Completed data parsing for this run")
        return binned_results
