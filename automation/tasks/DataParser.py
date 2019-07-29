import os
import multiprocessing
from subprocess import Popen, PIPE
import pandas as pd
import numpy as np
import json
import logging
import re
from itertools import repeat

import tempfile
import csv
import shutil

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

    # Additions

    @staticmethod
    def remove_vectors(json_fields, single=False):
        # Simple function to remove the vector from results in json file might be we remove that part from the json file
        if single:
            return json_fields.replace(":vector", "")
        for i in range(len(json_fields)):
            json_fields[i] = json_fields[i].replace(":vector", "")
        return json_fields

    @staticmethod
    def parse_vector_desc_line(line):
        # Converts a vector description line to a dictionary for use in parsing later
        node_id_pattern = re.compile("\[\d+\]")

        vector_line_dict = {"nodeID": None, "vectorName": None, "ETV": True}
        split_line = line.split(" ")
        vector_num = int(split_line[1])
        match = node_id_pattern.search(split_line[2])
        nodeID = int(match.group().strip("[]"))
        vector_name = split_line[3]
        vector_name = vector_name.split(":")[0]
        if "ETV" in split_line[4]:
            ETV = True
        else:
            ETV = False

        vector_line_dict["nodeID"] = nodeID
        vector_line_dict["vectorName"] = vector_name
        vector_line_dict["ETV"] = ETV

        return vector_num, vector_line_dict

    @staticmethod
    def parse_vector_line(line):
        # Simple function to split a vector line and convert to floats.
        split_nums = line.split()
        for i in range(len(split_nums)):
            split_nums[i] = float(split_nums[i])
        return split_nums

    @staticmethod
    def prepare_csv_line(vector_dict, vector_id, parsed_vec):
        # Parses the vector line information to be written to the csv file.
        node_id = vector_dict[vector_id]["nodeID"]
        vector_name = vector_dict[vector_id]["vectorName"]
        if vector_dict[vector_id]["ETV"]:
            etv = parsed_vec[1]
            time = parsed_vec[2]
            value = parsed_vec[3]
        else:
            etv = None
            time = parsed_vec[1]
            value = parsed_vec[2]

        csv_line = [node_id, etv, time, vector_name, value]
        return csv_line

    def read_vector_file(self, output_file, vector_path, stats):
        # Reads the csv file, parses it and writes to a temp file for use later in generating a DF and CSV file.
        vector_dict = {}
        no_interest_vectors = {}

        # Patterns which identify vector declaration lines and result lines
        vector_dec_line_pattern = re.compile("^vector")
        vector_res_line_pattern = re.compile("^\d+")

        vector_file = open(vector_path, "r")

        # Stores lines appearing before their declaration. Files are oddly formatted, this is purely safety ensuring we
        # don't accidentally miss anything.
        early_vectors = tempfile.NamedTemporaryFile(mode="r+")

        output_writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        # Prepare and write out first line format NodeID, EventNumber, Time, Stat1, Stat2, Stat3, ...
        title_line = ["NodeID", "EventNumber", "Time", "StatisticName", "Value"]

        output_writer.writerow(title_line)

        for line in vector_file:
            if vector_dec_line_pattern.match(line):
                # if line matches a vector declaration, parse the vector description
                vector_num, vec_dict = self.parse_vector_desc_line(line)
                if vec_dict["vectorName"] in stats:
                    # Vector is of interest, add it to our overall dictionary and update it's index.
                    vector_dict[vector_num] = vec_dict
                else:
                    # Mark this as a vector we don't care about.
                    no_interest_vectors[vector_num] = None

            elif vector_res_line_pattern.match(line):
                parsed_vec = self.parse_vector_line(line)
                # If the previous step fails then we can simply continue to the next line ignoring this line.
                if parsed_vec is None:
                    continue
                vector_id = parsed_vec[0]
                if vector_id in vector_dict:
                    # Write out to a csv file correctly
                    csv_line = self.prepare_csv_line(vector_dict, vector_id, parsed_vec)
                    output_writer.writerow(csv_line)
                else:
                    if vector_id not in no_interest_vectors:
                        # Write the line out in case we found it before declaration. Only if it is of possible interest.
                        early_vectors.write(line)

        # Rewind the early vectors file so we can search it for missed vectors
        early_vectors.seek(0)

        for line in early_vectors:
            # Parse the line again.
            parsed_vec = self.parse_vector_line(line)
            vector_id = parsed_vec[0]
            # check for the vector
            if vector_id in vector_dict:
                # If we have it create the csv line and write it our
                csv_line = self.prepare_csv_line(vector_dict, vector_id, parsed_vec)
                output_writer.writerow(csv_line)

        # Close our vector file.
        vector_file.close()

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

    def log_subprocess_output(self, pipe):
        for line in iter(pipe.readline, b''):  # b'\n'-separated lines
            self.logger.debug('Subprocess Line: %r', line)

    def bin_fields(self, df, fields, bin_width=10, bin_quantity=100):
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

        distance_col = self.config["results"]["distance"]

        for i in range(len(bins)):
            lower_b = bins[i][0]
            upper_b = bins[i][1]
            fields_temp = df[(df[distance_col] >= lower_b) & (df[distance_col] < upper_b)]
            for field in fields:
                if i < len(overall_fields[field]):
                    overall_fields[field][i] = (fields_temp[field].mean() + overall_fields[field][i]) / 2
                else:
                    overall_fields[field].append(fields_temp[field].mean())

        return overall_fields

    def parse_data(self, results_dirs, now):
        combined_results = {}

        configs = []
        for config in self.config["config_names"]:
            config_data = self.config["config_names"][config]
            if config_data["repeat"] != 0:
                if "naming" in config_data and len(config_data["naming"]) > 0:
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

            runs = []

            for run in os.listdir(result_dir):
                if ".vec" in run:
                    runs.append(run)

            num_processes = self.config["parallel_processes"]
            if num_processes > multiprocessing.cpu_count():
                self.logger.warning("Too many processes, going to revert to total - 1")
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

                multiple_results = pool.starmap(self.filter_data, zip(runs[i:i + num_processes], repeat(config_name), repeat(now), repeat(orig_loc)))

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

    @staticmethod
    def combine_results(combined, results):
        for result in results:
            for field in result:
                if field in combined:
                    for i in range(len(result[field])):
                        combined[field][i] = (combined[field][i] + result[field][i]) / 2
                else:
                    combined[field] = result[field]
        return combined

    def filter_data(self, raw_data_file, config_name, now, orig_loc):

        run_num = raw_data_file.split(".")[0]

        temp_file_name = run_num + ".csv"

        self.logger.info("File being parsed: {}".format(temp_file_name))

        output_csv_dir = "{}/data/raw_data/{}/{}".format(orig_loc, self.experiment_type, config_name)

        os.makedirs(output_csv_dir, exist_ok=True)

        output_csv = "{}/{}-{}.csv".format(output_csv_dir, run_num, now)

        self.logger.info("Raw output file: {}".format(output_csv))

        vector_df = self.tidy_data(temp_file_name, raw_data_file, self.results["filtered_vectors"], output_csv)

        self.logger.info("Completed tidying of dataframes")

        graphs = self.config["results"]["graphs"]
        self.logger.info("The data for the following graphs must be prepared {}".format(graphs))

        if ":vector" in self.config["results"]["decoded"]:
            # Assuming if decoded contains :vector then fails will too.
            self.config["results"]["decoded"]  = self.remove_vectors(self.config["results"]["decoded"], single=True)
            self.config["results"]["distance"] = self.remove_vectors(self.config["results"]["distance"], single=True)
            self.config["results"]["fails"]    = self.remove_vectors(self.config["results"]["fails"])

        fields = []
        if "pdr-dist" in graphs:
            self.logger.info("Calculating pdr for pdr graph")
            fields.append(self.results["decoded"])
        if "error-dist" in graphs:
            for fail in self.config["results"]["fails"]:
                fields.append(fail)
            fields.append(self.results["decoded"])

        self.logger.info("Binning all the necessary information for the graphs")
        binned_results = self.bin_fields(vector_df, fields)

        del vector_df

        self.logger.info("Completed data parsing for this run")
        return binned_results

    def tidy_data(self, temp_file, real_vector_path, json_fields, output_csv):
        temp_file_pt = open(temp_file, "w+")

        # Simply remove the :vector part of vector names from both sets of vectors.
        found_vector = False
        for field in json_fields:
            if ":vector" in field:
                found_vector = True
                break

        if found_vector:
            json_fields = self.remove_vectors(json_fields)

        self.logger.info("Beginning parsing of vector file: {}".format(real_vector_path))

        # Read the file and retrieve the list of vectors
        self.read_vector_file(temp_file_pt, real_vector_path, json_fields)

        self.logger.info("Finished parsing of vector file: {}".format(real_vector_path))

        # Ensure we are at the start of the file for sorting
        temp_file_pt.seek(0)

        over_all_df = pd.read_csv(temp_file_pt)

        # Parse the vector file to ensure it is formatted correclty.
        over_all_df['seq'] = over_all_df.groupby(["EventNumber", "StatisticName"]).cumcount()
        over_all_df = over_all_df.pivot_table("Value", ["EventNumber", "Time", "NodeID", "seq"], "StatisticName")
        over_all_df.reset_index(inplace=True)
        over_all_df = over_all_df.drop(["seq"], axis=1)

        # Write this out as our raw_results file
        over_all_df.to_csv(output_csv, index=False)
        self.logger.info("Wrote out the parsed vector file to: {}".format(output_csv))

        # Remove our temporary file.
        os.remove(temp_file_pt.name)
        self.logger.debug("Removed the temporary file")

        return over_all_df

