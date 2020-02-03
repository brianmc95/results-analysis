import os
import multiprocessing
import pandas as pd
import logging
import re
from itertools import repeat
from natsort import natsorted
import shutil
import csv
import numpy as np


class DataParser:

    def __init__(self, config, experiment_type, scalars=True, vectors=True, stats=None, all_types=False,
                 tidied_results=None):
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

        # Patterns which identify vector declaration lines and result lines
        self.vector_dec_line_pattern = re.compile("^vector")
        self.vector_res_line_pattern = re.compile("^\d+")

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

    def parse_vector_line(self, line):
        # Simple function to split a vector line and convert to floats.
        try:
            line = bytes(line, 'utf-8').decode('utf-8', 'ignore')
            split_nums = line.split()
            for i in range(len(split_nums)):
                split_nums[i] = float(split_nums[i])
            return split_nums
        except ValueError as e:
            self.logger.error("Line: {} could not be converted due to bad encoding".format(line))
            return

    @staticmethod
    def prepare_csv_line(vector_dict, vector_id, parsed_vec):
        # Parses the vector line information to be written to the csv file.
        node_id = vector_dict[vector_id]["nodeID"]
        vector_name = vector_dict[vector_id]["vectorName"]
        if vector_dict[vector_id]["ETV"]:
            time = parsed_vec[2]
            value = parsed_vec[3]
        else:
            time = parsed_vec[1]
            value = parsed_vec[2]

        csv_line = {"NodeID": node_id, "Time": time, "StatisticName": vector_name, "Value": value}
        return csv_line, time

    def csv_pivot(self, directory, stats):
        orig_loc = os.getcwd()
        os.chdir(directory)

        csv_files = os.listdir(os.getcwd())
        csv_files = natsorted(csv_files)
        header = True
        for csv_file in csv_files:
            if ".csv" in csv_file:
                self.logger.debug("Pivoting chunk file: {}".format(csv_file))
                chunk_df = pd.read_csv(csv_file)

                chunk_df = chunk_df.infer_objects()

                chunk_df = chunk_df.sort_values(by=["NodeID", "Time"])
                # Parse the vector file to ensure it is formatted correclty.
                chunk_df['seq'] = chunk_df.groupby(["Time", "NodeID", "StatisticName"]).cumcount()

                chunk_df = chunk_df.pivot_table("Value", ["Time", "NodeID", "seq"], "StatisticName")
                chunk_df.reset_index(inplace=True)
                chunk_df = chunk_df.drop(["seq"], axis=1)

                # Ensure all fields correctly filled
                for field in stats:
                    if field not in chunk_df.columns:
                        chunk_df[field] = np.nan

                # Ensure the order of the files is also correct
                chunk_df = chunk_df.reindex(sorted(chunk_df.columns), axis=1)

                chunk_df.to_csv(csv_file, index=False, header=header)
                header = False

                del chunk_df

        os.chdir(orig_loc)

    def combine_files(self, csv_directory, outfile):
        destination = open(outfile, 'wb')

        orig_loc = os.getcwd()
        os.chdir(csv_directory)

        csv_files = os.listdir(os.getcwd())
        csv_files = natsorted(csv_files)
        for csv_file in csv_files:
            if ".csv" in csv_file and csv_file != outfile:
                self.logger.debug("Merging chunk file: {} into {}".format(csv_file, outfile))
                shutil.copyfileobj(open(csv_file, 'rb'), destination)
                os.remove(csv_file)
        destination.close()

        os.chdir(orig_loc)

        os.rmdir(csv_directory)

    def setup_chunk_writer(self, output_file, chunk_num, title_line):
        # Setup our chunk writer

        # First create a folder to hold chunks
        chunk_folder = output_file.split(".")[0]
        os.makedirs(chunk_folder, exist_ok=True)
        chunk_name = "{}/chunk-{}.csv".format(chunk_folder, chunk_num)

        # Create the chunk file and create a csv writer which uses it
        self.logger.debug("Setting up new chunk: {}".format(chunk_name))
        temp_file_pt = open(chunk_name, "w+")
        output_writer = csv.DictWriter(temp_file_pt, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL,
                                       fieldnames=title_line)
        output_writer.writeheader()

        return temp_file_pt, output_writer

    def read_vector_file(self, output_file, vector_path, stats, chunk_size=4e+8):
        """
        chunk_size: the time between different files 1.5s as default
        """
        # Reads the csv file, parses it and writes to a temp file for use later in generating a DF and CSV file.
        vector_dict = {}
        no_interest_vectors = {}  # Probably don't need to remember one's we don't care for.

        chunk_times = []
        chunk_info = {}

        last_time = -1
        current_chunk_index = 0

        vector_file = open(vector_path, "r")

        # Prepare and write out first line format NodeID, EventNumber, Time, Stat1, Stat2, Stat3, ...
        title_line = ["NodeID", "Time", "StatisticName", "Value"]

        temp_file_pt, writer = self.setup_chunk_writer(output_file, current_chunk_index, title_line)
        chunk_info["CurrentChunk"] = {"file": temp_file_pt, "writer": writer}

        # Read 100,000 lines
        tmp_lines = vector_file.readlines(100000)
        while tmp_lines:
            new_lines, old_lines, time = self.process_chunk([line for line in tmp_lines], vector_dict,
                                                            no_interest_vectors, stats, last_time, chunk_times)

            chunk_info["CurrentChunk"]["writer"].writerows(new_lines)
            if chunk_info["CurrentChunk"]["file"].tell() >= chunk_size:
                self.logger.debug("Time ending this chunk:{}".format(time))

                # This chunk is old and as such can be placed into the previous chunks
                chunk_info[time] = {"file": chunk_info["CurrentChunk"]["file"],
                                    "writer": chunk_info["CurrentChunk"]["writer"]}
                chunk_times.append(time)
                last_time = time
                current_chunk_index += 1

                # This file is at max size, create a new writer
                temp_file_pt, writer = self.setup_chunk_writer(output_file, current_chunk_index, title_line)
                # Update current chunk writer to point at this new one.
                chunk_info["CurrentChunk"] = {"file": temp_file_pt, "writer": writer}

            if old_lines:
                for chunk_time in old_lines:
                    chunk_info[chunk_time]["writer"].writerows(old_lines[chunk_time])
            # Read the next 100,000 lines
            tmp_lines = vector_file.readlines(100000)

        vector_file.close()

    def process_chunk(self, lines, vector_dict, no_interest_vectors, stats, last_time, chunk_times):

        new_lines = []
        old_lines = {}
        max_time = -1

        for line in lines:
            if self.vector_dec_line_pattern.match(line):
                # if line matches a vector declaration, parse the vector description
                vector_num, vec_dict = self.parse_vector_desc_line(line)
                if vector_num is None and vec_dict is None:
                    continue
                if vec_dict["vectorName"] in stats:
                    # Vector is of interest, add it to our overall dictionary and update it's index.
                    vector_dict[vector_num] = vec_dict
                else:
                    # Mark this as a vector we don't care about.
                    no_interest_vectors[vector_num] = None

            elif self.vector_res_line_pattern.match(line):
                # {"nodeID": None, "vectorName": None, "ETV": True} This is what it looks like
                parsed_vec = self.parse_vector_line(line)
                # If the previous step fails then we can simply continue to the next line ignoring this line.
                if parsed_vec is None:
                    continue
                vector_id = parsed_vec[0]
                if vector_id in vector_dict:
                    # Write out to a csv file correctly
                    csv_line, time = self.prepare_csv_line(vector_dict, vector_id, parsed_vec)

                    if time > last_time:
                        new_lines.append(csv_line)
                        if time > max_time:
                            max_time = time

                    if time <= last_time:
                        for chunk_time in chunk_times:
                            if time < chunk_time:
                                if chunk_time in old_lines:
                                    old_lines[chunk_time].append(csv_line)
                                else:
                                    old_lines[chunk_time] = [csv_line]

                else:
                    if vector_id not in no_interest_vectors:
                        # Write the line out in case we found it before declaration. Only if it is of possible interest.
                        self.logger.warning("There was a line which appeared in the wrong place: {}".format(line))

        return new_lines, old_lines, max_time

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

    def bin_fields(self, df, fields, bin_width=10, bin_quantity=49):
        """
        Bins multiple dfs into a single dictionary that can be used as an average for multiple fields across multiple
        runs
        :param df: dataframe to bin
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
            results = []
            while i < len(runs):
                if len(runs) < num_processes:
                    num_processes = len(runs)
                self.logger.info(
                    "Starting up processes, batch {}/{}".format((i // num_processes) + 1, number_of_batches))
                pool = multiprocessing.Pool(processes=num_processes)

                results.append(pool.starmap(self.filter_data,
                                            zip(runs[i:i + num_processes],
                                                repeat(config_name),
                                                repeat(now),
                                                repeat(orig_loc))))

                pool.close()
                pool.join()

                self.logger.info("Batch {}/{} complete".format((i // num_processes) + 1, number_of_batches))

                i += num_processes

            self.logger.debug("Moving back to original location: {}".format(orig_loc))
            os.chdir(orig_loc)

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

        output_csv_dir = "{}/data/parsed_data/{}/{}-{}".format(orig_loc, self.experiment_type, config_name, now)

        os.makedirs(output_csv_dir, exist_ok=True)

        output_csv = "{}/{}.csv".format(output_csv_dir, run_num)

        self.logger.info("Raw output file: {}".format(output_csv))

        self.tidy_data(raw_data_file, self.results["filtered_vectors"], output_csv)

        self.logger.info("Completed tidying of dataframes")

    def tidy_data(self, real_vector_path, json_fields, output_csv):
        # Simply remove the :vector part of vector names from both sets of vectors.
        found_vector = False
        for field in json_fields:
            if ":vector" in field:
                found_vector = True
                break

        if found_vector:
            json_fields = self.remove_vectors(json_fields)

        self.logger.debug(json_fields)

        self.logger.info("Beginning parsing of vector file: {}".format(real_vector_path))

        # Read the vector file into a csv file
        chunk_folder = output_csv.split(".")[0]
        self.read_vector_file(output_csv, real_vector_path, json_fields)

        self.logger.info("File read, begin pivoting csv file: {}".format(real_vector_path))
        self.csv_pivot(chunk_folder, json_fields)

        self.logger.info("Pivot complete, consolidate chunk files for {}".format(output_csv))
        self.combine_files(chunk_folder, output_csv)

        self.logger.info("Finished parsing of vector file: {}".format(real_vector_path))
