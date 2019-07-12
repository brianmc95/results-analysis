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

        vector_line_dict = {"nodeID": None, "vectorName": None, "ETV": True, "index": 0}
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
    def prepare_csv_line(vector_names, vector_dict, vector_id, parsed_vec):
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

        csv_line = [node_id, etv, time]
        csv_tail = []
        for i in range(len(vector_names)):
            if i == vector_dict[vector_id]["index"]:
                csv_tail.append(value)
            else:
                csv_tail.append(None)
        csv_line = csv_line + csv_tail
        return csv_line

    @staticmethod
    def write_top_line(output_file, vector_names, run_num):
        # Takes the temp file and writes out the top line of the vectors.
        output_file.seek(0)

        output_name = output_file.name

        temp_file_name = "to_be_removed-{}.csv".format(run_num)

        os.rename(output_file.name, temp_file_name)

        new_file = open(output_name, "w+")

        output_writer = csv.writer(new_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        # Prepare and write out first line format NodeID, EventNumber, Time, Stat1, Stat2, Stat3, ...
        title_line = ["NodeID", "EventNumber", "Time"]
        title_line = title_line + vector_names

        output_writer.writerow(title_line)

        shutil.copyfileobj(output_file, new_file)

        os.remove(temp_file_name)

        return new_file

    def read_vector_file(self, output_file, vector_path, stats):
        # Reads the csv file, parses it and writes to a temp file for use later in generating a DF and CSV file.
        vector_dict = {}
        no_interest_vectors = {}
        vector_names = []

        # Patterns which identify vector declaration lines and result lines
        vector_dec_line_pattern = re.compile("^vector")
        vector_res_line_pattern = re.compile("^\d+")

        vector_file = open(vector_path, "r")

        # Stores lines appearing before their declaration. Files are oddly formatted, this is purely safety ensuring we
        # don't accidentally miss anything.
        early_vectors = tempfile.NamedTemporaryFile(mode="r+")

        output_writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        for line in vector_file:
            if vector_dec_line_pattern.match(line):
                # if line matches a vector declaration, parse the vector description
                vector_num, vec_dict = self.parse_vector_desc_line(line)
                if vec_dict["vectorName"] not in vector_names:
                    if vec_dict["vectorName"] in stats:
                        # If we haven't seen this vector before and we are interested, add it and mark it's index
                        vector_names.append(vec_dict["vectorName"])
                        vec_dict["index"] = len(vector_names) - 1
                    else:
                        # Mark this as a vector we don't care about.
                        no_interest_vectors[vector_num] = None
                if vec_dict["vectorName"] in vector_names:
                    # Vector is of interest, add it to our overall dictionary and update it's index.
                    vec_dict["index"] = vector_names.index(vec_dict["vectorName"])
                    vector_dict[vector_num] = vec_dict

            elif vector_res_line_pattern.match(line):
                parsed_vec = self.parse_vector_line(line)
                vector_id = parsed_vec[0]
                if vector_id in vector_dict:
                    # Write out to a csv file correctly
                    csv_line = self.prepare_csv_line(vector_names, vector_dict, vector_id, parsed_vec)
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
                csv_line = self.write_out(vector_names, vector_dict)
                output_writer.writerow(csv_line)

        # Close our vector file.
        vector_file.close()

        return vector_names

    def sort_csv_result(self, temp_file_pt, run_num):
        # Pulling it all together

        temp_unsorted_dir = "temp-unsorted-{}".format(run_num)
        temp_sorted_dir = "temp-sorted-{}".format(run_num)

        os.makedirs(temp_unsorted_dir, exist_ok=True)
        os.makedirs(temp_sorted_dir, exist_ok=True)

        self.logger.debug("Splitting file {} into parts".format(temp_file_pt.name))

        split_command = "split -l 2500000 {} {}/temp_csv_".format(temp_file_pt.name, temp_unsorted_dir)

        # Need to delete all the temp things then when I write the file back into my temp_file
        process = Popen(split_command, shell=True, stdout=PIPE)
        process.wait()

        all_splits = os.listdir(temp_unsorted_dir)
        num_splits = len(all_splits)

        self.logger.info("Splitting complete, split temp_file: {} into {} files".format(temp_file_pt.name, num_splits))

        split = 1
        for file_to_sort in all_splits:
            self.logger.info("Sorting split: {} of {}".format(split, num_splits))
            sort_split_commands = "sort -t ',' -n -k2,2 -s -o {}/{} {}/{}".format(temp_sorted_dir, file_to_sort, temp_unsorted_dir, file_to_sort)
            self.logger.debug("Sorting splits command: {}".format(sort_split_commands))

            # Need to delete all the temp things then when I write the file back into my temp_file
            process = Popen(sort_split_commands, shell=True, stdout=PIPE)
            process.wait()

            split += 1

        self.logger.info("Merging split files into single sorted csv file")

        fully_sorted_command = "sort -t ',' -n -k2,2 -m -s -o {} {}/temp_csv*".format(temp_file_pt.name, temp_sorted_dir)
        self.logger.debug("Sorting merge command: {}".format(fully_sorted_command))

        # Need to delete all the temp things then when I write the file back into my temp_file
        process = Popen(fully_sorted_command, shell=True, stdout=PIPE)
        process.wait()

        self.logger.debug("Merging complete")

        # Delete the temporary directory.
        shutil.rmtree(temp_unsorted_dir)
        shutil.rmtree(temp_sorted_dir)

    @staticmethod
    def normal_merge(df, exploding_vectors):
        new_df = pd.DataFrame()
        for vector in exploding_vectors:
            if new_df.empty:
                new_df = df[df[vector].notnull()]
                new_df = new_df.drop(exploding_vectors[1:], axis=1)
            else:
                series = df.loc[:, vector].dropna()
                new_df[vector] = series.values
        return new_df

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

    def odd_merge(self, df, exploding_vectors):
        new_df = pd.DataFrame()
        series_length = 0
        empty_cols = []
        non_empty_cols = []
        for vector in exploding_vectors:
            if df[vector].isnull().all():
                empty_cols.append(vector)
            else:
                non_empty_cols.append(vector)

        for vector in non_empty_cols:
            self.logger.debug("Non-empty vector: {}".format(vector))
            if new_df.empty:
                new_df = df[df[vector].notnull()]
                new_df = new_df.drop(exploding_vectors[1:], axis=1)
                series_length = len(new_df.index)
            else:
                series = df.loc[:, vector].dropna()
                new_df[vector] = series.values

        for vector in empty_cols:
            self.logger.debug("Empty vector: {}:{}:{}".format(vector, series_length, len(new_df.index)))
            series = pd.Series(np.nan for _ in range(series_length))
            new_df[vector] = series.values

        return new_df

    def parse_csv_chunks(self, file, merging_vectors, key="EventNumber", chunk_size=1e6):

        # Tell pandas to read the data in chunks
        chunks = pd.read_csv(file, chunksize=chunk_size)

        results = []
        odds = []
        orphans = pd.DataFrame()

        chunk_num = 1

        for chunk in chunks:

            self.logger.debug("Processing Chunk: {}".format(chunk_num))

            # Add the previous orphans to the chunk
            chunk = pd.concat((orphans, chunk))

            # Determine which rows are orphans
            last_val = chunk[key].iloc[-1]
            is_orphan = chunk[key] == last_val

            # Put the new orphans aside
            chunk, orphans = chunk[~is_orphan], chunk[is_orphan]

            merging_cols = ["NodeID", "EventNumber", "Time"] + merging_vectors

            merging_df = chunk[merging_cols]

            no_merging_df = chunk[chunk.columns.difference(merging_vectors)]

            no_merge_cols = []
            for col in no_merging_df:
                if col not in merging_cols:
                    no_merge_cols.append(col)

            # Broken the chunk into the parts that we needed, deleting it saves us some memory.
            del chunk

            # Deal with the columns which act individually, these we simply drop everything except them
            # and take the first available value
            no_merging_df = no_merging_df.groupby("EventNumber", as_index=False).agg("first")
            no_merging_df = no_merging_df.dropna(subset=no_merge_cols, how="all")

            # Make sure there are no empty rows here, if so delete them
            merging_df = merging_df.dropna(subset=merging_vectors, how="all")
            merging_df = merging_df.reset_index(drop=True)

            # Determine rows where not all merges can occur and remove them from merges
            odd_merging_df = merging_df.groupby('EventNumber').filter(lambda x: len(x) % len(merging_vectors) != 0)

            merging_df = merging_df[~merging_df["EventNumber"].isin(odd_merging_df["EventNumber"])]

            merging_df = self.normal_merge(merging_df, merging_vectors)

            if not odd_merging_df.empty:
                self.logger.debug("Need to do odd merging")
                odds.append(odd_merging_df)
            else:
                self.logger.debug("No odd merging to be done.")

            result = pd.concat([merging_df, no_merging_df], sort=False, ignore_index=True)

            results.append(result)

            chunk_num += 1

        if len(odds) != 0:
            self.logger.debug("Dealing with odd values")

            possible_odds_df = pd.concat(odds, sort=False, ignore_index=True)

            # Determine rows where not all merges can occur and remove them from merges
            odd_merging_df = possible_odds_df.groupby('EventNumber').filter(lambda x: len(x) % len(merging_vectors) != 0)

            normal_merging_df = possible_odds_df[~possible_odds_df["EventNumber"].isin(odd_merging_df["EventNumber"])]

            normal_merging_df = self.normal_merge(normal_merging_df, merging_vectors)

            odd_merging_df = self.odd_merge(odd_merging_df, merging_vectors)

            result = pd.concat([normal_merging_df, odd_merging_df], sort=False, ignore_index=True)

            results.append(result)
        else:
            self.logger.debug("No odd values to deal with")

        return pd.concat(results, sort=False, ignore_index=True)

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

        vector_df = self.tidy_data(temp_file_name, raw_data_file, self.results["filtered_vectors"],
                                   self.results["merging"], output_csv, run_num)

        self.logger.info("Completed tidying of dataframes")

        graphs = self.config["results"]["graphs"]
        self.logger.info("The data for the following graphs must be prepared {}".format(graphs))

        if ":vector" in self.config["results"]["decoded"]:
            # Assuming if decoded contains :vector then fails will too.
            self.config["results"]["decoded"] = self.remove_vectors(self.config["results"]["decoded"], single=True)
            self.config["results"]["distance"] = self.remove_vectors(self.config["results"]["distance"], single=True)
            self.config["results"]["fails"] = self.remove_vectors(self.config["results"]["fails"])

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

    def tidy_data(self, temp_file, real_vector_path, json_fields, merging_vectors, output_csv, run_num):
        temp_file_pt = open(temp_file, "w+")

        # Simply remove the :vector part of vector names from both sets of vectors.
        found_vector = False
        for field in json_fields:
            if ":vector" in field:
                found_vector = True
                break

        for field in merging_vectors:
            if ":vector" in field or found_vector:
                found_vector = True
                break

        if found_vector:
            json_fields = self.remove_vectors(json_fields)
            merging_vectors = self.remove_vectors(merging_vectors)

        self.logger.info("Beginning parsing of vector file: {}".format(real_vector_path))

        # Read the file and retrieve the list of vectors
        vector_names = self.read_vector_file(temp_file_pt, real_vector_path, json_fields)

        self.logger.info("Finished parsing of vector file: {}".format(real_vector_path))

        # Ensure we are at the start of the file for sorting
        temp_file_pt.seek(0)

        # Splits and sorts individual elements of the overall file
        self.logger.info("Beginning the split and sorting of vector file")
        self.sort_csv_result(temp_file_pt, run_num)
        self.logger.info("Split and sort complete")

        # Writes the top line of the temporary file used to store the DF
        temp_file_pt = self.write_top_line(temp_file_pt, vector_names, run_num)
        self.logger.info("Wrote the top line of the parsed csv file")

        temp_file_pt.seek(0)
        # Read the sorted file in chunks and create an overall_df from it.
        self.logger.info("Beginning the parsing of the vector file into condensed format")
        over_all_df = self.parse_csv_chunks(temp_file_pt, merging_vectors, key="EventNumber")
        self.logger.info("Vector file parsed into condensed format and available to be worked on.")

        # Write this out as our raw_results file
        over_all_df.to_csv(output_csv, index=False)
        self.logger.info("Writing out the parsed vector file to: {}".format(output_csv))

        # Remove our temporary file.
        os.remove(temp_file_pt.name)
        self.logger.debug("Removed the temporary file")

        return over_all_df

