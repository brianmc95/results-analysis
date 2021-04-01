import logging
import os
import math
import multiprocessing
import statistics
from itertools import combinations

from scipy.stats import t
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import natsort


class Grapher:

    def __init__(self, config, experiment_type, markers=True, line_types=False, image_format="png"):

        self.use_markers = markers
        self.config = config
        self.experiment_type = experiment_type
        self.use_line_types = line_types
        self.image_format = image_format
        self.figure_store = "{}/data/figures".format(os.getcwd())
        self.results = self.config["results"]
        self.logger = logging.getLogger("Grapher")

        # TODO: Possibly find a prefered order for these
        self.markers = [".", "o", "v", "^", "<", ">", "1", "2", "3", "4", "8", "s", "p", "P", "*", "h", "H", "+",
                        "x", "X", "D", "d", "|", "_", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

        self.p = self.results["confidence-interval"]
        self.confidence_intervals = self.results["graph-confidence-interval"]
        self.dateTime = self.results["include-date-time"]
        self.image_format = self.results["image-format"]

        plt.rc("font", size=14)
        plt.rc("axes", titlesize=14)
        plt.rc("axes", labelsize=14)
        plt.rc("xtick", labelsize=14)
        plt.rc("ytick", labelsize=14)
        plt.rc("legend", fontsize=14)
        plt.rc("figure", titlesize=14)

    def generate_graphs(self, result_folders, now):

        self.logger.info("Beginning graphing of result files: {}".format(result_folders))

        if not self.config["processed-result-dir"]:
            self.config["processed-result-dir"] = self.prepare_results(result_folders, now)

        for graph_title in self.results["graph-configurations"]:
            self.logger.info("Graphing configuration: {}".format(graph_title))
            folders_for_comparison = []
            graph_info = self.results["graph-configurations"][graph_title]
            for config_name in graph_info["config_name"]:
                for folder in self.config["processed-result-dir"]:
                    configuration = folder.split("/")[-1][:-20]
                    if configuration == config_name:
                        folders_for_comparison.append(folder)

            for graph_type in self.results["graphs"]:
                if graph_type in ["PDR-SCI", "PDR-TB", "PDR-A", "PDR-P", "IPG", "Arrivals", "Collisions"]:
                    self.distance_graph(folders_for_comparison, graph_type, graph_title, graph_info, now)
                elif graph_type == "PeriodicVAperiodic":
                    self.traffic_graph(folders_for_comparison, graph_type, graph_title, graph_info, now)
                elif graph_type == "CBR":
                    self.cbr_graph(folders_for_comparison, graph_type, graph_title, graph_info, now)
                elif graph_type == "CBR-PSCCH":
                    self.cbr_pscch_graph(folders_for_comparison, graph_type, graph_title, graph_info, now)
                elif graph_type == "deltaCol":
                    self.delta_col(folders_for_comparison, graph_type, graph_title, graph_info, now)
                elif graph_type == "deltaColPeriodicBreakdown":
                    self.delta_col_traffic_pattern(folders_for_comparison, graph_type, graph_title, graph_info, now)
                elif graph_type == "Errors":
                    self.errors_dist(folders_for_comparison, graph_type, graph_title, graph_info, now)
                elif graph_type == "ErrorsPeriodicBreakDown":
                    self.errors_dist_traffic_pattern(folders_for_comparison, graph_type, graph_title, graph_info, now)
                elif graph_type == "GrantBreaks":
                    self.grant_break_graph(folders_for_comparison, graph_type, graph_title, graph_info, now)
                elif graph_type == "ResourceOccupancy":
                    self.resource_occupancy_graph(folders_for_comparison, graph_type, graph_title, graph_info, now)

    def prepare_results(self, result_folders, now):

        num_processes = self.config["parallel_processes"]
        if num_processes > multiprocessing.cpu_count():
            self.logger.warning("Too many processes, going to revert to total - 1")
            num_processes = multiprocessing.cpu_count() - 1

        processed_results = []
        for folder in result_folders:
            config_name = folder.split("/")[-1][:-20]
            self.logger.info("Results for config: {}".format(config_name))
            folder_results = []
            files = natsort.natsorted(os.listdir(folder))

            filtered_files = []
            for i in range(len(files)):
                # Ensures we don't load files passed by accident
                if ".csv" in files[i]:
                    filtered_files.append("{}/{}".format(folder, files[i]))

            i = 0
            while i < len(filtered_files):
                if len(filtered_files) < num_processes:
                    num_processes = len(filtered_files)
                pool = multiprocessing.Pool(processes=num_processes)

                folder_results.append(pool.starmap(self.generate_results, zip(filtered_files[i: i + num_processes])))

                pool.close()
                pool.join()

                i += num_processes

            folder_results = [y for x in folder_results for y in x]
            # Go through each of the available stats and write them out to a csv file.
            output_csv_dir = "{}/data/processed_data/{}/{}-{}".format(os.getcwd(), self.experiment_type,
                                                                      config_name, now)

            os.makedirs(output_csv_dir, exist_ok=True)

            goodTransmissions = []
            unusedTransmissions = []
            natural_grant_breaks = []

            # Shortcut ensures we get the stats from the parsed results
            for stat in folder_results[0]:
                if stat == "CBR":
                    self.across_run_results_cbr(folder_results, output_csv_dir)
                elif stat == "CBR-PSCCH":
                    self.across_run_results_cbrPscch(folder_results, output_csv_dir)
                elif stat == "GrantBreaks":
                    grant_breaks = []
                    for i in range(len(folder_results)):
                        grant_breaks.append(folder_results[i][stat])
                        df = pd.DataFrame({"GrantBreaks": grant_breaks})
                        df.to_csv("{}/{}.csv".format(output_csv_dir, stat), index=False)
                # elif stat == "Collisions":
                #     self.across_run_results(folder_results, stat, output_csv_dir, "CollisionDistance")
                elif stat == "GoodTransmissions":
                    for i in range(len(folder_results)):
                        goodTransmissions.append(folder_results[i][stat])
                elif stat == "UnusedTransmissions":
                    for i in range(len(folder_results)):
                        unusedTransmissions.append(folder_results[i][stat])
                elif stat == "NaturalGrantBreaks":
                    for i in range(len(folder_results)):
                        natural_grant_breaks.append(folder_results[i][stat])
                else:
                    self.across_run_results(folder_results, stat, output_csv_dir, "Distance")

            good_trans_mean = statistics.mean(goodTransmissions)
            good_trans_dev = statistics.stdev(goodTransmissions)
            unused_trans_mean = statistics.mean(unusedTransmissions)
            unused_trans_dev = statistics.stdev(unusedTransmissions)
            natural_grant_breaks_mean = statistics.mean(natural_grant_breaks)
            natural_grant_breaks_dev = statistics.stdev(natural_grant_breaks)
            total_resources = 3 * 12 * 1000

            df = pd.DataFrame(
                {
                    "GoodTransmissions_Mean": good_trans_mean,
                    "UnusedTransmissions_Mean": unused_trans_mean,
                    "NaturalGrantBreaks_Mean": natural_grant_breaks_mean,
                    "GoodTransmissions_std": good_trans_dev,
                    "UnusedTransmissions_std": unused_trans_dev,
                    "NaturalGrantBreaks_std": natural_grant_breaks_dev,
                    "TotalResources": total_resources
                }, index=[0]
            )
            df.to_csv("{}/{}.csv".format(output_csv_dir, "resource_usage"), index=False)

            processed_results.append(output_csv_dir)

        self.logger.info("Folders processed: {}".format(processed_results))
        return processed_results

    def generate_results(self, output_csv):

        self.logger.info("Generating results for file: {}".format(output_csv))

        results = {}

        pdr_sci_agg = pd.DataFrame()
        pdr_tb_agg = pd.DataFrame()
        pdr_a_agg = pd.DataFrame()
        pdr_p_agg = pd.DataFrame()
        # pdr_tb_ignore_sci_agg = pd.DataFrame()
        ipg_agg = pd.DataFrame()
        cbr_agg = pd.DataFrame()
        cbrPscch_agg = pd.DataFrame()
        arrivals_agg = pd.DataFrame()
        unsensed_errors = pd.DataFrame()
        hd_errors = pd.DataFrame()
        prop_errors = pd.DataFrame()
        interference_errors = pd.DataFrame()

        aperiodic_unsensed_errors = pd.DataFrame()
        aperiodic_hd_errors = pd.DataFrame()
        aperiodic_prop_errors = pd.DataFrame()
        aperiodic_interference_errors = pd.DataFrame()

        periodic_unsensed_errors = pd.DataFrame()
        periodic_hd_errors = pd.DataFrame()
        periodic_prop_errors = pd.DataFrame()
        periodic_interference_errors = pd.DataFrame()
        # collisions_agg = pd.DataFrame()

        total_grant_breaks = 0
        good_transmission = 0
        unused_transmissions = 0
        natural_grant_breaks = 0

        error_dfs = {}
        aperiodic_errors = {}
        periodic_errors = {}
        # Need a new for loop through all the errors and adding them as a stat distance
        for error in self.results["errors"]:
            error_dfs[error] = pd.DataFrame()
            aperiodic_errors[error] = pd.DataFrame()
            periodic_errors[error] = pd.DataFrame()

        for chunk in pd.read_csv(output_csv, chunksize=10 ** 6):

            chunk = chunk[chunk["Time"] >= 500]
            if chunk.empty:
                continue

            # SCI PDR calculation
            pdr_sci_agg = self.stat_distance(pdr_sci_agg, chunk, "sciDecoded", "txRxDistanceSCI", True)

            # TB PDR calculation
            pdr_tb_agg = self.stat_distance(pdr_tb_agg, chunk, "tbDecoded", "txRxDistanceTB", True)

            aperiodic_chunk = chunk[chunk["periodic"] == 0]
            pdr_a_agg = self.stat_distance(pdr_a_agg, aperiodic_chunk, "tbDecoded", "txRxDistanceTB", True)

            periodic_chunk = chunk[chunk["periodic"] == 1]
            pdr_p_agg = self.stat_distance(pdr_p_agg, periodic_chunk, "tbDecoded", "txRxDistanceTB", True)

            # pdr_tb_ignore_sci_agg = self.stat_distance(pdr_tb_agg, chunk, "tbDecodedIgnoreSCI", "txRxDistanceTB", True)

            # collisions_agg = self.collisions_distance(collisions_agg, chunk)

            chunk_good_transmission, chunk_unused_transmissions, chunk_natural_grant_breaks = self.resource_usage(chunk)
            good_transmission += chunk_good_transmission
            unused_transmissions += chunk_unused_transmissions
            natural_grant_breaks += chunk_natural_grant_breaks

            arrivals_agg = self.total_distance(arrivals_agg, chunk, "tbDecoded", "txRxDistanceTB")

            # IPG calculation
            ipg_agg = self.stat_distance(ipg_agg, chunk, "interPacketDelay", "txRxDistanceTB", False)

            # CBR calculation doesn't aggregate the same way as the above so dealt with separatel
            cbr_df = chunk[chunk["cbr"].notnull()]
            cbr_df = cbr_df[["Time", "cbr"]]
            cbr_df = cbr_df[cbr_df["cbr"].notnull()]

            if cbr_agg.empty:
                cbr_agg = cbr_df
            else:
                cbr_agg = cbr_agg.append(cbr_df)

            # CBRPSCCH calculation doesn't aggregate the same way as the above so dealt with separatel
            cbrPscch_df = chunk[chunk["cbrPscch"].notnull()]
            cbrPscch_df = cbrPscch_df[["Time", "cbrPscch"]]
            cbrPscch_df = cbrPscch_df[cbrPscch_df["cbrPscch"].notnull()]

            if cbrPscch_agg.empty:
                cbrPscch_agg = cbrPscch_df
            else:
                cbrPscch_agg = cbrPscch_agg.append(cbrPscch_df)

            # Deal with the grant breaking
            for col in self.results["grantBreaking"]:
                total_grant_breaks += chunk[col].sum()

            chunk = chunk[chunk["tbReceived"] != -1]
            for error in error_dfs:
                if "sci" in error[0:3]:
                    error_dfs[error] = self.stat_distance(error_dfs[error], chunk, error, "txRxDistanceSCI", True)
                else:
                    error_dfs[error] = self.stat_distance(error_dfs[error], chunk, error, "txRxDistanceTB", True)

            aperiodic_loss = aperiodic_chunk[aperiodic_chunk["tbReceived"] != -1]
            for error in aperiodic_errors:
                if "sci" in error[0:3]:
                    aperiodic_errors[error] = self.stat_distance(aperiodic_errors[error], aperiodic_loss, error, "txRxDistanceSCI", True)
                else:
                    aperiodic_errors[error] = self.stat_distance(aperiodic_errors[error], aperiodic_loss, error, "txRxDistanceTB", True)

            periodic_loss = periodic_chunk[periodic_chunk["tbReceived"] != -1]
            for error in periodic_errors:
                if "sci" in error[0:3]:
                    periodic_errors[error] = self.stat_distance(periodic_errors[error], aperiodic_loss, error, "txRxDistanceSCI", True)
                else:
                    periodic_errors[error] = self.stat_distance(periodic_errors[error], periodic_loss, error, "txRxDistanceTB", True)

        results["PDR-SCI"] = pdr_sci_agg
        results["PDR-TB"] = pdr_tb_agg
        results["PDR-A"] = pdr_a_agg
        results["PDR-P"] = pdr_p_agg
        # results["PDR-IGNORE-SCI"] = pdr_tb_ignore_sci_agg
        results["IPG"] = ipg_agg
        results["CBR"] = cbr_agg
        results["CBR-PSCCH"] = cbrPscch_agg
        results["Arrivals"] = arrivals_agg
        results["GrantBreaks"] = total_grant_breaks
        # results["Collisions"] = collisions_agg
        results["GoodTransmissions"] = good_transmission
        results["UnusedTransmissions"] = unused_transmissions
        results["NaturalGrantBreaks"] = natural_grant_breaks

        for key, df in zip(["unsensed_errors", "hd_errors", "prop_errors", "interference_errors"],
                           [unsensed_errors, hd_errors, prop_errors, interference_errors]):
            for error in self.results[key]:
                if df.empty:
                    df = error_dfs[error]
                else:
                    # Combine mean errors
                    df["mean"] = df["mean"] + error_dfs[error]["mean"]

            results[key] = df

        for key, df in zip(["aperiodic_unsensed_errors", "aperiodic_hd_errors",
                            "aperiodic_prop_errors", "aperiodic_interference_errors"],
                           [aperiodic_unsensed_errors, aperiodic_hd_errors,
                            aperiodic_prop_errors, aperiodic_interference_errors]):

            sub_key = key

            if "unsensed" in key:
                sub_key = "unsensed_errors"
            elif "hd" in key:
                sub_key = "hd_errors"
            elif "prop" in key:
                sub_key = "prop_errors"
            else:
                sub_key = "interference_errors"

            for error in self.results[sub_key]:
                if df.empty:
                    df = aperiodic_errors[error]
                else:
                    # Combine mean errors
                    df["mean"] = df["mean"] + aperiodic_errors[error]["mean"]

                results[key] = df

        for key, df in zip(["periodic_unsensed_errors", "periodic_hd_errors",
                            "periodic_prop_errors", "periodic_interference_errors"],
                           [periodic_unsensed_errors, periodic_hd_errors,
                            periodic_prop_errors, periodic_interference_errors]):

            sub_key = key

            if "unsensed" in key:
                sub_key = "unsensed_errors"
            elif "hd" in key:
                sub_key = "hd_errors"
            elif "prop" in key:
                sub_key = "prop_errors"
            else:
                sub_key = "interference_errors"

            for error in self.results[sub_key]:
                if df.empty:
                    df = periodic_errors[error]
                else:
                    # Combine mean errors
                    df["mean"] = df["mean"] + periodic_errors[error]["mean"]

                results[key] = df

        return results

    def stat_distance(self, agg_df, df, stat, distance, percentage):

        # Reduce the size of the DF to what we're interested in.
        distance_df = df[df[stat].notnull()]
        # distance_df = distance_df[(distance_df["posX"] > 0) & (distance_df["posX"] < 2000)]
        # distance_df = distance_df[(distance_df["Time"] > 198)]
        distance_df = distance_df[["Time", "NodeID", stat, distance]]
        distance_df = distance_df[distance_df[stat] > -1]
        distance_df = distance_df.rename(columns={"Time": "Time", "NodeID": "NodeID", stat: stat, distance: "Distance"})

        # Only interested in max 500m simply as it's not all that relevant to go further.
        # Note that going to the max distance of the file can cause issues with how they are parsed.
        max_distance = min(530, distance_df["Distance"].max())

        # Get the mean, std, count for each distance
        distance_df = distance_df.groupby(
            pd.cut(distance_df["Distance"], np.arange(0, max_distance, 10))).agg(
            {stat: [np.mean, "count"]})

        # Remove over head column
        distance_df.columns = distance_df.columns.droplevel()

        if percentage:
            distance_df = distance_df.apply(lambda x: x * 100, axis=1)

        if agg_df.empty:
            agg_df = distance_df
        else:
            # combine_chunks
            agg_df = pd.merge(agg_df, distance_df, on="Distance", how='outer')
            agg_df = agg_df.apply(self.combine_line, axis=1, result_type='expand')
            agg_df = agg_df.rename({0: "mean", 1: "count"}, axis='columns')

        return agg_df

    def collisions_distance(self, agg_df, df):

        df = df[['NodeID', 'Time', 'grantStartTime', 'selectedSubchannelIndex',
                 'selectedNumSubchannels', 'posX', 'posY']]

        df = df.dropna(subset=['grantStartTime', 'selectedSubchannelIndex', 'selectedNumSubchannels', 'posX', 'posY'],
                       how='all')

        df = df.drop_duplicates()

        grant_df = df[["NodeID", "grantStartTime", "selectedSubchannelIndex"]]

        grant_df = grant_df.dropna(subset=['grantStartTime', 'selectedSubchannelIndex'], how='all')

        position_df = df[["NodeID", "Time", "posX", "posY"]]

        position_df = position_df.dropna(subset=['posX', 'posY'], how='all')
        position_df = position_df.drop_duplicates()

        result_df = pd.merge(grant_df, position_df, left_on=["NodeID", "grantStartTime"], right_on=["NodeID", "Time"],
                             how='left')

        result_df = result_df[["NodeID", "grantStartTime", "Time", "posX", "posY", "selectedSubchannelIndex"]]

        f = lambda x: pd.DataFrame(list(combinations(x.values, 2)),
                                   columns=['CoordA', 'CoordB'])

        result_df = result_df.dropna(how="any")

        if not result_df.empty:

            try:

                position_update_df = (result_df.groupby(['Time', 'selectedSubchannelIndex'])['posX', 'posY'].apply(f)
                                      .reset_index(level=1, drop=True)
                                      .reset_index())

                position_update_df[['X_a', 'Y_a']] = pd.DataFrame(position_update_df.CoordA.tolist(),
                                                                  index=position_update_df.index)
                position_update_df[['X_b', 'Y_b']] = pd.DataFrame(position_update_df.CoordB.tolist(),
                                                                  index=position_update_df.index)
                position_update_df = position_update_df.drop(['CoordA', 'CoordB'], axis=1)

                position_update_df = position_update_df.rename(columns={"level_1": "count"})

                position_update_df["CollisionDistance"] = np.sqrt(
                    np.square(position_update_df["X_a"] - position_update_df["X_b"]) +
                    np.square(position_update_df["Y_a"] - position_update_df["Y_b"]))

                position_update_df = position_update_df.drop(['X_a', 'Y_a', 'X_b', 'Y_b'], axis=1)
                position_update_df = position_update_df.reset_index(drop=True)

                max_distance = min(530, position_update_df["CollisionDistance"].max())
                position_update_df = position_update_df.groupby(
                    pd.cut(position_update_df["CollisionDistance"], np.arange(0, max_distance, 10))).agg({"count": "count"})

                if agg_df.empty:
                    agg_df = position_update_df
                else:
                    # combine_chunks
                    agg_df = pd.merge(agg_df, position_update_df, on="CollisionDistance", how='outer')
                    agg_df = agg_df.apply(self.count_line, axis=1, result_type='expand')
                    agg_df = agg_df.rename({0: "count"}, axis='columns')
            except (IndexError, ValueError) as e:
                self.logger.warning("Data frame could not be read fully for collisions, final time is: {}s".format(result_df["Time"].min()))
                self.logger.warning(e)
                return agg_df

        return agg_df

    def total_distance(self, agg_df, df, stat, distance):

        # Reduce the size of the DF to what we're interested in.
        distance_df = df[df[stat].notnull()]
        # distance_df = distance_df[(distance_df["posX"] > 0) & (distance_df["posX"] < 2000)]
        distance_df = distance_df[["Time", "NodeID", stat, distance]]
        distance_df = distance_df[distance_df[stat] > -1]
        distance_df = distance_df.rename(columns={"Time": "Time", "NodeID": "NodeID", stat: stat, distance: "Distance"})

        # Only interested in max 500m simply as it's not all that relevant to go further.
        # Note that going to the max distance of the file can cause issues with how they are parsed.
        max_distance = min(530, distance_df["Distance"].max())

        # Get the mean, std, count for each distance
        distance_df = distance_df.groupby(
            pd.cut(distance_df["Distance"], np.arange(0, max_distance, 10))).agg(
            {stat: ["sum"]})

        # Remove over head column
        distance_df.columns = distance_df.columns.droplevel()

        if agg_df.empty:
            agg_df = distance_df
        else:
            # combine_chunks
            agg_df = pd.merge(agg_df, distance_df, on="Distance", how='outer')
            agg_df = agg_df.apply(self.sum_line, axis=1, result_type='expand')
            agg_df = agg_df.rename({0: "sum"}, axis='columns')

        return agg_df

    def resource_usage(self, df):
        # This function determines the resource usage for chunk being processed.

        df = df[["NodeID", "Time", "grantBreak", "grantBreakMissedTrans", "grantBreakSize", "grantStartTime",
                 "selectedSubchannelIndex"]]

        df = df.dropna(subset=["grantBreak", "grantBreakMissedTrans", "grantBreakSize", "grantStartTime",
                               "selectedSubchannelIndex"], how="all")

        grant_description_df = df[df["grantStartTime"].notnull()]

        grant_description_df = grant_description_df[["NodeID", "grantStartTime",
                                                     "selectedSubchannelIndex"]].reset_index(drop=True)

        grant_breaking_df = df[["NodeID", "Time", "grantBreak", "grantBreakMissedTrans"]]

        grant_breaking_df = grant_breaking_df.dropna(subset=["grantBreak", "grantBreakMissedTrans"], how="all")

        grant_breaking_df["grantBreak"] = np.where(grant_breaking_df["grantBreakMissedTrans"].eq(1), 0,
                                                   grant_breaking_df["grantBreak"])

        combined_df = pd.merge(grant_description_df, grant_breaking_df, left_on=["NodeID"], right_on=["NodeID"],
                               how='right')

        combined_df = combined_df[(combined_df["grantStartTime"] < combined_df["Time"])]

        combined_df = combined_df.drop_duplicates(subset=["NodeID", "grantStartTime"], keep="first")

        combined_df = combined_df.reset_index(drop=True)

        combined_df = combined_df.rename(columns={"Time": "grantBreakTime"})

        combined_df["grantTransmissions"] = ((combined_df["grantBreakTime"] - combined_df["grantStartTime"]) // 0.1)

        # Each grant break from a missed transmission is ultimately an unused resource which was shown as occupied.
        count_unused_transmissions = combined_df["grantBreakMissedTrans"].sum()
        count_good_transmissions = combined_df["grantTransmissions"].sum()
        natural_grant_breaks = combined_df["grantBreak"].sum()

        return count_good_transmissions, count_unused_transmissions, natural_grant_breaks


    @staticmethod
    def sum_line(line):
        sum_a = line["sum_x"]

        sum_b = line["sum_y"]

        if np.isnan(sum_a) and np.isnan(sum_b):
            return [sum_a]
        elif np.isnan(sum_a) and not np.isnan(sum_b):
            return [sum_b]
        elif np.isnan(sum_b) and not np.isnan(sum_a):
            return [sum_a]
        else:
            overall_sum = sum_a + sum_b

            return [overall_sum]

    @staticmethod
    def count_line(line):
        sum_a = line["count_x"]

        sum_b = line["count_y"]

        if np.isnan(sum_a) and np.isnan(sum_b):
            return [sum_a]
        elif np.isnan(sum_a) and not np.isnan(sum_b):
            return [sum_b]
        elif np.isnan(sum_b) and not np.isnan(sum_a):
            return [sum_a]
        else:
            overall_sum = sum_a + sum_b

            return [overall_sum]

    @staticmethod
    def combine_line(line):
        mean_a = line["mean_x"]
        count_a = line["count_x"]

        mean_b = line["mean_y"]
        count_b = line["count_y"]

        if np.isnan(mean_a) and np.isnan(mean_b):
            return [mean_a, count_a]
        elif np.isnan(mean_a) and not np.isnan(mean_b):
            return [mean_b, count_b]
        elif np.isnan(mean_b) and not np.isnan(mean_a):
            return [mean_a, count_a]
        else:
            ex_a = mean_a * count_a
            ex_b = mean_b * count_b

            tx = ex_a + ex_b
            tn = count_a + count_b

            overall_mean = tx / tn
            overall_count = tn

            return [overall_mean, overall_count]

    def across_run_results(self, results, stat, output_csv_dir, merge_col):

        df = pd.DataFrame()
        self.logger.info("Statistic of interest: {}".format(stat))

        aperiodic = False
        periodic = False
        if stat == "aperiodic_unsensed_errors":
            stat = "unsensed_errors"
            aperiodic = True
        elif stat == "periodic_unsensed_errors":
            stat = "unsensed_errors"
            periodic = True
        elif stat == "aperiodic_hd_errors":
            stat = "hd_errors"
            aperiodic = True
        elif stat == "periodic_hd_errors":
            stat = "hd_errors"
            periodic = True
        elif stat == "aperiodic_prop_errors":
            stat = "prop_errors"
            aperiodic = True
        elif stat == "periodic_prop_errors":
            stat = "prop_errors"
            periodic = True
        elif stat == "aperiodic_interference_errors":
            stat = "interference_errors"
            aperiodic = True
        elif stat == "periodic_interference_errors":
            stat = "interference_errors"
            periodic = True

        for i in range(len(results)):
            if df.empty:
                df = results[i][stat]
            else:
                df = pd.merge(df, results[i][stat], how='outer', on=merge_col,
                              suffixes=(i, i + 1),
                              copy=True, indicator=False)

        if stat == "Arrivals":
            mean_cols = df.filter(regex='sum').columns
        elif stat == "Collisions":
            mean_cols = df.filter(regex='count').columns
        else:
            mean_cols = df.filter(regex='mean').columns

        n = len(mean_cols) - 1
        t_value = t.ppf(self.p, n)

        df = df.apply(self.combine_runs, axis=1, result_type='expand', args=(mean_cols, t_value,))
        df = df.rename({0: "Mean", 1: "Confidence-Interval"}, axis='columns')
        if aperiodic:
            df.to_csv("{}/aperiodic-{}.csv".format(output_csv_dir, stat))
        elif periodic:
            df.to_csv("{}/periodic-{}.csv".format(output_csv_dir, stat))
        else:
            df.to_csv("{}/{}.csv".format(output_csv_dir, stat))

    def across_run_results_cbr(self, results, output_csv_dir):
        earliest_time = float("inf")
        latest_time = -float("inf")

        raw_cbr_df = pd.DataFrame()
        for folder in results:

            start_time = folder["CBR"]["Time"].min()
            if start_time < earliest_time:
                earliest_time = start_time

            end_time = folder["CBR"]["Time"].max()
            if end_time > latest_time:
                latest_time = end_time

            if raw_cbr_df.empty:
                raw_cbr_df = folder["CBR"]
            else:
                raw_cbr_df.append(folder["CBR"])

        self.logger.debug("Earliest time: {}s Latest time: {}s".format(earliest_time, latest_time))

        cbr_df = pd.DataFrame(columns=["Mean", "Time", "Confidence-Interval"])
        last_time = earliest_time
        for i in np.arange(earliest_time, latest_time, 0.1):
            subsection_df = pd.DataFrame()
            for folder in results:
                df = folder["CBR"]
                if subsection_df.empty:
                    subsection_df = df[(df["Time"] < i) & (df["Time"] >= last_time) & (df["cbr"].notnull())]
                else:
                    subsection_df.append(df[(df["Time"] < i) & (df["Time"] >= last_time) & (df["cbr"].notnull())])

            last_time = i

            cbr_df = cbr_df.append({"Mean": subsection_df["cbr"].mean(),
                                    "Time": (i + last_time) / 2,
                                    "Confidence-Interval": subsection_df["cbr"].std()
                                    }, ignore_index=True)

        cbr_df.to_csv("{}/CBR.csv".format(output_csv_dir), index=False)
        raw_cbr_df.to_csv("{}/raw-CBR.csv".format(output_csv_dir), index=False)

    def across_run_results_cbrPscch(self, results, output_csv_dir):
        earliest_time = float("inf")
        latest_time = -float("inf")

        raw_cbr_df = pd.DataFrame()
        for folder in results:

            start_time = folder["CBR-PSCCH"]["Time"].min()
            if start_time < earliest_time:
                earliest_time = start_time

            end_time = folder["CBR-PSCCH"]["Time"].max()
            if end_time > latest_time:
                latest_time = end_time

            if raw_cbr_df.empty:
                raw_cbr_df = folder["CBR-PSCCH"]
            else:
                raw_cbr_df.append(folder["CBR-PSCCH"])

        self.logger.debug("Earliest time: {}s Latest time: {}s".format(earliest_time, latest_time))

        cbr_df = pd.DataFrame(columns=["Mean", "Time", "Confidence-Interval"])
        last_time = earliest_time
        for i in np.arange(earliest_time, latest_time, 0.1):
            subsection_df = pd.DataFrame()
            for folder in results:
                df = folder["CBR-PSCCH"]
                if subsection_df.empty:
                    subsection_df = df[(df["Time"] < i) & (df["Time"] >= last_time) & (df["cbrPscch"].notnull())]
                else:
                    subsection_df.append(df[(df["Time"] < i) & (df["Time"] >= last_time) & (df["cbrPscch"].notnull())])

            last_time = i

            cbr_df = cbr_df.append({"Mean": subsection_df["cbrPscch"].mean(),
                                    "Time": (i + last_time) / 2,
                                    "Confidence-Interval": subsection_df["cbrPscch"].std()
                                    }, ignore_index=True)

        cbr_df.to_csv("{}/CBR-PSCCH.csv".format(output_csv_dir), index=False)
        raw_cbr_df.to_csv("{}/raw-CBR-PSCCH.csv".format(output_csv_dir), index=False)

    @staticmethod
    def combine_runs(line, mean_cols, t_value):
        means = []
        for mean in mean_cols:
            means.append(line[mean])

        n = len(means)

        # Average Across runs
        xBar = sum(means) / n

        # Deviation between runs and average
        deviation = []
        for mean in means:
            deviation.append((mean - xBar) ** 2)
        s2 = sum(deviation) / (n - 1)

        # Confidence interval
        ci = t_value * math.sqrt(s2 / n)

        return [xBar, ci]

    ### Graphing utilities

    def traffic_graph(self, folders, graph_type, graph_title, graph_info, now):
        means = []
        cis = []
        distances = []
        labels = []
        linestyles = []
        markers = []

        orig_labels = graph_info["labels"]
        orig_linestyles = graph_info["linestyles"]
        orig_markers = graph_info["markers"]

        for i in range(len(folders)):
            if "periodic-012vpm" in folders[i]:
                df = pd.read_csv("{}/PDR-P.csv".format(folders[i]))
                means.append(list(df["Mean"]))
                if self.confidence_intervals:
                    cis.append(list(df["Confidence-Interval"]))
            # elif any(substring in folders[i] for substring in ["10pc", "25pc", "50pc"]):
            #     for csv_file_name in ["PDR-A", "PDR-P"]:
            #         df = pd.read_csv("{}/{}.csv".format(folders[i], csv_file_name))
            #         distances = (list(range(0, df.shape[0] * 10, 10)))
            #         means.append(list(df["Mean"]))
            #         if self.confidence_intervals:
            #             cis.append(list(df["Confidence-Interval"]))
            else:
                for csv_file_name in ["PDR-A", "PDR-P"]:
                    df = pd.read_csv("{}/{}.csv".format(folders[i], csv_file_name))

                    if csv_file_name == "PDR-A":
                        labels.append("{} - {}".format(graph_info["labels"][i], "Aperiodic"))
                    else:
                        labels.append("{} - {}".format(graph_info["labels"][i], "Periodic"))

                    markers.append(graph_info["markers"][i])
                    linestyles.append(graph_info["linestyles"][i])
                    distances = (list(range(0, df.shape[0] * 10, 10)))
                    means.append(list(df["Mean"]))
                    if self.confidence_intervals:
                        cis.append(list(df["Confidence-Interval"]))

        graph_info["means"] = means
        graph_info["cis"] = cis
        graph_info["labels"] = labels
        graph_info["linestyles"] = linestyles
        graph_info["markers"] = markers

        self.dist_graph_traffic(distances, graph_info, "{}-{}".format(graph_title, graph_type),
                                ylabel="Packet Delivery Rate %", now=now,
                                confidence_intervals=self.confidence_intervals,
                                show=False, store=True, percentage=True)

        graph_info["means"] = []
        graph_info["cis"] = []
        graph_info["labels"] = orig_labels
        graph_info["linestyles"] = orig_linestyles
        graph_info["markers"] = orig_markers

    def dist_graph_traffic(self, distances, graph_info, plot_name, ylabel, now, legend_pos="lower left",
                           confidence_intervals=None, show=True, store=False, percentage=False, error=False,
                           delta_col=False):

        fig, ax = plt.subplots()

        for i in range(len(graph_info["labels"])):
            if confidence_intervals:
                ax.errorbar(distances, graph_info["means"][i], yerr=graph_info["cis"][i],
                            label="{}".format(graph_info["labels"][i]),
                            fillstyle="none", marker=graph_info["markers"][i], markevery=5,
                            color=graph_info["periodic-colors"][i], linestyle=graph_info["linestyles"][i])
            else:
                ax.plot(distances, graph_info["means"][i],
                        label="{}".format(graph_info["labels"][i],),
                        fillstyle="none", marker=graph_info["markers"][i], markevery=5,
                        color=graph_info["periodic-colors"][i], linestyle=graph_info["linestyles"][i])

        ax.set(xlabel='Distance (m)', ylabel=ylabel)

        handles, labels = plt.gca().get_legend_handles_labels()
        newLabels, newHandles = [], []
        for handle, label in zip(handles, labels):
            if label not in newLabels:
                newLabels.append(label)
                newHandles.append(handle)

        if error:
            l4 = plt.legend(newHandles, newLabels, bbox_to_anchor=(0.02, 0.86, 0.96, 0.04), loc=legend_pos,
                            borderaxespad=0, ncol=2, mode="expand")
        elif delta_col:
            l4 = plt.legend(newHandles, newLabels, bbox_to_anchor=(0.02, 0.96, 0.96, 0.04), loc=legend_pos,
                            borderaxespad=0, ncol=3, mode="expand")
        else:
            l4 = plt.legend(newHandles, newLabels, loc=legend_pos, handlelength=1,
                            borderaxespad=0, ncol=3, bbox_to_anchor=(0.02, 0.02, 0.96, 0.02))

        ax.tick_params(direction='in')

        ax.set_xlim([0, 500])
        plt.xticks(np.arange(0, (max(distances) + 1), step=50))

        if percentage:
            ax.set_ylim([0, 100])
            plt.yticks(np.arange(0, 101, step=10))

        plt.grid(b=True, color="#d1d1d1")

        if show:
            fig.show()

        if store:
            if self.dateTime:
                fig.savefig("{}/{}-{}.{}".format(self.figure_store, plot_name, now, self.image_format), dpi=400)
            else:
                fig.savefig("{}/{}.{}".format(self.figure_store, plot_name, self.image_format), dpi=400)
        plt.close(fig)

    def distance_graph(self, folders, graph_type, graph_title, graph_info, now):
        means = []
        cis = []
        distances = []
        for folder in folders:
            df = pd.read_csv("{}/{}.csv".format(folder, graph_type))
            means.append(list(df["Mean"]))
            if self.confidence_intervals:
                cis.append(list(df["Confidence-Interval"]))
            distances = (list(range(0, df.shape[0] * 10, 10)))

        graph_info["means"] = means
        graph_info["cis"] = cis

        if graph_type in ["PDR-SCI", "PDR-TB", "PDR-A", "PDR-P"]:
            self.dist_graph(distances, graph_info, "{}-{}".format(graph_title, graph_type),
                            ylabel="Packet Delivery Rate %", now=now, confidence_intervals=self.confidence_intervals,
                            show=False, store=True, percentage=True)
        elif graph_type == "IPG":
            self.dist_graph(distances, graph_info, "{}-{}".format(graph_title, graph_type),
                            ylabel="Inter-Packet Gap (ms)", now=now, legend_pos="upper left",
                            confidence_intervals=self.confidence_intervals, error=True, show=False, store=True)
        elif graph_type == "Arrivals":
            self.dist_graph(distances, graph_info, "{}-{}".format(graph_title, graph_type),
                            ylabel="Total Decoded Packets", now=now, legend_pos="upper right",
                            confidence_intervals=self.confidence_intervals, show=False, store=True)
        elif graph_type == "Collisions":
            self.dist_graph(distances, graph_info, "{}-{}".format(graph_title, graph_type),
                            ylabel="Total Colliding Grants", now=now, legend_pos="upper right",
                            confidence_intervals=self.confidence_intervals, show=False, store=True)

    def cbr_graph(self, folders, graph_type, graph_title, graph_info, now):
        # Might change this to time based graph but CBR is fine for now
        times = []
        cbrs = []
        cis = []
        box_plot_data = []
        for folder in folders:
            df = pd.read_csv("{}/CBR.csv".format(folder))
            cbr = list(df["Mean"])
            ci = list(df["Confidence-Interval"])

            # Transform 0-1 to 0-100
            for i in range(len(cbr)):
                cbr[i] = cbr[i] * 100
                ci[i] = ci[i] * 100

            times.append(list(df["Time"]))
            cbrs.append(cbr)
            cis.append(ci)

            cbr_csv = "{}/raw-CBR.csv".format(folder)
            df = pd.read_csv(cbr_csv)
            # filtered_df = df[df["Time"] > 502]
            box_plot_data.append(100 * df["cbr"])

        graph_info["means"] = cbrs
        graph_info["times"] = times
        graph_info["cis"] = cis
        graph_info["boxplotData"] = box_plot_data

        self.cbr_plot(graph_info, "{}-{}".format(graph_title, graph_type), now=now,
                      confidence_intervals=self.confidence_intervals, show=False, store=True)

        # self.box_plot(graph_info, "{}-{}".format(graph_title, graph_type), now=now, ylabel="Channel Busy Ratio %",
        #               percentage=True, show=False, store=True)

    def cbr_pscch_graph(self, folders, graph_type, graph_title, graph_info, now):
        # Might change this to time based graph but CBR is fine for now
        times = []
        cbrs = []
        cis = []
        box_plot_data = []
        for folder in folders:
            df = pd.read_csv("{}/CBR-PSCCH.csv".format(folder))
            cbr = list(df["Mean"])
            ci = list(df["Confidence-Interval"])

            # Transform 0-1 to 0-100
            for i in range(len(cbr)):
                cbr[i] = cbr[i] * 100
                ci[i] = ci[i] * 100

            times.append(list(df["Time"]))
            cbrs.append(cbr)
            cis.append(ci)

            cbr_csv = "{}/raw-CBR-PSCCH.csv".format(folder)
            df = pd.read_csv(cbr_csv)
            #filtered_df = df[df["Time"] > 502]
            box_plot_data.append(100 * df["cbrPscch"])

        graph_info["means"] = cbrs
        graph_info["times"] = times
        graph_info["cis"] = cis
        graph_info["boxplotData"] = box_plot_data

        self.cbr_plot(graph_info, "{}-{}".format(graph_title, graph_type), now=now,
                      confidence_intervals=self.confidence_intervals, show=False, store=True)

        # self.box_plot(graph_info, "{}-{}".format(graph_title, graph_type), now=now, ylabel="Channel Busy Ratio %",
        #               percentage=True, show=False, store=True)

    def grant_break_graph(self, folders, graph_type, graph_title, graph_info, now):
        # Might change this to time based graph but CBR is fine for now
        box_plot_data = []
        for folder in folders:
            df = pd.read_csv("{}/GrantBreaks.csv".format(folder))
            box_plot_data.append(df["GrantBreaks"])

        graph_info["boxplotData"] = box_plot_data

        self.box_plot(graph_info, "{}-{}".format(graph_title, graph_type), now=now, ylabel="Total Grant Breaks",
                      show=False, store=True)

    def resource_occupancy_graph(self, folders, graph_type, graph_title, graph_info, now, show=False, store=True):

        natural_grant_breaks = []
        missed_transmissions = []
        good_transmissions = []
        for folder in folders:
            df = pd.read_csv("{}/resource_usage.csv".format(folder))
            natural_grant_breaks.append(df["NaturalGrantBreaks_Mean"].sum())
            missed_transmissions.append(df["UnusedTransmissions_Mean"].sum())
            good_transmissions.append(df["GoodTransmissions_Mean"].sum())

        fig, ax = plt.subplots()

        labels = graph_info["labels"]

        bar_width = 0.15  # the width of the bars

        r1 = np.arange(len(labels))
        r2 = [x + bar_width for x in r1]
        r3 = [x + bar_width for x in r2]

        ax.bar(r1, missed_transmissions, bar_width - .01, label='Unused')
        ax.bar(r2, good_transmissions,   bar_width - .01, label='Correctly Used')
        ax.bar(r3, natural_grant_breaks, bar_width - .01, label='Vacated')

        ax.set_ylabel('Resources')
        ax.set_title('Densities')
        plt.xticks([r + bar_width for r in range(len(labels))], labels)
        ax.set_xticklabels(labels)
        ax.legend()

        fig.tight_layout()

        if show:
            fig.show()

        if store:
            if self.dateTime:
                fig.savefig("{}/{}-{}.{}".format(self.figure_store, "{}-{}".format(graph_title, graph_type), now,
                                                 self.image_format), dpi=400)
            else:
                fig.savefig("{}/{}.{}".format(self.figure_store, "{}-{}".format(graph_title, graph_type),
                                              self.image_format))
        plt.close(fig)

    def dist_graph(self, distances, graph_info, plot_name, ylabel, now, legend_pos="lower left",
                   confidence_intervals=None, show=True, store=False, percentage=False, error=False, delta_col=False):
        fig, ax = plt.subplots()

        for i in range(len(graph_info["labels"])):
            if confidence_intervals:
                ax.errorbar(distances, graph_info["means"][i], yerr=graph_info["cis"][i],
                            label=graph_info["labels"][i],
                            fillstyle="none", marker=graph_info["markers"][i], markevery=5,
                            color=graph_info["colors"][i], linestyle=graph_info["linestyles"][i])
            else:
                ax.plot(distances, graph_info["means"][i], label=graph_info["labels"][i],
                        fillstyle="none", marker=graph_info["markers"][i], markevery=5,
                        color=graph_info["colors"][i], linestyle=graph_info["linestyles"][i])

        ax.set(xlabel='Distance (m)', ylabel=ylabel)

        handles, labels = plt.gca().get_legend_handles_labels()
        newLabels, newHandles = [], []
        for handle, label in zip(handles, labels):
            if label not in newLabels:
                newLabels.append(label)
                newHandles.append(handle)

        if error:
            l4 = plt.legend(newHandles, newLabels, bbox_to_anchor=(0.02, 0.94, 0.96, 0.04), loc=legend_pos,
                            borderaxespad=0, ncol=2, mode="expand")
        elif delta_col:
            l4 = plt.legend(newHandles, newLabels, bbox_to_anchor=(0.02, 0.94, 0.96, 0.04), loc=legend_pos,
                            borderaxespad=0, ncol=1, mode="expand")
        else:
            l4 = plt.legend(newHandles, newLabels, loc=legend_pos, handlelength=1,
                            borderaxespad=0, ncol=2, bbox_to_anchor=(0.01, 0.02, 0.96, 0.02))

        ax.tick_params(direction='in')

        ax.set_xlim([0, 500])
        plt.xticks(np.arange(0, (max(distances) + 1), step=50))

        if percentage:
            ax.set_ylim([0, 100])
            plt.yticks(np.arange(0, 101, step=10))

        plt.grid(b=True, color="#d1d1d1")

        if show:
            fig.show()

        if store:
            if self.dateTime:
                fig.savefig("{}/{}-{}.{}".format(self.figure_store, plot_name, now, self.image_format), dpi=400)
            else:
                fig.savefig("{}/{}.{}".format(self.figure_store, plot_name, self.image_format), dpi=400)
        plt.close(fig)

    def cbr_plot(self, graph_info, plot_name, now, confidence_intervals=None, show=True, store=False):

        fig, ax = plt.subplots()

        for i in range(len(graph_info["config_name"])):
            if confidence_intervals:
                ax.errorbar(graph_info["times"][i], graph_info["means"][i], yerr=graph_info["cis"][i],
                            label=graph_info["labels"][i],
                            fillstyle="none", color=graph_info["colors"][i], linestyle=graph_info["linestyles"][i])
            else:
                ax.plot(graph_info["times"][i], graph_info["means"][i], label=graph_info["labels"][i],
                        marker=graph_info["markers"][i], markevery=5, fillstyle="none",
                        color=graph_info["colors"][i], linestyle=graph_info["linestyles"][i])

        # ax.plot(graph_info["times"][0], [29] * len(graph_info["times"][0]), fillstyle="none",
        #         color="Black", linestyle="dashed")
        #
        # ax.plot(graph_info["times"][0], [51] * len(graph_info["times"][0]), fillstyle="none",
        #         color="Red", linestyle="dashed")
        #
        # ax.plot(graph_info["times"][0], [73] * len(graph_info["times"][0]), fillstyle="none",
        #         color="Blue", linestyle="dashed")
        #
        # ax.plot(graph_info["times"][0], [86] * len(graph_info["times"][0]), fillstyle="none",
        #         color="Green", linestyle="dashed")

        handles, labels = plt.gca().get_legend_handles_labels()
        newLabels, newHandles = [], []
        for handle, label in zip(handles, labels):
            if label not in newLabels:
                newLabels.append(label)
                newHandles.append(handle)


        l4 = plt.legend(newHandles, newLabels, loc="upper left", handlelength=1,
                        borderaxespad=0, ncol=2, bbox_to_anchor=(0.06, 0.96, 0.02, 0.02))

        # ax.legend(loc='upper left')
        ax.set(xlabel='Time (s)', ylabel='Channel Busy Ratio %')
        ax.tick_params(direction='in')

        ax.set_ylim([0, 100])
        plt.yticks(np.arange(0, 101, step=10))
        plt.grid(b=True, color="#d1d1d1")

        if show:
            fig.show()

        if store:
            if self.dateTime:
                fig.savefig("{}/{}-{}.{}".format(self.figure_store, plot_name, now, self.image_format), dpi=400)
            else:
                fig.savefig("{}/{}.{}".format(self.figure_store, plot_name, self.image_format), dpi=400)
        plt.close(fig)

    def box_plot(self, graph_info, plot_name, now, ylabel=None, percentage=False, show=False, store=True):

        fig, ax = plt.subplots()
        ax.boxplot(graph_info["boxplotData"], labels=graph_info["labels"])

        ax.set(xlabel='Scenario', ylabel=ylabel)
        ax.tick_params(direction='in')

        if percentage:
            ax.set_ylim([0, 100])
            plt.yticks(np.arange(0, 101, step=10))

        plt.grid(b=True, color="#d1d1d1")

        if show:
            fig.show()

        if store:
            if self.dateTime:
                fig.savefig("{}/{}-Box-{}.{}".format(self.figure_store, plot_name, now, self.image_format), dpi=400)
            else:
                fig.savefig("{}/{}-Box.{}".format(self.figure_store, plot_name, self.image_format))
        plt.close(fig)

    def errors_dist(self, folders, graph_type, graph_title, graph_info, now):
        means = []
        cis = []
        distances = []
        labels = []
        markers = []
        colors = []
        linestyles = []

        orig_labels = graph_info["labels"]
        orig_colors = graph_info["colors"]
        orig_linestyles = graph_info["linestyles"]
        orig_markers = graph_info["markers"]

        errors = ["hd_errors", "unsensed_errors", "prop_errors", "interference_errors"]

        for i in range(len(errors)):
            for j in range(len(folders)):
                df = pd.read_csv("{}/{}.csv".format(folders[j], errors[i]))
                means.append(list(df["Mean"]))
                if self.confidence_intervals:
                    cis.append(list(df["Confidence-Interval"]))
                distances = (list(range(0, 520, 10)))
                if errors[i] == "hd_errors":
                    labels.append(r'Half Duplex Errors, $\delta_{HD}$')
                elif errors[i] == "interference_errors":
                    labels.append(r'Packet Collisions, $\delta_{COL}$')
                elif errors[i] == "unsensed_errors":
                    labels.append(r'Sensing Errors, $\delta_{SEN}$')
                elif errors[i] == "prop_errors":
                    labels.append(r'Propagation Errors, $\delta_{PRO}$')
                linestyles.append(graph_info["linestyles"][j])
                markers.append(graph_info["error-markers"][i])
                colors.append(graph_info["error-colors"][i])

        graph_info["means"] = means
        graph_info["cis"] = cis
        graph_info["labels"] = labels
        graph_info["markers"] = markers
        graph_info["colors"] = colors
        graph_info["linestyles"] = linestyles

        self.dist_graph(distances, graph_info,
                        "{}-{}".format(graph_title, graph_type), ylabel="Packet Loss Attribution %", now=now,
                        confidence_intervals=cis, show=False, store=True, percentage=True, legend_pos="upper left",
                        error=True)

        graph_info["means"] = []
        graph_info["cis"] = []
        graph_info["labels"] = orig_labels
        graph_info["linestyles"] = orig_linestyles
        graph_info["colors"] = orig_colors
        graph_info["markers"] = orig_markers

    def errors_dist_traffic_pattern(self, folders, graph_type, graph_title, graph_info, now):
        means = []
        cis = []
        distances = []
        labels = []
        markers = []
        colors = []
        linestyles = []

        orig_labels = graph_info["labels"]
        orig_colors = graph_info["colors"]
        orig_linestyles = graph_info["linestyles"]
        orig_markers = graph_info["markers"]

        errors = ["hd_errors", "unsensed_errors", "prop_errors", "interference_errors"]

        for traffic_pattern in ["aperiodic", "periodic"]:
            for i in range(len(errors)):
                for j in range(len(folders)):
                    df = pd.read_csv("{}/{}-{}.csv".format(folders[j], traffic_pattern, errors[i]))
                    means.append(list(df["Mean"]))
                    if self.confidence_intervals:
                        cis.append(list(df["Confidence-Interval"]))
                    distances = (list(range(0, 520, 10)))
                    if errors[i] == "hd_errors":
                        labels.append(r'Half Duplex Errors, $\delta_{HD}$')
                    elif errors[i] == "interference_errors":
                        labels.append(r'Packet Collisions, $\delta_{COL}$')
                    elif errors[i] == "unsensed_errors":
                        labels.append(r'Sensing Errors, $\delta_{SEN}$')
                    elif errors[i] == "prop_errors":
                        labels.append(r'Propagation Errors, $\delta_{PRO}$')
                    linestyles.append(graph_info["linestyles"][j])
                    markers.append(graph_info["error-markers"][i])
                    colors.append(graph_info["error-colors"][i])

                graph_info["means"] = means
                graph_info["cis"] = cis
                graph_info["labels"] = labels
                graph_info["markers"] = markers
                graph_info["colors"] = colors
                graph_info["linestyles"] = linestyles

                self.dist_graph(distances, graph_info,
                                "{}-{}-{}".format(graph_title, traffic_pattern, graph_type),
                                ylabel="Packet Loss Attribution %", now=now,
                                confidence_intervals=cis, show=False, store=True, percentage=True, legend_pos="upper left",
                                error=True)

            graph_info["means"] = []
            graph_info["cis"] = []
            graph_info["labels"] = orig_labels
            graph_info["linestyles"] = orig_linestyles
            graph_info["colors"] = orig_colors
            graph_info["markers"] = orig_markers

    def delta_col(self, folders, graph_type, graph_title, graph_info, now):
        means = []
        cis = []
        distances = []
        labels = []
        markers = []
        colors = []
        linestyles = []

        orig_labels = graph_info["labels"]
        orig_colors = graph_info["colors"]
        orig_linestyles = graph_info["linestyles"]
        orig_markers = graph_info["markers"]

        for i in range(len(folders)):
            df = pd.read_csv("{}/interference_errors.csv".format(folders[i]))
            means.append(list(df["Mean"]))
            if self.confidence_intervals:
                cis.append(list(df["Confidence-Interval"]))
            distances = (list(range(0, 520, 10)))
            labels.append(r'$\delta_{COL}$' + ': {}'.format(graph_info["labels"][i]))
            linestyles.append(graph_info["linestyles"][i])
            markers.append(graph_info["error-markers"][i])
            colors.append(graph_info["error-colors"][i])

        graph_info["means"] = means
        graph_info["cis"] = cis
        graph_info["labels"] = labels
        graph_info["markers"] = markers
        graph_info["colors"] = colors
        graph_info["linestyles"] = linestyles

        self.dist_graph(distances, graph_info,
                        "{}-{}".format(graph_title, graph_type), ylabel="Packet Loss - Collisions %", now=now,
                        confidence_intervals=cis, show=False, store=True, percentage=True, legend_pos="upper left",
                        error=False, delta_col=True)

        graph_info["means"] = []
        graph_info["cis"] = []
        graph_info["labels"] = orig_labels
        graph_info["linestyles"] = orig_linestyles
        graph_info["colors"] = orig_colors
        graph_info["markers"] = orig_markers

    def delta_col_traffic_pattern(self, folders, graph_type, graph_title, graph_info, now):
        means = []
        cis = []
        distances = []
        labels = []
        markers = []
        colors = []
        linestyles = []

        orig_labels = graph_info["labels"]
        orig_colors = graph_info["colors"]
        orig_linestyles = graph_info["linestyles"]
        orig_markers = graph_info["markers"]

        for traffic_pattern in ["aperiodic", "periodic"]:
            for i in range(len(folders)):
                df = pd.read_csv("{}/{}-interference_errors.csv".format(folders[i], traffic_pattern))
                means.append(list(df["Mean"]))
                if self.confidence_intervals:
                    cis.append(list(df["Confidence-Interval"]))
                distances = (list(range(0, 520, 10)))
                labels.append(r'$\delta_{COL}$' + ': {}'.format(graph_info["labels"][i]))
                linestyles.append(graph_info["linestyles"][i])
                markers.append(graph_info["error-markers"][i])
                colors.append(graph_info["error-colors"][i])

            graph_info["means"] = means
            graph_info["cis"] = cis
            graph_info["labels"] = labels
            graph_info["markers"] = markers
            graph_info["colors"] = colors
            graph_info["linestyles"] = linestyles

            self.dist_graph(distances, graph_info,
                            "{}-{}-{}".format(graph_title, traffic_pattern, graph_type),
                            ylabel="Packet Loss - Collisions %", now=now,
                            confidence_intervals=cis, show=False, store=True, percentage=True, legend_pos="upper left",
                            error=False, delta_col=True)

            graph_info["means"] = []
            graph_info["cis"] = []
            graph_info["labels"] = orig_labels
            graph_info["linestyles"] = orig_linestyles
            graph_info["colors"] = orig_colors
            graph_info["markers"] = orig_markers