import logging
import os
import math
import multiprocessing

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
                if graph_type in ["PDR-SCI", "PDR-TB", "IPG"]:
                    self.distance_graph(folders_for_comparison, graph_type, graph_title, graph_info, now)
                elif graph_type == "CBR":
                    self.cbr_graph(folders_for_comparison, graph_type, graph_title, graph_info, now)
                elif graph_type == "Errors":
                    self.errors_dist(folders_for_comparison, graph_type, graph_title, graph_info, now)

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

            # Shortcut ensures we get the stats from the parsed results
            for stat in folder_results[0]:
                if stat == "CBR":
                    self.across_run_results(folder_results, stat, output_csv_dir, "Time")
                else:
                    self.across_run_results(folder_results, stat, output_csv_dir, "Distance")

            processed_results.append(output_csv_dir)

        self.logger.info("Folders processed: {}".format(processed_results))
        return processed_results

    def generate_results(self, output_csv):

        self.logger.info("Generating results for file: {}".format(output_csv))

        results = {}

        pdr_sci_agg = pd.DataFrame()
        pdr_tb_agg = pd.DataFrame()
        pdr_tb_ignore_sci_agg = pd.DataFrame()
        ipg_agg = pd.DataFrame()
        cbr_agg = pd.DataFrame()
        unsensed_errors = pd.DataFrame()
        hd_errors = pd.DataFrame()
        prop_errors = pd.DataFrame()
        interference_errors = pd.DataFrame()

        error_dfs = {}
        # Need a new for loop through all the errors and adding them as a stat distance
        for error in self.results["errors"]:
            error_dfs[error] = pd.DataFrame()

        for chunk in pd.read_csv(output_csv, chunksize=10 ** 6):

            # SCI PDR calculation
            pdr_sci_agg = self.stat_distance(pdr_sci_agg, chunk, "sciDecoded", "txRxDistanceSCI", True)

            # TB PDR calculation
            pdr_tb_agg = self.stat_distance(pdr_tb_agg, chunk, "tbDecoded", "txRxDistanceTB", True)

            pdr_tb_ignore_sci_agg = self.stat_distance(pdr_tb_agg, chunk, "tbDecodedIgnoreSCI", "txRxDistanceTB", True)

            # IPG calculation
            ipg_agg = self.stat_distance(ipg_agg, chunk, "interPacketDelay", "txRxDistanceTB", False)

            # CBR calculation doesn't aggregate the same way as the above so dealt with separately
            cbr_df = chunk[chunk["cbr"].notnull()]
            cbr_df = cbr_df[["Time", "cbr"]]
            cbr_df = cbr_df.groupby("Time").agg({"cbr": [np.mean, np.std, "count"]})
            cbr_df.columns = cbr_df.columns.droplevel()
            cbr_df = cbr_df.apply(lambda x: x * 100, axis=1)

            if cbr_agg.empty:
                cbr_agg = cbr_df
            else:
                # combine_chunks
                cbr_agg = cbr_agg.append(cbr_df)

            chunk = chunk[chunk["tbReceived"] != -1]
            for error in error_dfs:
                if "sci" in error[0:3]:
                    error_dfs[error] = self.stat_distance(error_dfs[error], chunk, error, "txRxDistanceSCI", True)
                else:
                    error_dfs[error] = self.stat_distance(error_dfs[error], chunk, error, "txRxDistanceTB", True)

        results["PDR-SCI"] = pdr_sci_agg
        results["PDR-TB"] = pdr_tb_agg
        results["PDR-IGNORE-SCI"] = pdr_tb_ignore_sci_agg
        results["IPG"] = ipg_agg
        results["CBR"] = cbr_agg

        for key, df in zip(["unsensed_errors", "hd_errors", "prop_errors", "interference_errors"],
                           [unsensed_errors, hd_errors, prop_errors, interference_errors]):
            for error in self.results[key]:
                if df.empty:
                    df = error_dfs[error]
                else:
                    # Combine mean errors
                    df["mean"] = df["mean"] + error_dfs[error]["mean"]

            results[key] = df

        return results

    def stat_distance(self, agg_df, df, stat, distance, percentage):

        # Reduce the size of the DF to what we're interested in.
        distance_df = df[df[stat].notnull()]
        distance_df = distance_df[(distance_df["posX"] > 0) & (distance_df["posX"] < 2000)]
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
        for i in range(len(results)):
            if df.empty:
                df = results[i][stat]
            else:
                df = pd.merge(df, results[i][stat], how='outer', on=merge_col,
                              suffixes=(i, i + 1),
                              copy=True, indicator=False)

        mean_cols = df.filter(regex='mean').columns

        n = len(mean_cols) - 1
        t_value = t.ppf(self.p, n)

        df = df.apply(self.combine_runs, axis=1, result_type='expand', args=(mean_cols, t_value,))
        df = df.rename({0: "Mean", 1: "Confidence-Interval"}, axis='columns')
        df.to_csv("{}/{}.csv".format(output_csv_dir, stat))

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

        if graph_type in ["PDR-SCI", "PDR-TB"]:
            self.dist_graph(distances, graph_info, "{}-{}".format(graph_title, graph_type),
                            ylabel="Packet Delivery Rate %", now=now, confidence_intervals=cis, show=False, store=True)
        elif graph_type == "IPG":
            self.dist_graph(distances, graph_info, "{}-{}".format(graph_title, graph_type),
                            ylabel="Inter-Packet Gap (ms)", now=now, legend_pos="upper left",
                            confidence_intervals=cis, show=False, store=True)

    def cbr_graph(self, folders, graph_type, graph_title, graph_info, now):
        # Might change this to time based graph but CBR is fine for now
        times = []
        cbr = []
        cis = []
        for folder in folders:
            df = pd.read_csv("{}/CBR.csv".format(folder))
            times.append(list(df["Time"]))
            cbr.append(list(df["Mean"]))
            if self.confidence_intervals:
                cis.append(list(df["Confidence-Interval"]))

        graph_info["means"] = cbr
        graph_info["times"] = times
        graph_info["cis"] = cis

        self.cbr_plot(graph_info, "{}-{}".format(graph_title, graph_type), now=now,
                      confidence_intervals=cis, show=False, store=True)

    def dist_graph(self, distances, graph_info, plot_name, ylabel, now, legend_pos="lower left",
                   confidence_intervals=None, show=True, store=False, percentage=False):
        fig, ax = plt.subplots()

        for i in range(len(graph_info["labels"])):
            if confidence_intervals:
                ax.errorbar(distances, graph_info["means"][i], yerr=confidence_intervals[i], label=graph_info["labels"][i],
                            fillstyle="none", marker=graph_info["markers"][i], markevery=5,
                            color=graph_info["colors"][i], linestyle=graph_info["linestyles"][i])
            else:
                ax.plot(distances, graph_info["means"][i], label=graph_info["labels"][i],
                        fillstyle="none", marker=graph_info["markers"][i], markevery=5,
                        color=graph_info["colors"][i], linestyle=graph_info["linestyles"][i])

        ax.set(xlabel='Distance (m)', ylabel=ylabel)
        ax.legend(loc=legend_pos)
        ax.tick_params(direction='in')

        ax.set_xlim([0, 500])
        plt.xticks(np.arange(0, (max(distances) + 1), step=50))

        if percentage:
            ax.set_ylim([0, 100])
            plt.yticks(np.arange(0, 101, step=10))

        plt.grid(b=True, alpha=0.5)

        if show:
            fig.show()

        if store:
            fig.savefig("{}/{}-{}.png".format(self.figure_store, plot_name, now), dpi=300)
        plt.close(fig)

    def cbr_plot(self, graph_info, plot_name, now, confidence_intervals=None, show=True, store=False):

        fig, ax = plt.subplots()

        for i in range(len(graph_info["config_name"])):
            if confidence_intervals:
                ax.errorbar(graph_info["times"][i], graph_info["means"][i], yerr=confidence_intervals[i],
                            label=graph_info["labels"][i],
                            fillstyle="none", color=graph_info["colors"][i], linestyle=graph_info["linestyles"][i])
            else:
                ax.plot(graph_info["times"][i], graph_info["means"][i], label=graph_info["labels"][i],
                        fillstyle="none", color=graph_info["colors"][i], linestyle=graph_info["linestyles"][i])

        ax.legend(loc='upper left')
        ax.set(xlabel='Time (s)', ylabel='Channel Busy Ratio %')
        ax.tick_params(direction='in')

        ax.set_ylim([0, 100])
        plt.yticks(np.arange(0, 101, step=10))
        plt.grid(b=True, alpha=0.5)

        if show:
            fig.show()

        if store:
            fig.savefig("{}/{}-{}.png".format(self.figure_store, plot_name, now), dpi=300)
        plt.close(fig)

    def errors_dist(self, folders, graph_type, graph_title, graph_info, now):
        means = []
        cis = []
        distances = []
        labels = []
        markers = []
        colors = []
        linestyles = []

        errors = ["hd_errors", "interference_errors", "unsensed_errors", "prop_errors"]

        for i in range(len(folders)):
            for j in range(len(errors)):
                df = pd.read_csv("{}/{}.csv".format(folders[i], errors[j]))
                means.append(list(df["Mean"]))
                if self.confidence_intervals:
                    cis.append(list(df["Confidence-Interval"]))
                distances = (list(range(0, 520, 10)))
                labels.append("{}-{}".format(graph_info["labels"][i], errors[j]))
                linestyles.append(graph_info["linestyles"][i])
                markers.append(graph_info["error-markers"][j])
                colors.append(graph_info["error-colors"][j])

        graph_info["means"] = means
        graph_info["cis"] = cis
        graph_info["labels"] = labels
        graph_info["markers"] = markers
        graph_info["colors"] = colors
        graph_info["linestyles"] = linestyles

        self.dist_graph(distances, graph_info,
                        "{}-{}".format(graph_title, graph_type), ylabel="Error Probability %", now=now,
                        confidence_intervals=cis, show=False, store=True, percentage=True, legend_pos="upper left")
