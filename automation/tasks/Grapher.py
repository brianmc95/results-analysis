import logging
import json
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

        # TODO: Add option to change this
        self.p = .95

    def generate_graphs(self, results_file, now):

        self.logger.info("Beginning graphing of result file: {}".format(results_file))

        self.prepare_results(results_file, now)

        exit(0)

        with open(results_file) as results_json:
            data = json.load(results_json)

        comparison_graphs = []
        individual_graphs = []

        pdr_field = self.results["decoded"].replace(":vector", "")

        for individual in self.results["individual"]:
            self.logger.info("Config we are graphing {}".format(individual))
            for graph in self.results["graphs"]:

                plot_name = "{}-{}-{}-{}".format(self.experiment_type, individual, graph, now)
                self.logger.info("Generating graph: {}".format(plot_name))

                if graph == "pdr-dist":
                    self.pdr_dist_individual(data[individual][pdr_field], data[individual]["distance"],
                                             "PDR", plot_name)

                elif graph == "error-dist":
                    errors = []
                    for error in self.results["fails"]:
                        errors.append(data[individual][error])

                    self.errors_dist_individual(data[individual]["distance"], data[individual][pdr_field],
                                                errors, self.results["error_labels"], plot_name)

                individual_graphs.append("{}.{}".format(plot_name, self.image_format))

        for compare in self.results["compare"]:
            self.logger.info("Comparing {}".format(self.results["compare"][compare]))
            for graph in self.results["graphs"]:

                plot_name = "{}-{}-{}-{}".format(self.experiment_type, compare, graph, now)
                self.logger.info("Generating graph: {}".format(plot_name))

                if graph == "pdr-dist":
                    pdrs = []
                    labels = []
                    distances = []

                    for config in self.results["compare"][compare]:
                        labels.append("PDR: {}".format(config))
                        pdrs.append(data[config][pdr_field])
                        distances = data[config]["distance"]

                    self.pdr_dist(pdrs, distances, labels, plot_name)

                elif graph == "error-dist":

                    decoded = []
                    decoded_labels = []
                    error_labels = []
                    distances = []
                    errors = []

                    for config in self.results["compare"][compare]:
                        decoded.append(data[config][pdr_field])
                        decoded_labels.append("{}: decoded".format(config))
                        distances = data[config]["distance"]

                        sub_errors = []
                        sub_error_labels = []
                        for i in range(len(self.results["fails"])):
                            sub_error_labels.append("{}: {}".format(config, self.results["error_labels"][i]))
                            sub_errors.append(data[config][self.results["fails"][i]])
                        errors.append(sub_errors)
                        error_labels.append(sub_error_labels)

                    self.errors_dist(distances, decoded, decoded_labels, errors, error_labels, plot_name)

                comparison_graphs.append("{}.{}".format(plot_name, self.image_format))

        return individual_graphs, comparison_graphs

    def pdr_dist_individual(self, pdr, distances, label, plot_name):
        fig, ax = plt.subplots()

        if self.use_markers:
            ax.plot(distances, pdr, label=label, marker=self.markers[0], markevery=3)

        elif self.use_line_types:
            # TODO: figure out the line types thing
            ax.plot(distances, pdr, label=label)

        else:
            ax.plot(distances, pdr, label=label)

        ax.set(xlabel='Distance (m)', ylabel='Packet Delivery Rate (PDR) %')
        ax.legend(loc='lower right')
        ax.grid()

        ax.set_ylim([0, 1.1])
        plt.yticks(np.arange(0, 1, step=.1))

        ax.set_xlim([0, (max(distances) + 1)])
        plt.xticks(np.arange(0, (max(distances) + 1), step=50))

        fig.suptitle(plot_name, fontsize=12)
        fig.savefig("{}/individual/{}.{}".format(self.figure_store, plot_name, self.image_format))

    def errors_dist_individual(self, distances, decoded, errors, error_labels, plot_name):

        fig, ax = plt.subplots()

        if self.use_markers:
            ax.plot(distances, decoded, label="Decoded", marker=self.markers[0], markevery=3)

            for i in range(len(errors)):
                ax.plot(distances, errors[i], label=error_labels[i], marker=self.markers[i+1], markevery=3)

        elif self.use_line_types:
            ax.plot(distances, decoded, label="Decoded")

            for i in range(len(errors)):
                ax.plot(distances, errors[i], label=error_labels[i])

        else:
            ax.plot(distances, decoded, label="Decoded")

            for i in range(len(errors)):
                ax.plot(distances, errors[i], label=error_labels[i])

        ax.legend(loc='center left')

        ax.set(xlabel='Distance (m)', ylabel='Packet Delivery Rate (PDR) %')
        ax.grid()

        ax.set_ylim([0, 1.1])
        plt.yticks(np.arange(0, 1.1, step=.1))

        ax.set_xlim([0, (max(distances) + 1)])
        plt.xticks(np.arange(0, (max(distances) + 1), step=50))

        fig.suptitle(plot_name, fontsize=12)
        fig.savefig("{}/individual/{}.{}".format(self.figure_store, plot_name, self.image_format))

    def pdr_dist(self, pdrs, distances, labels, plot_name):

        fig, ax = plt.subplots()

        if self.use_markers:
            for i in range(len(pdrs)):
                ax.plot(distances, pdrs[i], label=labels[i], marker=self.markers[i], markevery=3)

        elif self.use_line_types:
            # TODO: figure out the line types thing
            for i in range(len(pdrs)):
                ax.plot(distances, pdrs[i], label=labels[i])

        else:
            for i in range(len(pdrs)):
                ax.plot(distances, pdrs[i], label=labels[i])

        ax.set(xlabel='Distance (m)', ylabel='Packet Delivery Rate (PDR) %')
        ax.legend(loc='lower right')
        ax.grid()

        ax.set_ylim([0, 1.1])
        plt.yticks(np.arange(0, 1.1, step=.1))

        ax.set_xlim([0, (max(distances) + 1)])
        plt.xticks(np.arange(0, (max(distances) + 1), step=50))

        fig.suptitle(plot_name, fontsize=12)
        fig.savefig("{}/comparison/{}.{}".format(self.figure_store, plot_name, self.image_format))
        plt.close(fig)

    def errors_dist(self, distances, decoded, decoded_labels, errors, error_labels, plot_name):

        fig, ax = plt.subplots()

        if self.use_markers:
            for i in range(len(decoded)):
                ax.plot(distances, decoded[i], label=decoded_labels[i], marker=self.markers[i], markevery=3)

                for j in range(len(errors[i])):
                    ax.plot(distances, errors[i][j], label=error_labels[i][j], marker=self.markers[i + j])

        elif self.use_line_types:
            for i in range(len(decoded)):
                ax.plot(distances, decoded[i], label=decoded_labels[i])

                for j in range(len(errors[i])):
                    ax.plot(distances, errors[i][j], label=error_labels[i][j])

        else:
            for i in range(len(decoded)):
                ax.plot(distances, decoded[i], label=decoded_labels[i])

                for j in range(len(errors[i])):
                    ax.plot(distances, errors[i][j], label=error_labels[i][j])

        ax.legend(loc='center left')

        ax.set(xlabel='Distance (m)', ylabel='Packet Delivery Rate (PDR) %')
        ax.grid()

        ax.set_ylim([0, 1])
        plt.yticks(np.arange(0, 1.1, step=.1))

        ax.set_xlim([0, (max(distances) + 1)])
        plt.xticks(np.arange(0, (max(distances) + 1), step=50))

        fig.savefig("{}/comparison/{}.{}".format(self.figure_store, plot_name, self.image_format))
        plt.close(fig)

    def prepare_results(self, results, now):

        num_processes = self.config["parallel_processes"]
        if num_processes > multiprocessing.cpu_count():
            self.logger.warning("Too many processes, going to revert to total - 1")
            num_processes = multiprocessing.cpu_count() - 1

        for folder in results:
            config_name = folder.split("/")[-1]
            self.logger.debug("Generating results from folder: {}".format(folder))
            self.logger.info("Results for config: {}".format(config_name))
            folder_results = []
            files = natsort.natsorted(os.listdir(folder))

            for i in range(len(files)):
                files[i] = "{}/{}".format(folder, files[i])

            i = 0
            while i < len(files):
                if len(files) < num_processes:
                    num_processes = len(files)
                pool = multiprocessing.Pool(processes=num_processes)

                folder_results.append(pool.starmap(self.generate_results, zip(files[i: i+num_processes])))

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
                if "SCI" in stat:
                    self.across_run_results(folder_results, stat, output_csv_dir, "txRxDistanceSCI")
                elif stat == "CBR":
                    self.across_run_results(folder_results, stat, output_csv_dir, "Time")
                else:
                    self.across_run_results(folder_results, stat, output_csv_dir, "txRxDistanceTB")

    def generate_results(self, output_csv):

        self.logger.info("Generating results for file: {}".format(output_csv))

        results = {}

        pdr_sci_agg = pd.DataFrame()
        pdr_tb_agg = pd.DataFrame()
        ipg_agg = pd.DataFrame()
        cbr_agg = pd.DataFrame()

        for chunk in pd.read_csv(output_csv, chunksize=10 ** 6):

            # SCI PDR calculation
            pdr_sci_agg = self.stat_distance(pdr_sci_agg, chunk, "sciDecoded", "txRxDistanceSCI")

            # TB PDR calculation
            pdr_tb_agg = self.stat_distance(pdr_tb_agg, chunk, "tbDecoded", "txRxDistanceTB")

            # IPG calculation
            ipg_agg = self.stat_distance(ipg_agg, chunk, "interPacketDelay", "txRxDistanceTB")

            # CBR calculation doesn't aggregate the same way as the above so dealt with separately
            cbr_df = chunk[chunk["cbr"].notnull()]
            cbr_df = cbr_df[["Time", "cbr"]]
            cbr_df = cbr_df.groupby("Time").agg({"cbr": [np.mean, np.std, "count"]})
            cbr_df.columns = cbr_df.columns.droplevel()

            if cbr_agg.empty:
                cbr_agg = cbr_df
            else:
                # combine_chunks
                cbr_agg = cbr_agg.append(cbr_df)

        results["PDR-SCI"] = pdr_sci_agg
        results["PDR-TB"] = pdr_tb_agg
        results["IPG"] = ipg_agg
        results["CBR"] = cbr_agg

        return results

    def stat_distance(self, agg_df, df, stat, distance):

        # Reduce the size of the DF to what we're interested in.
        distance_df = df[df[stat].notnull()]
        distance_df = distance_df[["Time", "NodeID", stat, distance]]
        distance_df = distance_df[distance_df[stat] > -1]

        max_distance = distance_df[distance].max()

        # Get the mean, std, count for each distance
        distance_df = distance_df.groupby(
            pd.cut(distance_df[distance], np.arange(0, max_distance, 10))).agg(
            {stat: [np.mean, np.std, "count"]})

        # Remove over head column
        distance_df.columns = distance_df.columns.droplevel()

        if agg_df.empty:
            agg_df = distance_df
        else:
            # combine_chunks
            agg_df = pd.merge(agg_df, distance_df, on=distance, how='outer')
            agg_df = agg_df.apply(self.combine_line, axis=1, result_type='expand')
            agg_df = agg_df.rename({0: "mean", 1: "std", 2: "count"}, axis='columns')

        return agg_df

    @staticmethod
    def combine_line(line):
        mean_a = line["mean_x"]
        std_a = line["std_x"]
        count_a = line["count_x"]

        mean_b = line["mean_y"]
        std_b = line["std_y"]
        count_b = line["count_y"]

        ex_a = mean_a * count_a
        ex_b = mean_b * count_b
        ex_squared_a = ((std_a ** 2) * (count_a - 1)) + ((ex_a ** 2) / count_a)
        ex_squared_b = ((std_b ** 2) * (count_b - 1)) + ((ex_b ** 2) / count_b)

        tx = ex_a + ex_b
        txx = ex_squared_a + ex_squared_b
        tn = count_a + count_b

        overall_mean = tx / tn
        overall_std = math.sqrt((txx - tx ** 2 / tn) / (tn - 1))
        overall_count = tn

        return [overall_mean, overall_std, overall_count]

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
