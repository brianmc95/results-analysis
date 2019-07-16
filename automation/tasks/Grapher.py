import numpy as np
import logging
import json
import os

import matplotlib.pyplot as plt


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

    def generate_graphs(self, results_file, now):

        self.logger.info("Beginning graphing of result file: {}".format(results_file))

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

        ax.set_ylim([0, 1])
        plt.yticks(np.arange(0, 1, step=.1))

        ax.set_xlim([0, (max(distances) + 1)])
        plt.xticks(np.arange(0, (max(distances) + 1), step=100))

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

        ax.set_ylim([0, 1])
        plt.yticks(np.arange(0, 1.1, step=.1))

        ax.set_xlim([0, (max(distances) + 1)])
        plt.xticks(np.arange(0, (max(distances) + 1), step=100))

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

        ax.set_ylim([0, 1])
        plt.yticks(np.arange(0, 1.1, step=.1))

        ax.set_xlim([0, (max(distances) + 1)])
        plt.xticks(np.arange(0, (max(distances) + 1), step=100))

        fig.suptitle(plot_name, fontsize=12)
        fig.savefig("{}/comparison/{}.{}".format(self.figure_store, plot_name, self.image_format))

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
        plt.xticks(np.arange(0, (max(distances) + 1), step=100))

        fig.savefig("{}/comparison/{}.{}".format(self.figure_store, plot_name, self.image_format))
