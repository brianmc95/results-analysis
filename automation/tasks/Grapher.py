import numpy as np
import logging

import matplotlib.pyplot as plt


class Grapher:

    def __init__(self, config, markers=True, line_types=False, figure_store="figures", image_format="png"):

        self.use_markers = markers
        self.config = config
        self.use_line_types = line_types
        self.image_format = image_format
        self.figure_store = figure_store
        self.results = self.config["results"]
        self.logger = logging.getLogger("multi-process")

        # TODO: Possibly find a prefered order for these
        self.markers = [".", "o", "v", "^", "<", ">", "1", "2", "3", "4", "8", "s", "p", "P", "*", "h", "H", "+",
                        "x", "X", "D", "d", "|", "_", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    def pdr_dist(self, pdrs, distances, labels, plot_name):

        fig, ax = plt.subplots()

        if self.use_markers:
            for pdr, label, marker in pdrs, labels, self.markers:
                ax.plot(distances, pdr, label=label, marker=marker, markevery=3)

        elif self.use_line_types:
            # TODO: figure out the line types thing
            for pdr, label in pdrs, labels:
                ax.plot(distances, pdr, label=label)

        else:
            for pdr, label in pdrs, labels:
                ax.plot(distances, pdr, label=label)

        ax.set(xlabel='Distance (m)', ylabel='Packet Delivery Rate (PDR) %')
        ax.legend(loc='lower right')
        ax.grid()

        ax.set_ylim([0, 100])
        plt.yticks(np.arange(0, 101, step=10))

        ax.set_xlim([0, (max(distances) + 1)])
        plt.xticks(np.arange(0, (max(distances) +1), step=50))

        fig.savefig("{}/{}.{}".format(self.figure_store, plot_name, self.image_format))

    def errors_dist(self, distances, decoded, errors, error_labels, plot_name):

        fig, ax = plt.subplots()

        if self.use_markers:
            ax.plot(distances, decoded, label="Decoded", marker=self.markers[0], markevery=3)

            for error, error_label, marker in errors, error_labels, self.markers[1:]:
                ax.plot(distances, error, label=error_label, marker='+', markevery=3)

        elif self.use_line_types:
            ax.plot(distances, decoded, label="Decoded")

            for error, error_label in errors, error_labels:
                ax.plot(distances, error, label=error_label)

        else:
            ax.plot(distances, decoded, label="Decoded")

            for error, error_label in errors, error_labels:
                ax.plot(distances, error, label=error_label)

        ax.legend(loc='center left')

        ax.set(xlabel='Distance (m)', ylabel='Packet Delivery Rate (PDR) %')
        ax.grid()

        ax.set_ylim([0, 100])
        plt.yticks(np.arange(0, 101, step=10))

        ax.set_xlim([0, (max(distances) + 1)])
        plt.xticks(np.arange(0, (max(distances) + 1), step=50))

        fig.savefig("{}/{}.{}".format(self.figure_store, plot_name, self.image_format))
        plt.show()
