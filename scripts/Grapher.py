import pandas as pd
import numpy as np
import json

import matplotlib.pyplot as plt


class Grapher:

    def __init__(self, fields, markers=True, line_types=False, figure_store="figures", image_format="png"):

        self.use_markers = markers
        self.use_line_types = line_types
        self.image_format = image_format
        self.figure_store = figure_store

        # TODO: Possibly find a prefered order for these
        self.markers = [".", "o", "v", "^", "<", ">", "1", "2", "3", "4", "8", "s", "p", "P", "*", "h", "H", "+",
                        "x", "X", "D", "d", "|", "_", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]


        # JSON file to describe fields of interest
        with open(fields) as fields_file:
            self.fields = json.load(fields_file)


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

    def pdr_calc(self, df):
        """
        Calculates Packet Delivery Ratio for a DataFrame based on fields provided in self.fields JSON file.
        :param df: DataFrame which we will calculate df from.
        :return: new_df which included pdr in it.
        """

        new_df = pd.DataFrame()

        decoded = df[df["name"] == self.fields["decoded"]]
        decoded = decoded.reset_index(drop=True)
        dist = df[df["name"] == self.fields["distance"]]
        dist = dist.reset_index(drop=True)

        new_df["time"] = dist["vectime"]
        new_df["distance"] = dist["vecvalue"]
        new_df["decoded"] = decoded["vecvalue"]
        new_df["node"] = dist["node"]

        for i in range(len(self.fields["Fails"])):
            fail_df = df[df["name"] == self.fields["Fails"][i]]
            fail_df = fail_df.reset_index(drop=True)
            new_df[self.fields[self.fields["Fails"][i]]] = fail_df["vecvalue"]
            new_df["total_fails"] += fail_df

        new_df["pdr"] = ((new_df["decoded"]) / (df["decoded"] + df["total_fails"])) * 100

        return new_df

    def bin_pdr(self, dfs, bin_width=10, bin_quantity=49):
        """
        Bins multiple dfs pdr into a single pdr that can be used as an average
        :param dfs: list of dataframes to bin
        :param bin_width: width of each bin
        :param bin_quantity: total number of bins
        :return:
        """
        bins = self.create_bins(lower_bound=0, width=bin_width, quantity=bin_quantity)
        distances = []
        pdrs = []
        for interval in bins:
            upper_b = interval[1]
            distances.append(upper_b)

        for df in dfs:
            for i in range(len(bins)):
                lower_b = bins[i][0]
                upper_b = bins[i][1]
                pdr_temp = df[(df["distance"] >= lower_b) & (df["distance"] < upper_b)]
                if i < len(pdrs):
                    pdrs[i] = (pdr_temp["pdr"].mean() + pdrs[i]) / 2
                else:
                    pdrs.append(pdr_temp["pdr"].mean())

        return pdrs, distances

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
