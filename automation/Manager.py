import json
import argparse
import os
import logging.config
import time
import datetime

from slackclient import SlackClient

from tasks.DataParser import DataParser
from tasks.ExperimentRunner import ExperimentRunner
from tasks.Grapher import Grapher
from tasks.Uploader import Uploader


class Manager:
    """
    Class designed to manage the whole process of automated experimentation and results analysis
    """

    def __init__(self, experiment_type, experiment=True, parse=True, graph=True, upload=True, verbose=False, channel=None):

        self.experiment_type = experiment_type
        self.experiment = experiment
        self.parse = parse
        self.graph = graph
        self.upload = upload

        # Assuming you are running from the root of the project instead, this can throw an error
        config_path = os.path.join(os.getcwd(), "configs/{}.json".format(self.experiment_type))

        self.setup_logging(verbose=verbose)
        self.logger = logging.getLogger(__name__)

        if verbose:
            self.logger.setLevel(logging.DEBUG)

        with open(config_path) as json_file:
            self.config = json.load(json_file)[self.experiment_type]

        self.slack_api_token = self.config["slack-api-token"]
        if self.slack_api_token != "":
            self.slack_client = SlackClient(self.slack_api_token)

        self.channel = channel

        self.phases = 0

        if self.experiment:
            self.runner = ExperimentRunner(self.config, self.experiment_type)
            self.phases += 1

        if self.parse:
            self.phases += 1
            self.parser = DataParser(self.config, self.experiment_type)

        if self.graph:
            self.grapher = Grapher(self.config, self.experiment_type)
            self.phases += 1

        if self.upload:
            self.uploader = Uploader(self.config, self.experiment_type)
            self.phases += 1

    def send_slack_message(self, message):
        # Sends the response back to the channel
        self.logger.info("Message {} to be sent to channel {}".format(message, self.channel))
        if self.channel and self.slack_api_token != "":
            self.slack_client.api_call(
                "chat.postMessage",
                channel=self.channel,
                text=message
            )

    @staticmethod
    def setup_logging(default_path='automation/logger/logging.json', default_level=logging.INFO, verbose=False):
        """
        Setup logging configuration
        """
        path = default_path
        if os.path.exists(path):
            with open(path, 'rt') as f:
                config = json.load(f)
                if verbose:
                    config["handlers"]["console"]["level"] = "DEBUG"
            logging.config.dictConfig(config)
        else:
            logging.basicConfig(level=default_level)

    def run(self):
        """
        Runs the experiment &/OR parses data &/OR graphs data
        :return:
        """
        current_phase = 1

        self.send_slack_message("Beginning experiment: {}".format(self.experiment_type))

        overall_start = time.time()

        now = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")

        if self.experiment:

            start = time.time()

            self.send_slack_message("Beginning Experiment phase, phase {} of {}".format(current_phase, self.phases))

            self.logger.info("Experiment option set, moving into start experiment")
            try:
                self.config["result-dirs"] = self.runner.start_experiment(now)
                self.logger.info("result-dirs: {}".format(self.config["result-dirs"]))
            except Exception as e:
                self.logger.error("Experiment failed with error: {}".format(e))
                self.send_slack_message("Experiment phase failed")
                return

            end = time.time()
            elapsed = round(end - start)
            self.send_slack_message("Experiment phase complete, phase {} of {} in {}s".format(current_phase,
                                                                                              self.phases, elapsed))
            current_phase += 1

        if self.parse:

            start = time.time()

            self.send_slack_message("Beginning Parse phase, phase {} of {}".format(current_phase, self.phases))

            self.logger.info("Parsing option set, moving to parse raw data")
            try:
                self.config["processed-results"] = self.parser.parse_data(self.config["result-dirs"], now)
                self.logger.info("processed-results: {}".format(self.config["processed-results"]))
            except Exception as e:
                self.logger.error("Parse failed with error: {}".format(e))
                self.send_slack_message("Parsing phase failed")
                return

            end = time.time()
            elapsed = round(end - start)
            self.send_slack_message("Parse phase complete, phase {} of {} in {}s".format(current_phase,
                                                                                         self.phases, elapsed))
            current_phase += 1

        if self.graph:

            start = time.time()

            self.send_slack_message("Beginning Graph phase, phase {} of {}".format(current_phase, self.phases))

            self.logger.info("Graph option set, moving into Graphing stage")
            try:
                individual_graphs, comparison_graphs = self.grapher.generate_graphs(self.config["processed-results"], now)
                self.config["figures"]["individual"] = individual_graphs
                self.config["figures"]["comparison"] = comparison_graphs
                self.logger.info("figures, individual: {}".format(individual_graphs))
                self.logger.info("figures, comparison: {}".format(comparison_graphs))
            except Exception as e:
                self.logger.error("Graph failed with error: {}".format(e))
                self.send_slack_message("graph phase failed")
                return

            end = time.time()
            elapsed = round(end - start)
            self.send_slack_message("Graph phase complete, phase {} of {} in {}s".format(current_phase,
                                                                                         self.phases, elapsed))
            current_phase += 1

        if self.upload:

            start = time.time()

            self.send_slack_message("Beginning Upload phase, phase {} of {}".format(current_phase, self.phases))

            self.logger.info("Uploading data from run")
            try:
                self.uploader.upload_results(self.config)
            except Exception as e:
                self.logger.error("Upload failed with error: {}".format(e))
                self.send_slack_message("Upload phase failed")
                return

            end = time.time()
            elapsed = round(end - start)
            self.send_slack_message("Upload phase complete, phase {} of {} in {}s".format(current_phase,
                                                                                          self.phases, elapsed))
            current_phase += 1

        overall_end = time.time()
        overall_elapsed = round(overall_end - overall_start)
        self.logger.info("Experiment {} complete".format(self.experiment_type))
        self.send_slack_message("Experiment {} complete in {}".format(self.experiment_type, overall_elapsed))

        return


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Retrieve results from simulation and store to raw_data')
    parser.add_argument("-e", "--experiment_type", help="Type of the experiment")
    parser.add_argument("-x", "--experiment", type=str2bool, default=True,  help="Run experiments")
    parser.add_argument("-p", "--parse", type=str2bool, default=True, help="Parse results into graphable format")
    parser.add_argument("-g", "--graph", type=str2bool, default=True, help="Graph results of experiment")
    parser.add_argument("-u", "--upload", type=str2bool, default=True, help="Upload results of experiment")
    parser.add_argument("-v", "--verbose", type=str2bool, default=False, help="Turn on debug logging level")
    args = parser.parse_args()

    manager = Manager(args.experiment_type, args.experiment, args.parse, args.graph,
                      args.upload, args.verbose, channel=None)

    manager.run()

