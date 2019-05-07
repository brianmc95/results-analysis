import json
import argparse
import os
import logging.config

from tasks.DataParser import DataParser
from tasks.ExperimentRunner import ExperimentRunner
from tasks.Grapher import Grapher


class Manager:
    """
    Class designed to manage the whole process of automated experimentation and results analysis
    """

    def __init__(self, experiment_type, experiment, scave, parse, graph):

        self.experiment_type = experiment_type
        self.experiment = experiment
        self.scave = scave
        self.parse = parse
        self.graph = graph

        # Assuming you are running from the root of the project instead, this can throw an error
        config_path = os.path.join(os.getcwd(), "configs/{}.json".format(self.experiment_type))
        self.setup_logging()

        self.logger = logging.getLogger(__name__)

        with open(config_path) as json_file:
            self.config = json.load(json_file)[self.experiment_type]

        if self.experiment:
            self.runner = ExperimentRunner(self.config, self.experiment_type)

        if self.parse or self.graph:
            self.parser = DataParser(self.config, self.experiment_type)

        if self.graph:
            self.grapher = Grapher(self.config, self.experiment_type)

    @staticmethod
    def setup_logging(default_path='automation/logger/logging.json', default_level=logging.INFO, env_key='LOG_CFG'):
        """
        Setup logging configuration
        """
        path = default_path
        value = os.getenv(env_key, None)
        if value:
            path = value
        if os.path.exists(path):
            with open(path, 'rt') as f:
                config = json.load(f)
            logging.config.dictConfig(config)
        else:
            logging.basicConfig(level=default_level)

    def run(self):
        """
        Runs the experiment &/OR parses data &/OR graphs data
        :return:
        """
        if self.experiment:
            self.logger.info("Experiment option set, moving into start experiment")
            self.config["result-dirs"] = self.runner.start_experiment()

        if self.scave:
            self.logger.info("Scave option set, moving to extract raw data")
            self.parser.extract_raw_data(self.config["result-dirs"])

        if self.parse:
            self.logger.info("Parsing option set, moving to parse raw data")
            self.config["result-dirs"] = self.parser.parse_data(self.config["result-dirs"])

        if self.graph:
            self.logger.info("Graph option set, moving into Graphing stage")
            self.grapher.generate_graphs(self.config["result-dirs"])


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
    parser.add_argument("-s", "--scave", type=str2bool, default=True, help="Extract results from omnet output")
    parser.add_argument("-g", "--graph", type=str2bool, default=True, help="Graph results of experiment")
    args = parser.parse_args()

    manager = Manager(args.experiment_type, args.experiment, args.scave, args.parse, args.graph)

    manager.run()

