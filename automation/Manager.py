import json
import argparse
import os
import logging.config

#from tasks.DataParser import DataParser
from tasks.ExperimentRunner import ExperimentRunner
#from tasks.Grapher import Grapher


class Manager:
    """
    Class designed to manage the whole process of automated experimentation and results analysis
    """

    def __init__(self, experiment_type, experiment, parse, graph):

        self.experiment_type = experiment_type
        self.experiment = experiment
        self.parse = parse
        self.graph = graph
        self.setup_logging()
        self.logger = logging.getLogger(__name__)

        # TODO: Is this predetermined or passed in i.e. the config file?
        with open("../configs/run_configurations/cv2x.json") as json_file:
            self.config = json.load(json_file)[self.experiment_type]

        if self.experiment:
            self.runner = ExperimentRunner(self.config)

        # if self.parse:
        #     self.parser = DataParser(config=self.config)

        # if self.graph:
        #     self.grapher = Grapher("../configs/fields/cv2x.json")

    @staticmethod
    def setup_logging(default_path='logger/logging.json', default_level=logging.INFO, env_key='LOG_CFG'):
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
        Runs the experiment &/OR parsing &/OR Graphing
        :return:
        """
        if self.experiment:
            self.logger.info("Experiment option set, moving into start experiment")
            self.runner.start_experiment()

        # if self.parse:
        #     # This is not yet paralleled
        #     self.logger.info("Experiment option set, moving into start experiment")
        #     self.parser.extract_raw_data()
        #
        # if self.graph:
        #     self.logger.info("Experiment option set, moving into start experiment")
        #     self.parser.tidy_data()
        #     self.grapher.


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
    parser.add_argument("-p", "--parse", type=str2bool, default=True, help="Extract results from experiments")
    parser.add_argument("-g", "--graph", type=str2bool, default=True, help="Extract results from experiments")
    args = parser.parse_args()

    manager = Manager(args.experiment_type, args.experiment, args.parse, args.graph)

    manager.run()

