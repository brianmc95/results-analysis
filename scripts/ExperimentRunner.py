import multiprocessing
import subprocess
import json
import os


class ExperimentRunner:

    def __init__(self, config_file, experiment_type):
        self.experiment_type = experiment_type
        self.processors = multiprocessing.cpu_count()
        with open(config_file) as config:
            self.config = json.load(config)

    def start_experiment(self, num_processes, config_names):
        if num_processes > self.processors:
            print("Too many processes, going to revert to total - 1")
            num_processes = self.processors - 1

        if __name__ == "__main__":
            for i in range(num_processes):
                p = multiprocessing.Process(target=self.run_experiment, args=(self, config_names[i],))
                p.start()

    def run_experiment(self, config_name):
        name = multiprocessing.current_process().name
        if config_name not in self.config:
            raise Exception("Process: {} with config: {} does not exist in config files".format(name, config_name))

        orig_loc = os.getcwd()

        os.chdir(self.config[self.experiment_type]["cmake_dir"])

        scenario_config = "SCENARIO_CONFIG={}".format(config_name)
        setup_command = ["cmake", "-D", scenario_config, "--build", "."]
        subprocess.run(setup_command, shell=True)

        os.chdir(self.config[self.experiment_type]["project_path"])

        run_command = ["cmake", "--build", self.config[self.experiment_type]["cmake_dir"],
                       "--target", self.config[self.experiment_type]["target"]]
        subprocess.run(run_command, shell=True)

        os.chdir(orig_loc)

    def update_config(self, config_name, config_variant):
        """
        Allow for th changing of a config to a different configuration of the same type.
        e.g. change probability of resourceKeep across multiple simulations.
        :param config_name:
        :param config_variant:
        :return:
        """
        # TODO: Actually implement this as it is a bit of a extended project.
        return
