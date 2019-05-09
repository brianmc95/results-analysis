import multiprocessing
import logging
from subprocess import Popen, PIPE, STDOUT
import time
import datetime
import os
import re


class ExperimentRunner:

    def __init__(self, config, experiment_type):
        self.processors = multiprocessing.cpu_count()
        self.config = config
        self.experiment_type = experiment_type
        self.logger = logging.getLogger("ExperimentRunner")

    def start_experiment(self):

        result_dirs = []
        now = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")

        for config in self.config["config_names"]:

            if self.config["config_names"][config] <= 0:
                continue

            output_data_path = "{}/data/omnet/{}/{}-{}".format(os.getcwd(), self.experiment_type, config, now)

            os.mkdir(output_data_path)

            result_dirs.append(os.path.basename(os.path.normpath(output_data_path)))

            num_processes = self.config["parallel_processes"]
            if num_processes > self.processors:
                self.logger.warn("Too many processes, going to revert to total - 1")
                num_processes = self.processors - 1

            self.logger.info("Beginning simulation of config: {}, total of {} runs of this configuration".format(config, self.config["config_names"][config]))

            configs = []
            for _ in range(self.config["config_names"][config]):
                configs.append(config)

            self.logger.debug("Configurations list: {}".format(configs))
            number_of_batches = len(configs)//num_processes
            if number_of_batches == 0:
                number_of_batches = 1

            if config not in self.config["config_names"]:
                self.logger.error(
                    "Config: {} does not exist in config files".format(config))
                raise Exception("Config: {} does not exist in config files".format(config))

            orig_loc = os.getcwd()

            os.chdir(self.config["cmake_dir"])

            self.logger.debug("Moved into the cmake directory {}".format(os.getcwd()))

            setup_command = ["cmake", "-D", "SCENARIO_CONFIG={}".format(config), "--build", "."]

            self.logger.debug("Command to be run: {}".format(setup_command))

            process = Popen(setup_command, stdout=PIPE, stderr=STDOUT)
            with process.stdout:
                self.log_subprocess_output(process.stdout)
            exitcode = process.wait()  # 0 means success

            if exitcode != 0:
                self.logger.exception("Received exit code {} from config setup, exiting".format(exitcode))
                raise Exception("Received exit code {} from config setup, exiting".format(exitcode))

            os.chdir(orig_loc)

            self.logger.debug("Moved backed to original location {}".format(os.getcwd()))

            self.logger.info("Completed scenario setup")

            i = 0
            while i < len(configs):
                if len(configs) < num_processes:
                    num_processes = len(configs)
                self.logger.info("Starting up processes, batch {}/{}".format((i//num_processes)+1, number_of_batches))
                pool = multiprocessing.Pool(processes=num_processes)

                pool.map(self.run_experiment, list(range(num_processes)))

                self.logger.info("Batch {}/{} complete".format((i // num_processes) + 1, number_of_batches))

                self.store_results(config, i, now)

                i += num_processes

            self.logger.info("Removing {} dir from results".format(config))

            old_output = "{}/data/omnet/{}/{}".format(os.getcwd(), self.experiment_type, config)
            os.removedirs(old_output)

        self.logger.info(result_dirs)
        return result_dirs

    def log_subprocess_output(self, pipe):
        for line in iter(pipe.readline, b''):  # b'\n'-separated lines
            self.logger.debug('Subprocess Line: %r', line)

    def run_experiment(self, wait):

        name = multiprocessing.current_process().name
        self.logger.info("Starting process {}".format(name))

        self.logger.info("Waiting {}s".format(wait))
        time.sleep(wait)
        self.logger.info("Wait complete")

        orig_loc = os.getcwd()

        os.chdir(self.config["project_path"])

        self.logger.debug("Moved into the project directory {}".format(os.getcwd()))

        self.logger.info("Beginning simulation run")

        run_command = ["cmake", "--build", self.config["cmake_dir"], "--target", self.config["target"]]
        process = Popen(run_command, stdout=PIPE, stderr=STDOUT)
        with process.stdout:
            self.log_subprocess_output(process.stdout)
        exitcode = process.wait()  # 0 means success

        if exitcode != 0:
            self.logger.error("Non-Zero exit code for simulation run, code is {}".format(exitcode))

        self.logger.info("Completed simulation run.")

        os.chdir(orig_loc)

        self.logger.debug("Moved backed to original location {}".format(os.getcwd()))

    def store_results(self, config, runs_so_far, now):
        self.logger.info("Moving results for config {} to a known location".format(config))
        self.logger.info("Current directory {}".format(os.getcwd()))

        pattern = "\d+\."
        run_num = runs_so_far
        previous_process = None

        output_data_path = "{}/data/omnet".format(os.getcwd())

        result_files = os.listdir("{}/{}/{}".format(output_data_path, self.experiment_type, config))
        result_files.sort()

        for result_file in result_files:
            self.logger.debug(re.findall(pattern, result_file)[0])
            if previous_process != re.findall(pattern, result_file)[0]:
                previous_process = re.findall(pattern, result_file)[0]
                run_num += 1
                self.logger.debug("Moving onto next run {}".format(run_num))

            extension = os.path.splitext(result_file)[1]
            new_loc = "{}/{}/{}-{}/run-{}{}".format(output_data_path, self.experiment_type,
                                                    config, now, run_num, extension)
            result_file = "{}/{}/{}/{}".format(output_data_path, self.experiment_type, config, result_file)

            os.rename(result_file, new_loc)
            self.logger.debug("Moved file {} to {}".format(result_file, new_loc))

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
