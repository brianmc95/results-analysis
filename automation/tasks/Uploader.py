from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os
import tarfile
import logging


class Uploader:

    def __init__(self, config, experiment_type):

        self.config = config
        self.experiment_type = experiment_type

        self.logger = logging.getLogger("uploader")

        self.gauth = None
        self.drive = None
        self.results_id = None
        self.figures_id = None

        self.drive_setup()

    def drive_setup(self):

        self.logger.info("Beginning google authentication")
        self.gauth = GoogleAuth()
        # Try to load saved client credentials
        self.gauth.LoadCredentialsFile("mycreds.txt")
        if self.gauth.credentials is None:
            self.logger.info("No google drive creds file found must manually setup.")
            # Authenticate if they're not there
            self.gauth.LocalWebserverAuth()
        elif self.gauth.access_token_expired:
            # Refresh them if expired
            self.gauth.Refresh()
        else:
            # Initialize the saved creds
            self.logger.info("Google cred file found automatic authentication")
            self.gauth.Authorize()
        # Save the current credentials to a file
        self.logger.info("Automatically storing credentials.")
        self.gauth.SaveCredentialsFile("mycreds.txt")

        self.drive = GoogleDrive(self.gauth)

        file_list = self.drive.ListFile({'q': "'root' in parents and trashed=false"}).GetList()
        for root_file in file_list:
            if root_file["title"] == "automated-results":
                results_folder_id = root_file["id"]
                results_folder_files = self.drive.ListFile(
                    {'q': "'{}' in parents and trashed=false".format(results_folder_id)}).GetList()
                for res_file in results_folder_files:
                    if res_file["title"] == "raw_results":
                        self.results_id = res_file["id"]
                    if res_file["title"] == "figures":
                        self.figures_id = res_file["id"]

    def tar_results(self):
        orig_loc = os.getcwd()
        self.logger.info("Original location: {}".format(orig_loc))
        for tar_ball in self.config["raw-results"]:

            os.chdir("data/raw_data/{}/".format(self.experiment_type))
            self.logger.info("Moved to {}".format(os.getcwd()))

            self.logger.info("Tarring up {} folder".format(tar_ball))
            tf = tarfile.open("{}.tar.xz".format(tar_ball), mode="w:xz")
            tf.add(tar_ball)
            tf.close()
            self.logger.info("Folder: {}, successfully tarred".format(tar_ball))

            os.chdir(orig_loc)
            self.logger.debug("Moved to {}".format(os.getcwd()))
            self.upload_tar_ball(tf.name)

            self.logger.info("Deleting Tar file")
            os.remove(tf.name)
        self.logger.info("Moving back to original directory: {}".format(os.getcwd()))

    def upload_tar_ball(self, tar_file_path):

        self.logger.info("Uploading tarfile: {}".format(os.path.basename(tar_file_path)))

        # TODO: Between this method and the figure uploader lots of repeated code, would be better to tidy this.
        found = False
        file_list = self.drive.ListFile({'q': "'{}' in parents and trashed=false".format(self.results_id)}).GetList()
        for raw_results_file in file_list:
            if raw_results_file["title"] == self.experiment_type:
                found = True
                experiment_folder_id = raw_results_file["id"]
                self.logger.info("Found the experiment folder: {}, id: {}".format(self.experiment_type,
                                                                                  experiment_folder_id))

        if not found:
            self.logger.info("Experiment {} folder not found creating it.".format(self.experiment_type))
            # Folder creation
            folder = self.drive.CreateFile({'title': self.experiment_type,
                                            "parents": [{"id": self.results_id}],
                                            "mimeType": "application/vnd.google-apps.folder"})
            folder.Upload()
            experiment_folder_id = folder["id"]
            self.logger.info("Experiment {} folder created with id {}".format(self.experiment_type,
                                                                              experiment_folder_id))

        # File upload
        self.logger.info("Beginning the upload of {} to results folder automated-results/raw_results/{}".format(os.path.basename(tar_file_path), self.experiment_type))
        tarball = self.drive.CreateFile(
            {"parents": [{"kind": "drive#fileLink", "id": experiment_folder_id}],
             'title': os.path.basename(tar_file_path)})
        tarball.SetContentFile(tar_file_path)
        tarball.Upload()

    def upload_figures(self):
        self.logger.info("Uploading figures")

        found = False
        file_list = self.drive.ListFile({'q': "'{}' in parents and trashed=false".format(self.figures_id)}).GetList()
        for figures_file in file_list:
            if figures_file["title"] == self.experiment_type:
                found = True
                experiment_folder_id = figures_file["id"]
                self.logger.info("Found the experiment folder: {}, id: {}".format(self.experiment_type,
                                                                                  experiment_folder_id))

        if not found:
            self.logger.info("Experiment {} folder not found creating it.".format(self.experiment_type))
            # Folder creation
            folder = self.drive.CreateFile({'title': self.experiment_type,
                                            "parents": [{"id": self.figures_id}],
                                            "mimeType": "application/vnd.google-apps.folder"})
            folder.Upload()
            experiment_folder_id = folder["id"]
            self.logger.info("Experiment {} folder created with id {}".format(self.experiment_type,
                                                                              experiment_folder_id))

        comparison_found = False
        individual_found = False
        file_list = self.drive.ListFile({'q': "'{}' in parents and trashed=false".format(experiment_folder_id)}).GetList()
        for exp_file in file_list:
            if exp_file["title"] == "comparison":
                comparison_found = True
                comparison_folder_id = exp_file["id"]
                self.logger.info("Found the comparison folder id: {}".format(comparison_folder_id))
            elif exp_file["title"] == "individual":
                individual_found = True
                individual_folder_id = exp_file["id"]
                self.logger.info("Found the comparison folder id: {}".format(individual_folder_id))

        if not comparison_found:
            self.logger.info("Experiment {} folder not found creating it.".format(self.experiment_type))
            # Folder creation
            folder = self.drive.CreateFile({'title': "comparison",
                                            "parents": [{"id": experiment_folder_id}],
                                            "mimeType": "application/vnd.google-apps.folder"})
            folder.Upload()
            comparison_folder_id = folder["id"]
            self.logger.info("Comparison folder created with id {}".format(comparison_folder_id))

        if not individual_found:
            self.logger.info("Experiment {} folder not found creating it.".format(self.experiment_type))
            # Folder creation
            folder = self.drive.CreateFile({'title': "individual",
                                            "parents": [{"id": experiment_folder_id}],
                                            "mimeType": "application/vnd.google-apps.folder"})
            folder.Upload()
            individual_folder_id = folder["id"]
            self.logger.info("Comparison folder created with id {}".format(individual_folder_id))

        for comparison in self.config["figures"]["comparison"]:
            figure_path = os.path.join(os.getcwd(), "data/figures/comparison/{}".format(comparison))

            self.logger.info("Beginning the upload of {} to results folder automated-results/figures/{}/comparison".format(
                os.path.basename(figure_path), self.experiment_type))

            fig = self.drive.CreateFile(
                {"parents": [{"kind": "drive#fileLink", "id": comparison_folder_id}],
                 'title': os.path.basename(figure_path)})
            fig.SetContentFile(figure_path)
            fig.Upload()

            self.logger.info("File successfully uploaded")

        for individual in self.config["figures"]["individual"]:
            figure_path = os.path.join(os.getcwd(), "data/figures/individual/{}".format(individual))

            self.logger.info(
                "Beginning the upload of {} to results folder automated-results/figures/{}/individual".format(
                    os.path.basename(figure_path), self.experiment_type))

            fig = self.drive.CreateFile(
                {"parents": [{"kind": "drive#fileLink", "id": individual_folder_id}],
                 'title': os.path.basename(figure_path)})
            fig.SetContentFile(figure_path)
            fig.Upload()

            self.logger.info("File successfully uploaded")

    def upload_results(self, config):
        self.config = config
        self.logger.info("Beginning the tarring of result files")
        self.tar_results()
        self.upload_figures()
