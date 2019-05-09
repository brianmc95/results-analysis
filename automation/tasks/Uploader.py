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
        for tar_ball in self.config["raw-results"]:

            os.chdir("../../data/raw_data/{}/".format(self.experiment_type))
            self.logger.debug("Moved to {}".format(os.getcwd()))

            self.logger.info("Tarring up {} folder".format(tar_ball))
            tf = tarfile.open("{}.tar.gz".format(tar_ball), mode="w:xz")
            tf.add(tar_ball)
            tf.close()
            self.logger.info("Folder: {}, successfully tarred".format(tar_ball))

            os.chdir(orig_loc)
            self.logger.debug("Moved to {}".format(os.getcwd()))
            self.upload_tar_ball(tf)

            self.logger.info("Deleting Tar file")
            os.remove(tf.name)
        self.logger.info("Moving back to original directory: {}".format(os.getcwd()))

    def upload_tar_ball(self, tarfile):

        self.logger.info("Uploading tarfile: {}".format(os.path.basename(tarfile.name)))

        found = False
        file_list = self.drive.ListFile({'q': "'{}' in parents and trashed=false".format(self.results_id)}).GetList()
        for raw_results_file in file_list:
            if raw_results_file["title"] == experiment_type:
                found = True
                experiment_folder_id = raw_results_file["id"]
                self.logger.info("Found the experiment folder: {}, id: {}".format(self.experiment_type, experiment_folder_id))

        if not found:
            self.logger.info("Experiment {} folder not found creating it.".format(self.experiment_type))
            # Folder creation
            folder = self.drive.CreateFile({'title': self.experiment_type,
                                            "parents": [{"id": self.results_id}],
                                            "mimeType": "application/vnd.google-apps.folder"})
            folder.Upload()
            experiment_folder_id = folder["id"]
            self.logger.info("Experiment {} folder created with id {}".format(self.experiment_type, experiment_folder_id))

        # File upload
        self.logger.info("Beginning the upload of {} to results folder automated-results/raw_results/{}".format(os.path.basename(tarfile.name), self.experiment_type))
        tarball = self.drive.CreateFile(
            {"parents": [{"kind": "drive#fileLink", "id": experiment_folder_id}], 'title': os.path.basename(tarfile.name)})
        tarball.SetContentFile(results_file)
        tarball.Upload()

    def upload_figures(self):
        pass

    def upload_results(self):
        self.tar_results()


if __name__ == "__main__":
    results_file = "/home/brian/git_repos/results-analysis/data/raw_data/cv2x/Base-2019-05-02-14:54:06.tar.xz"
    experiment_type = "cv2x"

    # import json
    #
    # config_file = "../../configs/cv2x.json"
    #
    # with open(config_file) as json_config:
    #     config = json.load(json_config)["cv2x"]
    #     up = Uploader("cv2x", config)
    #     up.tar_results()

    #
    # found = False
    # file_list3 = drive.ListFile({'q': "'{}' in parents and trashed=false".format(figures_id)}).GetList()
    # for file1 in file_list:
    #     if file1["title"] == experiment_type:
    #         found = True
    #         experiment_folder = file1["id"]
    #
    # if not found:
    #     # Folder creation
    #     folder = drive.CreateFile({'title': experiment_type,
    #                               "parents": [{"id": figures_id}],
    #                               "mimeType": "application/vnd.google-apps.folder"})
    #     folder.Upload()
    #     experiment_folder = folder["id"]
