import time
import re
import os
import json
import logging.config
import threading
from slackclient import SlackClient

from Manager import Manager


class MySlackBot:

    def __init__(self):
        # instantiate Slack client
        with open("configs/cv2x.json") as json_file:
            config = json.load(json_file)

        slack_api_token = config["cv2x"]["slack-api-token"]

        self.setup_logging()
        self.logger = logging.getLogger("slackbot")
        self.slack_client = SlackClient(slack_api_token)
        # starterbot's user ID in Slack: value is assigned after the bot starts up
        self.starterbot_id = None

        # constants
        self.RTM_READ_DELAY = 1  # 1 second delay between reading from RTM
        self.EXAMPLE_COMMAND = "run"
        self.MENTION_REGEX = "^<@(|[WU].+?)>(.*)"

        self.experiment_thread = None

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

    def parse_bot_commands(self, slack_events):
        """
            Parses a list of events coming from the Slack RTM API to find bot commands.
            If a bot command is found, this function returns a tuple of command and channel.
            If its not found, then this function returns None, None.
        """
        for event in slack_events:
            if event["type"] == "message" and not "subtype" in event:
                user_id, message = self.parse_direct_mention(event["text"])
                if user_id == self.starterbot_id:
                    self.logger.debug("Received the an event from user {} on channel {}".format(event["channel"],
                                                                                               event["user"]))
                    return message, event["channel"]
        return None, None

    def parse_direct_mention(self, message_text):
        """
            Finds a direct mention (a mention that is at the beginning) in message text
            and returns the user ID which was mentioned. If there is no direct mention, returns None
        """
        matches = re.search(self.MENTION_REGEX, message_text)
        # the first group contains the username, the second group contains the remaining message
        return (matches.group(1), matches.group(2).strip()) if matches else (None, None)

    def handle_command(self, command, channel):
        """
            Executes bot command if the command is known
        """
        # Default response is help text for the user
        default_response = "Not sure what you mean. Try *{}*.".format(self.EXAMPLE_COMMAND)

        # This is where you start to implement more commands!
        if command.startswith(self.EXAMPLE_COMMAND):
            sub_sections = command.split(" ")
            if sub_sections[1].lower() == "help" or sub_sections[1].lower() == "h":
                response = "Command is of the following structure: run configuration name {experiment} {parse} {scave} {graph}"
                self.logger.info("Prepared the following response: {}".format(response))

                # Sends the response back to the channel
                self.slack_client.api_call(
                    "chat.postMessage",
                    channel=channel,
                    text=response or default_response
                )

            else:
                experiment_type = sub_sections[1].lower()
                if len(sub_sections) == 2:
                    manager = Manager(experiment_type, channel=channel)
                else:
                    experiment = False
                    scave = False
                    parse = False
                    graph = False
                    upload = False
                    if "experiment" in sub_sections or "x" in sub_sections:
                        experiment = True
                    if "scave" in sub_sections or "s" in sub_sections:
                        scave = True
                    if "parse" in sub_sections or "p" in sub_sections:
                        parse = True
                    if "graph" in sub_sections or "g" in sub_sections:
                        graph = True
                    if "upload" in sub_sections or "u" in sub_sections:
                        upload = True

                    manager = Manager(experiment_type, experiment, scave, parse, graph, upload, channel=channel)

                self.experiment_thread = threading.Thread(target=manager.run, args=())
                self.experiment_thread.daemon = True  # Daemonize thread
                self.experiment_thread.start()  # Start the execution
        else:
            # Sends the response back to the channel
            self.slack_client.api_call(
                "chat.postMessage",
                channel=channel,
                text= default_response
            )

    def run_bot(self):
        if self.slack_client.rtm_connect(with_team_state=False):
            self.logger.info("Starter Bot connected and running!")
            # Read bot's user ID by calling Web API method `auth.test`
            self.starterbot_id = self.slack_client.api_call("auth.test")["user_id"]

            running_experiment = False

            while True:
                if self.experiment_thread:
                    if self.experiment_thread.is_alive():
                        running_experiment = True
                        self.logger.debug("Running experiment")
                    else:
                        self.logger.info("Experiment complete killing thread")
                        self.experiment_thread.join()
                        self.experiment_thread = None
                        running_experiment = False

                command, channel = self.parse_bot_commands(self.slack_client.rtm_read())
                if command:
                    if not running_experiment:
                        self.handle_command(command, channel)
                    else:
                        # Sends the response back to the channel
                        self.slack_client.api_call(
                            "chat.postMessage",
                            channel=channel,
                            text="Currently running an experiment, please wait before triggering another"
                        )
                time.sleep(self.RTM_READ_DELAY)
        else:
            self.logger.error("Connection failed. Exception traceback printed above.")


if __name__ == "__main__":
    mybot = MySlackBot()
    mybot.run_bot()
