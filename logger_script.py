import os
import sys
import logging

from datetime import datetime
from logging import StreamHandler, Formatter
from utils import constants


def logger_handlers_exist(logger: logging.Logger) -> bool:
    """
    This function checks whether there is already an existing logger.
    :param logger: The logger object.
    :return: Two bools indicating is stream handler and file handler exist.
    """

    stream_handler_exists = False
    file_handler_exists = False

    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            stream_handler_exists = True
        if isinstance(handler, logging.FileHandler):
            file_handler_exists = True

    return stream_handler_exists and file_handler_exists


def initialize_logger(logger_name: str, log_file_path: str, log_level: str = "DEBUG") -> logging.Logger:
    """
    This function initializes a logger together with stream and file handler. Performs a check whether a logger
    already exists, if so returns that one, else it initializes a logger by creating a stream handler and file handler.
    :param logger_name: Name of the logger
    :param log_file_path: Path to store the log file (if None, no file is stored)
    :param log_level: Level of log messages to display
    :return: Logger object
    """

    # Get or create logger
    logger = logging.getLogger(logger_name)

    # Define logger format
    logger_format = "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    logger_formatter = logging.Formatter(logger_format)

    if not any(isinstance(handler, logging.StreamHandler) for handler in logger.handlers):
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(logger_formatter)
        logger.addHandler(stream_handler)

    if log_file_path is not None and not any(isinstance(handler, logging.FileHandler) for handler in logger.handlers):
        log_dir = os.path.dirname(log_file_path)
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(logger_formatter)
        logger.addHandler(file_handler)

    # Set logger to desired logging level
    logger.setLevel(log_level)

    return logger


def get_logger(script_type: str):
    """
    Fetches the logger for the specific script type.
    :param script_type: String that represents the script that is logged.
    :return: Logger object
    """
    logging_folder = os.path.join(constants.REPO_PATH, "logger_files")
    logger_file_name = f"Logger_{script_type}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"

    logger_path = os.path.join(logging_folder, logger_file_name)

    return initialize_logger(logger_name=script_type, log_file_path=logger_path, log_level="DEBUG")
