import os
import shutil
import logging
from datetime import datetime


def set_logger(name: str = None) -> tuple:
    # set parameters
    current_path = os.path.dirname(os.path.abspath(__file__))
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_folder = "log_" + current_time + ("_" + name if name else "")
    log_path = os.path.join(current_path, log_folder)
    code_path = os.path.join(log_path, "code")

    # copy code
    os.makedirs(code_path)
    for filename in os.listdir(current_path):
        if filename.endswith(".py"):
            file = os.path.join(current_path, filename)
            shutil.copy(file, code_path)

    # set logger
    filename = os.path.join(log_path, current_time + ("_" + name if name else "") + ".log")
    formatter = logging.Formatter("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    log = logging.getLogger()
    log.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.DEBUG)
    log.addHandler(console_handler)
    file_handler = logging.FileHandler(filename)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    log.addHandler(file_handler)

    log.debug("Successfully set logger.")
    log.debug(f"Successfully made directory \"{log_folder}\".")
    log.debug(f"Successfully copied code to directory \"{log_folder}/code\".")

    return log, log_folder
