import os
from datetime import datetime
import logging
import structlog


class CustomLogger:
    def __init__(self, log_dir = "logs"):
        # ensure log directory exists
        self.logs_dir = os.path.join(os.getcwd(), log_dir)
        os.makedirs(self.logs_dir, exist_ok = True)

        # timestamped log file (for persistence)
        log_file = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
        self.log_file_path = os.path.join(self.logs_dir, log_file)

    def get_logger(self, name=__file__):
        logger_name = os.path.basename(name)

        #configure ogging for file and console
        file_handler = logging.FileHandler(self.log_file_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter("%(message)s")) # raw JSON lines

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter("%(message)s"))

        logging.basicConfig(
            level = logging.INFO,
            format = "%(message)s",
            handlers = [file_handler, console_handler]
        )

        # configure structlog for JSON structured logging
        structlog.configure(
            processors = [
                structlog.processors.TimeStamper(fmt="est", utc=True, key = "timestamp"),
                structlog.processors.add_log_level,
                structlog.processors.EventRenamer(to="event"),
                structlog.processors.JSONRenderer()
            ],
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )

        return structlog.get_logger(logger_name)
    
    # --example--
    
    if __name__ == "__main__":
        logger = CustomLogger().get_logger(__file__)
        logger.info("logger has started")
        logger.error("failed to start logger")
