import logging
import os
from datetime import datetime

# Log directory
LOGS_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

# Log file
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
LOG_FILE_PATH = os.path.join(LOGS_DIR, LOG_FILE)

# Logging setup
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format='[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
)

if __name__ == "__main__":
    logging.info("Logging has been set up.")
    logging.info(f"Log file created at: {LOG_FILE_PATH}")
    logging.info("Logging level set to INFO.")
    logging.info("Log format: [%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s")
