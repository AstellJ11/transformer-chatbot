import os
import sys
import logging


# Logging function
def init_logging(debug=False):
    format_str = "%(levelname)s %(asctime)s - %(message)s"
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(format=format_str, level=level)


# Remove file at path location
def remove_file(path):
    try:
        os.remove(path)
    except OSError as e:
        print("Error: %s : %s" % (path, e.strerror))


# Initialise logger
init_logging()
logger = logging.getLogger(__name__)

# Access the parent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

git_url = "https://github.com/google-research-datasets/dstc8-schema-guided-dialogue.git"
file_dir = "raw_data"
data_dir = os.path.join(current_dir, file_dir)  # Combine the new 'data' directory with the parent directory

if __name__ == "__main__":
    check_exists = os.path.isdir(data_dir)
    if check_exists:
        logger.info("This dataset has already been successfully downloaded!")
    else:
        os.makedirs(file_dir, exist_ok=True)  # Create the 'data' directory to hold the dataset

        logger.info("Downloading the Schema-Guided-Dialogue dataset...")
        os.chdir(data_dir)  # Change the working directory
        os.system(f"git clone {git_url}")  # Execute the system command to clone the repository

        remove_file("dstc8-schema-guided-dialogue/LICENSE.txt")
        remove_file("dstc8-schema-guided-dialogue/README.md")
        remove_file("dstc8-schema-guided-dialogue/dstc8.md")
        remove_file("dstc8-schema-guided-dialogue/schema_guided_overview.png")
        logger.info("Download Complete!")
