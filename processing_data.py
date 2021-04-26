import json
import os
import sys
import logging
import re
from downloading_data import init_logging, remove_file

from pathlib import Path
import tqdm as tqdm
from pick import pick
import csv
import pandas as pd


def preprocess_sentence(w):
    w = (w.lower().strip())

    # Create a space between a word and the punctuation following it
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

    # Replace everything with space except (a-z, A-Z, ".", "?", "!", ",", all numbers, "-" and "$" for the slotted values)
    w = re.sub(r"[^a-zA-Z?.!,¿$_0123456789-]+", " ", w)
    w = w.strip()

    return w


# Allow the user to choose a domain
def domain_choice():
    all_domains = []  # List array for domain names (including duplicates)
    unique_list = []  # List array to find unique domain names
    domain_names = []  # Final list array for all unique domains

    # Gather list of domains from dataset
    for file_name in tqdm.tqdm(os.listdir(train_dir), desc='Preparing dataset'):  # Display progress bar during loop

        if 'schema.json' in file_name:
            continue

        file_path = os.path.join(train_dir, file_name)  # Create new file path for each run of loop

        # Open each dialogue file
        with open(file_path, "r") as f:
            data = json.load(f)

        # Extract all domains from within 'services' dict
        temp_domains = []
        for dialogue in data:
            theme = [dialogue['services']]
            if theme not in temp_domains:
                temp_domains.extend(theme)
        all_domains.extend(temp_domains)

        # Extract all unique domain names
        for item in all_domains:
            if item not in unique_list:
                unique_list.append(item)

    # Convert list of list to flat base list
    flat_list = [item for sublist in unique_list for item in sublist]

    # Remove "_1" etc. from all domains, to ensure the entire domain is selected
    flat_list = [s.replace("_", "").replace("1", "").replace("2", "").replace("3", "").replace("4", "")
                 for s in flat_list]

    # Duplicates can still exist within list, final check and pass to domain_names
    for name in flat_list:
        if name not in domain_names:
            domain_names.append(name)

    # Allow the user to choose the domain being trained
    domain_question = 'Please choose a domain from the list you wish to talk about:  '
    domain_option, index = pick(domain_names, domain_question)

    return domain_option


# Extract every utterance for chosen domain
def extract_utterance(inp_dir, out_dir):
    all_dialogs = []  # List array for final extracted dialogs

    for file_name in tqdm.tqdm(os.listdir(inp_dir), desc='Extracting utterances'):  # Display progress bar during loop

        if 'schema.json' in file_name:
            continue

        file_path = os.path.join(inp_dir, file_name)

        with open(file_path, "r") as f:
            data = json.load(f)

        temp_dialogs = []

        for dialogue in data:
            substring_in_domain = any(domain_option in string for string in dialogue['services'])
            if substring_in_domain == True:
                for item in dialogue['turns']:
                    utterance = [item['utterance']]  # Extract the system and user speech
                    s_utterance = utterance[0]  # Single list element to string
                    w = preprocess_sentence(s_utterance)  # String to function
                    w_utterance = [w]  # Return value back to list
                    temp_dialogs.extend(w_utterance)
        all_dialogs.extend(temp_dialogs)  # Add all elements of new dialogue to overall list

    # Process data into required format: I \t R \n I \t R...
    deli = "\n"  # Initialising delimiter
    temp_string = list(map(str, all_dialogs))  # Convert each list element to a string

    res = deli.join(temp_string)  # Add each individual utterance to a new line

    lines = res.splitlines()  # Split on new line

    # For every utterance create a tab break, for every other utterance create a new line
    processed = ''
    step_size = 2
    for i in range(0, len(lines), step_size):
        processed += '\t'.join(lines[i:i + step_size]) + '\n'

    # File Saving
    temp_txt_file = "temp.txt"
    temp_csv_file = "temp.csv"
    csv_file_path = Path(current_dir, out_dir)

    logger.info(f"Saving Schema dialogue data to {csv_file_path}")

    # Saving tab separated data
    save_path = open(temp_txt_file, "w")
    n = save_path.write(str(processed))
    save_path.close()

    # Converting tab separated data to csv
    in_txt = csv.reader(open(temp_txt_file, "r"), delimiter='\t')
    out_csv = csv.writer(open(temp_csv_file, 'w'))
    out_csv.writerows(in_txt)

    df = pd.read_csv(temp_csv_file)
    df.to_csv(csv_file_path, index=False)

    logger.info("Processing Complete!")


# Extract all utterances with slotted values replaced
def slotted_utterance(inp_dir, out_dir):
    all_dialogs = []  # List array for final extracted dialogs

    for file_name in tqdm.tqdm(os.listdir(inp_dir), desc='Extracting utterances'):  # Display progress bar during loop

        if 'schema.json' in file_name:
            continue

        file_path = os.path.join(inp_dir, file_name)

        with open(file_path, "r") as f:
            data = json.load(f)

        temp_dialogs = []

        for dialogue in data:
            # Check selected domain substring against string values
            substring_in_domain = any(domain_option in string for string in dialogue['services'])
            if substring_in_domain == True:
                for item in dialogue['turns']:
                    utterance = item['utterance']  # Extract the system and user speech
                    for item2 in item['frames']:
                        for item3 in item2['actions']:
                            canonical_value = item3['values']  # Extract canonical values
                            slot_value = '$' + item3['slot']  # Extract replacement slot values

                            # Replace each canonical value with its respected slot value
                            for i in canonical_value:
                                utterance = utterance.replace(i, slot_value)

                            w = preprocess_sentence(utterance)  # String to function

                    temp_dialogs.append(w)
        all_dialogs.extend(temp_dialogs)

    # Process data into required format: I \t R \n I \t R...
    deli = "\n"  # Initialising delimiter
    temp_string = list(map(str, all_dialogs))  # Convert each list element to a string

    res = deli.join(temp_string)  # Add each individual utterance to a new line

    lines = res.splitlines()  # Split on new line

    # For every utterance create a tab break, for every other utterance create a new line
    processed = ''
    step_size = 2
    for i in range(0, len(lines), step_size):
        processed += '\t'.join(lines[i:i + step_size]) + '\n'

    # File Saving
    temp_txt_file = "temp.txt"
    temp_csv_file = "temp.csv"
    csv_file_path = Path(current_dir, out_dir)

    logger.info(f"Saving Schema dialogue data to {csv_file_path}")

    # Saving tab separated data
    save_path = open(temp_txt_file, "w")
    n = save_path.write(str(processed))
    save_path.close()

    # Converting tab separated data to csv
    in_txt = csv.reader(open(temp_txt_file, "r"), delimiter='\t')
    out_csv = csv.writer(open(temp_csv_file, 'w'))
    out_csv.writerows(in_txt)

    df = pd.read_csv(temp_csv_file)
    df.to_csv(csv_file_path, index=False)

    logger.info("Processing Complete!")


# Initialise logger
init_logging()
logger = logging.getLogger(__name__)

# Access the parent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Unprocessed train + test directories
train_dir = "raw_data/dstc8-schema-guided-dialogue/train"
test_dir = "raw_data/dstc8-schema-guided-dialogue/test"
val_dir = "raw_data/dstc8-schema-guided-dialogue/dev"

# Processed train + test directories
output_train_dir = "processed_data/train/all_training_dialogue.csv"
output_test_dir = "processed_data/test/all_testing_dialogue.csv"
output_val_dir = "processed_data/val/all_val_dialogue.csv"

if __name__ == "__main__":
    # Create processed_data folders for all final dialogues
    new_train_dir = Path(current_dir, "processed_data/train")
    new_test_dir = Path(current_dir, "processed_data/test")
    new_test_dir2 = Path(current_dir, "processed_data/BLEU")
    new_val_dir = Path(current_dir, "processed_data/val")

    try:
        os.makedirs(new_train_dir)
    except OSError:
        logger.info("Creation of the directory '%s' failed. It may already exist." % new_train_dir)
    else:
        logger.info("Successfully created the directory '%s'!" % new_train_dir)
    try:
        os.makedirs(new_test_dir)
    except OSError:
        logger.info("Creation of the directory '%s' failed. It may already exist." % new_test_dir)
    else:
        logger.info("Successfully created the directory '%s'!" % new_test_dir)
    try:
        os.makedirs(new_val_dir)
    except OSError:
        logger.info("Creation of the directory '%s' failed. It may already exist." % new_val_dir)
    else:
        logger.info("Successfully created the directory '%s'!" % new_val_dir)
    try:
        os.makedirs(new_test_dir2)
    except OSError:
        logger.info("Creation of the directory '%s' failed. It may already exist." % new_test_dir2)
    else:
        logger.info("Successfully created the directory '%s'!" % new_test_dir2)

    with open('status.txt', 'w') as filetowrite:
        filetowrite.write("No model has been trained yet!")

    # Gather user chosen domain
    domain_option = domain_choice()

    # Allow the user to choose the domain being trained
    slot_question = 'Would you like to train on the default dataset or a dataset with replaced slotted values?  '
    slot_answers = ['Default Dataset', 'Slotted Dataset']
    slot_option, index = pick(slot_answers, slot_question)

    if index == 0:
        # Extract training + testing utterances
        extract_utterance(train_dir, output_train_dir)
        extract_utterance(test_dir, output_test_dir)
        extract_utterance(val_dir, output_val_dir)

        remove_file("temp.txt")
        remove_file("temp.csv")
    else:
        # Data without slot values
        slotted_utterance(train_dir, output_train_dir)
        slotted_utterance(test_dir, output_test_dir)
        slotted_utterance(val_dir, output_val_dir)

        remove_file("temp.txt")
        remove_file("temp.csv")
