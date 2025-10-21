import os

ROOT_DIR = "data/"
TEXT_FILES = [ROOT_DIR + f for f in os.listdir(ROOT_DIR)]
sample_file = TEXT_FILES[0]
with open(sample_file, "r") as f:
    text = f.read().rstrip().splitlines()

def load_data():
    prepped_data = []
    for file in TEXT_FILES:
        with open(file, "r") as f:
            text_lines = f.read().rstrip().splitlines()

        title = text_lines[0]
        author = text_lines[1]
        play_text = "\n".join(text_lines[8:])
        data_list = title + "\n" + author + "\n" + "\n" + play_text
        prepped_data.append(data_list)
    return prepped_data

