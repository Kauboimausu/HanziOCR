from datasets import load_dataset
import argparse
import os
from hanzi_ocr import utils

def save_dataset(opts):
    """
    Saves the chinese wikipedia dataset in the specified location
    """
    # First we'll find the project's root
    root = utils.find_project_root()
    ds = load_dataset("zetavg/zh-tw-wikipedia", split="train", revision="0c0463b2ae0643db0018e68f6286a256f0e88c85")

    # If the desired save location doesn't exist we'll create it 
    if not os.path.exists(os.path.join(root, opts.data_folder, opts.save_location)):
        os.makedirs(os.path.join(root, opts.data_folder, opts.save_location))

    # And then we'll save the dataset there
    ds.save_to_disk(os.path.join(root, opts.data_folder, opts.save_location))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_folder", type=str, default="data/", help="Root folder for the data inside the root project repository"
    )

    parser.add_argument(
        "--save_location", type=str, default="zh_wiki_data/", help="Desired save location for the Chinese Wikipedia data"
    )

    opts = parser.parse_args()
    save_dataset(opts)

if __name__ == "__main__":
    main()