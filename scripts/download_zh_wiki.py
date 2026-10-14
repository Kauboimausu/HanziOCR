import argparse
import os

from datasets import load_dataset

from hanzi_ocr import utils


def save_dataset(root, opts):
    """
    Saves the chinese wikipedia dataset in the specified location
    Parameters
    -----------
    root: str or Path
        Project's root directory
    opts: argparse.Namespace
        Parameters given by the user
    """
    ds = load_dataset(
        "YCWTG/wikipedia_zh_cleaned_latest",
        split="train",
        revision="d76825105fb6e3c238c29fb105bcf5731facaf5e",
    )

    # If the desired save location doesn't exist we'll create it
    if not os.path.exists(os.path.join(root, opts.data_folder, opts.save_location)):
        os.makedirs(os.path.join(root, opts.data_folder, opts.save_location))

    # And then we'll save the dataset there
    ds.save_to_disk(os.path.join(root, opts.data_folder, opts.save_location))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_folder",
        type=str,
        default="data/",
        help="Root folder for the data inside the root project repository",
    )

    parser.add_argument(
        "--save_location",
        type=str,
        default="zh_wiki_data/",
        help="Desired save location for the Chinese Wikipedia data",
    )

    parser.add_argument(
        "--delete_dir_files",
        type=utils.str2bool,
        default=True,
        help="Deletes any existing files in the target extraction directory",
    )

    opts = parser.parse_args()
    save_dataset(utils.find_project_root(), opts)


if __name__ == "__main__":
    main()
