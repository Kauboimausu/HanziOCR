import pandas as pd
import os
import argparse
from pathlib import Path

def find_project_root():
    """
    Finds directory's root

    Returns
    ---------
    Path object with the directory's root
    """
    leaf = Path(__file__).parent()
    while not Path(leaf/"pyproject.toml").exists():
        if leaf.parent == leaf:
            raise Exception("Directory's root couldn't be found")
        leaf = leaf.parent
    return leaf

def make_char_list(opts):
    """
    Generates a pd dataframe from the given files and saves it in the given ç
    desired location as a csv
    Parameters
    ------------
    opts: argparse.Namespace
        Parameters given by the user
    """

    dir_root = find_project_root()
    files = opts.hsk_level
    unique_hanzi = []
    unique_hanzi_df = pd.DataFrame(columns=["codepoint", "simplified", "traditional"])
    for file in files:
        hanzi_df = pd.read_csv(
            os.path.join(dir_root, opts.root_folder, opts.hsk_list_folder, file),
            delimiter="\t",
            header=None,
            names=["traditional", "simplified", "pinyin", "definition"],
        )

        hanzi_df.head(5)
        for _, row in hanzi_df.iterrows():
            for simplified, traditional in zip(row["simplified"], row["traditional"]):
                # print(f"simplified: {simplified}")
                # print(f"traditional: {traditional}")
                if simplified not in unique_hanzi:
                    unique_hanzi.append(simplified)
                    unique_hanzi_df.loc[len(unique_hanzi_df)] = [
                        ord(simplified),
                        simplified,
                        traditional,
                    ]

    unique_hanzi_df.set_index("codepoint", inplace=True)
    unique_hanzi_df.sort_index(inplace=True)
    unique_hanzi_df.to_csv(
        os.path.join(dir_root, opts.root_folder, opts.save_location, opts.save_name)
    )


# Source - https://stackoverflow.com/a/43357954
# Posted by Maxim, modified by community. See post 'Timeline' for change history
# Retrieved 2026-06-16, License - CC BY-SA 4.0
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_folder",
        type=str,
        default="data/",
        help="Root folder for the data inside the root project directory",
    )

    parser.add_argument(
        "--hsk_list_folder",
        type=str,
        default="Old_HSK_tsvs/",
        help="Folder with the HSK words to be read",
    )

    parser.add_argument(
        "--hsk_level",
        type=str,
        nargs="+",
        default=["HSK 1.tsv"],
        help="In the given folder, which files are to be added to the set",
    )

    parser.add_argument(
        "--save_location",
        type=str,
        default="Characters/",
        help="Directory in which to save the csv with the unique characters",
    )

    parser.add_argument(
        "--save_name",
        type=str,
        default="characters.csv",
        help="Name for the file in which to save the unique characters",
    )

    opts = parser.parse_args()
    make_char_list(opts)


if __name__ == "__main__":
    main()
