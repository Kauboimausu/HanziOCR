from PIL import PcfFontFile, Image, ImageDraw, ImageFont
import sys
import os
import argparse
from hanzi_ocr import utils

root_ = utils.find_project_root()


def get_fonts_list(opts):
    """
    Returns the files stored in the fonts directory, that is, returns a list of all the fonts to be rendered

    Parameters 
    ----------
    opts: argparse.Namespace
        Parameters given by the user

    Returns
    ----------
    List of fonts stored in specified directory
    """
    return os.listdir(os.path.join(root_, opts.data_folder, opts.font_location))



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_folder",
        type=str,
        default="data/",
        help="Root folder for the data inside the root project directory",
    )

    parser.add_argument(
        "--font_location",
        type=str,
        default="fonts/",
        help="Folder with the fonts to be read",
    )

    parser.add_argument(
        "--save_location",
        type=str,
        default="character_images/",
        help="Directory in which to save the images with the characters",
    )

    opts = parser.parse_args()
    fonts_list = get_fonts_list(opts)


if __name__ == "__main__":
    main()
