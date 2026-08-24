from PIL import Image, ImageDraw, ImageFont
import os
import argparse
from hanzi_ocr import utils
import pandas as pd
from fontTools.ttLib import TTFont
from math import ceil
import numpy as np


def get_hanzi_list(opts) -> pd.DataFrame:
    """
    Reads the csv in which the hanzis are stored and returns the dataframe

    Parameters
    ----------
    opts: argparse.Namespace
    Parameters given by the user

    Returns
    ---------
    Pandas dataframe with the hanzi data
    """
    root = utils.find_project_root()
    try:
        hanzi_df = pd.read_csv(
            os.path.join(
                root, opts.data_folder, opts.hanzi_location, opts.hanzi_file_name
            ),
            index_col="codepoint",
        )
        return hanzi_df
    except Exception as e:
        print(f"File could not be found at given location: {e}")


def write_images(opts):
    """
    Generates the training images by loading each font stored locally,
    iterating through them one at a time, then for each font it iterates
    through each character stored in the character list, for each of these it
    generates an image using Pillow

    Parameters
    -----------
    opts: argparse.Namespace
    Parameters given by the user
    """
    root = utils.find_project_root()
    # If it doesn't exist we create a folder for the fonts
    if not os.path.exists(os.path.join(root, opts.data_folder, opts.font_location)):
        print(
            f"WARNING: folder with fonts doesn't exist, make sure the name is properly given"
        )
        return

    # We want to generate a manifest to keep track of what character, font, size, etc is being generated on each file.
    manifest_df = pd.DataFrame(columns=["file name", "text", "font", "type"])
    # We'll also generate a manifest for the characters that were not able to be rendered
    error_manifest_df = pd.DataFrame(columns=["text", "font", "type"])
    manifest_arr = []
    error_manifest_arr = []

    # If the specified folder for the generated images doesn't already exist we'll create it
    if not os.path.exists(
        os.path.join(root, opts.data_folder, opts.image_save_location)
    ):
        os.makedirs(os.path.join(root, opts.data_folder, opts.image_save_location))

    # We'll get the list of saved fonts and iterate through them
    fonts = utils.get_fonts_list(opts.data_folder, opts.font_location)
    hanzi_df = get_hanzi_list(opts)

    if "s" not in opts.hanzi_styles.lower() and "t" not in opts.hanzi_styles.lower():
        print(
            "ERROR: Either 'S' (Simplified) or 'T' (Traditional) has to be included in the '--hanzi_styles' flag"
        )
        return

    # Weĺl load each font and iterate through our previously generated character list
    for font_name in fonts:
        # reserved is the name for fonts that i dont want to render yet
        if "reserved" in font_name:
            continue
        try:
            font = ImageFont.truetype(
                font=os.path.join(
                    root, opts.data_folder, opts.font_location, font_name
                ),
                size=opts.character_size,
            )

            tfont = TTFont(
                os.path.join(root, opts.data_folder, opts.font_location, font_name)
            )
            font_cmap = tfont.getBestCmap()
        except Exception as e:
            print(f"ERROR: Could not load font, reason: {e}")
        print(f"Generating images for font {font_name}")
        for _, row in hanzi_df.iterrows():
            for script in opts.hanzi_styles:
                if script.lower() == "s":
                    script_name = "simplified"
                elif script.lower() == "t":
                    script_name = "traditional"
                else:
                    continue

            char = row[script_name]
            if ord(char) in font_cmap:
                left, top, right, bottom = font.getbbox(char)
                # we'll calculate the required dimensions from the results, and we'll add a little wiggle room
                required_width = ceil(abs((right - left) + 2 * opts.width_padding))
                img_height = ceil(abs((bottom - top) + 2 * opts.height_padding))
                img1 = Image.new("RGB", (required_width, img_height), "white")
                draw1 = ImageDraw.Draw(img1)
                draw1.text(
                    (opts.width_padding, img_height / 2),
                    text=char,
                    font=font,
                    fill="black",
                    anchor="lm",
                    align="left",
                )
                file_name = f"{ord(char)}_{script_name}_{font_name}.png"
                img1.save(
                    os.path.join(
                        root,
                        opts.data_folder,
                        opts.image_save_location,
                        file_name,
                    )
                )

                manifest_arr = manifest_arr + [file_name, char, font_name, script_name]
                manifest_df.loc[len(manifest_df)] = [
                   file_name,
                   char,
                   font_name,
                   script_name,
                ]
            else:
                error_manifest_arr = error_manifest_arr + [char, font_name, script_name]
                error_manifest_df.loc[len(error_manifest_df)] = [
                   char,
                   font_name,
                   script_name,
                ]
        print("Done")

    # At the end we'll save our manifest df as a csv, this is important since this stores our ys for each X, the X being the image
    if not os.path.exists(
        os.path.join(root, opts.data_folder, opts.manifest_save_location)
    ):
        os.makedirs(
            os.path.join(root, opts.data_folder, opts.manifest_save_location)
        )
    manifest_df.to_csv(
        os.path.join(
            root,
            opts.data_folder,
            opts.manifest_save_location,
            opts.manifest_name,
        ),
        index=False,
    )
    # We will also save our error manifest, this tells us which characters were not able to be rendered with a particular font, we'll save it in the same location as the previous one, but with a different name of course
    error_manifest_df.to_csv(
        os.path.join(
            root,
            opts.data_folder,
            opts.manifest_save_location,
            opts.error_manifest_name,
        ),
        index=False,
    )


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
        "--hanzi_location",
        type=str,
        default="hsk_characters/",
        help="Folder containing the created hanzi vocabulary list",
    )

    parser.add_argument(
        "--hanzi_file_name",
        type=str,
        default="characters.csv",
        help="Name of the file containing the hanzi character list",
    )

    parser.add_argument(
        "--image_save_location",
        type=str,
        default="hanzi_imgs/",
        help="Directory in which to save the images with the rendered text lines",
    )

    parser.add_argument(
        "--manifest_save_location",
        type=str,
        default="hanzi_images_manifest/",
        help="Location in which to save the manifest file for the generated images",
    )

    parser.add_argument(
        "--manifest_name",
        type=str,
        default="manifest.csv",
        help="Name for the manifest file, it stores the ground truth, image_name, font used, etc",
    )

    parser.add_argument(
        "--error_manifest_name",
        type=str,
        default="error_manifest.csv",
        help="Name for the manifest of the text lines that could not be rendered due to at least one of the characters not being in the target font, stores problem hanzi, its codepoint and font ",
    )

    parser.add_argument(
        "--character_size",
        type=int,
        default=100,
        help="Character size, in pixels, characters are more or less squares in their dimensions",
    )

    parser.add_argument(
        "--width_padding",
        type=int,
        default=50,
        help="Padding added on the sides of the image's borders to avoid text overflowing",
    )

    parser.add_argument(
        "--height_padding",
        type=int,
        default=50,
        help="Padding added on the top and bottom of the image's borders to avoid text overflowing",
    )

    parser.add_argument(
        "--hanzi_styles",
        type=str,
        default="S",
        help="Which hanzi types to include: S - Simplified, T - Traditional, both can be included in any order, and with any capitalization, any other letters will be ignored",
    )

    opts = parser.parse_args()
    utils.delete_images(
        opts.data_folder, opts.image_save_location, opts.manifest_save_location
    )
    write_images(opts)


if __name__ == "__main__":
    main()
