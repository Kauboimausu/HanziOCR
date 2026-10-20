import argparse
import json
import os

import pandas as pd
from fontTools.ttLib import TTFont
from PIL import ImageFont

from hanzi_ocr import utils
from hanzi_ocr.utils import GENERATION_PARAMETERS_NAME


def get_hanzi_list(root, data_folder, hanzi_location, hanzi_file_name) -> pd.DataFrame:
    """
    Reads the csv in which the hanzis are stored and returns the dataframe

    Parameters
    ----------
    root: str or Path
        Project's root directory
    data_folder: str
        Root folder for project data
    hanzi_location: str
        Folder containing the hanzi vocabulary list
    hanzi_file_name: str
        Name of the csv file with the hanzi vocabulary list

    Returns
    ---------
    Pandas dataframe with the hanzi data
    """
    hanzi_df = pd.read_csv(
        os.path.join(root, data_folder, hanzi_location, hanzi_file_name),
        index_col="codepoint",
    )
    return hanzi_df


def get_font_weights(font, wanted_weights):
    """Returns the weights the font will render"""
    try: 
        available = [name.decode() for name in font.get_variation_names()]
    except OSError:
        return [None]
    available_weights = [available_weight for available_weight in available if available_weight in wanted_weights]
    if len(available_weights):
        return available_weights
    else:
        return [None]



def write_images(root, opts):
    """
    Generates the training images by loading each font stored locally,
    iterating through them one at a time, then for each font it iterates
    through each character stored in the character list, for each of these it
    generates an image using Pillow

    Parameters
    -----------
    root: str or Path
        Project's root directory
    opts: argparse.Namespace
        Parameters given by the user
    """
    # If it doesn't exist we create a folder for the fonts
    if not os.path.exists(os.path.join(root, opts.data_folder, opts.font_location)):
        print(
            "WARNING: folder with fonts doesn't exist, make sure the name is properly given"
        )
        return

    # We want to generate a manifest to keep track of what character, font, size, etc is being generated on each file.
    manifest_df = pd.DataFrame(
        columns=["file name", "codepoint", "text", "font", "weight", "type"]
    )
    # We'll also generate a manifest for the characters that were not able to be rendered
    error_manifest_df = pd.DataFrame(columns=["codepoint", "text", "font", "type"])

    # If the specified folder for the generated images doesn't already exist we'll create it
    if not os.path.exists(
        os.path.join(root, opts.data_folder, opts.image_save_location)
    ):
        os.makedirs(os.path.join(root, opts.data_folder, opts.image_save_location))

    # We'll get the list of saved fonts and iterate through them
    fonts = utils.get_fonts_list(root, opts.data_folder, opts.font_location)
    hanzi_df = get_hanzi_list(
        root, opts.data_folder, opts.hanzi_location, opts.hanzi_file_name
    )

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
        font_path = os.path.join(root, opts.data_folder, opts.font_location, font_name)
        font = ImageFont.truetype(font=font_path, size=opts.character_size)
        tfont = TTFont(font_path)
        font_cmap = tfont.getBestCmap()

        font_variations = get_font_weights(font, opts.font_weights)
        for variation in font_variations:
            if variation is not None:
                font.set_variation_by_name(variation)
            variation_name = variation or "default"
            print(f"Generating images for font {font_name} {variation_name}")
            for _, row in hanzi_df.iterrows():
                for script in opts.hanzi_styles:
                    if script.lower() == "s":
                        script_name = "simplified"
                    elif script.lower() == "t":
                        script_name = "traditional"
                    else:
                        continue

                    char = row[script_name]
                    if not utils.is_hanzi(char):
                        continue
                    if ord(char) in font_cmap:
                        img = utils.render_text(
                            char, font, opts.width_padding, opts.height_padding
                        )
                        file_name = f"{ord(char)}_{script_name}_{font_name}_{variation_name}.png"
                        img.save(
                            os.path.join(
                                root,
                                opts.data_folder,
                                opts.image_save_location,
                                file_name,
                            )
                        )
                        manifest_df.loc[len(manifest_df)] = [
                            file_name,
                            ord(char),
                            char,
                            font_name,
                            variation_name,
                            script_name,
                        ]
                    else:
                        error_manifest_df.loc[len(error_manifest_df)] = [
                            ord(char),
                            char,
                            font_name,
                            script_name,
                        ]
                        break
        print("Done")

    # At the end we'll save our manifest df as a csv, this is important since this stores our ys for each X, the X being the image
    if not os.path.exists(
        os.path.join(root, opts.data_folder, opts.manifest_save_location)
    ):
        os.makedirs(os.path.join(root, opts.data_folder, opts.manifest_save_location))
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
    # We save the parameters used to generate the images next to the manifest, training copies them into the model's folder
    with open(
        os.path.join(
            root,
            opts.data_folder,
            opts.manifest_save_location,
            GENERATION_PARAMETERS_NAME,
        ),
        "w",
    ) as jsonfile:
        json.dump(vars(opts), jsonfile, indent=4)


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
        type=str.lower,
        choices=["s", "t", "st", "ts"],
        required=True,
        help="Which hanzi types to include: S - Simplified, T - Traditional, both can be included in any order, and with any capitalization",
    )

    parser.add_argument(
        "--font_weights",
        type=str,
        nargs="+",
        default=["Light", "Regular", "Bold"],
        help="Weights rendered for variable fonts, written as the font names them, e.g. Light Regular Bold. Fonts with a single weight are rendered once regardless",
    )

    opts = parser.parse_args()
    root = utils.find_project_root()
    utils.delete_images(
        root, opts.data_folder, opts.image_save_location, opts.manifest_save_location
    )
    write_images(root, opts)


if __name__ == "__main__":
    main()
