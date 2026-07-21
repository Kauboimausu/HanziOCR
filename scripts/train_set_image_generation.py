from PIL import Image, ImageDraw, ImageFont
import os, shutil
import argparse
from hanzi_ocr import utils
import pandas as pd
from fontTools.ttLib import TTFont
from datasets import load_from_disk, Dataset
from opencc import OpenCC
import re


def take_wiki_excerpt(wiki_entry, excerpt_min_length, excerpt_max_length, random):
    """
    Takes a random excerpt from the wiki entry and returns it

    Parameters
    ----------
    wiki_entry
        An entry from Chinese Wikipedia

    excerpt_min_length: int
        Minimum length of the excerpt

    excerpt_man_length: int
        Maximum length of the excerpt

    random: random
        An initialized random instance, for reproducibility's sake

    Returns
    ---------
    An excerpt from the wikipedia article of the given length range, or None if the requested string is not possible
    """

    # First we'll see how long our excerpt will be
    # It is quite unlikely that the user has requested a string that is longer than
    # the length of the article, but just in case we'll check that it isn't
    # if the minimum length is longer than the article itself then we'll just skip this entry
    if excerpt_min_length > len(wiki_entry):
        return None
    elif excerpt_min_length == len(wiki_entry):
        return wiki_entry
    excerpt_length = random.randint(
        excerpt_min_length, min(excerpt_max_length, len(wiki_entry))
    )

    start_pointer = random.randint(0, len(wiki_entry) - excerpt_length)

    return wiki_entry[start_pointer : (start_pointer + excerpt_length)]


def load_wiki_dataset(opts):
    """
    Loads the dataset from the indicated disk location so the

    Parameters
    ------------
    opts: argparse.Namespace
        Parameters given by the user
    """
    root = utils.find_project_root()
    try:
        ds = load_from_disk(os.path.join(root, opts.data_folder, opts.wiki_ds_data))
        return ds
    except Exception as e:
        print(f"Dataset could not be loaded: {e}")


def delete_images(opts):
    """
    If requested by the user, deletes all the hanzi images previously generated

    Parameters
    -----------
    opts: argparse.Namespace
        Parameters given by the user
    """
    # Source - https://stackoverflow.com/a/185941
    # Posted by Nick Stinemates, modified by community. See post 'Timeline' for change history
    # Retrieved 2026-07-08, License - CC BY-SA 4.0

    root = utils.find_project_root()
    if opts.delete_previous_images:
        folder = os.path.join(root, opts.data_folder, opts.save_location)
        if not os.path.exists(folder):
            os.makedirs(folder)
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print("Failed to delete %s. Reason: %s" % (file_path, e))


def get_fonts_list(opts) -> list[str]:
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
    root = utils.find_project_root()
    try:
        return os.listdir(os.path.join(root, opts.data_folder, opts.font_location))
    except Exception as e:
        raise Exception(f"Folder could not be found: {e}")


def load_wikipedia_data(opts) -> Dataset:
    """
    Retrieves the dataset from the disk and returns it

    Parameters
    ----------
    opts: argparse.Namespace
        Parameters given by the user

    Returns
    ----------
    List of fonts stored in specified directory
    """
    root = utils.find_project_root()
    try:
        return load_from_disk(
            os.path.join(root, opts.data_folder, opts.zh_wiki_location)
        )
    except Exception as e:
        print(f"File could not be found at given location: {e}")


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


def clean_entry_text(wiki_entry, hanzi_styles, converter):
    """
    Removes garbage text from the wiki's entry. Also converts from traditional to simpliified, if need be

    Parameters
    -----------
    wiki_entry
        The entry to clean up

    Returns
    -----------
    Returns the entry split by newlines in the order: Simplified, Traditional. If either was not requested it won't be returned at all (the number of returns will change, None will not be returned in its place)
    """

    wiki_entry = re.sub(r"={2,}", "", wiki_entry)
    wiki_entry = re.sub(r"\\[a-zA-Z]+(\{[a-zA-Z0-9]+\})?", "", wiki_entry)
    wiki_entry = wiki_entry.replace("|", "")

    if "S" in hanzi_styles.upper() and "T" in hanzi_styles.upper():
        simplified_entry = converter.convert(wiki_entry)
        simplified_splits = simplified_entry.split("\n")
        traditional_splits = wiki_entry.split("\n")

        return simplified_splits, traditional_splits
    elif "S" in hanzi_styles.upper():
        simplified_entry = converter.convert(wiki_entry)
        simplified_splits = simplified_entry.split("\n")

        return simplified_splits
    elif "T" in hanzi_styles.upper():
        traditional_splits = wiki_entry.split("\n")

        return traditional_splits


def audit_font(
    font_location: str, hanzi_df: pd.DataFrame, audit_traditional: bool
) -> dict:
    """
    Tells the program which characters are able to be rendered by a
    particular font, those that aren't will be skipped so as to not
    generate something useless or even harmful for training

    Parameters
    ---------
    font_location: str
        The location of the font to be loaded
    hanzi_df: pd.Dataframe
        Pandas DataFrame generated by the previous script (make_char_list.py)
    audit_traditional: bool
        Whether or not to check if traditional versions of the characters
        are supported as well

    Returns
    ---------
        Dictionary with each character and whether it can be rendered
        or not by the font being audited
    """
    supported = {}
    try:
        with TTFont(font_location) as font:
            cmap = font.getBestCmap()
            for r in hanzi_df.itertuples(index=True):
                # Despite being saved in different columns in the same row,
                # traditional and simplified characters, if different, have
                # different code points, so we'll make sure to audit them
                # separately if indicated
                if r.simplified != r.traditional:
                    if audit_traditional:
                        traditional_codepoint = ord(r.traditional)
                        supported[traditional_codepoint] = traditional_codepoint in cmap

                    # we do not have to look up the codepoint for the simplified
                    # as we already have it in the codepoint column, which is the index
                    supported[r.Index] = r.Index in cmap
                # if the two characters are the same then we don't have to
                # do anything special
                else:
                    # Check if the code point exists in the cmap
                    supported[r.Index] = r.Index in cmap
    except Exception as e:
        print(f"Error loading font: {e}")
        return supported

    return supported


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
    manifest_df = pd.DataFrame(columns=["file name", "character", "font"])
    # We'll also generate a manifest for the characters that were not able to be rendered
    error_manifest_df = pd.DataFrame(columns=["codepoint", "character", "font"])

    # If the specified folder for the generated images doesn't already exist we'll create it
    if not os.path.exists(os.path.join(root, opts.data_folder, opts.save_location)):
        os.makedirs(os.path.join(root, opts.data_folder, opts.save_location))

    # We'll get the list of saved fonts and iterate through them
    fonts = get_fonts_list(opts)
    # hanzi_df = get_hanzi_list(opts)
    wikipedia_ds = load_wikipedia_data(opts)
    print(wikipedia_ds.column_names)
    print(type(wikipedia_ds[0]['text']))
    return
    num_img = 0
    # Weĺl load each font and iterate through our previously generated character list
    for font_name in fonts:
        try:
            font = ImageFont.truetype(
                font=os.path.join(
                    root, opts.data_folder, opts.font_location, font_name
                ),
                size=opts.character_size,
            )
        except Exception as e:
            print(f"ERROR: Could not load font {e}")

        supported_chars = audit_font(
            font_location=os.path.join(
                root, opts.data_folder, opts.font_location, font_name
            ),
            hanzi_df=hanzi_df,
            audit_traditional=opts.save_location,
        )
        for r in hanzi_df.itertuples(index=True):
            # for hanzi in ["影", "一", "口"]:
            # Some characters differ in their simplified and traditional forms, it this happens we'll save both if the save_traditional flag is True
            # Otherwise we'll keep only the simplified character
            if r.traditional != r.simplified:
                if opts.save_traditional:
                    trad_codepoint = ord(r.traditional)
                    # We'll make sure the character is able to even be rendered
                    if supported_chars[trad_codepoint] == True:
                        img1 = Image.new(
                            "RGB", (opts.image_width, opts.image_height), "white"
                        )
                        draw1 = ImageDraw.Draw(img1)
                        # textbox_val = draw1.textbbox((opts.image_width/2, opts.image_height/2), text=r.traditional, font=font, anchor="mm", align="center")
                        # print(textbox_val)
                        file_name = f"hanzi_{trad_codepoint}_{font_name}.png"
                        draw1.text(
                            (opts.image_width / 2, opts.image_height / 2),
                            text=r.traditional,
                            font=font,
                            fill="black",
                            anchor="mm",
                            align="center",
                        )
                        img1.save(
                            os.path.join(
                                root,
                                opts.data_folder,
                                opts.save_location,
                                file_name,
                            )
                        )
                        manifest_df.loc[len(manifest_df)] = [
                            file_name,
                            r.traditional,
                            font_name,
                        ]
                        num_img += 1
                    else:
                        # If it's not then we'll add it to the manifest for characters that couldn't be rendered
                        error_manifest_df.loc[len(error_manifest_df)] = [
                            trad_codepoint,
                            r.traditional,
                            font_name,
                        ]

                if supported_chars[r.Index] == True:
                    img2 = Image.new(
                        "RGB", (opts.image_width, opts.image_height), "white"
                    )
                    draw2 = ImageDraw.Draw(img2)
                    # textbox_val = draw2.textbbox((opts.image_width/2, opts.image_height/2), text=r.traditional, font=font, anchor="mm", align="center")
                    # print(textbox_val)
                    file_name = f"hanzi_{r.Index}_{font_name}.png"
                    draw2.text(
                        (opts.image_width / 2, opts.image_height / 2),
                        text=r.simplified,
                        font=font,
                        fill="black",
                        anchor="mm",
                        align="center",
                    )
                    img2.save(
                        os.path.join(
                            root,
                            opts.data_folder,
                            opts.save_location,
                            file_name,
                        )
                    )
                    manifest_df.loc[len(manifest_df)] = [
                        file_name,
                        r.simplified,
                        font_name,
                    ]
                    num_img += 1
                else:
                    error_manifest_df.loc[len(error_manifest_df)] = [
                        ord(r.simplified),
                        r.simplified,
                        font_name,
                    ]

            # If they're the same it doesn't really matter which we save, but we'll save the traditional since that's our priority
            else:
                if supported_chars[r.Index] == True:
                    img = Image.new(
                        "RGB", (opts.image_width, opts.image_height), "white"
                    )
                    draw = ImageDraw.Draw(img)
                    # textbox_val = draw.textbbox((opts.image_width/2, opts.image_height/2), text=r.simplified, font=font, anchor="mm", align="center")
                    # print(textbox_val)
                    file_name = f"hanzi_{r.Index}_{font_name}.png"
                    draw.text(
                        (opts.image_width / 2, opts.image_height / 2),
                        text=r.simplified,
                        font=font,
                        fill="black",
                        anchor="lm",
                        align="center",
                    )
                    img.save(
                        os.path.join(
                            root,
                            opts.data_folder,
                            opts.save_location,
                            file_name,
                        )
                    )
                    manifest_df.loc[len(manifest_df)] = [
                        file_name,
                        r.simplified,
                        font_name,
                    ]
                    num_img += 1
                else:
                    error_manifest_df.loc[len(error_manifest_df)] = [
                        ord(r.simplified),
                        r.simplified,
                        font_name,
                    ]

        # At the end we'll save our manifest df as a csv, this is important since this stores our ys for each X, the X being the image
        if not os.path.exists(
            os.path.join(root, opts.data_folder, opts.manifest_location)
        ):
            os.makedirs(os.path.join(root, opts.data_folder, opts.manifest_location))
        manifest_df.to_csv(
            os.path.join(
                root, opts.data_folder, opts.manifest_location, opts.manifest_name
            ),
            index=False,
        )
        # We will also save our error manifest, this tells us which characters were not able to be rendered with a particular font, we'll save it in the same location as the previous one, but with a different name of course
        error_manifest_df.to_csv(
            os.path.join(
                root,
                opts.data_folder,
                opts.manifest_location,
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
        "--zh_wiki_location",
        type=str,
        default="zh_wiki_data/",
        help="Folder with Mandarin Wikipedia texts",
    )

    parser.add_argument(
        "--save_location",
        type=str,
        default="zh_text_imgs/",
        help="Directory in which to save the images with the rendered text lines",
    )

    parser.add_argument(
        "--wiki_ds_location",
        type=str,
        default="zh_wiki_data/",
        help="Data where the Chinese wikipedia data was saved, if it hasn't been saved run the download_zh_wiki.py script",
    )

    parser.add_argument(
        "--manifest_location",
        type=str,
        default="images_manifest/",
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
        "--image_height",
        type=int,
        default=64,
        help="Height of the generated image, the width is calculated dynamically",
    )

    parser.add_argument(
        "--snippet_length_range",
        type=int,
        nargs=2,
        default=[1, 15],
        help="Range of the length of the snippets to be picked, first number is the minimum (inclusive), the second is the maximum (inclusive)",
    )

    parser.add_argument(
        "--character_size",
        type=int,
        default=60,
        help="Character size, in pixels. You want this to be smaller than the image height",
    )

    parser.add_argument(
        "--save_traditional",
        type=utils.str2bool,
        default=False,
        help="By default the data is pulled from traditional characters and converted into its simplified version, with the traditional version thrown away, if this is set to True it will not be thrown away",
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        default=21,
        help="Random seed used for picking out snippets of text to render",
    )

    parser.add_argument(
        "--hanzi_types",
        type=str,
        default="S",
        help="Which hanzi types to include: S - Simplified, T - Traditional, both can be included in any order, and with any capitalization",
    )

    parser.add_argument(
        "--delete_previous_images",
        type=utils.str2bool,
        default=True,
        help="If True deletes all previously generated images in the destination folder",
    )

    opts = parser.parse_args()
    if opts.delete_previous_images:
        delete_images(opts)
    write_images(opts)


if __name__ == "__main__":
    main()
