from PIL import Image, ImageDraw, ImageFont
import os, shutil
import argparse
from hanzi_ocr import utils
import pandas as pd
from fontTools.ttLib import TTFont
from datasets import load_from_disk, Dataset
import opencc
import re
from random import Random
from math import ceil


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
    if wiki_entry is None or excerpt_min_length > len(wiki_entry):
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

    # wiki_entry = re.sub(r"={2,}", "", wiki_entry)
    # wiki_entry = re.sub(r"\\[a-zA-Z]+(\{[a-zA-Z0-9]+\})?", "", wiki_entry)
    # wiki_entry = re.sub(
    #    r"(</?p>|</p *+>|</?b>|</b *+>|</?span>|</?span id=" * +">|</?h[1-6]>)|",
    #    "",
    #    wiki_entry,
    # )

    wiki_entry = wiki_entry.replace("**", "")
    wiki_entry = wiki_entry.replace("__", "")
    wiki_entry = wiki_entry.replace("##", "")
    wiki_entry = re.sub(r"$.+$", "", wiki_entry)
    wiki_entry = wiki_entry.replace("「!」", "")
    wiki_entry = wiki_entry.replace("( ! )", "")
    wiki_entry = wiki_entry.replace("「?」", "")
    wiki_entry = wiki_entry.replace("( ? )", "")
    # wiki_entry = re.sub(r"( ! )|「!」", "!", wiki_entry)
    # wiki_entry = re.sub(r"( ? )|「?」", "!", wiki_entry)

    if ("S" in hanzi_styles or "s" in hanzi_styles) and (
        "T" in hanzi_styles or "t" in hanzi_styles
    ):
        simplified_entry = converter.convert(wiki_entry)
        simplified_splits = simplified_entry.split("\n")
        traditional_splits = wiki_entry.split("\n")

        return (simplified_splits, "simplified"), (traditional_splits, "traditional")
    elif "S" in hanzi_styles:
        simplified_entry = converter.convert(wiki_entry)
        simplified_splits = simplified_entry.split("\n")

        return (simplified_splits, "simplified")
    elif "T" in hanzi_styles:
        traditional_splits = wiki_entry.split("\n")

        return (traditional_splits, "traditional")


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

    # If the specified folder for the generated images doesn't already exist we'll create it
    if not os.path.exists(os.path.join(root, opts.data_folder, opts.save_location)):
        os.makedirs(os.path.join(root, opts.data_folder, opts.save_location))

    # We'll get the list of saved fonts and iterate through them
    fonts = get_fonts_list(opts)
    # hanzi_df = get_hanzi_list(opts)
    wikipedia_ds = load_wikipedia_data(opts)

    if "S" not in opts.hanzi_styles or "T" not in opts.hanzi_styles:
        print(
            "ERROR: Either 'S' (Simplified) or 'T' (Traditional) has to be included in the '--hanzi_styles' flag"
        )
        return

    # trial_counter = 5
    # We'll initialize our rng and traditional to simplified chinese converter
    random = Random(opts.random_seed)
    converter = opencc.OpenCC("t2s.json")

    # Weĺl load each font and iterate through our previously generated character list
    for font_name in fonts:
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
            print(f"ERROR: Could not load font {e}")

        # We'll iterate through each dataset entry
        for wiki_page in wikipedia_ds:
            # We'll obtain the parts of the page we need, we'll keep the page id for the manifest and the markdown for the text
            pid = wiki_page["pageid"]
            pmd = wiki_page["markdown"]

            # We'll clean the text as much as possible from formatting artifacts
            cleaned_split_text = clean_entry_text(pmd, opts.hanzi_styles, converter)

            # Depending on the parameters given we will have more than one script type (traditional and simplified), we'll render an image for each
            for script_type_splits in cleaned_split_text:
                # We'll unpack the result, the first part is the splits and the second is the name of the script, which we'll use for the manifest
                splits, script_name = script_type_splits
                # We split the articles by newlines, we will take an excerpt for each of these
                for split_num, split in enumerate(splits):
                    # If '\n\n' was encountered it will result in 0 length splits, we will skip these as they are useless
                    excerpt = take_wiki_excerpt(
                        split,
                        opts.snippet_length_range[0],
                        opts.snippet_length_range[1],
                        random,
                    )

                    if excerpt is None or len(excerpt) == 0:
                        continue

                    # We will audit each character against the font lest we get a "tofu", which is useless and even harmful to our application
                    renderable = True
                    for char in excerpt:
                        if char not in font_cmap:
                            # If a character is not able to be rendered we will skip the section altogether
                            # but before skipping it we'll register it in the error manifest
                            error_manifest_df.loc[len(manifest_df)] = [
                                char,
                                font_name,
                                script_name,
                            ]
                            renderable = False

                    # Otherwise we will render the character
                    # Before creating the image we'll see how much space we need to fit the text
                    if renderable:
                        left, top, right, bottom = font.getbbox(excerpt)
                        # we'll calculate the required dimensions from the results, and we'll add a little wiggle room
                        required_width = ceil(abs((right - left) + 2 * opts.width_padding))
                        img_height = ceil(abs((bottom - top) + 2 * opts.height_padding))
                        img1 = Image.new("RGB", (required_width, img_height), "white")
                        draw1 = ImageDraw.Draw(img1)
                        draw1.text(
                            (opts.width_padding, img_height / 2),
                            text=excerpt,
                            font=font,
                            fill="black",
                            anchor="lm",
                            align="left",
                        )
                        file_name = f"{pid}_{split_num}_{script_name}_{font_name}.png"
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
                            excerpt,
                            font_name,
                            script_name,
                        ]

            # trial_counter -= 1
            # if trial_counter < 0:
            #    return

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
        "--character_size",
        type=int,
        default=60,
        help="Character size, in pixels, characters are more or less squares in their dimensions",
    )

    parser.add_argument(
        "--width_padding",
        type=int,
        default="30",
        help="Padding added on the sides of the image's borders to avoid text overflowing",
    )

    parser.add_argument(
        "--height_padding",
        type=int,
        default=20,
        help="Padding added on the top and bottom of the image's borders to avoid text overflowing",
    )

    parser.add_argument(
        "--snippet_length_range",
        type=int,
        nargs=2,
        default=[3, 15],
        help="Range of the length of the snippets to be picked, first number is the minimum (inclusive), the second is the maximum (inclusive)",
    )

    parser.add_argument(
        "--hanzi_styles",
        type=str,
        nargs="+",
        default=["S", "T"],
        help="Which hanzi types to include: S - Simplified, T - Traditional, both can be included in any order, and with any capitalization, any other letters will be ignored",
    )

    parser.add_argument(
        "--delete_previous_images",
        type=utils.str2bool,
        default=True,
        help="If True deletes all previously generated images in the destination folder",
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        default=21,
        help="Random seed used for picking out snippets of text to render",
    )

    opts = parser.parse_args()
    if opts.delete_previous_images:
        delete_images(opts)
    write_images(opts)


if __name__ == "__main__":
    main()
