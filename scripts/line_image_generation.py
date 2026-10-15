import argparse
import os
import re
from math import ceil

import numpy as np
import opencc
import pandas as pd
from datasets import Dataset, load_from_disk
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont

from hanzi_ocr import utils


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

    random: np.random
        An initialized numpy random instance, for reproducibility's sake

    Returns
    ---------
    An excerpt from the wikipedia article of the given length range, or None if the requested string is not possible
    """

    # First we'll see how long our excerpt will be
    # It is quite unlikely that the user has requested a string that is longer than
    # the length of the article, but just in case we'll check that it isn't
    # if the minimum length is longer than the article itself then we'll just skip this entry
    if excerpt_min_length > excerpt_max_length:
        raise ValueError("The minimum length cannot be higher than the maximum length")
    elif wiki_entry is None or excerpt_min_length > len(wiki_entry):
        return None
    
    excerpt_length = random.integers(
        excerpt_min_length,
        min(excerpt_max_length, len(wiki_entry)),
        endpoint=True,
    )

    start_pointer = random.integers(0, len(wiki_entry) - excerpt_length, endpoint=True)

    return wiki_entry[start_pointer : (start_pointer + excerpt_length)]


def get_excerpt_hanzi_proportion(excerpt):
    """
    Returns the percentage of hanzi to total characters in a given string
    Parameters
    -----------
    excerpt: str
        The string to be measured
    Returns
    -----------
    percentage: float
        the percentage of hanzi to total characters in the excerpt
    """

    total_hanzi = 0
    for char in excerpt:
        if utils.is_hanzi(char):
            total_hanzi += 1

    return (total_hanzi / len(excerpt)) * 100


def load_wikipedia_data(root, data_folder, zh_wiki_location) -> Dataset:
    """
    Retrieves the dataset from the disk and returns it

    Parameters
    ----------
    root: str or Path
        Project's root directory
    data_folder: str
        Root folder for project data
    zh_wiki_location: str
        Folder in which the Chinese Wikipedia dataset was saved

    Returns
    ----------
    The Chinese Wikipedia dataset
    """
    return load_from_disk(os.path.join(root, data_folder, zh_wiki_location))


def clean_entry_text(
    wiki_entry, hanzi_styles, simplified_converter, traditional_converter
):
    """
    Removes garbage text from the wiki's entry. Also converts from traditional to simpliified, if need be

    Parameters
    -----------
    wiki_entry: str
        The entry to clean up
    hanzi_styles: str
        The hanzi types to be included:
            ``S`` for simplified
            ``T`` for traditional
            Both can be included
    simplified_converter: openCC.OpenCC
        Initialized OpenCC instance to convert from traditional to simplified
    traditional_converter: openCC.OpenCC
        Initialized OpenCC instance to convert from simplified to traditional

    Returns
    -----------
    Returns the entry split by newlines in the order: Simplified, Traditional. If either was not requested it won't be returned at all (the number of returns will change, None will not be returned in its place)
    """
    wiki_entry = wiki_entry.replace("**", "")
    wiki_entry = wiki_entry.replace("__", "")
    wiki_entry = wiki_entry.replace("##", "")
    wiki_entry = re.sub(r"\$\$[^$]*\$\$|\$[^$]*\$", "", wiki_entry)
    wiki_entry = wiki_entry.replace("「!」", "")
    wiki_entry = wiki_entry.replace("( ! )", "")
    wiki_entry = wiki_entry.replace("「?」", "")
    wiki_entry = wiki_entry.replace("( ? )", "")
    wiki_entry = wiki_entry.replace("[io]", "")

    if "s" in hanzi_styles.lower() and "t" in hanzi_styles.lower():
        simplified_entry = simplified_converter.convert(wiki_entry)
        simplified_splits = simplified_entry.split("\n")

        traditional_entry = traditional_converter.convert(wiki_entry)
        traditional_splits = traditional_entry.split("\n")

        return [(simplified_splits, "simplified"), (traditional_splits, "traditional")]
    elif "s" in hanzi_styles.lower():
        simplified_entry = simplified_converter.convert(wiki_entry)
        simplified_splits = simplified_entry.split("\n")

        return [(simplified_splits, "simplified")]
    elif "t" in hanzi_styles.lower():
        traditional_entry = traditional_converter.convert(wiki_entry)
        traditional_splits = traditional_entry.split("\n")

        return [(traditional_splits, "traditional")]


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
    manifest_df = pd.DataFrame(columns=["file name", "text", "font", "type"])
    # We'll also generate a manifest for the characters that were not able to be rendered
    error_manifest_df = pd.DataFrame(columns=["text", "font", "type"])

    # If the specified folder for the generated images doesn't already exist we'll create it
    if not os.path.exists(os.path.join(root, opts.data_folder, opts.img_save_location)):
        os.makedirs(os.path.join(root, opts.data_folder, opts.img_save_location))

    # We'll get the list of saved fonts and iterate through them
    fonts = utils.get_fonts_list(root, opts.data_folder, opts.font_location)
    wikipedia_ds = load_wikipedia_data(root, opts.data_folder, opts.zh_wiki_location)

    if "s" not in opts.hanzi_styles.lower() and "t" not in opts.hanzi_styles.lower():
        print(
            "ERROR: Either 'S' (Simplified) or 'T' (Traditional) has to be included in the '--hanzi_styles' flag"
        )
        return

    # trial_counter = 5
    # We'll initialize our rng and traditional to simplified chinese converter
    rng = np.random.default_rng(opts.random_seed)
    simplified_converter = opencc.OpenCC("t2s.json")
    traditional_converter = opencc.OpenCC("s2t.json")

    # Weĺl load each font and iterate through our previously generated character list
    for font_name in fonts:
        # reserved is the name for fonts that i dont want to render yet
        if "reserved" in font_name:
            continue
        font = ImageFont.truetype(
            font=os.path.join(root, opts.data_folder, opts.font_location, font_name),
            size=opts.character_size,
        )

        tfont = TTFont(
            os.path.join(root, opts.data_folder, opts.font_location, font_name)
        )
        font_cmap = tfont.getBestCmap()

        # print(len(wikipedia_ds)) - 2,533,212
        # We'll iterate through each dataset entry
        random_entries = rng.choice(
            len(wikipedia_ds), opts.pages_per_font, replace=False
        )
        font_subset = wikipedia_ds.select(random_entries)
        for wiki_page in font_subset:
            # We'll obtain the parts of the page we need, we'll keep the page id for the manifest and the markdown for the text
            pid = wiki_page["pageid"]
            pmd = wiki_page["markdown"]

            # We'll clean the text as much as possible from formatting artifacts
            cleaned_split_text = clean_entry_text(
                pmd, opts.hanzi_styles, simplified_converter, traditional_converter
            )

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
                        rng,
                    )

                    if excerpt is None or len(excerpt) == 0:
                        continue

                    hanzi_perc = get_excerpt_hanzi_proportion(excerpt)
                    if hanzi_perc < opts.min_hanzi_proportion:
                        continue

                    # We will audit each character against the font lest we get a "tofu", which is useless and even harmful to our application
                    renderable = True
                    for char in excerpt:
                        if ord(char) not in font_cmap:
                            # If a character is not able to be rendered we will skip the section altogether
                            # but before skipping it we'll register it in the error manifest
                            error_manifest_df.loc[len(error_manifest_df)] = [
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
                        required_width = ceil(
                            abs((right - left) + 2 * opts.width_padding)
                        )
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
                                opts.img_save_location,
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
            os.path.join(root, opts.data_folder, opts.manifest_save_location)
        ):
            os.makedirs(
                os.path.join(root, opts.data_folder, opts.manifest_save_location)
            )
        manifest_df.to_csv(
            os.path.join(
                root, opts.data_folder, opts.manifest_save_location, opts.manifest_name
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
        "--zh_wiki_location",
        type=str,
        default="zh_wiki_data/",
        help="Folder with Mandarin Wikipedia texts",
    )

    parser.add_argument(
        "--img_save_location",
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
        "--manifest_save_location",
        type=str,
        default="text_images_manifest/",
        help="Location in which to save the manifest file for the generated images",
    )

    parser.add_argument(
        "--pages_per_font",
        type=int,
        default=5000,
        help="How many pages to use to make images per font, note that each page likely renders more than one image",
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
        "--min_hanzi_proportion",
        type=int,
        default=70,
        help="The minimum percentage of hanzi that an excerpt is required to have in order to be rendered, accepted as a percentage (out of a hundred)",
    )

    parser.add_argument(
        "--hanzi_styles",
        type=str.lower,
        choices=["s", "t", "st", "ts"],
        required=True,
        help="Which hanzi types to include: S - Simplified, T - Traditional, both can be included in any order, and with any capitalization",
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        default=21,
        help="Random seed used for picking out snippets of text to render",
    )

    opts = parser.parse_args()
    root = utils.find_project_root()
    utils.delete_images(
        root, opts.data_folder, opts.img_save_location, opts.manifest_save_location
    )
    write_images(root, opts)


if __name__ == "__main__":
    main()
