import argparse
import os
import shutil
from math import ceil
from pathlib import Path

from PIL import Image, ImageDraw

# File with the parameters used to generate a dataset, saved next to its manifest
GENERATION_PARAMETERS_NAME = "generation_parameters.json"


def find_project_root():
    """
    Finds directory's root

    Returns
    ---------
    Path object with the directory's root
    """
    leaf = Path(__file__).parent
    while not Path(leaf / "pyproject.toml").exists():
        if leaf.parent == leaf:
            raise FileNotFoundError(
                "Directory's root couldn't be found, are you certain a 'pyproject.toml file exists at the project directory?'"
            )
        leaf = leaf.parent
    return leaf


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


def get_fonts_list(root, data_folder, font_folder) -> list[str]:
    """
    Returns the files stored in the fonts directory, that is, returns a list of all the fonts to be rendered

    Parameters
    ----------
    root: str or Path
        Project's root directory
    data_folder: str
        Root folder for project data
    font_folder: str
        Folder in which the fonts are located

    Returns
    ----------
    List of fonts stored in specified directory
    """
    return os.listdir(os.path.join(root, data_folder, font_folder))


def delete_images(root, data_folder, image_location, manifest_location):
    """
    Deletes all the images and the corresponding manifest in the specified location

    Parameters
    -----------
    root: str or Path
        Project's root directory
    data_folder: str
        Root folder for project data
    image_location: str
        Folder with the images to be deleted
    manifest_location: str
        Folder with the manifests to be deleted
    """
    # Source - https://stackoverflow.com/a/185941
    # Posted by Nick Stinemates, modified by community. See post 'Timeline' for change history
    # Retrieved 2026-07-08, License - CC BY-SA 4.0

    imgs_folder = os.path.join(root, data_folder, image_location)
    manifest_folder = os.path.join(root, data_folder, manifest_location)

    if not os.path.exists(imgs_folder):
        os.makedirs(imgs_folder)
    if not os.path.exists(manifest_folder):
        os.makedirs(manifest_folder)

    for filename in os.listdir(imgs_folder):
        file_path = os.path.join(imgs_folder, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)

    for filename in os.listdir(manifest_folder):
        file_path = os.path.join(manifest_folder, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)


# Source - https://stackoverflow.com/a/37311125
# Posted by alvas, modified by community. See post 'Timeline' for change history
# Retrieved 2026-08-17, License - CC BY-SA 3.0
def is_hanzi(character):
    """ "
    Checks whether character is CJK.

    :param character: The character that needs to be checked.
    :type character: char
    :return: bool
    """
    return any(
        start <= ord(character) <= end
        for start, end in [
            (0x4E00, 0x9FFF),  # CJK Unified Ideographs
            (0x3400, 0x4DBF),  # Extension A
            (0x20000, 0x2A6DF),  # Extension B
            (0x2A700, 0x2B73F),  # Extension C
            (0x2B740, 0x2B81F),  # Extension D
            (0x2B820, 0x2CEAF),  # Extension E
            (0x2CEB0, 0x2EBEF),  # Extension F
            (0x2EBF0, 0x2EE5F),  # Extension I
            (0x30000, 0x3134F),  # Extension G
            (0x31350, 0x323AF),  # Extension H
            (0xF900, 0xFAFF),  # CJK Compatibility Ideographs
            (0x2F800, 0x2FA1F),  # CJK Compatibility Ideographs Supplement
        ]
    )


def render_text(text, font, width_padding, height_padding):
    """
    Renders text, a single character or a whole line, black on a white background

    Parameters
    ----------
    text: str
        Text that will be rendered in the image
    font: PIL.ImageFont.FreeTypeFont
        Loaded font in which the text will be written
    width_padding: int
        Padding, in pixels, added on the left and right of the text
    height_padding: int
        Padding, in pixels, added above and below the text

    Returns
    ----------
    PIL.Image.Image
        RGB image with the rendered text, sized to fit the text plus the padding
    """
    left, top, right, bottom = font.getbbox(text)
    # The image is as big as the text's bounding box plus the padding on each side
    required_width = ceil(abs((right - left) + 2 * width_padding))
    img_height = ceil(abs((bottom - top) + 2 * height_padding))
    img = Image.new("RGB", (required_width, img_height), "white")
    draw = ImageDraw.Draw(img)
    draw.text(
        (width_padding, img_height / 2),
        text=text,
        font=font,
        fill="black",
        anchor="lm",
        align="left",
    )

    return img
