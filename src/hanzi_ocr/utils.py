import argparse
import os
import shutil
from pathlib import Path


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


def get_fonts_list(data_folder, font_folder) -> list[str]:
    """
    Returns the files stored in the fonts directory, that is, returns a list of all the fonts to be rendered

    Parameters
    ----------
    data_folder: str
        Root folder for project data
    font_folder: str
        Folder in which the fonts are located

    Returns
    ----------
    List of fonts stored in specified directory
    """
    root = find_project_root()
    return os.listdir(os.path.join(root, data_folder, font_folder))


def delete_images(data_folder, image_location, manifest_location):
    """
    Deletes all the images and the corresponding manifest in the specified location

    Parameters
    -----------
    opts: argparse.Namespace
    Parameters given by the user
    """
    # Source - https://stackoverflow.com/a/185941
    # Posted by Nick Stinemates, modified by community. See post 'Timeline' for change history
    # Retrieved 2026-07-08, License - CC BY-SA 4.0

    root = find_project_root()
    imgs_folder = os.path.join(root, data_folder, image_location)
    manifest_folder = os.path.join(root, data_folder, manifest_location)

    if not os.path.exists(imgs_folder):
        os.makedirs(imgs_folder)
    if not os.path.exists(manifest_folder):
        os.makedirs(manifest_folder)

    for filename in os.listdir(imgs_folder):
        file_path = os.path.join(imgs_folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except ReferenceError as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

    manifest_folder = os.path.join(root, data_folder, manifest_location)
    for filename in os.listdir(manifest_folder):
        file_path = os.path.join(manifest_folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except ReferenceError as e:
            print(f"Failed to delete {file_path}. Reason: {e}")


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
            (ord("\u3300"), ord("\u33ff")),
            (ord("\ufe30"), ord("\ufe4f")),
            (ord("\uf900"), ord("\ufaff")),
            (ord("\U0002f800"), ord("\U0002fa1f")),
            (ord("\u4e00"), ord("\u9fff")),
            (ord("\u3400"), ord("\u4dbf")),
            (ord("\U00020000"), ord("\U0002a6df")),
            (ord("\U0002a700"), ord("\U0002b73f")),
            (ord("\U0002b740"), ord("\U0002b81f")),
            (ord("\U0002b820"), ord("\U0002ceaf")),
        ]
    )
