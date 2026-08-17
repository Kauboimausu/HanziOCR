from pathlib import Path
import argparse
import os
import shutil

def find_project_root():
    """
    Finds directory's root

    Returns
    ---------
    Path object with the directory's root
    """
    leaf = Path(__file__).parent
    while not Path(leaf/"pyproject.toml").exists():
        if leaf.parent == leaf:
            raise Exception("Directory's root couldn't be found")
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
    try:
        return os.listdir(os.path.join(root, data_folder, font_folder))
    except Exception as e:
        raise Exception(f"Folder could not be found: {e}")


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
    if not os.path.exists(imgs_folder):
        os.makedirs(imgs_folder)
        for filename in os.listdir(imgs_folder):
            file_path = os.path.join(imgs_folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print("Failed to delete %s. Reason: %s" % (file_path, e))

                manifest_folder = os.path.join(root, data_folder, manifest_location)
                for filename in os.listdir(manifest_folder):
                    file_path = os.path.join(manifest_folder, filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as e:
                        print("Failed to delete %s. Reason: %s" % (file_path, e))