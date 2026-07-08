from PIL import PcfFontFile, Image, ImageDraw, ImageFont
import os, shutil
import argparse
from hanzi_ocr import utils
import pandas as pd

root_ = utils.find_project_root()

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
    if opts.delete_previous:
        folder = os.path.join(root_, opts.data_folder, opts.save_location)
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))


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


def get_hanzi_list(opts):
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
    hanzi_df = pd.read_csv(
        os.path.join(root_, opts.data_folder, opts.hanzi_location, opts.hanzi_name_file)
    )
    return hanzi_df


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
    fonts = get_fonts_list(opts)
    hanzi_df = get_hanzi_list(opts)
    num_img = 0
    for font in fonts:
        with open(
            os.path.join(root_, opts.data_folder, opts.font_location, font), "rb"
        ) as fp:
            font = PcfFontFile.PcfFontFile(fp)
            imagefont = font.to_imagefont()
            for r in hanzi_df.itertuples(index=False):
                if r.traditional != r.simplified:
                    img1 = Image.new("RGB", (400, 300), "white")
                    draw1 = ImageDraw.Draw(img1)
                    draw1.text((0, 0), r.traditional, font=imagefont)
                    img1.save(
                        os.path.join(
                            root_,
                            opts.data_folder,
                            opts.save_location,
                            f"hanzi_img_{num_img}",
                        )
                    )
                    num_img += 1

                    img2 = Image.new("RGB", (400, 300), "white")
                    draw2 = ImageDraw.Draw(img2)
                    draw2.text((0, 0), r.simplified, font=imagefont)
                    img2.save(
                        os.path.join(
                            root_,
                            opts.data_folder,
                            opts.save_location,
                            f"hanzi_img_{num_img}",
                        )
                    )
                    num_img += 1
                else:
                    img = Image.new("RGB", (400, 300), "white")
                    draw = ImageDraw.Draw(img)
                    draw.text((0, 0), r.traditional, font=imagefont)
                    img.save(
                        os.path.join(
                            root_,
                            opts.data_folder,
                            opts.save_location,
                            f"hanzi_img_{num_img}",
                        )
                    )
                    num_img += 1


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
        default="characters",
        help="Folder with processed hanzi list",
    )

    parser.add_argument(
        "--hanzi_file_name",
        type=str,
        default="characters.csv",
        help="File in which the hanzi list is stored",
    )

    parser.add_argument(
        "--save_location",
        type=str,
        default="character_images/",
        help="Directory in which to save the images with the characters",
    )

    parser.add_argument(
        "--delete_previous_images",
        type=utils.str2bool,
        default=True,
        help="If True deletes all previously generated images in the destinatio folder"
        )

    opts = parser.parse_args()
    if opts.delete_previous_images:
        delete_images(opts)
    write_images(opts)


if __name__ == "__main__":
    main()
