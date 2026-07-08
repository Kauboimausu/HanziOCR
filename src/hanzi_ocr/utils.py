from pathlib import Path
import argparse

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