from pathlib import Path

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