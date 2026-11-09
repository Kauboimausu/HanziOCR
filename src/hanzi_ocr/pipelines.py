import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import get_worker_info


def seed_worker(worker_id):
    """Gives each DataLoader worker its own augmentation seed, meant to be used as worker_init_fn

    Every worker receives a copy of the dataset with the same augmentation pipeline,
    so without reseeding all workers would produce the same augmentations.
    The worker's seed comes from the DataLoader's generator plus the worker id,
    so it is different for each worker and reproducible between runs.

    Parameters
    -----------
    worker_id: int
        Id of the worker, passed by the DataLoader. Unused, get_worker_info already includes it in the seed
    """
    info = get_worker_info()
    info.dataset.augmentation_pipeline.set_random_seed(info.seed)

def augmentation_pipeline(random_state):
    return A.Compose(
        [
            A.Perspective(fit_output=True, keep_size=False, p=0.5),
            A.SafeRotate((-30, 30), p=0.6),
            A.Illumination(p=0.4),
            A.InvertImg(p=0.2),
            A.RandomShadow(p=0.7),
            A.AdditiveNoise(p=0.3),
            A.MotionBlur(p=0.25),
            A.Defocus(p=0.15),
        ],
        seed=random_state,
    )   

def normalization_pipeline():
    return A.Compose(
    [
        A.Resize(64, 64),
        A.Normalize(mean=0.5, std=0.5, max_pixel_value=255),
        ToTensorV2(),
    ]
)