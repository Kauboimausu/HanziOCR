import albumentations as A
from albumentations.pytorch import ToTensorV2


def augmentation_pipeline(random_state):
    return A.Compose(
        [
            A.Illumination(p=0.4),
            A.RandomShadow(p=0.7),
            A.AdditiveNoise(p=0.3),
            A.InvertImg(p=0.15),
            A.MotionBlur(p=0.25),
            A.Defocus(p=0.15),
            A.Perspective(fit_output=True, keep_size=False, p=0.5),
            A.SafeRotate((-30, 30), p=0.6),
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