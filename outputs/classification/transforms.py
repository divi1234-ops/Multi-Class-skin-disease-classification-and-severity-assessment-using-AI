import albumentations as A
from albumentations.pytorch import ToTensorV2
from configs.config import IMAGE_SIZE
train_transform = A.Compose([

    A.Resize(IMAGE_SIZE, IMAGE_SIZE),

    A.HorizontalFlip(p=0.5),

    A.VerticalFlip(p=0.5),

    A.Rotate(limit=30, p=0.5),

    A.ShiftScaleRotate(
        shift_limit=0.08,
        scale_limit=0.08,
        rotate_limit=20,
        border_mode=0,
        p=0.5
    ),

    A.RandomBrightnessContrast(
        brightness_limit=0.2,
        contrast_limit=0.2,
        p=0.5
    ),

    A.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1,
        p=0.4
    ),

    A.GaussNoise(
        std_range=(0.02, 0.08),
        p=0.2
    ),

    A.CoarseDropout(
        num_holes_range=(1, 6),
        hole_height_range=(16, 32),
        hole_width_range=(16, 32),
        fill=0,
        p=0.3
    ),

    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),

    ToTensorV2()

])


val_transform = A.Compose([

    A.Resize(IMAGE_SIZE, IMAGE_SIZE),

    A.Normalize(
        mean=(0.485,0.456,0.406),
        std=(0.229,0.224,0.225)
    ),

    ToTensorV2()

])