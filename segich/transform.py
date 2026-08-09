import torch
from torchvision import tv_tensors
from torchvision.transforms import v2


class _BaseTransform:
    transforms: v2.Compose

    def __call__(self, image, mask):
        img_tv = tv_tensors.Image(image)
        mask_tv = tv_tensors.Mask(mask)
        transformed_img, transformed_mask = self.transforms(img_tv, mask_tv)

        return transformed_img, transformed_mask


class TrainTransform(_BaseTransform):
    def __init__(self, size: tuple[int, int] = (512, 512), dtype: torch.dtype = torch.float32):
        self.transforms = v2.Compose(
            [
                v2.Resize(size=size, antialias=True),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
                v2.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1), scale=(0.9, 1.1)),
                v2.ToDtype(dtype=dtype, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )


class TestTransform(_BaseTransform):
    def __init__(self, size: tuple[int, int] = (512, 512), dtype: torch.dtype = torch.float32):
        self.transforms = v2.Compose(
            [
                v2.Resize(size=size, antialias=True),
                v2.ToDtype(dtype=dtype, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
