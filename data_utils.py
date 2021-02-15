import math
import warnings
from os import listdir
from os.path import join
from random import choice, uniform

import torch
from PIL import Image
from numpy.random.mtrand import lognormal, normal
from torch.utils.data.dataset import Dataset, T_co
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize
from torchvision.transforms.functional import hflip, affine, resize, adjust_brightness, adjust_contrast, adjust_hue, \
    adjust_saturation, pad, center_crop


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def train_hr_transform(crop_size):
    return Compose([
        RandomCrop(crop_size),
        ToTensor(),
    ])


def train_lr_transform(crop_size, upscale_factor):
    return Compose([
        ToPILImage(),
        Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC),
        ToTensor(),
    ])


def display_transform():
    return Compose([
        Resize(400),
        CenterCrop(400),
    ])


class TrainDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, patch_size, upscale_factor):
        super(TrainDatasetFromFolder, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        patch_size = calculate_valid_crop_size(patch_size, upscale_factor)
        self.hr_transform = train_hr_transform(patch_size)
        self.lr_transform = train_lr_transform(patch_size, upscale_factor)

    def __getitem__(self, index):
        hr_image = self.hr_transform(Image.open(self.image_filenames[index]))
        lr_image = self.lr_transform(hr_image)
        return lr_image, hr_image

    def __len__(self):
        return len(self.image_filenames)


class ValDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(ValDatasetFromFolder, self).__init__()
        self.upscale_factor = upscale_factor
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]

    def __getitem__(self, index):
        """
        :return:
            * lr_image: downsampled and center-cropped images
            * hr_restore_image = upsampling of lr_image with bicubic interpolation
            * hr_image = center-cropped original image

        """
        hr_image = Image.open(self.image_filenames[index])
        w, h = hr_image.size
        crop_size = calculate_valid_crop_size(min(w, h), self.upscale_factor)
        lr_scale = Resize(crop_size // self.upscale_factor, interpolation=Image.BICUBIC)
        hr_scale = Resize(crop_size, interpolation=Image.BICUBIC)
        hr_image = CenterCrop(crop_size)(hr_image)
        lr_image = lr_scale(hr_image)
        hr_restore_img = hr_scale(lr_image)
        return ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

    def __len__(self):
        return len(self.image_filenames)


class HrValDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir):
        super(HrValDatasetFromFolder, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]

    def __getitem__(self, index):
        hr_image = Image.open(self.image_filenames[index])
        return (ToTensor()(hr_image) * 255).to(torch.uint8)

    def __len__(self):
        return len(self.image_filenames)


class TestDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(TestDatasetFromFolder, self).__init__()
        self.lr_path = dataset_dir + '/SRF_' + str(upscale_factor) + '/data/'
        self.hr_path = dataset_dir + '/SRF_' + str(upscale_factor) + '/target/'
        self.upscale_factor = upscale_factor
        self.lr_filenames = [join(self.lr_path, x) for x in listdir(self.lr_path) if is_image_file(x)]
        self.hr_filenames = [join(self.hr_path, x) for x in listdir(self.hr_path) if is_image_file(x)]

    def __getitem__(self, index):
        image_name = self.lr_filenames[index].split('/')[-1]
        lr_image = Image.open(self.lr_filenames[index])
        w, h = lr_image.size
        hr_image = Image.open(self.hr_filenames[index])
        hr_scale = Resize((self.upscale_factor * h, self.upscale_factor * w), interpolation=Image.BICUBIC)
        hr_restore_img = hr_scale(lr_image)
        return image_name, ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

    def __len__(self):
        return len(self.lr_filenames)


class SingleTensorDataset(Dataset):
    def __init__(self, data: torch.tensor):
        super(SingleTensorDataset, self).__init__()
        self.data = data

    def __getitem__(self, index) -> T_co:
        return self.data[index]

    def __len__(self):
        return len(self.data)


def rgb_to_ycbcr(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an RGB image to YCbCr.

    Args:
        image (torch.Tensor): RGB Image to be converted to YCbCr.

    Returns:
        torch.Tensor: YCbCr version of the image.
    """

    if not torch.is_tensor(image):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                         .format(image.shape))

    r: torch.Tensor = image[..., 0, :, :]
    g: torch.Tensor = image[..., 1, :, :]
    b: torch.Tensor = image[..., 2, :, :]

    delta = .5
    y: torch.Tensor = .299 * r + .587 * g + .114 * b
    cb: torch.Tensor = (b - y) * .564 + delta
    cr: torch.Tensor = (r - y) * .713 + delta
    return torch.stack((y, cb, cr), -3)


def ycbcr_to_rgb(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an YCbCr image to RGB.

    The image data is assumed to be in the range of (0, 1).

    Args:
        image (torch.Tensor): YCbCr Image to be converted to RGB with shape :math:`(*, 3, H, W)`.

    Returns:
        torch.Tensor: RGB version of the image with shape :math:`(*, 3, H, W)`.

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = ycbcr_to_rgb(input)  # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                         .format(image.shape))

    y: torch.Tensor = image[..., 0, :, :]
    cb: torch.Tensor = image[..., 1, :, :]
    cr: torch.Tensor = image[..., 2, :, :]

    delta: float = 0.5
    cb_shifted: torch.Tensor = cb - delta
    cr_shifted: torch.Tensor = cr - delta

    r: torch.Tensor = y + 1.403 * cr_shifted
    g: torch.Tensor = y - 0.714 * cr_shifted - 0.344 * cb_shifted
    b: torch.Tensor = y + 1.773 * cb_shifted
    return torch.stack([r, g, b], -3)


def augment_batch(images: torch.Tensor, p: float) -> torch.Tensor:
    warnings.warn("augment_batch is deprecated", DeprecationWarning)
    batch_size, channels, h_orig, w_orig = images.size()
    images = pad(images, padding=(w_orig - 1, h_orig - 1, w_orig - 1, h_orig - 1), padding_mode='reflect')
    batch_size, channels, h, w = images.size()
    mask = (torch.rand(batch_size) < p).logical_and(torch.rand(batch_size) < 0.5)
    images[mask] = hflip(images[mask])
    output_images = images.new_zeros((batch_size, channels, h_orig, w_orig))

    translate = (0, 0)
    angle_step = choice([0, 1, 2, 3])
    angle = -90 * angle_step

    scale_iso_mask = torch.rand(batch_size) < p
    scale_iso = lognormal(0, 0.2 * math.log(2))
    scale = (scale_iso, scale_iso)

    p_rot = 1 - math.sqrt(1 - p)
    rot_mask = torch.rand(batch_size) < p_rot
    theta = uniform(-180, 180)
    angle += theta

    scale_mask = torch.rand(batch_size) < p
    scale_factor = lognormal(0, 0.2 * math.log(2))
    scale_x, scale_y = scale
    scale = (scale_x * scale_factor, scale_y / scale_factor)
    new_size = (int(h * scale[0]), int(w * scale[1]))

    if torch.any(rot_mask):
        affine_transformed = affine(images[rot_mask], angle=angle, translate=list(translate), shear=[0., 0.], scale=1)
        images[rot_mask] = affine_transformed

    resize_mask = scale_iso_mask.logical_and(scale_mask)
    resized_images = resize(images[resize_mask], list(new_size))
    output_images[resize_mask.logical_not()] = center_crop(images[resize_mask.logical_not()], (h_orig, w_orig))
    output_images[resize_mask] = center_crop(resized_images, (h_orig, w_orig))

    images = output_images

    mask = torch.rand(batch_size) < p
    brightness = normal(1, 0.2)
    images[mask] = adjust_brightness(images[mask], brightness)

    mask = torch.rand(batch_size) < p
    contrast = lognormal(0, (0.5 * math.log(2)))
    images[mask] = adjust_contrast(images[mask], contrast)

    mask = torch.rand(batch_size) < p
    image_data = rgb_to_ycbcr(images[mask])
    image_data[..., 0, :, :] = (1 - image_data[..., 0, :, :])
    images[mask] = ycbcr_to_rgb(image_data)

    mask = torch.rand(batch_size) < p
    if torch.any(mask):
        hue_factor = uniform(-0.5, 0.5)
        images[mask] = adjust_hue(images[mask], hue_factor)

    mask = torch.rand(batch_size) < p
    saturation = lognormal(0, math.log(2))
    images[mask] = adjust_saturation(images[mask], saturation)

    mask = torch.rand(batch_size) < p
    std_dev = abs(normal(0, 0.1))
    noise_images = torch.randn_like(images[mask]) * std_dev
    images[mask] += noise_images.clamp(0, 1)

    return images
