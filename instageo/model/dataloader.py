# ------------------------------------------------------------------------------
# This code is licensed under the Attribution-NonCommercial-ShareAlike 4.0
# International (CC BY-NC-SA 4.0) License.
#
# You are free to:
# - Share: Copy and redistribute the material in any medium or format
# - Adapt: Remix, transform, and build upon the material
#
# Under the following terms:
# - Attribution: You must give appropriate credit, provide a link to the license,
#   and indicate if changes were made. You may do so in any reasonable manner,
#   but not in any way that suggests the licensor endorses you or your use.
# - NonCommercial: You may not use the material for commercial purposes.
# - ShareAlike: If you remix, transform, or build upon the material, you must
#   distribute your contributions under the same license as the original.
#
# For more details, see https://creativecommons.org/licenses/by-nc-sa/4.0/
# ------------------------------------------------------------------------------

"""Dataloader Module."""

import os
import random
from functools import partial
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd
import rasterio
import torch
from absl import logging
from PIL import Image
from torchvision import transforms

from instageo.data.hls_utils import open_mf_tiff_dataset


def random_crop_and_flip(
    ims: List[Image.Image], label: Image.Image, im_size: int
) -> Tuple[List[Image.Image], Image.Image]:
    """Apply random cropping and flipping transformations to the given images and label.

    Args:
        ims (List[Image.Image]): List of PIL Image objects representing the images.
        label (Image.Image): A PIL Image object representing the label.

    Returns:
        Tuple[List[Image.Image], Image.Image]: A tuple containing the transformed list of
        images and label.
    """
    i, j, h, w = transforms.RandomCrop.get_params(ims[0], (im_size, im_size))

    ims = [transforms.functional.crop(im, i, j, h, w) for im in ims]
    label = transforms.functional.crop(label, i, j, h, w)

    if random.random() > 0.5:
        ims = [transforms.functional.hflip(im) for im in ims]
        label = transforms.functional.hflip(label)

    if random.random() > 0.5:
        ims = [transforms.functional.vflip(im) for im in ims]
        label = transforms.functional.vflip(label)

    return ims, label


def normalize_and_convert_to_tensor(
    ims: List[Image.Image],
    label: Image.Image | None,
    mean: List[float],
    std: List[float],
    temporal_size: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Normalize the images and label and convert them to PyTorch tensors.

    Args:
        ims (List[Image.Image]): List of PIL Image objects representing the images.
        label (Image.Image | None): A PIL Image object representing the label.
        mean (List[float]): The mean of each channel in the image
        std (List[float]): The standard deviation of each channel in the image
        temporal_size: The number of temporal steps

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple of tensors representing the normalized
        images and label.
    """
    norm = transforms.Normalize(mean, std)
    ims_tensor = torch.stack([transforms.ToTensor()(im).squeeze() for im in ims])
    _, h, w = ims_tensor.shape
    ims_tensor = ims_tensor.reshape([temporal_size, -1, h, w])  # T*C,H,W -> T,C,H,W
    ims_tensor = torch.stack([norm(im) for im in ims_tensor]).permute(
        [1, 0, 2, 3]
    )  # T,C,H,W -> C,T,H,W
    if label:
        label = torch.from_numpy(np.array(label)).squeeze()
    return ims_tensor, label


def process_and_augment(
    x: np.ndarray,
    y: np.ndarray | None,
    mean: List[float],
    std: List[float],
    temporal_size: int = 1,
    im_size: int = 224,
    augment: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Process and augment the given images and labels.

    Args:
        x (np.ndarray): Numpy array representing the images.
        y (np.ndarray): Numpy array representing the label.
        mean (List[float]): The mean of each channel in the image
        std (List[float]): The standard deviation of each channel in the image
        temporal_size: The number of temporal steps
        augment: Flag to perform augmentations in training mode.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple of tensors representing the processed
        and augmented images and label.
    """
    ims = x.copy()
    label = None
    # convert to PIL for easier transforms
    ims = [Image.fromarray(im) for im in ims]
    if y is not None:
        label = Image.fromarray(y.copy().squeeze())
    if augment:
        ims, label = random_crop_and_flip(ims, label, im_size)
    ims, label = normalize_and_convert_to_tensor(ims, label, mean, std, temporal_size)
    return ims, label


def crop_array(
    arr: np.ndarray, left: int, top: int, right: int, bottom: int
) -> np.ndarray:
    """Crop Numpy Image.

    Crop a given array (image) using specified left, top, right, and bottom indices.

    This function supports cropping both grayscale (2D) and color (3D) images.

    Args:
        arr (np.ndarray): The input array (image) to be cropped.
        left (int): The left boundary index for cropping.
        top (int): The top boundary index for cropping.
        right (int): The right boundary index for cropping.
        bottom (int): The bottom boundary index for cropping.

    Returns:
        np.ndarray: The cropped portion of the input array (image).

    Raises:
        ValueError: If the input array is not 2D or 3D.
    """
    if len(arr.shape) == 2:  # Grayscale image (2D array)
        return arr[top:bottom, left:right]
    elif len(arr.shape) == 3:  # Color image (3D array)
        return arr[:, top:bottom, left:right]
    elif len(arr.shape) == 4:  # Color image (3D array)
        return arr[:, :, top:bottom, left:right]
    else:
        raise ValueError("Input array must be a 2D, 3D or 4D array")


def process_test(
    x: np.ndarray,
    y: np.ndarray,
    mean: List[float],
    std: List[float],
    temporal_size: int = 1,
    img_size: int = 512,
    crop_size: int = 224,
    stride: int = 224,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Process and augment test data.

    Args:
        x (np.ndarray): Input image array.
        y (np.ndarray): Corresponding mask array.
        mean (List[float]): Mean values for normalization.
        std: (List[float]): Standard deviation values for normalization.
        temporal_size (int, optional): Temporal dimension size. Defaults to 1.
        img_size (int, optional): Size of the input images. Defaults to
            512.
        crop_size (int, optional): Size of the crops to be extracted from the
            images. Defaults to 224.
        stride (int, optional): Stride for cropping. Defaults to 224.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of tensors containing the processed
            images and masks.
    """
    preprocess_func = partial(
        process_and_augment,
        mean=mean,
        std=std,
        temporal_size=temporal_size,
        augment=False,
    )

    img_crops, mask_crops = [], []
    width, height = img_size, img_size

    for top in range(0, height - crop_size + 1, stride):
        for left in range(0, width - crop_size + 1, stride):
            bottom = top + crop_size
            right = left + crop_size

            img_crops.append(crop_array(x, left, top, right, bottom))
            mask_crops.append(crop_array(y, left, top, right, bottom))

    samples = [preprocess_func(x, y) for x, y in zip(img_crops, mask_crops)]
    imgs = torch.stack([sample[0] for sample in samples])
    labels = torch.stack([sample[1] for sample in samples])
    return imgs, labels


def get_raster_data(
    filepath: str,
    bands: List[int],
    no_data_value: int = -9999,
    constant_multiplier: float = 1.0,
) -> np.ndarray:
    """獲取柵格數據.

    Args:
        filepath: 文件路徑
        bands: 波段列表 (1-based indexing)
        no_data_value: 無數據值
        constant_multiplier: 乘數因子

    Returns:
        np.ndarray: 處理後的數據
    """
    with rasterio.open(filepath) as src:
        # 確保波段索引從1開始
        bands = [b + 1 for b in bands] if bands[0] == 0 else bands
        
        # 檢查波段索引是否有效
        valid_bands = list(range(1, src.count + 1))
        if not all(b in valid_bands for b in bands):
            raise ValueError(f"Invalid band indices. Valid bands are {valid_bands}")
            
        # 讀取數據
        band = src.read(bands).astype(np.float32)
        
        # 應用乘數因子
        if constant_multiplier != 1.0:
            band = band * constant_multiplier
            
        # 處理無效值
        band[band == no_data_value] = 0
        
    return band


def process_data(
    im_fname: str,
    mask_fname: str | None = None,
    no_data_value: int | None = -9999,
    reduce_to_zero: bool = False,
    replace_label: Tuple | None = None,
    bands: List[int] | None = None,
    constant_multiplier: float = 1.0,
    mask_cloud: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """處理圖像和掩碼數據.

    Args:
        im_fname: 圖像文件名
        mask_fname: 掩碼文件名
        bands: 波段索引列表
        no_data_value: 無數據值
        reduce_to_zero: 是否將標籤索引從0開始
        replace_label: 要替換的標籤值元組
        constant_multiplier: 圖像乘數
        mask_cloud: 是否進行雲掩碼

    Returns:
        Tuple[np.ndarray, np.ndarray]: 處理後的圖像和掩碼數據
    """
    try:
        arr_x = get_raster_data(
            im_fname,
            bands=bands,
            no_data_value=no_data_value,
            constant_multiplier=constant_multiplier,
        )
    except Exception as e:
        print(f"Error processing image {im_fname}: {str(e)}")
        raise

    if mask_fname:
        try:
            arr_y = get_raster_data(
                mask_fname, 
                bands=[1],  # 掩碼通常只有一個波段
                no_data_value=no_data_value
            )
            if replace_label:
                arr_y = np.where(arr_y == replace_label[0], replace_label[1], arr_y)
            if reduce_to_zero:
                arr_y -= 1
        except Exception as e:
            print(f"Error processing mask {mask_fname}: {str(e)}")
            raise
    else:
        arr_y = None

    return arr_x, arr_y


def load_data_from_csv(fname: str, input_root: str) -> List[Tuple[str, str | None]]:
    """Load data file paths from a CSV file."""
    file_paths = []
    print(f"Loading CSV from: {fname}")
    data = pd.read_csv(fname)
    print(f"CSV columns: {data.columns.tolist()}")
    print(f"Total rows in CSV: {len(data)}")
    
    label_present = True if "Label" in data.columns else False
    for idx, row in data.iterrows():
        # 不要添加 input_root，因為 CSV 中已經包含了完整路徑
        im_path = row["Input"]
        mask_path = None if not label_present else row["Label"]
        
        # 檢查文件是否存在
        full_im_path = os.path.join(input_root, im_path)
        full_mask_path = os.path.join(input_root, mask_path) if mask_path else None
        
        print(f"Checking file {idx}: {full_im_path}")
        if os.path.exists(full_im_path):
            try:
                with rasterio.open(full_im_path) as src:
                    _ = src.crs
                file_paths.append((full_im_path, full_mask_path))
            except Exception as e:
                print(f"Error reading file {full_im_path}: {e}")
                continue
        else:
            print(f"File not found: {full_im_path}")
            
    print(f"Total valid files found: {len(file_paths)}")
    return file_paths


class InstaGeoDataset(torch.utils.data.Dataset):
    """InstaGeo PyTorch Dataset for Loading and Handling HLS Data."""

    def __init__(
        self,
        fname: str,
        input_root: str,
        bands: List[int],
        mean: List[float],
        std: List[float],
        temporal_size: int = 1,
        img_size: int = 224,
        no_data_value: int | None = -9999,
        reduce_to_zero: bool = False,
        replace_label: Tuple | None = None,
        constant_multiplier: float = 1.0,
        augment: bool = True,
        mask_cloud: bool = False,
    ) -> None:
        """Initialize the InstaGeo Dataset.

        Args:
            fname: CSV文件名
            input_root: 輸入根目錄
            bands: 波段列表
            mean: 均值列表
            std: 標準差列表
            temporal_size: 時間維度大小
            img_size: 圖像大小
            no_data_value: 無數據值
            reduce_to_zero: 是否將標籤從0開始
            replace_label: 替換標籤值
            constant_multiplier: 乘數因子
            augment: 是否進行數據增強
            mask_cloud: 是否進行雲掩碼
        """
        super().__init__()
        self.file_paths = load_data_from_csv(fname, input_root)
        self.bands = bands
        self.mean = mean
        self.std = std
        self.temporal_size = temporal_size
        self.img_size = img_size
        self.no_data_value = no_data_value
        self.reduce_to_zero = reduce_to_zero
        self.replace_label = replace_label
        self.constant_multiplier = constant_multiplier
        self.augment = augment
        self.mask_cloud = mask_cloud

    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns:
            int: 數據集長度
        """
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """獲取數據集中的樣本.

        Args:
            idx: 樣本索引

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (圖像數據, 標籤數據)
        """
        im_fname, mask_fname = self.file_paths[idx]
        arr_x, arr_y = process_data(
            im_fname,
            mask_fname,
            bands=self.bands,
            no_data_value=self.no_data_value,
            reduce_to_zero=self.reduce_to_zero,
            replace_label=self.replace_label,
            constant_multiplier=self.constant_multiplier,
            mask_cloud=self.mask_cloud,
        )
        return process_and_augment(
            arr_x,
            arr_y,
            mean=self.mean,
            std=self.std,
            temporal_size=self.temporal_size,
            im_size=self.img_size,
            augment=self.augment,
        )
