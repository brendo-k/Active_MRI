# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# from activemri.baselines.loupe_codes.transforms import normalize
import math
import pathlib
from typing import Callable, Collection, List, Optional, Sequence, Tuple


import fastmri
import h5py
import numpy as np
import torch.utils.data
import activemri
import scipy.ndimage as ndimage

ifft_2d_img = lambda x, dim=(-1, -2): np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x, axes=dim), axes=dim, norm='ortho'), axes=dim)
rss = lambda x, dim=1: np.sqrt(np.sum(np.power(np.abs(x), 2), axis=dim))

class RealBrainData(torch.utils.data.Dataset):
    # This is the same as fastMRI singlecoil_knee, except we provide a custom test split
    # and also normalize images by the mean norm of the k-space over training data
#     KSPACE_WIDTH = 368
#     KSPACE_HEIGHT = 640
#     START_PADDING = 166
#     END_PADDING = 202
#     CENTER_CROP_SIZE = 320

    def __init__(
        self,
        root: pathlib.Path,
        image_shape: Tuple[int, int],
        transform: Callable,
        noise_type: str,
        noise_level: float = 5e-5,
        num_cols: Optional[int] = None,
        num_volumes: Optional[int] = None,
        num_rand_slices: Optional[int] = None,
        custom_split: Optional[str] = None,
        random_rotate=False 
    ):
        self.image_shape = image_shape
        self.transform = transform
        self.noise_type = noise_type
        self.noise_level = noise_level
        self.examples: List[Tuple[pathlib.PurePath, int]] = []


        self.num_rand_slices = num_rand_slices
        self.rng = np.random.RandomState(1234)
        self.recons_key = 'reconstruction_rss'

        files = []
        for fname in list(pathlib.Path(root).iterdir()):
            data = h5py.File(fname, "r")
            if 'reconstruction_rss' not in data.keys():
                continue
            files.append(fname)

        self.train_mode = False 

        if num_volumes is not None:
            self.rng.shuffle(files)
            files = files[:num_volumes]

        for volume_i, fname in enumerate(sorted(files)):
            data = h5py.File(fname, "r")
            kspace = data["kspace"]

            if num_rand_slices is None:
                num_slices = data['reconstruction_rss'].shape[0]
                self.examples += [(fname, slice_id) for slice_id in range(num_slices)]
            else:
                assert 0 
                slice_ids = list(range(kspace.shape[0]))
                self.rng.seed(seed=volume_i)
                self.rng.shuffle(slice_ids)
                self.examples += [
                    (fname, slice_id) for slice_id in slice_ids[:num_rand_slices]
                ]

        self.random_rotate = random_rotate
        if self.random_rotate:
            np.random.seed(42)
            self.random_angles = np.random.random(len(self)) * 30- 15

    def center_crop(self, data, shape):
        """
        (Same as the one in activemri/baselines/policy_gradient_codes/helpers/transforms.py)
        Apply a center crop to the input real image or batch of real images.

        Args:
            data (torch.Tensor): The input tensor to be center cropped. It should have at
                least 2 dimensions and the cropping is applied along the last two dimensions.
            shape (int, int): The output shape. The shape should be smaller than the
                corresponding dimensions of data.

        Returns:
            torch.Tensor: The center cropped image
        """
        assert 0 < shape[0] <= data.shape[-2]
        assert 0 < shape[1] <= data.shape[-1]
        data = data[:, ::2, :]
        w_from = (data.shape[-2] - shape[0]) // 2
        h_from = (data.shape[-1] - shape[1]) // 2
        w_to = w_from + shape[0]
        h_to = h_from + shape[1]
        return data[..., w_from:w_to, h_from:h_to]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        fname, slice_id = self.examples[i]
        try:
            kspace = None
            attr = {}
            with h5py.File(fname, 'r') as data:
                kspace = torch.from_numpy(data['kspace'][slice_id])
                attr = dict(data.attrs)

            kspace = self.center_crop(kspace, self.image_shape)

            center_k = self.get_center_k_space(kspace)
            norm_factor = np.max(rss(ifft_2d_img(center_k)))
            
            kspace = kspace / norm_factor
            
            target = rss(ifft_2d_img(kspace.numpy()), dim=0)

            kspace = torch.view_as_real(kspace)


            return self.transform(
                kspace,
                torch.zeros(kspace.shape[1]),
                target,
                attr,
                fname.name,
                slice_id
            )
        except Exception:
            print(repr(fname), slice_id, flush=True)
            raise


    # Get center cropped k-space (for computing normalization factor)
    def get_center_k_space(self, kspace, center_size: Sequence[int] = [10, 10]):
        assert len(center_size) == 2
        mask = torch.zeros_like(kspace)
        mask[..., self.image_shape[0]//2-center_size[0]//2:self.image_shape[0]//2+math.ceil(center_size[0]/2), self.image_shape[1]//2-center_size[1]//2:self.image_shape[1]//2+math.ceil(center_size[1]/2)] = 1
        return kspace * mask


"""from activemri.envs.envs import ActiveMRIEnv
data = RealBrainData(
    'datasets/brain/train_no_kspace',
    (128, 128),
    ActiveMRIEnv._void_transform,
    noise_type='none'
)

data[0]
"""