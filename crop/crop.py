from __future__ import annotations
import numpy as np
import torch
from collections.abc import Hashable, Mapping, Sequence
from typing import Any
from copy import deepcopy
from monai.config import  KeysCollection
from monai.transforms.croppad.array import (
    BorderPad,
    BoundingRect,
    CenterScaleCrop,
    CenterSpatialCrop,
    Crop,
    CropForeground,
    DivisiblePad,
    Pad,
    RandCropByLabelClasses,
    RandCropByPosNegLabel,
    RandScaleCrop,
    RandSpatialCrop,
    RandSpatialCropSamples,
    RandWeightedCrop,
    ResizeWithPadOrCrop,
    SpatialCrop,
    SpatialPad,
)
from monai.transforms.traits import LazyTrait, MultiSampleTrait
from monai.config.type_definitions import NdarrayOrTensor
from monai.data.utils import get_random_patch, get_valid_patch_size
from monai.transforms.croppad.functional import crop_func, pad_func
from monai.transforms.inverse import InvertibleTransform
from monai.transforms.transform import LazyTransform, Randomizable, MapTransform
from monai.utils import (
    TraceKeys,
    TransformBackends,
    deprecated_arg_default,
    fall_back_tuple,
    pytorch_after,
convert_to_tensor, ensure_tuple)



from monai.data.meta_obj import get_track_meta
from monai.data.meta_tensor import MetaTensor
from monai.transforms.inverse import TraceableTransform
from monai.transforms.utils import create_translate
from monai.transforms.croppad.dictionary import Cropd

__all__ = [
    "Pad",
    "SpatialPad",
    "BorderPad",
    "DivisiblePad",
    "Crop",
    "SpatialCrop",
    "CenterSpatialCrop",
    "CenterScaleCrop",
    "RandSpatialCrop",
    "RandScaleCrop",
    "RandSpatialCropSamples",
    "CropForeground",
    "RandWeightedCrop",
    "RandCropByPosNegLabel",
    "RandCropByLabelClasses",
    "ResizeWithPadOrCrop",
    "BoundingRect",
    "pad_func",
    "crop_func",
    # "RandCropByLabelClassesd"
]




def crop_func(img: torch.Tensor, slices: tuple[slice, ...], lazy: bool, transform_info: dict) -> torch.Tensor:
    """
    Functional implementation of cropping a MetaTensor. This function operates eagerly or lazily according
    to ``lazy`` (default ``False``).

    Args:
        img: data to be transformed, assuming `img` is channel-first and cropping doesn't apply to the channel dim.
        slices: the crop slices computed based on specified `center & size` or `start & end` or `slices`.
        lazy: a flag indicating whether the operation should be performed in a lazy fashion or not.
        transform_info: a dictionary with the relevant information pertaining to an applied transform.
    """
    img_size = img.peek_pending_shape() if isinstance(img, MetaTensor) else img.shape[1:]
    spatial_rank = img.peek_pending_rank() if isinstance(img, MetaTensor) else 3
    cropped = np.asarray([[s.indices(o)[0], o - s.indices(o)[1]] for s, o in zip(slices[1:], img_size)])
    extra_info = {"cropped": cropped.flatten().tolist()}
    to_shift = []
    for i, s in enumerate(ensure_tuple(slices)[1:]):
        if s.start is not None:
            to_shift.append(img_size[i] + s.start if s.start < 0 else s.start)
        else:
            to_shift.append(0)
    shape = [s.indices(o)[1] - s.indices(o)[0] for s, o in zip(slices[1:], img_size)]
    meta_info = TraceableTransform.track_transform_meta(
        img,
        sp_size=shape,
        affine=create_translate(spatial_rank, to_shift),
        extra_info=extra_info,
        orig_size=img_size,
        transform_info=transform_info,
        lazy=lazy,
    )
    out = convert_to_tensor(img.as_tensor() if isinstance(img, MetaTensor) else img, track_meta=get_track_meta())
    if lazy:
        return out.copy_meta_from(meta_info) if isinstance(out, MetaTensor) else meta_info  # type: ignore
    out = out[slices]
    return out.copy_meta_from(meta_info) if isinstance(out, MetaTensor) else out  # type: ignore

class Crop(InvertibleTransform, LazyTransform):
    """
    Perform crop operations on the input image.

    This transform is capable of lazy execution. See the :ref:`Lazy Resampling topic<lazy_resampling>`
    for more information.

    Args:
        lazy: a flag to indicate whether this transform should execute lazily or not. Defaults to False.
    """

    backend = [TransformBackends.TORCH]

    def __init__(self, lazy: bool = False):
        LazyTransform.__init__(self, lazy)

    @staticmethod
    def compute_slices(
        roi_center: Sequence[int] | NdarrayOrTensor | None = None,
        roi_size: Sequence[int] | NdarrayOrTensor | None = None,
        roi_start: Sequence[int] | NdarrayOrTensor | None = None,
        roi_end: Sequence[int] | NdarrayOrTensor | None = None,
        roi_slices: Sequence[slice] | None = None,
    ) -> tuple[slice]:
        """
        Compute the crop slices based on specified `center & size` or `start & end` or `slices`.

        Args:
            roi_center: voxel coordinates for center of the crop ROI.
            roi_size: size of the crop ROI, if a dimension of ROI size is larger than image size,
                will not crop that dimension of the image.
            roi_start: voxel coordinates for start of the crop ROI.
            roi_end: voxel coordinates for end of the crop ROI, if a coordinate is out of image,
                use the end coordinate of image.
            roi_slices: list of slices for each of the spatial dimensions.

        """
        roi_start_t: torch.Tensor

        if roi_slices:
            if not all(s.step is None or s.step == 1 for s in roi_slices):
                raise ValueError(f"only slice steps of 1/None are currently supported, got {roi_slices}.")
            return ensure_tuple(roi_slices)  # type: ignore
        else:
            if roi_center is not None and roi_size is not None:
                roi_center_t = convert_to_tensor(data=roi_center, dtype=torch.int16, wrap_sequence=True, device="cpu")
                roi_size_t = convert_to_tensor(data=roi_size, dtype=torch.int16, wrap_sequence=True, device="cpu")
                _zeros = torch.zeros_like(roi_center_t)
                half = (
                    torch.divide(roi_size_t, 2, rounding_mode="floor")
                    if pytorch_after(1, 8)
                    else torch.floor_divide(roi_size_t, 2)
                )
                roi_start_t = torch.maximum(roi_center_t - half, _zeros)
                roi_end_t = torch.maximum(roi_start_t + roi_size_t, roi_start_t)
            else:
                if roi_start is None or roi_end is None:
                    raise ValueError("please specify either roi_center, roi_size or roi_start, roi_end.")
                roi_start_t = convert_to_tensor(data=roi_start, dtype=torch.int16, wrap_sequence=True)
                roi_start_t = torch.maximum(roi_start_t, torch.zeros_like(roi_start_t))
                roi_end_t = convert_to_tensor(data=roi_end, dtype=torch.int16, wrap_sequence=True)
                roi_end_t = torch.maximum(roi_end_t, roi_start_t)
            # convert to slices (accounting for 1d)
            if roi_start_t.numel() == 1:
                return ensure_tuple([slice(int(roi_start_t.item()), int(roi_end_t.item()))])  # type: ignore
            return ensure_tuple(  # type: ignore
                [slice(int(s), int(e)) for s, e in zip(roi_start_t.tolist(), roi_end_t.tolist())]
            )

    def __call__(  # type: ignore[override]
        self, img: torch.Tensor, slices: tuple[slice, ...], lazy: bool | None = None
    ) -> torch.Tensor:
        """
        Apply the transform to `img`, assuming `img` is channel-first and
        slicing doesn't apply to the channel dim.

        """
        slices_ = list(slices)
        sd = len(img.peek_pending_shape() if isinstance(img, MetaTensor) else img.shape[1:])  # spatial dims
        if len(slices_) < sd:
            slices_ += [slice(None)] * (sd - len(slices_))
        # Add in the channel (no cropping)
        slices_ = list([slice(None)] + slices_[:sd])

        img_t: MetaTensor = convert_to_tensor(data=img, track_meta=get_track_meta())
        lazy_ = self.lazy if lazy is None else lazy
        return crop_func(img_t, tuple(slices_), lazy_, self.get_transform_info())

    def inverse(self, img: MetaTensor) -> MetaTensor:
        transform = self.pop_transform(img)
        cropped = transform[TraceKeys.EXTRA_INFO]["cropped"]
        # the amount we pad is equal to the amount we cropped in each direction
        inverse_transform = BorderPad(cropped)
        # Apply inverse transform
        with inverse_transform.trace_transform(False):
            return inverse_transform(img)  # type: ignore


class CenterSpatialCrop(Crop):
    """
    Crop at the center of image with specified ROI size.
    If a dimension of the expected ROI size is larger than the input image size, will not crop that dimension.
    So the cropped result may be smaller than the expected ROI, and the cropped results of several images may
    not have exactly the same shape.

    This transform is capable of lazy execution. See the :ref:`Lazy Resampling topic<lazy_resampling>`
    for more information.

    Args:
        roi_size: the spatial size of the crop region e.g. [224,224,128]
            if a dimension of ROI size is larger than image size, will not crop that dimension of the image.
            If its components have non-positive values, the corresponding size of input image will be used.
            for example: if the spatial size of input data is [40, 40, 40] and `roi_size=[32, 64, -1]`,
            the spatial size of output data will be [32, 40, 40].
        lazy: a flag to indicate whether this transform should execute lazily or not. Defaults to False.
    """

    def __init__(self, roi_size: Sequence[int] | int, lazy: bool = False) -> None:
        super().__init__(lazy=lazy)
        self.roi_size = roi_size

    def compute_slices(self, spatial_size: Sequence[int]) -> tuple[slice]:  # type: ignore[override]
        roi_size = fall_back_tuple(self.roi_size, spatial_size)
        roi_center = [i // 2 for i in spatial_size]
        return super().compute_slices(roi_center=roi_center, roi_size=roi_size)

    def __call__(self, img: torch.Tensor, lazy: bool | None = None) -> torch.Tensor:  # type: ignore[override]
        """
        Apply the transform to `img`, assuming `img` is channel-first and
        slicing doesn't apply to the channel dim.

        """
        lazy_ = self.lazy if lazy is None else lazy
        return super().__call__(
            img=img,
            slices=self.compute_slices(img.peek_pending_shape() if isinstance(img, MetaTensor) else img.shape[1:]),
            lazy=lazy_,
        )

class RandSpatialCrop(Randomizable, Crop):
    """
    Crop image with random size or specific size ROI. It can crop at a random position as center
    or at the image center. And allows to set the minimum and maximum size to limit the randomly generated ROI.

    Note: even `random_size=False`, if a dimension of the expected ROI size is larger than the input image size,
    will not crop that dimension. So the cropped result may be smaller than the expected ROI, and the cropped results
    of several images may not have exactly the same shape.

    This transform is capable of lazy execution. See the :ref:`Lazy Resampling topic<lazy_resampling>`
    for more information.

    Args:
        roi_size: if `random_size` is True, it specifies the minimum crop region.
            if `random_size` is False, it specifies the expected ROI size to crop. e.g. [224, 224, 128]
            if a dimension of ROI size is larger than image size, will not crop that dimension of the image.
            If its components have non-positive values, the corresponding size of input image will be used.
            for example: if the spatial size of input data is [40, 40, 40] and `roi_size=[32, 64, -1]`,
            the spatial size of output data will be [32, 40, 40].
        max_roi_size: if `random_size` is True and `roi_size` specifies the min crop region size, `max_roi_size`
            can specify the max crop region size. if None, defaults to the input image size.
            if its components have non-positive values, the corresponding size of input image will be used.
        random_center: crop at random position as center or the image center.
        random_size: crop with random size or specific size ROI.
            if True, the actual size is sampled from `randint(roi_size, max_roi_size + 1)`.
        lazy: a flag to indicate whether this transform should execute lazily or not. Defaults to False.
    """

    @deprecated_arg_default("random_size", True, False, since="1.1", replaced="1.3")
    def __init__(
        self,
        roi_size: Sequence[int] | int,
        max_roi_size: Sequence[int] | int | None = None,
        random_center: bool = True,
        random_size: bool = True,
        lazy: bool = False,
    ) -> None:
        super().__init__(lazy)
        self.roi_size = roi_size
        self.max_roi_size = max_roi_size
        self.random_center = random_center
        self.random_size = random_size
        self._size: Sequence[int] | None = None
        self._slices: tuple[slice, ...]

    def randomize(self, img_size: Sequence[int], label: torch.Tensor, selected_class: int = 2, num_pixels: int = 100) -> None:
        self._size = fall_back_tuple(self.roi_size, img_size)
        if self.random_size:
            max_size = img_size if self.max_roi_size is None else fall_back_tuple(self.max_roi_size, img_size)
            if any(i > j for i, j in zip(self._size, max_size)):
                raise ValueError(f"min ROI size: {self._size} is larger than max ROI size: {max_size}.")
            self._size = tuple(self.R.randint(low=self._size[i], high=max_size[i] + 1) for i in range(len(img_size)))
        if self.random_center:
            valid_size = get_valid_patch_size(img_size, self._size)
            while True:
                self._slices = get_random_patch(img_size, valid_size, self.R)
                if torch.sum(label[(slice(None), *self._slices)] == selected_class) > num_pixels:
                    break


    def __call__(self, img: torch.Tensor, randomize: bool = True, lazy: bool | None = None) -> torch.Tensor:  # type: ignore
        """
        Apply the transform to `img`, assuming `img` is channel-first and
        slicing doesn't apply to the channel dim.

        """
        img_size = img.peek_pending_shape() if isinstance(img, MetaTensor) else img.shape[1:]
        if randomize:
            self.randomize(img_size, img)
        if self._size is None:
            raise RuntimeError("self._size not specified.")
        lazy_ = self.lazy if lazy is None else lazy
        if self.random_center:
            return super().__call__(img=img, slices=self._slices, lazy=lazy_)
        cropper = CenterSpatialCrop(self._size, lazy=lazy_)
        return super().__call__(img=img, slices=cropper.compute_slices(img_size), lazy=lazy_)

class Cropd(MapTransform, InvertibleTransform, LazyTransform):
    """
    Dictionary-based wrapper of abstract class :py:class:`monai.transforms.Crop`.

    This transform is capable of lazy execution. See the :ref:`Lazy Resampling topic<lazy_resampling>`
    for more information.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.compose.MapTransform`
        cropper: crop transform for the input image.
        allow_missing_keys: don't raise exception if key is missing.
        lazy: a flag to indicate whether this transform should execute lazily or not. Defaults to False.
    """

    backend = Crop.backend

    def __init__(self, keys: KeysCollection, cropper: Crop, allow_missing_keys: bool = False, lazy: bool = False):
        MapTransform.__init__(self, keys, allow_missing_keys)
        LazyTransform.__init__(self, lazy)
        self.cropper = cropper

    @LazyTransform.lazy.setter  # type: ignore
    def lazy(self, value: bool) -> None:
        self._lazy = value
        if isinstance(self.cropper, LazyTransform):
            self.cropper.lazy = value

    def __call__(self, data: Mapping[Hashable, torch.Tensor], lazy: bool | None = None) -> dict[Hashable, torch.Tensor]:
        d = dict(data)
        lazy_ = self.lazy if lazy is None else lazy
        for key in self.key_iterator(d):
            d[key] = self.cropper(d[key], lazy=lazy_)  # type: ignore
        return d

    def inverse(self, data: Mapping[Hashable, MetaTensor]) -> dict[Hashable, MetaTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.cropper.inverse(d[key])
        return d
#
class RandCropd(Cropd, Randomizable):
    """
    Base class for random crop transform.

    This transform is capable of lazy execution. See the :ref:`Lazy Resampling topic<lazy_resampling>`
    for more information.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.compose.MapTransform`
        cropper: random crop transform for the input image.
        allow_missing_keys: don't raise exception if key is missing.
        lazy: a flag to indicate whether this transform should execute lazily or not. Defaults to False.
    """

    backend = Crop.backend

    def __init__(self, keys: KeysCollection, cropper: Crop, allow_missing_keys: bool = False, lazy: bool = False):
        super().__init__(keys, cropper=cropper, allow_missing_keys=allow_missing_keys, lazy=lazy)

    def set_random_state(self, seed: int | None = None, state: np.random.RandomState | None = None) -> RandCropd:
        super().set_random_state(seed, state)
        if isinstance(self.cropper, Randomizable):
            self.cropper.set_random_state(seed, state)
        return self

    def randomize(self, img_size: Sequence[int], label: torch.FloatTensor) -> None:
        if isinstance(self.cropper, Randomizable):
            self.cropper.randomize(img_size, label)

    def __call__(self, data: Mapping[Hashable, torch.Tensor], lazy: bool | None = None) -> dict[Hashable, torch.Tensor]:
        d = dict(data)
        # the first key must exist to execute random operations
        first_item = d[self.first_key(d)]
        self.randomize(first_item.peek_pending_shape() if isinstance(first_item, MetaTensor) else first_item.shape[1:], data['label'])
        lazy_ = self.lazy if lazy is None else lazy
        if lazy_ is True and not isinstance(self.cropper, LazyTrait):
            raise ValueError(
                "'self.cropper' must inherit LazyTrait if lazy is True "
                f"'self.cropper' is of type({type(self.cropper)}"
            )
        for key in self.key_iterator(d):
            kwargs = {"randomize": False} if isinstance(self.cropper, Randomizable) else {}
            if isinstance(self.cropper, LazyTrait):
                kwargs["lazy"] = lazy_
            d[key] = self.cropper(d[key], **kwargs)  # type: ignore
        return d

class RandSpatialCropd(RandCropd):
    """
    Dictionary-based version :py:class:`monai.transforms.RandSpatialCrop`.
    Crop image with random size or specific size ROI. It can crop at a random position as
    center or at the image center. And allows to set the minimum and maximum size to limit the randomly
    generated ROI. Suppose all the expected fields specified by `keys` have same shape.

    Note: even `random_size=False`, if a dimension of the expected ROI size is larger than the input image size,
    will not crop that dimension. So the cropped result may be smaller than the expected ROI, and the cropped
    results of several images may not have exactly the same shape.

    This transform is capable of lazy execution. See the :ref:`Lazy Resampling topic<lazy_resampling>`
    for more information.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: monai.transforms.MapTransform
        roi_size: if `random_size` is True, it specifies the minimum crop region.
            if `random_size` is False, it specifies the expected ROI size to crop. e.g. [224, 224, 128]
            if a dimension of ROI size is larger than image size, will not crop that dimension of the image.
            If its components have non-positive values, the corresponding size of input image will be used.
            for example: if the spatial size of input data is [40, 40, 40] and `roi_size=[32, 64, -1]`,
            the spatial size of output data will be [32, 40, 40].
        max_roi_size: if `random_size` is True and `roi_size` specifies the min crop region size, `max_roi_size`
            can specify the max crop region size. if None, defaults to the input image size.
            if its components have non-positive values, the corresponding size of input image will be used.
        random_center: crop at random position as center or the image center.
        random_size: crop with random size or specific size ROI.
            if True, the actual size is sampled from:
            `randint(roi_scale * image spatial size, max_roi_scale * image spatial size + 1)`.
        allow_missing_keys: don't raise exception if key is missing.
        lazy: a flag to indicate whether this transform should execute lazily or not. Defaults to False.
    """

    @deprecated_arg_default("random_size", True, False, since="1.1", replaced="1.3")
    def __init__(
        self,
        keys: KeysCollection,
        roi_size: Sequence[int] | int,
        max_roi_size: Sequence[int] | int | None = None,
        random_center: bool = True,
        random_size: bool = True,
        allow_missing_keys: bool = False,
        lazy: bool = False,
    ) -> None:
        cropper = RandSpatialCrop(roi_size, max_roi_size, random_center, random_size, lazy=lazy)
        super().__init__(keys, cropper=cropper, allow_missing_keys=allow_missing_keys, lazy=lazy)

class RandCropByLabelClassesd(Randomizable, MapTransform, LazyTransform, MultiSampleTrait):
    """
    Dictionary-based version :py:class:`monai.transforms.RandCropByLabelClasses`.
    Crop random fixed sized regions with the center being a class based on the specified ratios of every class.
    The label data can be One-Hot format array or Argmax data. And will return a list of arrays for all the
    cropped images. For example, crop two (3 x 3) arrays from (5 x 5) array with `ratios=[1, 2, 3, 1]`::

        cropper = RandCropByLabelClassesd(
            keys=["image", "label"],
            label_key="label",
            spatial_size=[3, 3],
            ratios=[1, 2, 3, 1],
            num_classes=4,
            num_samples=2,
        )
        data = {
            "image": np.array([
                [[0.0, 0.3, 0.4, 0.2, 0.0],
                [0.0, 0.1, 0.2, 0.1, 0.4],
                [0.0, 0.3, 0.5, 0.2, 0.0],
                [0.1, 0.2, 0.1, 0.1, 0.0],
                [0.0, 0.1, 0.2, 0.1, 0.0]]
            ]),
            "label": np.array([
                [[0, 0, 0, 0, 0],
                [0, 1, 2, 1, 0],
                [0, 1, 3, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]]
            ]),
        }
        result = cropper(data)

        The 2 randomly cropped samples of `label` can be:
        [[0, 1, 2],     [[0, 0, 0],
         [0, 1, 3],      [1, 2, 1],
         [0, 0, 0]]      [1, 3, 0]]

    If a dimension of the expected spatial size is larger than the input image size,
    will not crop that dimension. So the cropped result may be smaller than expected size, and the cropped
    results of several images may not have exactly same shape.
    And if the crop ROI is partly out of the image, will automatically adjust the crop center to ensure the
    valid crop ROI.

    This transform is capable of lazy execution. See the :ref:`Lazy Resampling topic<lazy_resampling>`
    for more information.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.compose.MapTransform`
        label_key: name of key for label image, this will be used for finding indices of every class.
        spatial_size: the spatial size of the crop region e.g. [224, 224, 128].
            if a dimension of ROI size is larger than image size, will not crop that dimension of the image.
            if its components have non-positive values, the corresponding size of `label` will be used.
            for example: if the spatial size of input data is [40, 40, 40] and `spatial_size=[32, 64, -1]`,
            the spatial size of output data will be [32, 40, 40].
        ratios: specified ratios of every class in the label to generate crop centers, including background class.
            if None, every class will have the same ratio to generate crop centers.
        num_classes: number of classes for argmax label, not necessary for One-Hot label.
        num_samples: number of samples (crop regions) to take in each list.
        image_key: if image_key is not None, only return the indices of every class that are within the valid
            region of the image (``image > image_threshold``).
        image_threshold: if enabled `image_key`, use ``image > image_threshold`` to
            determine the valid image content area and select class indices only in this area.
        indices_key: if provided pre-computed indices of every class, will ignore above `image` and
            `image_threshold`, and randomly select crop centers based on them, expect to be 1 dim array
            of spatial indices after flattening. a typical usage is to call `ClassesToIndices` transform first
            and cache the results for better performance.
        allow_smaller: if `False`, an exception will be raised if the image is smaller than
            the requested ROI in any dimension. If `True`, any smaller dimensions will remain
            unchanged.
        allow_missing_keys: don't raise exception if key is missing.
        warn: if `True` prints a warning if a class is not present in the label.
        max_samples_per_class: maximum length of indices in each class to reduce memory consumption.
            Default is None, no subsampling.
        lazy: a flag to indicate whether this transform should execute lazily or not. Defaults to False.
    """

    backend = RandCropByLabelClasses.backend

    def __init__(
        self,
        keys: KeysCollection,
        label_key: str,
        spatial_size: Sequence[int] | int,
        ratios: list[float | int] | None = None,
        num_classes: int | None = None,
        num_samples: int = 1,
        image_key: str | None = None,
        image_threshold: float = 0.0,
        indices_key: str | None = None,
        allow_smaller: bool = False,
        allow_missing_keys: bool = False,
        warn: bool = True,
        max_samples_per_class: int | None = None,
        lazy: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        LazyTransform.__init__(self, lazy)
        self.label_key = label_key
        self.image_key = image_key
        self.indices_key = indices_key
        self.cropper = RandCropByLabelClasses(
            spatial_size=spatial_size,
            ratios=ratios,
            num_classes=num_classes,
            num_samples=num_samples,
            image_threshold=image_threshold,
            allow_smaller=allow_smaller,
            warn=warn,
            max_samples_per_class=max_samples_per_class,
            lazy=lazy,
        )

    def set_random_state(
        self, seed: int | None = None, state: np.random.RandomState | None = None
    ) -> RandCropByLabelClassesd:
        super().set_random_state(seed, state)
        self.cropper.set_random_state(seed, state)
        return self

    def randomize(
        self, label: torch.Tensor, indices: list[NdarrayOrTensor] | None = None, image: torch.Tensor | None = None
    ) -> None:
        self.cropper.randomize(label=label, indices=indices, image=image)

    @LazyTransform.lazy.setter  # type: ignore
    def lazy(self, value: bool) -> None:
        self._lazy = value
        self.cropper.lazy = value

    @property
    def requires_current_data(self):
        return True

    def __call__(self, data: Mapping[Hashable, Any], lazy: bool | None = None) -> list[dict[Hashable, torch.Tensor]]:
        d = dict(data)
        self.randomize(d.get(self.label_key), d.pop(self.indices_key, None), d.get(self.image_key))  # type: ignore

        # initialize returned list with shallow copy to preserve key ordering
        ret: list = [dict(d) for _ in range(self.cropper.num_samples)]
        # deep copy all the unmodified data
        for i in range(self.cropper.num_samples):
            for key in set(d.keys()).difference(set(self.keys)):
                ret[i][key] = deepcopy(d[key])

        lazy_ = self.lazy if lazy is None else lazy
        for key in self.key_iterator(d):
            for i, im in enumerate(self.cropper(d[key], randomize=False, lazy=lazy_)):
                ret[i][key] = im
        return ret[0]
