"""Input preparation (augmentation, JPEG compression, quantized DCT + quantization table) and the DiffNet wrapper."""
import io
import os
import random
from typing import List, Union

import cv2
import numpy as np
import PIL
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as TF
from PIL import Image, ImageColor
from torchvision import transforms


brightness=transforms.ColorJitter(brightness=[0.3,2.0],contrast=0,saturation=0)
contrast=transforms.ColorJitter(contrast=[0.3,2.0],brightness=0,saturation=0)
saturation=transforms.ColorJitter(saturation=[0.3,2.0],brightness=0,contrast=0)

DCT_MAX = 20  # |quantized DCT| is clipped to [0, 20]

qt_per_path_dict={}

def color_customized(color, mode):
    if isinstance(color, str):
        color = ImageColor.getcolor(color, mode)
    return color


def expand_customized(image, border, fill=0):

    left, top, right, bottom = border
    width = left + image.size[0] + right
    height = top + image.size[1] + bottom
    color = color_customized(fill, image.mode)
    if image.palette:
        raise RuntimeError("didn't expect palette")
    out = Image.new(image.mode, (width, height), color)
    out.paste(image, (left, top))
    return out

def generate_homography(image: Image.Image, mask: Image.Image):
    """Apply random homography transformation to image and mask."""
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    mask_cv = cv2.cvtColor(np.array(mask), cv2.COLOR_GRAY2BGR)

    src_pts = np.float32([[0, 0], [0, image.width], [image.width, image.height], [image.height, 0]])

    dst_pts = np.float32([
        [np.random.randint(50, 100), np.random.randint(100, 150)],
        [0, image.width],
        [image.width, np.random.randint(image.width-100, image.width)],
        [np.random.randint(image.height-100, image.height), 0]
    ])

    H, _ = cv2.findHomography(src_pts, dst_pts)

    image_transformed = cv2.warpPerspective(image_cv, H, (image.width, image.height))
    mask_transformed = cv2.warpPerspective(mask_cv, H, (mask.width, mask.height))

    image_transformed = Image.fromarray(cv2.cvtColor(image_transformed, cv2.COLOR_BGR2RGB))
    mask_transformed = Image.fromarray(cv2.cvtColor(mask_transformed, cv2.COLOR_BGR2GRAY))

    return image_transformed, mask_transformed


def random_crop_pad_white(image, mask, base_width, base_height):
    """Pad image/mask if smaller than base size, random crop if larger."""
    width, height = image.size

    left, top, right, bottom= 0,0,0,0
    if width < base_width or height < base_height:

        left = max(0, (base_width - width) // 2)
        top = max(0, (base_height - height) // 2)
        right = max((base_width - width) - left,0)
        bottom = max((base_height - height) - top,0)

        image = expand_customized(image, border=(left, top, right, bottom), fill='white')
        mask = expand_customized(mask, border=(left, top, right, bottom), fill='black')


    w_img, h_img = image.size
    mask_array = np.array(mask)

    if w_img > base_width or h_img > base_height:

        def random_crop_coords():
            x0 = random.randint(0, w_img - base_width)
            y0 = random.randint(0, h_img - base_height)
            return x0, y0

        total_pos = int((mask_array > 0).sum())
        k = min(total_pos, 32)

        if k == 0 or random.random() > 0.75:
            x0, y0 = random_crop_coords()
        else:
            for _ in range(100):
                x0, y0 = random_crop_coords()
                patch = mask_array[y0 : y0 + base_height, x0 : x0 + base_width]
                if int((patch > 0).sum()) >= k:
                    break

        x1 = x0 + base_width
        y1 = y0 + base_height

        image = image.crop((x0, y0, x1, y1))
        mask  = mask.crop((x0, y0, x1, y1))



    return image, mask


def _blockwise_cv2_dct(y: np.ndarray) -> np.ndarray:
    """8x8 block DCT without quantization."""
    Hp, Wp = y.shape
    blocks = y.reshape(Hp // 8, 8, Wp // 8, 8).transpose(0, 2, 1, 3)  # (bh, bw, 8, 8)
    bh, bw, _, _ = blocks.shape

    for i in range(bh):
        for j in range(bw):
            blocks[i, j] = cv2.dct(blocks[i, j])

    dct_hw = blocks.transpose(0, 2, 1, 3).reshape(Hp, Wp)
    return dct_hw


def doctamper_dct_opencv(pil_img: Image.Image) -> np.ndarray:
    """Blockwise DCT of the level-shifted luminance (DocTamper style)."""
    L = np.array(pil_img.convert("L"), dtype=np.float32)
    L = L - 128.0

    dct = _blockwise_cv2_dct(L)
    return dct


def compute_qt_from_quality(quality):
    """Compute JPEG quantization table from quality (libjpeg formula)."""
    STD_LUMINANCE_QT = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ], dtype=np.float64)

    # jpeg_quality_scaling from jcparam.c
    if quality <= 0:
        quality = 1
    if quality > 100:
        quality = 100

    if quality < 50:
        scale_factor = 5000 // quality
    else:
        scale_factor = 200 - quality * 2

    # jpeg_add_quant_table formula
    qt = np.zeros((8, 8), dtype=np.int32)
    for i in range(8):
        for j in range(8):
            temp = (STD_LUMINANCE_QT[i, j] * scale_factor + 50) // 100
            qt[i, j] = max(1, int(temp))

    return qt

def qt_from_pillow(image_path: str) -> np.ndarray:
    """Luminance quantization table (8, 8) int32 of a JPEG file, or None."""
    with Image.open(image_path) as im:
        qdict = getattr(im, "quantization", None)
        if not qdict or 0 not in qdict:
            return None
        q = np.array(qdict[0], dtype=np.int32)
        if q.size != 64:
            return None
        return q.reshape(8, 8)


def compress_image(img,max_compressions,min_max_compression_rate,image_path=False,qualitys=False,do_compress=True) :
    """Optionally JPEG-compress the image, then return (RGB image, clipped |quantized DCT|, quantization table).

    qualitys=False draws the number of compressions and their qualities at random (training); a list gives them
    explicitly ([] = no compression). The quantization table is the one of the last compression, else the one of
    the JPEG file at image_path, else all ones.
    """
    if qualitys==False :
        if random.random() <0.33:
            num1=0
        elif  random.random() <0.33:
            num1=1
        else:
            num1 = max(random.randint(1, max_compressions),1)

        nums2 = [random.randint(min_max_compression_rate[0], min_max_compression_rate[1]) for _ in range(num1)]
    else :
        nums2 = list(qualitys)
        num1=len(nums2)
    im=img
    image=False
    qt=None
    last_quality = nums2[-1] if (num1!=0 and do_compress) else None
    shared_buff = io.BytesIO()
    if num1!=0  and do_compress:

        for index,el in enumerate(nums2) :
            shared_buff.seek(0)
            shared_buff.truncate(0)
            im.save(shared_buff, format="JPEG",quality=el)

            shared_buff.seek(0)
            with Image.open(shared_buff) as _im:
                _im.load()
                im = _im.copy()

        image = im if im.mode == 'RGB' else im.convert('RGB')
    else:
        image = img if isinstance(img, Image.Image) and img.mode == 'RGB' else img.convert('RGB')

    shared_buff.close()

    dct = doctamper_dct_opencv(image)

    if last_quality is not None:
        qt = compute_qt_from_quality(last_quality)
    elif image_path and str(image_path).lower().endswith((".jpg", ".jpeg")):
        try:
            if isinstance(image_path, str):
                if image_path in qt_per_path_dict.keys():
                    qt=qt_per_path_dict[image_path]
                else:
                    qt = qt_from_pillow(image_path)
                    if qt is None:
                        print("qt extracted with pillow is None setting qt to ones")
                        qt = np.ones((8, 8), dtype=np.int32)
                    qt_per_path_dict[image_path] = qt
                    if len(qt_per_path_dict) > 5:
                        del qt_per_path_dict[next(iter(qt_per_path_dict))]
            else:
                qt = np.ones((8, 8), dtype=np.int32)
        except Exception as e:
            print(f"Failed to extract QT from {image_path}: {e}, using ones")
            qt = np.ones((8, 8), dtype=np.int32)
    else:
        qt = np.ones((8, 8), dtype=np.int32)

    H, W = dct.shape
    H_blocks, W_blocks = H // 8, W // 8
    qt_tiled = np.tile(qt, (H_blocks, W_blocks))

    dct_quantized = np.round(dct / qt_tiled).astype(np.int32)

    dct = np.clip(np.abs(dct_quantized), 0, DCT_MAX).astype(np.int32)

    qt = np.expand_dims(qt, axis=0).astype(np.int32)

    return image,dct,qt

def augment(img,mask,probabilities):
    """Apply spatial, visual and flip augmentations to image and mask."""
    blur = lambda img : (TF.gaussian_blur(img, kernel_size=random.choice([3, 5, 7])))


    angle = random.randint(-5, 5)
    rotate = lambda img, mask: (TF.rotate(img, angle=angle, fill=(255,255,255)), TF.rotate(mask, angle=angle, fill=0))

    hflip = lambda img, mask: (TF.hflip(img), TF.hflip(mask))

    vflip = lambda img, mask: (TF.vflip(img), TF.vflip(mask))


    rot90   = lambda img, mask: (TF.rotate(img,  90, fill=(255,255,255)), TF.rotate(mask,  90, fill=0))
    rotm90  = lambda img, mask: (TF.rotate(img, -90, fill=(255,255,255)), TF.rotate(mask, -90, fill=0))
    rot180  = lambda img, mask: (TF.rotate(img, 180, fill=(255,255,255)), TF.rotate(mask, 180, fill=0))
    rotate_ninety_ops = [rot90, rotm90, rot180]
    index=np.random.randint(0, len(rotate_ninety_ops))
    rotate_harsh=rotate_ninety_ops[index]

    flips=[hflip,vflip]
    index=np.random.randint(0, len(flips))
    flip_chosen=flips[index]

    process=[brightness,contrast,saturation,blur]

    flip=[flip_chosen,rotate_harsh]

    move=[random_shrink_rescale,rotate]

    data_aug=[(move,probabilities["spatial"]),(process,probabilities["visual"]),(flip,probabilities["flip"])]


    for index,aug_operation in enumerate(data_aug) :
        if index==0 or index==2:
            for operation in aug_operation[0]:
                if random.random()<aug_operation[1] :
                    img, mask = operation(img, mask)
        else :
            for operation in aug_operation[0]:
                if random.random()<aug_operation[1] :
                    img=operation(img)



    return img,mask

def random_shrink_rescale(image,mask):
        original_size = image.size
        scale = 0.5+np.random.rand()*0.5
        if random.random() >0.5 :
            scale=1/scale

        new_size = (int(original_size[0] * scale), int(original_size[1] * scale))



        FILTERS = [
            Image.BILINEAR,
            Image.BICUBIC,
            Image.LANCZOS,
        ]

        chosen_filter = random.choice(FILTERS)

        image = image.resize(new_size, chosen_filter)

        mask = mask.resize(new_size, Image.NEAREST)


        return image,mask


def load_weights(net, path):
    """Strict load of DiffNet weights from a training checkpoint (*.pth / Lightning *.ckpt) or an exported
    *.safetensors / *.pth state_dict. Key prefixes added by Lightning, DDP and torch.compile are stripped."""
    if str(path).endswith(".safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(str(path))
    else:
        state_dict = torch.load(path, map_location='cpu', weights_only=False)
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
    normalized = {}
    for key, value in state_dict.items():
        key = key.replace("_orig_mod.", "")
        for prefix in ("module.model.encoder.model.", "model.encoder.model.", "module."):
            if key.startswith(prefix):
                key = key[len(prefix):]
                break
        normalized[key] = value
    expected = net.state_dict()
    missing = [key for key in expected if key not in normalized]
    if missing:
        raise RuntimeError(f"Missing checkpoint keys: {missing}")
    head_keys = [key for key in expected if key.startswith("offset_head.")]
    trainable = {"offset_head." + name for name, _ in net.offset_head.named_parameters()}
    for key in head_keys:
        if key not in trainable and not torch.equal(normalized[key], expected[key]):
            raise RuntimeError(f"Checkpoint/model operating-point mismatch: {key}")
    # Lightning saves also contain the focal-loss buffers outside the network.
    net.load_state_dict({key: normalized[key] for key in expected}, strict=True)


class Encoder(nn.Module):
    """DiffNet network plus its input preparation."""

    def __init__(
        self,
        input_size: List[int],
        pretrained_model_name_or_path: Union[str, bytes, os.PathLike] = None,
        data_aug_config = None,
        load_imagenet_weights: bool = True,
    ):
        super().__init__()

        self.input_size = input_size
        self.data_aug_config = data_aug_config

        self.to_tensor = transforms.Compose(
            [
                transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.455, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

        self.to_tensor_mask = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        from models.diffnet import build_diffnet
        # The ImageNet initialisation is skipped when a full DiffNet checkpoint is loaded right after.
        self.model = build_diffnet(num_classes=2,
                                   load_imagenet_weights=load_imagenet_weights and not pretrained_model_name_or_path)

        if pretrained_model_name_or_path:
            print("Loading weights from", pretrained_model_name_or_path)
            load_weights(self.model, pretrained_model_name_or_path)
            print("Weights loaded")

    def forward(self, x, dcts, qt, return_raw=False):
        """(thresholded logits, image logits[, raw logits]); see DiffNet.forward."""
        extras = {'x': x, 'dct': dcts, 'qtb': qt}
        if return_raw:
            extras['return_raw'] = True
        return self.model(extras)

    def prepare_input(self, img ,mask, aug=True,qualitys=False,crop_pad=True,image_path_to_compute_qt=None):
        """(image tensor, mask tensor, dct, qt) for one sample; aug=False for evaluation/inference."""
        try:
            prob_disable=self.data_aug_config["prob_disable"]
        except:
            prob_disable=0

        if  random.random()<prob_disable:
            aug=False
            qualitys=[]

        aug= aug and self.data_aug_config["enabled"]

        if type(img)==str :
            img=cv2.imread(img)

        if not isinstance(img, PIL.Image.Image) :
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)

        base_width=self.input_size[1]
        base_height=self.input_size[0]

        if type(mask)==str :
            mask=cv2.imread(mask,cv2.IMREAD_GRAYSCALE)
        if not isinstance(mask, PIL.Image.Image) :
            mask = mask / 255
            mask = Image.fromarray(mask)

        if aug:
            img,mask=augment(img,mask,probabilities=self.data_aug_config["probs"] )

        if crop_pad :
            img,mask=random_crop_pad_white(img, mask, base_width, base_height)

        try :

            if aug and random.random()>0.9:
                img,mask=generate_homography(img,mask)

        except :
            print("homography failed")

        if aug and self.data_aug_config is not None:
            img,dct,qt=compress_image(img,self.data_aug_config["compression"]["max_num"],self.data_aug_config["compression"]["quality_range"],qualitys=qualitys,do_compress=aug)

        else :
            assert image_path_to_compute_qt is not None
            img,dct,qt=compress_image(img,None,None,image_path=image_path_to_compute_qt,qualitys=qualitys,do_compress=aug)

        image_tensor=self.to_tensor(img)
        return image_tensor, (self.to_tensor_mask(mask)>0).long(), torch.from_numpy(dct), torch.from_numpy(qt)


class DocMDetector(nn.Module):
    """Top-level module; its `encoder` holds the network (checkpoint keys start with `model.encoder.`)."""

    def __init__(self, input_size, pretrained_model_name_or_path=None, data_aug_config=None,
                 load_imagenet_weights=True):
        super().__init__()
        self.encoder = Encoder(
            input_size=input_size,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            data_aug_config = data_aug_config,
            load_imagenet_weights=load_imagenet_weights,
        )

        self.encoder = nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder)

    def forward(self, x, dcts, qt):
        return self.encoder(x, dcts, qt)
