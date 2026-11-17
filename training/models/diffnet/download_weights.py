"""Download the ImageNet-1K ConvNeXt-V2 checkpoints used to initialise DiffNet for training:
base (RGB backbone) and pico (stage-2 blocks reused by the DCT branch and the refinement blocks).

  python models/diffnet/download_weights.py
"""
import os
import urllib.request

URL = 'https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_{}_1k_224_ema.pt'
WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pretrained_weights')


def main():
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    for variant in ('base', 'pico'):
        dest = os.path.join(WEIGHTS_DIR, f'convnextv2_{variant}_1k_224_ema.pt')
        if os.path.exists(dest):
            print(f'exists: {dest}')
            continue
        print(f'downloading {URL.format(variant)} -> {dest}')
        urllib.request.urlretrieve(URL.format(variant), dest + '.tmp')
        os.replace(dest + '.tmp', dest)


if __name__ == '__main__':
    main()
