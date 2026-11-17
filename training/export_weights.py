"""Export the network weights of a training checkpoint (periodic.pth / epoch<N>.pth / Lightning *.ckpt) to a
weights-only file loadable by infer.py and the evaluation scripts (--weights / --weights-path).

  python export_weights.py --checkpoint <run>/checkpoints/epoch30.pth --output diffnet.safetensors
"""
import argparse

import torch

from model import load_weights
from models.diffnet import build_diffnet


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", required=True, help="*.safetensors or *.pth")
    args = ap.parse_args()

    net = build_diffnet(num_classes=2, load_imagenet_weights=False)
    load_weights(net, args.checkpoint)  # strict: every network key must be present
    state_dict = {k: v.detach().cpu().contiguous() for k, v in net.state_dict().items()}
    if args.output.endswith(".safetensors"):
        from safetensors.torch import save_file
        save_file(state_dict, args.output)
    else:
        torch.save(state_dict, args.output)
    print(f"wrote {len(state_dict)} tensors to {args.output}")


if __name__ == "__main__":
    main()
