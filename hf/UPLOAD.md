# Uploading weights to Hugging Face

Target repository: [`MohamedDhouib1/DiffNet`](https://huggingface.co/MohamedDhouib1/DiffNet) (public model repo).
Requires `pip install huggingface_hub`.

## 1. Export the weights

Upload only weights-only files written by `training/export_weights.py`, never a training checkpoint. From
`training/`:

```bash
python export_weights.py --checkpoint <doc run>/checkpoints/epoch30.pth --output ../diffnet_doc_protocol.safetensors
python export_weights.py --checkpoint <tdoc run>/checkpoints/epoch2.pth --output ../diffnet_tdoc_protocol.safetensors
```

The output is the network state_dict, in safetensors format for a `.safetensors` name (`.pth` gives a
`torch.save` file); it loads strictly with `infer.py --weights`, the evaluators' `--weights-path` and `INIT`.
The file name is kept on the Hub (`<protocol>/<file name>`).

Upload the repository configs (`training/configs/<protocol>/diffnet.yaml`), not the `config.yaml` saved in a run
folder, which contains local paths.

## 2. Authenticate

The token is read from the environment or the `hf` login cache, never passed as an argument:

```bash
export HF_TOKEN=...        # or: hf auth login
```

## 3. Create the repository

From the repository root. Model card plus one placeholder README per protocol folder; check first with
`--dry-run` (lists the uploads, no network):

```bash
python hf/upload_weights.py --repo-id MohamedDhouib1/DiffNet --model-card hf/MODEL_CARD.md \
    --init-placeholder doc_protocol --init-placeholder tdoc_protocol --dry-run
python hf/upload_weights.py --repo-id MohamedDhouib1/DiffNet --model-card hf/MODEL_CARD.md \
    --init-placeholder doc_protocol --init-placeholder tdoc_protocol
```

## 4. Upload the weights

Doc Protocol (uploads `doc_protocol/diffnet_doc_protocol.safetensors` and `doc_protocol/diffnet.yaml`, and removes
the `doc_protocol/README.md` placeholder):

```bash
python hf/upload_weights.py --repo-id MohamedDhouib1/DiffNet --protocol doc_protocol \
    --weights diffnet_doc_protocol.safetensors --config training/configs/doc_protocol/diffnet.yaml --dry-run
python hf/upload_weights.py --repo-id MohamedDhouib1/DiffNet --protocol doc_protocol \
    --weights diffnet_doc_protocol.safetensors --config training/configs/doc_protocol/diffnet.yaml
```

Syn2Real-TDoc protocol, once trained (uploads `tdoc_protocol/diffnet_tdoc_protocol.safetensors` and
`tdoc_protocol/diffnet.yaml`, and removes the `tdoc_protocol/README.md` placeholder):

```bash
python hf/upload_weights.py --repo-id MohamedDhouib1/DiffNet --protocol tdoc_protocol \
    --weights diffnet_tdoc_protocol.safetensors --config training/configs/tdoc_protocol/diffnet.yaml
```

After each weights upload, update the status column of `hf/MODEL_CARD.md` and upload it again with
`--model-card hf/MODEL_CARD.md` (it can be added to the same command). Afterwards run `unset HF_TOKEN`.
