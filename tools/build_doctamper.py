"""Build the DocTamper part of the Doc Protocol data from the official DocTamper LMDBs.

Outputs (under --out), each split as images/<idx:06d>.jpg + masks/<idx:06d>.png (0/255):
  DocTamperV1-TrainingSet_extracted/      raw LMDB JPEG bytes, not re-encoded (training compresses on the fly)
  DocTamperV1/DocTamperV1-{TestingSet,FCD,SCD}/
                                          Doc Protocol test sets: decoded RGB image saved once with PIL at the final
                                          quality factor record[-1] of the official pks/<set>_75.pk (minimum
                                          quality 75, the default of the official evaluation)
  DocTamperV1_uncompressed/DocTamperV1-{TestingSet,FCD,SCD}-uncompressed/
                                          same test images as raw LMDB JPEG bytes (no protocol compression)
The LMDB images are 512x512 RGB JPEGs at quality 100 (all-ones quantization tables).

Usage: python tools/build_doctamper.py <split> --lmdb-root <dir with DocTamperV1-<split>/> --pks <dir with
       DocTamperV1-<split>_75.pk> --out <doc_protocol data root> [--limit N]
       split in TrainingSet, TestingSet, FCD, SCD
"""
import argparse
import io
import os
import pickle
import sys

import cv2
import lmdb
import numpy as np
from PIL import Image

MINQ = 75


def write_atomic(path, data):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split", choices=["TrainingSet", "TestingSet", "FCD", "SCD"])
    ap.add_argument("--lmdb-root", required=True)
    ap.add_argument("--pks", help="folder with the official DocTamperV1-<split>_75.pk (test splits)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="only the first N samples")
    a = ap.parse_args()
    LMDB_ROOT, PKS, OUT = a.lmdb_root, a.pks, a.out
    if a.split != "TrainingSet" and not PKS:
        sys.exit("--pks is required for the test splits")
    name = f"DocTamperV1-{a.split}"
    env = lmdb.open(os.path.join(LMDB_ROOT, name), readonly=True, lock=False, readahead=False, meminit=False)
    with env.begin() as txn:
        n = int(txn.get(b"num-samples"))
    if a.limit:
        n = min(n, a.limit)

    if a.split == "TrainingSet":
        outs = {"raw": os.path.join(OUT, f"{name}_extracted")}
        record = None
    else:
        outs = {"protocol": os.path.join(OUT, "DocTamperV1", name),
                "raw": os.path.join(OUT, "DocTamperV1_uncompressed", f"{name}-uncompressed")}
        with open(os.path.join(PKS, f"{name}_{MINQ}.pk"), "rb") as f:
            record = pickle.load(f)
        if len(record) < n:
            sys.exit(f"{name}: pickle has {len(record)} records < {n} samples")
    for d in outs.values():
        os.makedirs(os.path.join(d, "images"), exist_ok=True)
        os.makedirs(os.path.join(d, "masks"), exist_ok=True)

    qf_hist = {}
    with env.begin() as txn:
        for i in range(n):
            imgbuf = txn.get(b"image-%09d" % i)
            lblbuf = txn.get(b"label-%09d" % i)
            if imgbuf is None or lblbuf is None:
                sys.exit(f"{name}: missing key at index {i}")
            mask = (cv2.imdecode(np.frombuffer(lblbuf, dtype=np.uint8), 0) != 0).astype(np.uint8) * 255
            im = Image.open(io.BytesIO(imgbuf))
            if (im.height, im.width) != mask.shape:
                sys.exit(f"{name}: image/mask size mismatch at {i}: {im.size} vs {mask.shape}")
            ok, mpng = cv2.imencode(".png", mask)
            assert ok
            stem = f"{i:06d}"
            # Files are written atomically, so an existing pair is complete (skipped samples are not counted in the
            # quality histogram).
            if all(os.path.exists(os.path.join(d, "images", stem + ".jpg")) and
                   os.path.exists(os.path.join(d, "masks", stem + ".png")) for d in outs.values()):
                continue
            for kind, d in outs.items():
                if kind == "raw":
                    data = bytes(imgbuf)
                else:
                    q = int(record[i][-1])
                    qf_hist[q] = qf_hist.get(q, 0) + 1
                    buf = io.BytesIO()
                    im.convert("RGB").save(buf, "JPEG", quality=q)
                    data = buf.getvalue()
                write_atomic(os.path.join(d, "images", stem + ".jpg"), data)
                write_atomic(os.path.join(d, "masks", stem + ".png"), mpng.tobytes())
            if (i + 1) % 10000 == 0:
                print(f"{name}: {i + 1}/{n}", flush=True)
    for kind, d in outs.items():
        with open(os.path.join(d, "BUILD_INFO.txt"), "w") as f:
            f.write(f"source={name} samples={n} kind={kind}"
                    + (f" pk={name}_{MINQ}.pk qf=record[-1] single PIL save RGB" if kind == "protocol" else
                       " raw LMDB JPEG bytes (quality 100), not re-encoded") + "\n")
    print(f"{name}: done {n} samples -> {outs}; final-QF histogram {dict(sorted(qf_hist.items()))}")


if __name__ == "__main__":
    main()
