"""
extract_icwt.py - CLIP features of iCubWorld Transformations (iCWT), frame by frame (then use streams.py).

iCWT (Pasquale et al., Robotics and Autonomous Systems 2019; Zenodo record 3970624, CC BY 4.0):
20 categories x 10 objects, each object held by a person in front of the iCub cameras and filmed
while undergoing 5 transformations (ROT2D, ROT3D, SCALE, TRANSL, MIX), on 2 different days,
with 2 cameras (left, right). About 150-200 frames per sequence at ~20 fps (MIX about twice as long).

Download (Zenodo): part1_cropped.tar.gz ... part4_cropped.tar.gz (~1.6 GB each, 256x256 crops around
the object); part1.tar.gz ... part4.tar.gz (~5.7 GB each) are the full 640x480 frames. Each part has
5 categories. Extract them in one folder, e.g. (left camera only, to halve the disk space):
    tar -xzf part1_cropped.tar.gz --wildcards '*/left/*'

Folder layout read by this script (any part, cropped or full):
    <root>/partX_cropped/<category>/<object>/<TRANSFORMATION>/<dayN>/<left|right>/<frame>.jpg

  python extract_icwt.py --root ~/tda_data/icwt --backbone RN50
      -> RESULTS_DIR/features/icwt__RN50.pt   (--name icwt_full for the full frames)
  options: --camera left (default) | right | both;  --every 4 (1 frame out of 4 in each sequence);
           --transforms ROT2D,ROT3D,SCALE,TRANSL,MIX (default all);  --max-objects N (quick test)

Saved meta per frame (same keys as CORe50, so every stream order of streams.py works):
  session  = acquisition day of the object, RELATIVE: 1 = its first day, 2 = its second day.
             The day numbers are not shared across categories (e.g. book: day5/day6, mouse: day7/day8),
             so the absolute day would group frames by class; the relative day is the CORe50-like
             "session" (every object appears in both).
  object   = one video = one (object, transformation, day, camera) sequence
  frame    = position of the frame inside its sequence (0, 1, 2, ...), in temporal order
  instance = object name (e.g. mug3), transform, day (absolute dayN), camera, file_frame, path
Classes: the category folder names, made readable (cellphone -> cell phone), prompt "a photo of a {}.".
Use the same frozen hyper-parameters as CORe50: streams.py --cfg-like caltech101.
"""

import argparse
import os
import re
import sys
import time

TDA_HOME = os.environ.get("TDA_HOME", os.path.dirname(os.path.abspath(__file__)))
REPO_DIR = os.environ.get("REPO_DIR", os.path.join(TDA_HOME, "TDA"))
RESULTS_DIR = os.environ.get("RESULTS_DIR", os.path.join(TDA_HOME, "results"))
FEATURE_DIR = os.path.join(RESULTS_DIR, "features")
sys.path[:0] = [REPO_DIR, TDA_HOME]

import torch
from PIL import Image
from tqdm import tqdm

import clip
import tda_replica as tda

TEMPLATES = ["a photo of a {}."]                         # same prompt as CORe50
TRANSFORMS = ["ROT2D", "ROT3D", "SCALE", "TRANSL", "MIX"]
READABLE = {"bodylotion": "body lotion", "cellphone": "cell phone", "hairbrush": "hair brush",
            "hairclip": "hair clip", "ovenglove": "oven glove", "pencilcase": "pencil case",
            "remote": "remote control", "ringbinder": "ring binder", "soapdispenser": "soap dispenser",
            "sodabottle": "soda bottle", "squeezer": "lemon squeezer"}


def feature_path(name, backbone):
    return os.path.join(FEATURE_DIR, f"{name}__{backbone.replace('/', '-')}.pt")


def find_sequences(root, cameras, transforms):
    """{(category, object, transform, day, camera): [image paths sorted by frame number]}."""
    seqs = {}
    for dirpath, _, files in os.walk(root):
        parts = dirpath.replace("\\", "/").rstrip("/").split("/")
        if len(parts) < 5:
            continue
        cat, obj, tr, day, cam = parts[-5:]
        if cam not in cameras or tr not in transforms or not re.fullmatch(r"day\d+", day):
            continue
        imgs = [f for f in files if f.lower().endswith((".jpg", ".png", ".ppm"))]
        if not imgs:
            continue
        imgs.sort(key=lambda f: int(re.findall(r"\d+", f)[-1]))
        seqs[(cat, obj, tr, day, cam)] = [os.path.join(dirpath, f) for f in imgs]
    if not seqs:
        raise SystemExit(f"No iCWT sequences under {root} (expected .../<category>/<object>/<TRANSF>/<dayN>/<left|right>/*.jpg)")
    return seqs


def build_rows(seqs, every, max_objects):
    """Rows sorted by (relative day, category, object, transform, camera, frame)."""
    days_of = {}
    for cat, obj, tr, day, cam in seqs:
        days_of.setdefault((cat, obj), set()).add(int(day[3:]))
    objects = sorted(days_of)
    if max_objects:
        per_cat = {}
        for cat, obj in objects:
            per_cat.setdefault(cat, []).append(obj)
        objects = [(c, o) for c, objs in per_cat.items() for o in objs[:max_objects]]
    keep = set(objects)
    categories = sorted({c for c, _ in objects})
    label_of = {c: k for k, c in enumerate(categories)}
    one_day = [o for o in objects if len(days_of[o]) < 2]
    if one_day:
        print(f"WARNING: {len(one_day)} objects have only one day (e.g. {one_day[:3]}): S2 will be unbalanced")
    keys = sorted((k for k in seqs if (k[0], k[1]) in keep),
                  key=lambda k: (sorted(days_of[(k[0], k[1])]).index(int(k[3][3:])), k))
    rows = []
    for video, key in enumerate(keys):
        cat, obj, tr, day, cam = key
        session = sorted(days_of[(cat, obj)]).index(int(day[3:])) + 1
        for pos, path in enumerate(seqs[key][::every]):
            rows.append({"path": path, "label": label_of[cat], "session": session, "object": video,
                         "frame": pos, "instance": obj, "transform": tr, "day": day, "camera": cam,
                         "file_frame": int(re.findall(r"\d+", os.path.basename(path))[-1])})
    return rows, categories


class Frames(torch.utils.data.Dataset):
    def __init__(self, rows, preprocess):
        self.rows, self.preprocess = rows, preprocess

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.preprocess(Image.open(self.rows[i]["path"]).convert("RGB")), self.rows[i]["label"]


@torch.no_grad()
def extract(args):
    path = feature_path(args.name, args.backbone)
    if os.path.isfile(path) and not args.overwrite:
        raise SystemExit(f"{path} already exists: use --overwrite to replace it, or another --name.")
    cameras = ["left", "right"] if args.camera == "both" else [args.camera]
    transforms = [t.strip().upper() for t in args.transforms.split(",") if t.strip()]
    seqs = find_sequences(args.root, cameras, transforms)
    rows, categories = build_rows(seqs, args.every, args.max_objects)
    print(f"icwt: {len(rows)} frames, {len({r['object'] for r in rows})} sequences, "
          f"{len({r['instance'] for r in rows})} objects, {len(categories)} categories, "
          f"camera {args.camera}, 1 frame out of {args.every}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(args.backbone, device=device) if device == "cpu" else clip.load(args.backbone)
    model.eval()
    loader = torch.utils.data.DataLoader(Frames(rows, preprocess), batch_size=args.batch_size,
                                         shuffle=False, num_workers=args.workers, pin_memory=device == "cuda")
    feats, start = [], time.time()
    for images, _ in tqdm(loader, desc=f"{args.name} {args.backbone}", mininterval=30):
        feats.append(model.encode_image(images.to(device)).cpu())
    classnames = [READABLE.get(c, c) for c in categories]
    text_weights = tda.build_text_classifier(classnames, TEMPLATES, model, device).cpu()
    meta = {k: [r[k] for r in rows] for k in
            ("path", "session", "object", "frame", "instance", "transform", "day", "camera", "file_frame")}
    out = {"dataset": "icwt", "name": args.name, "backbone": args.backbone, "root": args.root,
           "feats": torch.cat(feats), "labels": torch.tensor([r["label"] for r in rows]).long(), "meta": meta,
           "classnames": classnames, "templates": TEMPLATES, "text_weights": text_weights,
           "every": args.every, "camera": args.camera, "transforms": transforms,
           "batch_size": args.batch_size, "time": time.strftime("%Y-%m-%d %H:%M")}
    os.makedirs(FEATURE_DIR, exist_ok=True)
    torch.save(out, path)
    print(f"saved {path}: {tuple(out['feats'].shape)} features, {len(classnames)} classes "
          f"({', '.join(classnames)}), {(time.time() - start) / 60:.1f} min")


def main():
    p = argparse.ArgumentParser(description="CLIP features of iCubWorld Transformations, frame by frame")
    p.add_argument("--root", required=True, help="folder containing the extracted partX(_cropped) folders")
    p.add_argument("--backbone", required=True, choices=["RN50", "ViT-B/16"])
    p.add_argument("--name", default="icwt", help="feature file name (default icwt; e.g. icwt_full for 640x480)")
    p.add_argument("--camera", default="left", choices=["left", "right", "both"])
    p.add_argument("--every", type=int, default=4, help="keep 1 frame out of N in each sequence (default 4)")
    p.add_argument("--transforms", default=",".join(TRANSFORMS), help="comma-separated (default all five)")
    p.add_argument("--max-objects", type=int, default=0, help="quick test: only N objects per category")
    p.add_argument("--batch-size", type=int, default=1,
                   help="1 (default) gives features bit-identical to the batch-size-1 TDA stream")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    args.root = os.path.abspath(os.path.expanduser(args.root))
    if os.path.isdir(REPO_DIR):
        os.chdir(REPO_DIR)
    extract(args)


if __name__ == "__main__":
    main()
