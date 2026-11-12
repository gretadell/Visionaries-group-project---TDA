"""
extract_ucf101_video.py - CLIP features of UCF101 *videos*, frame by frame (then use streams.py).

Why: in the TDA paper UCF101 is one image per video (the middle frame). Here every test video
becomes a short clip of consecutive frames, so TDA sees UCF101 as a real video stream. The output
has the same format as extract_features.py, so streams.py (orders S0-S3, M2, oracle, smoothing)
and analyze_traces.py work unchanged.

Input (no unzip needed, the .avi files are read straight from the zip):
  --video-zip   UCF-101.zip   (13,320 .avi, paths UCF-101/<Class>/v_<Class>_gXX_cYY.avi)
  --split-zip   UCF101TrainTestSplits-RecognitionTask.zip   (classInd.txt, testlist01.txt, ...)
Both are on Hugging Face (quchenyuan/UCF101-ZIP) and on the official UCF site.

  python extract_ucf101_video.py --video-zip UCF-101.zip --split-zip UCF101TrainTestSplits-RecognitionTask.zip \
                                 --backbone RN50 [--max-per-class 3 --name ucf101_video_pilot]
      -> RESULTS_DIR/features/<name>__<backbone>.pt   (default name: ucf101_video)

Frames: --frames 16 consecutive frames at --fps 5 (one every 0.2 s at the usual 25 fps), centred on
the middle of the video, i.e. around the frame the paper uses. Short videos give fewer frames.

Saved meta per frame (same keys as CORe50, so every stream order works):
  session = UCF101 group gXX (same actor and background; testlist01 = groups 1-7)
  object  = video id (one clip)     frame = position of the frame inside its clip (0..frames-1)
  video_frame = frame index in the original video, path = video file, clip = "gXX_cYY"
Classes: the 101 UCF101 classes in classInd.txt order, names split from CamelCase
("ApplyEyeMakeup" -> "Apply Eye Makeup"), prompt "a photo of a person doing {}." as in the paper.
Use the official UCF101 hyper-parameters with streams.py --cfg-like ucf101.
"""

import argparse
import io
import os
import re
import sys
import tempfile
import time
import zipfile

TDA_HOME = os.environ.get("TDA_HOME", os.path.dirname(os.path.abspath(__file__)))
REPO_DIR = os.environ.get("REPO_DIR", os.path.join(TDA_HOME, "TDA"))
RESULTS_DIR = os.environ.get("RESULTS_DIR", os.path.join(TDA_HOME, "results"))
FEATURE_DIR = os.path.join(RESULTS_DIR, "features")
sys.path[:0] = [REPO_DIR, TDA_HOME]

import cv2
import torch
from PIL import Image
from tqdm import tqdm

import clip
import tda_replica as tda

TEMPLATES = ["a photo of a person doing {}."]          # the paper's UCF101 prompt


def feature_path(name, backbone):
    return os.path.join(FEATURE_DIR, f"{name}__{backbone.replace('/', '-')}.pt")


def camel_to_words(name):
    """ApplyEyeMakeup -> Apply Eye Makeup, YoYo -> Yo Yo, PlayingDaf -> Playing Daf."""
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", " ", name)


def read_split(split_zip, split):
    """(class folders in label order, [(class folder, video file name)]) of the official test list."""
    with zipfile.ZipFile(split_zip) as z:
        files = {os.path.basename(n): n for n in z.namelist()}
        ind = z.read(files["classInd.txt"]).decode().split()
        test = z.read(files[f"testlist0{split}.txt"]).decode().split()
    classes = [ind[i] for i in range(1, len(ind), 2)]            # "1 ApplyEyeMakeup 2 ..."
    videos = [(p.split("/")[0], p.split("/")[-1]) for p in test]
    return classes, videos


def pick_videos(videos, max_per_class):
    """All test videos, or `max_per_class` per class spread over different groups (pilot)."""
    if not max_per_class:
        return sorted(videos)
    by_class = {}
    for cls, name in sorted(videos):
        by_class.setdefault(cls, []).append(name)
    out = []
    for cls, names in by_class.items():
        by_group = {}
        for n in names:
            by_group.setdefault(re.search(r"_g(\d+)_", n).group(1), []).append(n)
        groups = sorted(by_group)
        picked, k = [], 0
        while len(picked) < max_per_class and any(by_group.values()):
            g = groups[k % len(groups)]
            if by_group[g]:
                picked.append(by_group[g].pop(0))
            k += 1
        out += [(cls, n) for n in picked]
    return out


def frame_indices(n_frames, video_fps, frames, fps):
    """`frames` frames, one every video_fps/fps, centred on the middle of the video."""
    step = max(1, round((video_fps or 25.0) / fps))
    span = (frames - 1) * step
    start = max(0, (n_frames - 1 - span) // 2)
    idx = [start + k * step for k in range(frames)]
    return sorted({min(i, max(n_frames - 1, 0)) for i in idx})


def read_frames(video_bytes, frames, fps):
    """Decode only the needed frames of one .avi (given as bytes). Returns ([PIL images], [indices])."""
    with tempfile.NamedTemporaryFile(suffix=".avi", delete=False) as f:
        f.write(video_bytes)
        tmp = f.name
    try:
        cap = cv2.VideoCapture(tmp)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        wanted = frame_indices(n, cap.get(cv2.CAP_PROP_FPS), frames, fps) if n > 0 else list(range(frames))
        images, kept, i = [], [], 0
        last = max(wanted)
        while i <= last:
            ok, frame = cap.read()
            if not ok:
                break
            if i in wanted:
                images.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                kept.append(i)
            i += 1
        cap.release()
    finally:
        os.remove(tmp)
    return images, kept


class VideoClips(torch.utils.data.Dataset):
    """One item = all sampled frames of one video, already preprocessed: [t, 3, H, W]."""

    def __init__(self, video_zip, items, preprocess, frames, fps):
        self.video_zip, self.items, self.preprocess = video_zip, items, preprocess
        self.frames, self.fps, self.zip = frames, fps, None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        if self.zip is None:                       # one handle per DataLoader worker
            self.zip = zipfile.ZipFile(self.video_zip)
        member = self.items[i]["member"]
        images, kept = read_frames(self.zip.read(member), self.frames, self.fps)
        if not images:
            return None, i, []
        return torch.stack([self.preprocess(im) for im in images]), i, kept


@torch.no_grad()
def extract(args):
    name = args.name or ("ucf101_video_pilot" if args.max_per_class else "ucf101_video")
    path = feature_path(name, args.backbone)
    if os.path.isfile(path) and not args.overwrite:
        raise SystemExit(f"{path} already exists: use --overwrite to replace it, or another --name.")

    classes, test = read_split(args.split_zip, args.split)
    label_of = {c: k for k, c in enumerate(classes)}
    with zipfile.ZipFile(args.video_zip) as z:
        members = {os.path.basename(n).lower(): n for n in z.namelist() if n.lower().endswith(".avi")}
    items, missing = [], 0
    for cls, fname in pick_videos(test, args.max_per_class):
        member = members.get(fname.lower())
        if member is None:
            missing += 1
            continue
        g, c = re.search(r"_g(\d+)_c(\d+)", fname).groups()
        items.append({"member": member, "label": label_of[cls], "group": int(g), "clip": f"g{g}_c{c}"})
    if missing:
        print(f"WARNING: {missing} videos of the test list are not in the zip")
    print(f"ucf101 split {args.split}: {len(items)} videos, {len(classes)} classes, "
          f"{args.frames} frames at {args.fps} fps per video")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(args.backbone, device=device) if device == "cpu" else clip.load(args.backbone)
    model.eval()
    loader = torch.utils.data.DataLoader(VideoClips(args.video_zip, items, preprocess, args.frames, args.fps),
                                         batch_size=None, shuffle=False, num_workers=args.workers)
    feats, labels, meta, bad, start = [], [], {k: [] for k in
                                             ("path", "session", "object", "frame", "video_frame", "clip")}, 0, time.time()
    for frames, i, kept in tqdm(loader, desc=f"{name} {args.backbone}", mininterval=30):
        if frames is None:
            bad += 1
            continue
        it = items[i]
        feats.append(model.encode_image(frames.to(device)).cpu())
        for pos, vf in enumerate(kept):
            labels.append(it["label"])
            meta["path"].append(it["member"])
            meta["session"].append(it["group"])
            meta["object"].append(i)                 # video id
            meta["frame"].append(pos)
            meta["video_frame"].append(vf)
            meta["clip"].append(it["clip"])
    if bad:
        print(f"WARNING: {bad} videos could not be decoded and were skipped")
    classnames = [camel_to_words(c) for c in classes]
    text_weights = tda.build_text_classifier(classnames, TEMPLATES, model, device).cpu()
    out = {"dataset": "ucf101_video", "name": name, "backbone": args.backbone,
           "feats": torch.cat(feats), "labels": torch.tensor(labels).long(), "meta": meta,
           "classnames": classnames, "templates": TEMPLATES, "text_weights": text_weights,
           "frames": args.frames, "fps": args.fps, "split": args.split, "max_per_class": args.max_per_class,
           "time": time.strftime("%Y-%m-%d %H:%M")}
    os.makedirs(FEATURE_DIR, exist_ok=True)
    torch.save(out, path)
    print(f"saved {path}: {tuple(out['feats'].shape)} features from {len(set(meta['object']))} videos, "
          f"{len(set(meta['session']))} groups, {(time.time() - start) / 60:.1f} min")


def main():
    p = argparse.ArgumentParser(description="CLIP features of UCF101 videos, frame by frame")
    p.add_argument("--video-zip", required=True, help="UCF-101.zip (the .avi files)")
    p.add_argument("--split-zip", required=True, help="UCF101TrainTestSplits-RecognitionTask.zip")
    p.add_argument("--backbone", required=True, choices=["RN50", "ViT-B/16"])
    p.add_argument("--split", type=int, default=1, choices=[1, 2, 3], help="official test list (default 1)")
    p.add_argument("--frames", type=int, default=16, help="consecutive frames per video (default 16)")
    p.add_argument("--fps", type=float, default=5, help="sampling rate of those frames (default 5 per second)")
    p.add_argument("--max-per-class", type=int, default=0,
                   help="pilot: only N videos per class, from different groups (default: all test videos)")
    p.add_argument("--name", default=None, help="feature file name (default ucf101_video, or ucf101_video_pilot)")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--workers", type=int, default=2)
    args = p.parse_args()
    args.video_zip, args.split_zip = os.path.abspath(args.video_zip), os.path.abspath(args.split_zip)
    if os.path.isdir(REPO_DIR):
        os.chdir(REPO_DIR)
    extract(args)


if __name__ == "__main__":
    main()
