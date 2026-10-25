"""
prepare_data.py - download and arrange the datasets exactly as the official loaders expect
(layout of docs/DATASETS.md in the official repo, CoOp splits). Called by `bash run.sh data`.

  python prepare_data.py                  # every automatic dataset except SUN397
  python prepare_data.py dtd eurosat      # only some datasets
  python prepare_data.py --with-sun397    # also SUN397 (37 GB download)

Folders (environment variables):
  DATA_DIR     where the datasets go                     (default: ~/tda_data)
  ARCHIVE_DIR  where downloaded archives are kept/looked up (default: DATA_DIR/_archives)
               Colab points this to Google Drive so archives survive the session.

Re-running is safe: anything already in place is skipped. Archives are deleted after
extraction, unless --keep-archives or smaller than --keep-archives-under-gb.

Two datasets cannot be downloaded automatically. Put their files in DATA_DIR/manual/ and
re-run this script, it will arrange them:
  * ImageNet val (needs an account at image-net.org):
        DATA_DIR/manual/ILSVRC2012_img_val.tar
        DATA_DIR/manual/ILSVRC2012_devkit_t12.tar.gz
  * Stanford Cars (original links are dead; e.g. Kaggle mirror of the original files):
        DATA_DIR/manual/stanford_cars/cars_test.tgz
        DATA_DIR/manual/stanford_cars/cars_test_annos_withlabels.mat
        DATA_DIR/manual/stanford_cars/car_devkit.tgz      (optional)
"""
import argparse
import glob
import os
import shutil
import subprocess
import traceback

DATA_DIR = os.environ.get("DATA_DIR", os.path.expanduser("~/tda_data"))
ARCH = os.environ.get("ARCHIVE_DIR", os.path.join(DATA_DIR, "_archives"))
MANUAL = os.environ.get("MANUAL_DIR", os.path.join(DATA_DIR, "manual"))
TMP = os.path.join(DATA_DIR, "_tmp")
KEEP_ARCHIVES = False
KEEP_UNDER_GB = 0.0
IMAGENET_CLASSNAMES_GID = "1-61f_ol79pViBFDG_IDlUQSwoLcn2XXF"


# ----------------------------------------------------------------------------- helpers
def sh(cmd):
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


def fetch(name, urls=(), gid=None):
    """Download archive `name` (Google Drive id first, then the URLs in order). Returns its path."""
    os.makedirs(ARCH, exist_ok=True)
    dst = os.path.join(ARCH, name)
    if os.path.isfile(dst) and os.path.getsize(dst) > 0:
        return dst
    if gid:
        import gdown
        try:
            if gdown.download(id=gid, output=dst, quiet=False) is not None:
                return dst
        except Exception as e:
            print("  Google Drive download failed:", e)
    for url in urls:
        if subprocess.run(f'curl -L --fail --retry 5 --progress-bar -o "{dst}" "{url}"', shell=True).returncode == 0:
            return dst
    if os.path.exists(dst):
        os.remove(dst)
    raise RuntimeError(f"could not download {name}. Download it by hand into {ARCH}/ (same name) and re-run.")


def extract(archive, dest):
    os.makedirs(dest, exist_ok=True)
    if archive.endswith(".zip"):
        sh(f'unzip -q -o "{archive}" -d "{dest}"')
    else:
        sh(f'tar -xf "{archive}" -C "{dest}"')
    if not KEEP_ARCHIVES and archive.startswith(ARCH) and os.path.getsize(archive) >= KEEP_UNDER_GB * 1e9:
        os.remove(archive)


def small(name, gid, dest_dir):
    """Small files from Google Drive (splits, class names): kept, copied into the dataset folder."""
    os.makedirs(dest_dir, exist_ok=True)
    target = os.path.join(dest_dir, name)
    if not os.path.isfile(target):
        shutil.copy(fetch(name, gid=gid), target)


def d(*p):
    return os.path.join(DATA_DIR, *p)


# ----------------------------------------------------------------------------- cross-domain
def prep_dtd():
    if not os.path.isdir(d("dtd", "images")):
        extract(fetch("dtd-r1.0.1.tar.gz", ["https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz"]), DATA_DIR)
    small("split_zhou_DescribableTextures.json", "1u3_QfB467jqHgNXC00UIzbLZRQCg2S7x", d("dtd"))


def prep_caltech101():
    root = d("caltech-101")
    if not os.path.isdir(os.path.join(root, "101_ObjectCategories")):
        try:   # the Google Drive file torchvision uses
            arc = fetch("101_ObjectCategories.tar.gz", gid="137RyRjvTBkBiIfeYBNZBtViDHQ6_Ewsp")
        except RuntimeError:
            z = fetch("caltech-101.zip", ["https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip"])
            extract(z, os.path.join(TMP, "caltech"))
            arc = glob.glob(os.path.join(TMP, "caltech", "**", "101_ObjectCategories.tar.gz"), recursive=True)[0]
        extract(arc, root)
    small("split_zhou_Caltech101.json", "1hyarUivQE36mY6jSomru6Fjd-JzwcCzN", root)


def prep_eurosat():
    root = d("eurosat")
    if not os.path.isdir(os.path.join(root, "2750")):
        extract(fetch("EuroSAT.zip", [
            "https://huggingface.co/datasets/torchgeo/eurosat/resolve/c877bcd43f099cd0196738f714544e355477f3fd/EuroSAT.zip",
            "http://madm.dfki.de/files/sentinel/EuroSAT.zip"]), root)
    small("split_zhou_EuroSAT.json", "1Ip7yaCWFi0eaOFUGga0lUdVi_DDQth1o", root)


def prep_oxford_pets():
    root = d("oxford_pets")
    if not os.path.isdir(os.path.join(root, "images")):
        extract(fetch("pets_images.tar.gz", ["https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"]), root)
    if not os.path.isdir(os.path.join(root, "annotations")):
        extract(fetch("pets_annotations.tar.gz", ["https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"]), root)
    small("split_zhou_OxfordPets.json", "1501r8Ber4nNKvmlFVQZ8SeUHTcdTTEqs", root)


def prep_oxford_flowers():
    root = d("oxford_flowers")
    if not os.path.isdir(os.path.join(root, "jpg")):
        extract(fetch("102flowers.tgz", ["https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"]), root)
    if not os.path.isfile(os.path.join(root, "imagelabels.mat")):
        shutil.move(fetch("imagelabels.mat", ["https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat"]), root)
    small("cat_to_name.json", "1AkcxCXeK_RCGCEC_GvmWxjcjaNhu-at0", root)
    small("split_zhou_OxfordFlowers.json", "1Pp0sRXzZFZq15zVOzKjKBu4A9i01nozT", root)


def prep_fgvc():
    root = d("fgvc_aircraft")
    if not os.path.isdir(os.path.join(root, "images")):
        extract(fetch("fgvc-aircraft-2013b.tar.gz",
                      ["https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz"]),
                os.path.join(TMP, "fgvc"))
        shutil.move(os.path.join(TMP, "fgvc", "fgvc-aircraft-2013b", "data"), root)   # keep only data/


def prep_food101():
    if not os.path.isdir(d("food-101", "images")):
        extract(fetch("food-101.tar.gz", ["http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"]), DATA_DIR)
    small("split_zhou_Food101.json", "1QK0tGi096I0Ba6kggatX1ee6dJFIcEJl", d("food-101"))


def prep_ucf101():
    root = d("ucf101")
    if not os.path.isdir(os.path.join(root, "UCF-101-midframes")):
        extract(fetch("UCF-101-midframes.zip", gid="10Jqome3vtUA2keJkNanAiFpgbyC9Hc2O"), root)
    small("split_zhou_UCF101.json", "1I0S0q91hJfsV9Gf4xDIjgDq4AqBNJb1y", root)


def prep_sun397():
    root = d("sun397")
    if not os.path.isdir(os.path.join(root, "SUN397")):
        os.makedirs(root, exist_ok=True)   # 37 GB: streamed straight into tar, no archive on disk
        sh(f'curl -L --fail --retry 5 "http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz" | tar -xz -C "{root}"')
    small("split_zhou_SUN397.json", "1y2RD81BYuiyvebdN-JymPfyWYcd8_MUq", root)


def prep_stanford_cars():   # manual files -> arranged here
    root, src = d("stanford_cars"), os.path.join(MANUAL, "stanford_cars")
    if not os.path.isdir(os.path.join(root, "cars_test")):
        if not os.path.isfile(os.path.join(src, "cars_test.tgz")):
            raise RuntimeError(f"manual dataset: put cars_test.tgz and cars_test_annos_withlabels.mat in {src}/")
        extract(os.path.join(src, "cars_test.tgz"), root)
        shutil.copy(os.path.join(src, "cars_test_annos_withlabels.mat"), root)
        if os.path.isfile(os.path.join(src, "car_devkit.tgz")):
            extract(os.path.join(src, "car_devkit.tgz"), root)
    small("split_zhou_StanfordCars.json", "1ObCFbaAgVu0I-k_Au-gIUcefirdAuizT", root)


# ----------------------------------------------------------------------------- OOD
def prep_imagenet():        # manual files -> arranged here (torchvision builds images/val/<wnid>/)
    img_root = d("imagenet", "images")
    if not os.path.isdir(os.path.join(img_root, "val")):
        needed = ["ILSVRC2012_img_val.tar", "ILSVRC2012_devkit_t12.tar.gz"]
        if not all(os.path.isfile(os.path.join(MANUAL, f)) for f in needed):
            raise RuntimeError(f"manual dataset: put {' and '.join(needed)} in {MANUAL}/")
        os.makedirs(img_root, exist_ok=True)
        for f in needed:
            shutil.copy(os.path.join(MANUAL, f), img_root)
        import torchvision
        torchvision.datasets.ImageNet(img_root, split="val")
        os.remove(os.path.join(img_root, "ILSVRC2012_img_val.tar"))
    small("classnames.txt", IMAGENET_CLASSNAMES_GID, d("imagenet"))


def prep_imagenet_a():
    root = d("imagenet-adversarial")
    if not os.path.isdir(os.path.join(root, "imagenet-a")):
        extract(fetch("imagenet-a.tar", ["https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar"]), root)
    small("classnames.txt", IMAGENET_CLASSNAMES_GID, root)


def prep_imagenet_r():
    root = d("imagenet-rendition")
    if not os.path.isdir(os.path.join(root, "imagenet-r")):
        extract(fetch("imagenet-r.tar", ["https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar"]), root)
    small("classnames.txt", IMAGENET_CLASSNAMES_GID, root)


def prep_imagenet_v():
    root = d("imagenetv2")
    if not os.path.isdir(os.path.join(root, "imagenetv2-matched-frequency-format-val")):
        extract(fetch("imagenetv2-matched-frequency.tar.gz", [
            "https://s3-us-west-2.amazonaws.com/imagenetv2public/imagenetv2-matched-frequency.tar.gz",
            "https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/main/imagenetv2-matched-frequency.tar.gz"]), root)
    small("classnames.txt", IMAGENET_CLASSNAMES_GID, root)


def prep_imagenet_s():
    # Best effort (Hugging Face mirror of the official zip). Otherwise download it from
    # https://github.com/HaohanWang/ImageNet-Sketch and put the 1000 n* folders in DATA_DIR/imagenet-sketch/images
    root = d("imagenet-sketch")
    if not os.path.isdir(os.path.join(root, "images")):
        z = fetch("ImageNet-Sketch.zip", ["https://huggingface.co/datasets/songweig/imagenet_sketch/resolve/main/data/ImageNet-Sketch.zip"])
        extract(z, os.path.join(TMP, "sketch"))
        first = glob.glob(os.path.join(TMP, "sketch", "**", "n01440764"), recursive=True)[0]
        os.makedirs(root, exist_ok=True)
        shutil.move(os.path.dirname(first), os.path.join(root, "images"))
    small("classnames.txt", IMAGENET_CLASSNAMES_GID, root)


AUTOMATIC = {"dtd": prep_dtd, "caltech101": prep_caltech101, "eurosat": prep_eurosat,
             "oxford_pets": prep_oxford_pets, "oxford_flowers": prep_oxford_flowers, "fgvc": prep_fgvc,
             "ucf101": prep_ucf101, "food101": prep_food101,
             "imagenet-a": prep_imagenet_a, "imagenet-r": prep_imagenet_r,
             "imagenet-v": prep_imagenet_v, "imagenet-s": prep_imagenet_s}
OPTIONAL = {"sun397": prep_sun397}
MANUAL_SETS = {"imagenet": prep_imagenet, "stanford_cars": prep_stanford_cars}
ALL = {**AUTOMATIC, **OPTIONAL, **MANUAL_SETS}


def main():
    global KEEP_ARCHIVES, KEEP_UNDER_GB
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("datasets", nargs="*", help=f"subset of: {', '.join(ALL)}")
    p.add_argument("--with-sun397", action="store_true", help="also download SUN397 (37 GB)")
    p.add_argument("--keep-archives", action="store_true", help="never delete downloaded archives")
    p.add_argument("--keep-archives-under-gb", type=float, default=0.0,
                   help="keep archives smaller than this many GB (Colab: cache them on Drive)")
    args = p.parse_args()
    KEEP_ARCHIVES = args.keep_archives
    KEEP_UNDER_GB = args.keep_archives_under_gb

    names = args.datasets or (list(AUTOMATIC) + (["sun397"] if args.with_sun397 else []) + list(MANUAL_SETS))
    print(f"Data folder: {DATA_DIR}\nDatasets: {names}\n", flush=True)
    status = {}
    for name in names:
        print(f"\n################ {name}", flush=True)
        try:
            ALL[name]()
            status[name] = "ready"
        except Exception as e:
            status[name] = f"NOT ready - {e}"
            if name not in MANUAL_SETS:
                traceback.print_exc()
    shutil.rmtree(TMP, ignore_errors=True)

    print("\n================ SUMMARY ================")
    for name, s in status.items():
        print(f"  {'OK ' if s == 'ready' else '-- '} {name:15s} {s}")
    print("\nNext step:  bash run.sh check")
    sh(f'du -sh "{DATA_DIR}" || true')


if __name__ == "__main__":
    main()
