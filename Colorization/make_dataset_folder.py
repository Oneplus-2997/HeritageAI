from pathlib import Path
import json, os, shutil
import argparse



# # Choose conditioning source:
# # cond_dir = Path("datasets/preprocessed_imagenet_L_filtered_flat/train")     # grayscale L
# cond_dir = Path("datasets/canny_preprocess_L_filterd/train")                 # canny
# rgb_dir  = Path("datasets/preprocessed_imagenet_filtered_flat/train")
# captions_file = Path("datasets/captions_imagenet_train.txt")

def parse_args():
    parser = argparse.ArgumentParser(description="Build ControlNet imagefolder dataset.")
    parser.add_argument("--rgb_dir", type=Path, default=Path("datasets/preprocessed_imagenet_filtered_flat/val"))
    parser.add_argument("--canny_dir", type=Path, default=Path("datasets/canny_preprocess_L_filterd/val"))
    parser.add_argument("--l_dir", type=Path, default=Path("datasets/preprocessed_imagenet_L_filtered_flat/val"))
    parser.add_argument("--captions_file", type=Path, default=Path("datasets/captions_imagenet_val.txt"))
    parser.add_argument("--out_root", type=Path, default=Path("datasets/controlnet_imagefolder_imagenet"))
    parser.add_argument("--overwrite", action="store_true")

    return parser.parse_args()

# out_root = Path("datasets/controlnet_imagefolder_imagenet")

args = parse_args()

rgb_dir = args.rgb_dir
canny_dir = args.canny_dir
l_dir = args.l_dir
captions_file = args.captions_file
out_root = args.out_root

img_out = out_root / "val" / "images"
con_out = out_root / "val" / "conditioning"
l_out = out_root / "val" / "l_images"
meta = out_root / "val" / "metadata.jsonl"

img_out.mkdir(parents=True, exist_ok=True)
con_out.mkdir(parents=True, exist_ok=True)
l_out.mkdir(parents=True, exist_ok=True)

rgb = {p.name: p for p in rgb_dir.glob("*") if p.is_file()}
canny = {p.name: p for p in canny_dir.glob("*") if p.is_file()}
l_imgs = {p.name: p for p in l_dir.glob("*") if p.is_file()}


captions = {}
with captions_file.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.rstrip("\n")
        if not line or "," not in line:
            continue
        img_path, caption = line.split(",", 1)
        caption = caption.strip()
        if not caption:
            continue
        name = Path(img_path.strip()).name
        if not name:
            continue
        captions[name] = caption


names = sorted(set(rgb) & set(canny) & set(l_imgs) & set(captions))

if not names:
    raise RuntimeError("No paired images with non-empty captions found")

def link_or_copy(src: Path, dst: Path):
    if dst.exists() or dst.is_symlink():
        if not args.overwrite:
            return
        dst.unlink()
    try:
        os.symlink(src.resolve(), dst)
    except OSError:
        shutil.copy2(src, dst)

with meta.open("w", encoding="utf-8") as f:
    for n in names:
        link_or_copy(rgb[n], img_out / n)
        link_or_copy(canny[n], con_out / n)
        link_or_copy(l_imgs[n], l_out / n)

        rec = {
            "file_name": f"images/{n}",
            "conditioning_image_file_name": f"conditioning/{n}",
            "l_image_file_name": f"l_images/{n}",
            "text": captions[n],
        }
        f.write(json.dumps(rec) + "\n")


print("rgb files:", len(rgb))
print("conditioning files:", len(canny))
print("captioned files:", len(captions))
print("paired+captioned samples:", len(names))
print("metadata:", meta)
print("L files:", len(l_imgs))
print("L dir:", l_out)

