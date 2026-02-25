from pathlib import Path
import shutil

# 1) Bu scriptin bulunduğu klasör (datasetler aynı klasördeyse yeter)
ROOT = Path(__file__).resolve().parent

# 2) Dataset klasör isimleri (ekrandaki isimlere göre)
DATASETS = [
    {"name": "gun",   "dir": ROOT / "gun.v3i.yolov8",               "map": {0: 1}},
    {"name": "knife", "dir": ROOT / "Knife.v2i.yolov8",             "map": {0: 2}},
    {"name": "person","dir": ROOT / "Person Detection.v1i.yolov8",  "map": {0: 0}},
]

# 3) Çıktı klasörü
OUT = ROOT / "combined_dataset"
SPLITS = [("train", "train"), ("valid", "val"), ("test", "test")]  # valid -> val

def remap_label_file(src_txt: Path, dst_txt: Path, class_map: dict[int,int]):
    lines = src_txt.read_text(encoding="utf-8").strip().splitlines()
    new_lines = []

    for line in lines:
        if not line.strip():
            continue
        parts = line.split()
        # YOLO format: class x y w h
        cls = int(float(parts[0]))  # bazen "0" veya "0.0" gibi gelebiliyor
        if cls in class_map:
            parts[0] = str(class_map[cls])
        else:
            # class_map'te yoksa olduğu gibi bırak
            parts[0] = str(cls)
        new_lines.append(" ".join(parts))

    dst_txt.parent.mkdir(parents=True, exist_ok=True)
    dst_txt.write_text("\n".join(new_lines) + ("\n" if new_lines else ""), encoding="utf-8")

def copy_images(src_img_dir: Path, dst_img_dir: Path, prefix: str):
    if not src_img_dir.exists():
        return
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    for p in src_img_dir.iterdir():
        if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
            shutil.copy2(p, dst_img_dir / f"{prefix}__{p.name}")

def process_dataset(ds):
    ds_name = ds["name"]
    ds_dir = ds["dir"]
    class_map = ds["map"]

    for split_in, split_out in SPLITS:
        src_split = ds_dir / split_in

        src_img_dir = src_split / "images"
        src_lbl_dir = src_split / "labels"

        dst_img_dir = OUT / "images" / split_out
        dst_lbl_dir = OUT / "labels" / split_out

        # images kopyala
        copy_images(src_img_dir, dst_img_dir, ds_name)

        # labels remap + kopyala
        if src_lbl_dir.exists():
            dst_lbl_dir.mkdir(parents=True, exist_ok=True)
            for txt in src_lbl_dir.glob("*.txt"):
                remap_label_file(
                    txt,
                    dst_lbl_dir / f"{ds_name}__{txt.name}",
                    class_map
                )

def write_data_yaml():
    # YOLOv8 için tek yaml
    yaml = f"""path: {OUT.as_posix()}
train: images/train
val: images/val
test: images/test

nc: 3
names: [person, gun, knife]
"""
    (OUT / "data.yaml").write_text(yaml, encoding="utf-8")

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for ds in DATASETS:
        if not ds["dir"].exists():
            raise FileNotFoundError(f"Klasör bulunamadı: {ds['dir']}")
        process_dataset(ds)
    write_data_yaml()
    print("✅ Bitti! combined_dataset oluşturuldu + label class'lar remap edildi.")
    print(f"➡️ YAML: {OUT / 'data.yaml'}")

if __name__ == "__main__":
    main()