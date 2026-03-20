"""
Import & Export dataset dalam berbagai format (YOLO, COCO, Pascal VOC, CSV).
Mirip fitur Roboflow.
"""
import os
import json
import zipfile
import tempfile
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

# Format yang didukung
IMPORT_FORMATS = {
    "yolo": {"name": "YOLO", "desc": "ZIP: images/ + labels/ (.txt)", "ext": ".zip"},
    "coco": {"name": "COCO JSON", "desc": "ZIP: images + annotations.json", "ext": ".zip,.json"},
    "voc": {"name": "Pascal VOC", "desc": "ZIP: images + XML per gambar", "ext": ".zip"},
    "csv": {"name": "CSV", "desc": "ZIP: images + CSV (path,x,y,w,h,class)", "ext": ".zip,.csv"},
}

EXPORT_FORMATS = {
    "yolo": {"name": "YOLO", "desc": "images + labels + data.yaml", "ext": ".zip"},
    "coco": {"name": "COCO JSON", "desc": "File JSON standar", "ext": ".json"},
    "voc": {"name": "Pascal VOC", "desc": "images + XML annotations", "ext": ".zip"},
}


def _normalize_bbox(x_center: float, y_center: float, width: float, height: float) -> Tuple[float, float, float, float]:
    """Pastikan bbox dalam range 0-1."""
    return (
        max(0, min(1, x_center)),
        max(0, min(1, y_center)),
        max(0, min(1, width)),
        max(0, min(1, height)),
    )


def _pixel_to_normalized(x: float, y: float, w: float, h: float, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    """Convert COCO/VOC pixel bbox [x,y,w,h] ke normalized [x_center, y_center, w, h]."""
    if img_w <= 0 or img_h <= 0:
        return 0.5, 0.5, 0.1, 0.1
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    return _normalize_bbox(x_center, y_center, nw, nh)


# --- IMPORT ---

def import_yolo(zip_path: Path, projects_folder: str, project_id: int) -> Tuple[int, int, List[str], List[Dict]]:
    """
    Import dari zip YOLO: images/ dan labels/ (atau train/val).
    Returns: (images_count, annotations_count, class_names, result_images)
    """
    project_dir = Path(projects_folder) / str(project_id)
    images_dir = project_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    class_names = []
    result_images = []
    
    with zipfile.ZipFile(zip_path, "r") as zf:
        all_files = zf.namelist()
        image_files = [f for f in all_files if f.lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"))]
        
        # Cek data.yaml untuk class names
        for f in all_files:
            if "data.yaml" in f or "data.yml" in f:
                with zf.open(f) as fp:
                    data_yaml = fp.read().decode("utf-8", errors="ignore")
                for line in data_yaml.split("\n"):
                    line = line.strip()
                    if ":" in line and not line.startswith("#") and "names" not in line.lower():
                        try:
                            idx, name = line.split(":", 1)
                            name = name.strip().strip("'\"")
                            if name and idx.strip().isdigit():
                                i = int(idx.strip())
                                while len(class_names) <= i:
                                    class_names.append(f"class_{len(class_names)}")
                                class_names[i] = name
                        except (ValueError, IndexError):
                            pass
                break
        
        img_bases = {}
        for f in image_files:
            base = Path(f).stem
            img_bases[base] = img_bases.get(base, {})
            img_bases[base]["img"] = f
        for f in all_files:
            if f.endswith(".txt"):
                base = Path(f).stem
                if base in img_bases:
                    img_bases[base]["lbl"] = f
        
        for base, paths in img_bases.items():
            img_path = paths.get("img")
            lbl_path = paths.get("lbl")
            if not img_path:
                continue
            
            ext = Path(img_path).suffix
            unique_name = f"{base}{ext}"
            dest = images_dir / unique_name
            with zf.open(img_path) as src:
                with open(dest, "wb") as dst:
                    dst.write(src.read())
            
            rel_path = os.path.relpath(dest, projects_folder)
            anns = []
            img_w, img_h = 640, 480
            try:
                from PIL import Image as PILImage
                with PILImage.open(dest) as pil:
                    img_w, img_h = pil.size
            except Exception:
                pass
            
            if lbl_path:
                with zf.open(lbl_path) as fp:
                    for line in fp:
                        line = line.decode("utf-8", errors="ignore").strip()
                        if not line:
                            continue
                        parts = line.split()
                        if len(parts) >= 5:
                            cid = int(parts[0])
                            xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                            xc, yc, w, h = _normalize_bbox(xc, yc, w, h)
                            while len(class_names) <= cid:
                                class_names.append(f"class_{len(class_names)}")
                            anns.append({"class_name": class_names[cid], "class_id": cid, "x_center": xc, "y_center": yc, "width": w, "height": h})
            
            result_images.append({
                "filepath": rel_path,
                "filename": unique_name,
                "width": img_w,
                "height": img_h,
                "annotations": anns,
            })
    
    ann_count = sum(len(r["annotations"]) for r in result_images)
    return len(result_images), ann_count, class_names, result_images


def import_coco_json_only(json_path: Path, projects_folder: str, project_id: int, images_dir: Optional[Path] = None) -> Tuple[int, int, List[str], List[Dict]]:
    """
    Import dari COCO JSON saja - gambar harus ada di images_dir (mis. dari zip).
    Returns: (img_count, ann_count, class_names, result_images)
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    project_dir = Path(projects_folder) / str(project_id)
    dest_images_dir = project_dir / "images"
    dest_images_dir.mkdir(parents=True, exist_ok=True)
    search_dir = images_dir or json_path.parent
    
    categories = {c["id"]: c["name"] for c in data.get("categories", [])}
    class_names = list(dict.fromkeys(categories.values()))
    ann_by_img = {}
    for a in data.get("annotations", []):
        iid = a["image_id"]
        if iid not in ann_by_img:
            ann_by_img[iid] = []
        ann_by_img[iid].append(a)
    
    result_images = []
    for img_info in data.get("images", []):
        fname = img_info.get("file_name", "")
        if not fname:
            continue
        src = search_dir / fname
        if not src.exists():
            for p in [json_path.parent, json_path.parent.parent]:
                s = p / fname
                if s.exists():
                    src = s
                    break
        if not src.exists():
            continue
        
        unique_name = f"{img_info['id']}_{Path(fname).name}"
        dest = dest_images_dir / unique_name
        shutil.copy2(src, dest)
        rel_path = os.path.relpath(dest, projects_folder)
        img_w = img_info.get("width", 640)
        img_h = img_info.get("height", 480)
        anns = []
        for a in ann_by_img.get(img_info["id"], []):
            bbox = a.get("bbox", [])
            if len(bbox) >= 4:
                x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
                xc, yc, nw, nh = _pixel_to_normalized(x, y, w, h, img_w, img_h)
                cname = categories.get(a["category_id"], f"class_{a['category_id']}")
                anns.append({"class_name": cname, "class_id": a["category_id"], "x_center": xc, "y_center": yc, "width": nw, "height": nh})
        result_images.append({"filepath": rel_path, "filename": unique_name, "width": img_w, "height": img_h, "annotations": anns})
    
    ann_count = sum(len(r["annotations"]) for r in result_images)
    return len(result_images), ann_count, class_names, result_images


def import_coco_with_images(zip_path: Path, projects_folder: str, project_id: int) -> Tuple[int, int, List[str], List[Dict]]:
    """
    Import COCO dari zip: berisi images + annotations.json
    Returns: (img_count, ann_count, class_names, list of {filepath, width, height, annotations})
    """
    project_dir = Path(projects_folder) / str(project_id)
    images_dir = project_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    result_images = []
    class_names = []
    
    with zipfile.ZipFile(zip_path, "r") as zf:
        all_files = zf.namelist()
        json_files = [f for f in all_files if f.endswith(".json")]
        image_files = [f for f in all_files if f.lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"))]
        
        coco_data = None
        for jf in json_files:
            if "annotations" in jf.lower() or jf.endswith("coco.json") or "instances" in jf:
                with zf.open(jf) as fp:
                    coco_data = json.load(fp)
                break
        if not coco_data:
            for jf in json_files:
                with zf.open(jf) as fp:
                    d = json.load(fp)
                if "images" in d and "annotations" in d:
                    coco_data = d
                    break
        
        if not coco_data:
            return 0, 0, [], []
        
        categories = {c["id"]: c["name"] for c in coco_data.get("categories", [])}
        class_names = list(dict.fromkeys(categories.values()))
        ann_by_img = {}
        for a in coco_data.get("annotations", []):
            iid = a["image_id"]
            if iid not in ann_by_img:
                ann_by_img[iid] = []
            ann_by_img[iid].append(a)
        
        img_by_id = {img["id"]: img for img in coco_data.get("images", [])}
        img_by_fname = {}
        for img in coco_data.get("images", []):
            fname = img.get("file_name", "")
            if fname:
                img_by_fname[Path(fname).name] = img
        
        img_count = 0
        ann_count = 0
        for zip_f in image_files:
            fname = Path(zip_f).name
            img_info = img_by_fname.get(fname)
            if not img_info:
                for img in coco_data.get("images", []):
                    if img.get("file_name", "").endswith(fname) or fname in img.get("file_name", ""):
                        img_info = img
                        break
            if not img_info:
                img_info = {"id": img_count, "width": 640, "height": 480}
            
            dest_name = f"{img_info['id']}_{fname}"
            dest = images_dir / dest_name
            with zf.open(zip_f) as src:
                with open(dest, "wb") as dst:
                    dst.write(src.read())
            
            rel_path = os.path.relpath(dest, projects_folder)
            img_w = img_info.get("width", 640)
            img_h = img_info.get("height", 480)
            
            anns = []
            for a in ann_by_img.get(img_info["id"], []):
                bbox = a.get("bbox", [])
                if len(bbox) >= 4:
                    x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
                    xc, yc, nw, nh = _pixel_to_normalized(x, y, w, h, img_w, img_h)
                    cname = categories.get(a["category_id"], f"class_{a['category_id']}")
                    anns.append({"class_name": cname, "class_id": a["category_id"], "x_center": xc, "y_center": yc, "width": nw, "height": nh})
                    ann_count += 1
            
            result_images.append({
                "filepath": rel_path,
                "filename": fname,
                "width": img_w,
                "height": img_h,
                "annotations": anns,
            })
            img_count += 1
    
    return img_count, ann_count, class_names, result_images


def import_voc(zip_path: Path, projects_folder: str, project_id: int) -> Tuple[int, int, List[str], List[Dict]]:
    """
    Import Pascal VOC dari zip: images + XML annotations.
    Returns: (img_count, ann_count, class_names, result_images)
    """
    project_dir = Path(projects_folder) / str(project_id)
    images_dir = project_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    result_images = []
    class_names = []
    
    with zipfile.ZipFile(zip_path, "r") as zf:
        all_files = zf.namelist()
        xml_files = [f for f in all_files if f.lower().endswith(".xml")]
        image_files = [f for f in all_files if f.lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"))]
        
        for zip_f in image_files:
            fname = Path(zip_f).name
            base = Path(zip_f).stem
            xml_f = next((x for x in xml_files if Path(x).stem == base), None)
            if not xml_f:
                xml_f = next((x for x in xml_files if base in x), None)
            
            dest = images_dir / fname
            with zf.open(zip_f) as src:
                with open(dest, "wb") as dst:
                    dst.write(src.read())
            
            rel_path = os.path.relpath(dest, projects_folder)
            img_w, img_h = 640, 480
            anns = []
            
            if xml_f:
                try:
                    with zf.open(xml_f) as fp:
                        tree = ET.parse(fp)
                    root = tree.getroot()
                    size = root.find("size")
                    if size is not None:
                        w = size.find("width")
                        h = size.find("height")
                        if w is not None and h is not None:
                            img_w, img_h = int(w.text), int(h.text)
                    for obj in root.findall("object"):
                        name_el = obj.find("name")
                        if name_el is None:
                            continue
                        cname = name_el.text
                        if cname not in class_names:
                            class_names.append(cname)
                        cid = class_names.index(cname)
                        bnd = obj.find("bndbox")
                        if bnd is None:
                            continue
                        xmin = float(bnd.find("xmin").text)
                        ymin = float(bnd.find("ymin").text)
                        xmax = float(bnd.find("xmax").text)
                        ymax = float(bnd.find("ymax").text)
                        x = xmin
                        y = ymin
                        w = xmax - xmin
                        h = ymax - ymin
                        xc, yc, nw, nh = _pixel_to_normalized(x, y, w, h, img_w, img_h)
                        anns.append({"class_name": cname, "class_id": cid, "x_center": xc, "y_center": yc, "width": nw, "height": nh})
                except Exception:
                    pass
            
            result_images.append({
                "filepath": rel_path,
                "filename": fname,
                "width": img_w,
                "height": img_h,
                "annotations": anns,
            })
    
    ann_count = sum(len(r["annotations"]) for r in result_images)
    return len(result_images), ann_count, class_names, result_images


def import_csv(csv_path: Path, projects_folder: str, project_id: int, image_base_dir: Optional[Path] = None) -> Tuple[int, int, List[str], List[Dict]]:
    """
    CSV format: image_path,x_center,y_center,width,height,class_name (normalized 0-1)
    Atau: image_path,x_min,y_min,x_max,y_max,class_name (pixel)
    """
    project_dir = Path(projects_folder) / str(project_id)
    images_dir = project_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    result_images = []
    class_names = []
    img_data = {}
    
    with open(csv_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line or (i == 0 and "image" in line.lower() and "class" in line.lower()):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue
            try:
                img_path_s = parts[0]
                v1, v2, v3, v4 = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                cname = parts[5]
            except (ValueError, IndexError):
                continue
            if cname not in class_names:
                class_names.append(cname)
            cid = class_names.index(cname)
            
            # Auto-detect: normalized (xc,yc,w,h) or pixel (xmin,ymin,xmax,ymax)
            if v1 <= 1 and v2 <= 1 and v3 <= 1 and v4 <= 1:
                xc, yc, nw, nh = v1, v2, v3, v4
            else:
                xc = (v1 + v3) / 2
                yc = (v2 + v4) / 2
                nw, nh = abs(v3 - v1), abs(v4 - v2)
                xc, yc, nw, nh = _pixel_to_normalized(v1, v2, nw, nh, 640, 480)
            
            xc, yc, nw, nh = _normalize_bbox(xc, yc, nw, nh)
            
            key = img_path_s
            if key not in img_data:
                img_data[key] = {"annotations": [], "path": img_path_s}
            img_data[key]["annotations"].append({"class_name": cname, "class_id": cid, "x_center": xc, "y_center": yc, "width": nw, "height": nh})
    
    # Copy images if base dir provided
    base = image_base_dir or csv_path.parent
    for img_path_s, data in img_data.items():
        src = base / img_path_s
        if not src.exists():
            src = Path(img_path_s)
        if not src.exists():
            continue
        fname = src.name
        dest = images_dir / fname
        shutil.copy2(src, dest)
        rel_path = os.path.relpath(dest, projects_folder)
        result_images.append({
            "filepath": rel_path,
            "filename": fname,
            "width": 640,
            "height": 480,
            "annotations": data["annotations"],
        })
    
    ann_count = sum(len(r["annotations"]) for r in result_images)
    return len(result_images), ann_count, class_names, result_images


# --- EXPORT ---

def _get_classes_from_project(project) -> List[Tuple[int, str]]:
    """Extract (class_id, class_name) from project annotations."""
    classes = set()
    for img in project.images:
        for ann in img.annotations:
            classes.add((ann.class_id, ann.class_name))
    return sorted(list(classes), key=lambda x: x[0])


def export_yolo(project, projects_folder: str, models_folder: str) -> Path:
    """Export project ke zip YOLO format."""
    project_dir = Path(projects_folder) / str(project.id)
    classes = _get_classes_from_project(project)
    class_names = [c[1] for c in classes]
    annotated = [img for img in project.images if img.annotated]
    
    tmp = tempfile.mkdtemp()
    try:
        out_dir = Path(tmp) / "dataset"
        train_img = out_dir / "images" / "train"
        train_lbl = out_dir / "labels" / "train"
        val_img = out_dir / "images" / "val"
        val_lbl = out_dir / "labels" / "val"
        for d in [train_img, train_lbl, val_img, val_lbl]:
            d.mkdir(parents=True, exist_ok=True)
        
        n = len(annotated)
        split = int(n * 0.8)
        train_imgs = annotated[:split]
        val_imgs = annotated[split:]
        
        def write_set(imgs, img_dst, lbl_dst):
            for img in imgs:
                src = Path(projects_folder) / img.filepath
                if not src.exists():
                    continue
                ext = img.filename.rsplit(".", 1)[-1] if "." in img.filename else "jpg"
                dst_name = f"{img.id}.{ext}"
                shutil.copy2(src, img_dst / dst_name)
                lbl_path = lbl_dst / f"{img.id}.txt"
                with open(lbl_path, "w") as f:
                    for ann in img.annotations:
                        f.write(f"{ann.class_id} {ann.x_center} {ann.y_center} {ann.width} {ann.height}\n")
        
        write_set(train_imgs, train_img, train_lbl)
        write_set(val_imgs, val_img, val_lbl)
        
        data_yaml = f"path: {out_dir.absolute()}\ntrain: images/train\nval: images/val\nnames:\n"
        for i, name in enumerate(class_names):
            data_yaml += f"  {i}: {name}\n"
        (out_dir / "data.yaml").write_text(data_yaml)
        
        zip_path = Path(tmp) / f"dataset_yolo_{project.id}.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in out_dir.rglob("*"):
                if f.is_file():
                    zf.write(f, f.relative_to(out_dir.parent))
        return zip_path
    finally:
        pass  # cleanup by caller


def export_coco(project, projects_folder: str) -> Path:
    """Export project ke COCO JSON."""
    classes = _get_classes_from_project(project)
    class_names = [c[1] for c in classes]
    cat_map = {name: i for i, name in enumerate(class_names)}
    
    images = []
    annotations = []
    ann_id = 1
    
    for img in project.images:
        if not img.annotated:
            continue
        src = Path(projects_folder) / img.filepath
        if not src.exists():
            continue
        try:
            from PIL import Image as PILImage
            with PILImage.open(src) as pil:
                w, h = pil.size
        except Exception:
            w, h = 640, 480
        
        images.append({"id": img.id, "file_name": img.filename, "width": w, "height": h})
        for ann in img.annotations:
            x_center = ann.x_center * w
            y_center = ann.y_center * h
            bw = ann.width * w
            bh = ann.height * h
            x = x_center - bw / 2
            y = y_center - bh / 2
            annotations.append({
                "id": ann_id,
                "image_id": img.id,
                "category_id": ann.class_id,
                "bbox": [round(x, 2), round(y, 2), round(bw, 2), round(bh, 2)],
                "area": round(bw * bh, 2),
                "iscrowd": 0,
            })
            ann_id += 1
    
    categories = [{"id": i, "name": n} for i, n in enumerate(class_names)]
    coco = {"images": images, "annotations": annotations, "categories": categories}
    
    tmp = tempfile.mkdtemp()
    out = Path(tmp) / f"dataset_coco_{project.id}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2, ensure_ascii=False)
    return out


def export_voc(project, projects_folder: str) -> Path:
    """Export project ke Pascal VOC (zip)."""
    project_dir = Path(projects_folder) / str(project.id)
    
    tmp = tempfile.mkdtemp()
    out_dir = Path(tmp) / "VOC"
    images_dir = out_dir / "images"
    ann_dir = out_dir / "annotations"
    images_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)
    
    for img in project.images:
        if not img.annotated:
            continue
        src = Path(projects_folder) / img.filepath
        if not src.exists():
            continue
        shutil.copy2(src, images_dir / img.filename)
        
        root = ET.Element("annotation")
        ET.SubElement(root, "filename").text = img.filename
        size = ET.SubElement(root, "size")
        ET.SubElement(size, "width").text = str(img.width or 640)
        ET.SubElement(size, "height").text = str(img.height or 480)
        ET.SubElement(size, "depth").text = "3"
        
        for ann in img.annotations:
            obj = ET.SubElement(root, "object")
            ET.SubElement(obj, "name").text = ann.class_name
            ET.SubElement(obj, "pose").text = "Unspecified"
            ET.SubElement(obj, "truncated").text = "0"
            ET.SubElement(obj, "difficult").text = "0"
            bnd = ET.SubElement(obj, "bndbox")
            w = img.width or 640
            h = img.height or 480
            xc, yc = ann.x_center * w, ann.y_center * h
            bw, bh = ann.width * w, ann.height * h
            xmin = xc - bw / 2
            ymin = yc - bh / 2
            xmax = xc + bw / 2
            ymax = yc + bh / 2
            ET.SubElement(bnd, "xmin").text = str(int(xmin))
            ET.SubElement(bnd, "ymin").text = str(int(ymin))
            ET.SubElement(bnd, "xmax").text = str(int(xmax))
            ET.SubElement(bnd, "ymax").text = str(int(ymax))
        
        tree = ET.ElementTree(root)
        ann_name = Path(img.filename).stem + ".xml"
        tree.write(ann_dir / ann_name, encoding="utf-8", xml_declaration=True, default_namespace="")
    
    zip_path = Path(tmp) / f"dataset_voc_{project.id}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in out_dir.rglob("*"):
            if f.is_file():
                zf.write(f, f.relative_to(out_dir.parent))
    return zip_path
