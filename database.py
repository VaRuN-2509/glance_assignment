"""Part A: Indexer — builds TWO collections:
  - "scenes":   one CLIP embedding per full image  (captures environment/context)
  - "segments": one CLIP embedding per annotated garment crop (captures category + color/attributes)
Uses Fashionpedia's instances_attributes json (COCO-style) for bboxes/categories/attributes.
"""
import os
import json
import chromadb
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

IMAGE_DIR = "home/test/val_varun/datasets"                                   # train2020/ folder
ANNOTATION_FILE = "instances_attributes_train2020.json"
DB_DIR = "chroma_db"

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

client = chromadb.PersistentClient(path=DB_DIR)
scenes = client.get_or_create_collection("scenes")
segments = client.get_or_create_collection("segments")


def embed_image(img):
    inputs = processor(images=img, return_tensors="pt")

    with torch.no_grad():
        outputs = model.vision_model(pixel_values=inputs["pixel_values"])
        feat = model.visual_projection(outputs.pooler_output)
        feat = feat / feat.norm(dim=-1, keepdim=True)

    return feat[0].cpu().numpy().tolist()


def build_index():
    with open(ANNOTATION_FILE) as f:
        data = json.load(f)

    cat_name = {c["id"]: c["name"] for c in data["categories"]}
    attr_name = {a["id"]: a["name"] for a in data["attributes"]}
    image_info = {img["id"]: img for img in data["images"]}

    # save the vocab so the retriever can hand it to the LLM query-parser
    # without needing to re-open the (large) annotations file
    os.makedirs(DB_DIR, exist_ok=True)
    with open(os.path.join(DB_DIR, "vocab.json"), "w") as f:
        json.dump({
            "categories": sorted(set(cat_name.values())),
            "attributes": sorted(set(attr_name.values())),
        }, f)

    # 1) Scene-level: one embedding per full image
    for i, (img_id, info) in enumerate(image_info.items()):
        path = os.path.join(IMAGE_DIR, info["file_name"])
        if not os.path.exists(path):
            continue
        img = Image.open(path).convert("RGB")
        scenes.add(
            ids=[str(img_id)],
            embeddings=[embed_image(img)],
            metadatas=[{"file_name": info["file_name"], "path": path}],
        )
        if i % 100 == 0:
            print(f"[scenes] {i}/{len(image_info)}")

    # 2) Segment-level: one embedding per annotated garment crop
    for i, ann in enumerate(data["annotations"]):
        img_info = image_info.get(ann["image_id"])
        if img_info is None:
            continue
        path = os.path.join(IMAGE_DIR, img_info["file_name"])
        if not os.path.exists(path):
            continue

        x, y, w, h = ann["bbox"]
        img = Image.open(path).convert("RGB")
        crop = img.crop((x, y, x + w, y + h))
        if crop.width < 5 or crop.height < 5:
            continue

        category = cat_name.get(ann["category_id"], "")
        attributes = ", ".join(attr_name.get(a, "") for a in ann.get("attribute_ids", []))

        segments.add(
            ids=[str(ann["id"])],
            embeddings=[embed_image(crop)],
            metadatas=[{
                "file_name": img_info["file_name"],
                "path": path,
                "category": category,
                "attributes": attributes,
            }],
        )
        if i % 200 == 0:
            print(f"[segments] {i}/{len(data['annotations'])}")

    print("Done indexing scenes + segments.")


if __name__ == "__main__":
    build_index()