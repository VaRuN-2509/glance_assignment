"""Part B: Retriever — the query is first turned into structured JSON by an LLM
(query_parser.parse_query), mirroring Fashionpedia's own category/attribute schema.
Each garment in that structure is then matched two ways against the "segments"
collection:
  - symbolic score: exact category match + fraction of requested attributes present
    in the segment's metadata (this is what actually fixes compositionality —
    "red tie" can only match a segment that IS a tie AND IS red)
  - soft CLIP score: embedding similarity, as a fallback for attributes not in
    Fashionpedia's fixed vocabulary (e.g. loose style words)
The "scene" field is matched separately against the "scenes" collection (full image).
"""
import chromadb
from transformers import CLIPModel, CLIPProcessor
from query_parser import format_text
import torch

DB_DIR = "chroma_db"
SYMBOLIC_WEIGHT = 0.6
CLIP_WEIGHT = 0.4
GARMENT_WEIGHT = 0.65
SCENE_WEIGHT = 0.35

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

client = chromadb.PersistentClient(path=DB_DIR)
scenes = client.get_collection("scenes")
segments = client.get_collection("segments")


def embed_text(text):
    inputs = processor(text=[text], return_tensors="pt", padding=True)

    with torch.no_grad():
        outputs = model.get_text_features(**inputs)

        feat = outputs.pooler_output
        feat = feat / feat.norm(dim=-1, keepdim=True)

    return feat[0].cpu().numpy().tolist()


def clip_similarity(distance):
    # chroma cosine distance is in [0, 2]; convert to a [0, 1] similarity
    return max(0.0, 1.0 - distance / 2.0)


def symbolic_similarity(garment, meta):
    cat_match = 1.0 if garment["category"].lower() == meta.get("category", "").lower() else 0.0
    wanted = {a.lower() for a in garment.get("attributes", [])}
    have = {a.strip().lower() for a in meta.get("attributes", "").split(",") if a.strip()}
    attr_frac = len(wanted & have) / len(wanted) if wanted else 0.0
    return 0.6 * cat_match + 0.4 * attr_frac


def best_match_for_garment(garment, pool=50):
    text = garment["category"] + " " + " ".join(garment.get("attributes", []))
    print(text)
    hits = segments.query(query_embeddings=[embed_text(text)], n_results=pool)

    best_per_image = {}
    for meta, dist in zip(hits["metadatas"][0], hits["distances"][0]):
        print(f"meta : {meta}, dist = {dist}")
        score = SYMBOLIC_WEIGHT * symbolic_similarity(garment, meta) + CLIP_WEIGHT * clip_similarity(dist)
        fname = meta["file_name"]
        if fname not in best_per_image or score > best_per_image[fname]:
            best_per_image[fname] = score
    return best_per_image


def scene_scores(scene_text, pool=100):
    if not scene_text:
        return {}
    hits = scenes.query(query_embeddings=[embed_text(scene_text)], n_results=pool)
    return {
        m["file_name"]: clip_similarity(d)
        for m, d in zip(hits["metadatas"][0], hits["distances"][0])
    }


def search(query, k=5):
    structured = format_text(model_name = "Qwen/Qwen3-8B",prompt=query)
    garments = structured.get("garments", [])
    scene_text = structured.get("scene", "")

    garment_maps = [best_match_for_garment(g) for g in garments]
    scene_map = scene_scores(scene_text)

    candidates = set(scene_map)
    for gm in garment_maps:
        candidates |= set(gm)

    results = []
    for fname in candidates:
        # image must match EVERY mentioned garment reasonably well -> take the min,
        # not the average, so "red tie + white shirt" needs both, not just one
        garment_score = min((gm.get(fname, 0.0) for gm in garment_maps), default=0.0)
        s_score = scene_map.get(fname, 0.0)
        total = GARMENT_WEIGHT * garment_score + SCENE_WEIGHT * s_score
        results.append((fname, total))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:k]


if __name__ == "__main__":
    query = "A red tie and a white shirt in a formal setting."
    for fname, score in search(query):
        print(f"/home/test/val_varun/datasets/{fname}  (score={score:.4f})")