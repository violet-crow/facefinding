import os, sys, argparse, csv, json, warnings
import cv2
import numpy as np
from sklearn.cluster import DBSCAN

# Optional: suppress InsightFace face_align FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, module=r".*insightface\.utils\.face_align.*")

# Prefer CUDA if available; InsightFace will still run on CPU if CUDA not present
os.environ.setdefault("INSIGHTFACE_ONNXRT_PROVIDERS", "CUDAExecutionProvider,CPUExecutionProvider")

from insightface.app import FaceAnalysis
import onnxruntime as ort

def normalize(v):
    n = np.linalg.norm(v) + 1e-8
    return v / n

def get_face_embedding(face):
    emb = getattr(face, 'normed_embedding', None)
    if emb is None:
        emb = getattr(face, 'embedding', None)
    return emb

def init_face_app(det_size=(640, 640), prefer_gpu=True):
    av = set(ort.get_available_providers())
    use_gpu = prefer_gpu and ("CUDAExecutionProvider" in av)
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=(0 if use_gpu else -1), det_size=det_size)
    return app, use_gpu

def extract_embeddings(app, video_paths, sample_every=3, min_det_score=0.4, max_faces_per_frame=10):
    samples = []  # each: {'vid': int, 'frame': int, 'ts': float, 'bbox': [x1,y1,x2,y2], 'emb': np.array}
    meta = []
    for vid_idx, path in enumerate(video_paths):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        meta.append({'fps': fps, 'W': W, 'H': H})
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % sample_every != 0:
                frame_idx += 1
                continue
            faces = app.get(frame)
            if not faces:
                frame_idx += 1
                continue
            faces = sorted(faces, key=lambda f: float(getattr(f, 'det_score', 0.0)), reverse=True)[:max_faces_per_frame]
            for f in faces:
                score = float(getattr(f, 'det_score', 0.0))
                if score < min_det_score:
                    continue
                emb = get_face_embedding(f)
                if emb is None:
                    continue
                emb = normalize(np.asarray(emb, dtype=np.float32))
                bbox = f.bbox.astype(int).tolist()
                ts = frame_idx / fps
                samples.append({'vid': vid_idx, 'frame': frame_idx, 'ts': ts, 'bbox': bbox, 'emb': emb})
            frame_idx += 1
        cap.release()
    return samples, meta

def cluster_identities(samples, eps=0.35, min_samples=5):
    if not samples:
        return np.zeros((0,), dtype=int), []
    X = np.stack([s['emb'] for s in samples], axis=0)
    db = DBSCAN(eps=eps, metric='cosine', min_samples=min_samples)
    raw = db.fit_predict(X)  # -1 noise
    # Build clusters of non-noise
    raw_to_indices = {}
    for i, lbl in enumerate(raw):
        if lbl == -1:
            continue
        raw_to_indices.setdefault(lbl, []).append(i)
    # Sort clusters by size desc and remap to 0..K-1
    sorted_raw = sorted(raw_to_indices.keys(), key=lambda l: len(raw_to_indices[l]), reverse=True)
    raw_to_new = {lbl: i for i, lbl in enumerate(sorted_raw)}
    labels = np.full_like(raw, -1)
    centers = []
    for raw_lbl in sorted_raw:
        idxs = raw_to_indices[raw_lbl]
        center = normalize(np.mean(np.stack([X[i] for i in idxs], axis=0), axis=0))
        new_id = raw_to_new[raw_lbl]
        centers.append({'cluster': new_id, 'center': center, 'count': len(idxs)})
        for i in idxs:
            labels[i] = new_id
    return labels, centers  # labels aligned with samples

def assign_clusters_to_names(video_paths, samples, labels, centers):
    # Names from filenames (stem), preserve original case
    names = [os.path.splitext(os.path.basename(p))[0] for p in video_paths]
    M = len(names); K = len(centers)
    if K == 0:
        return {}, {}

    # Count occurrences per cluster per video
    counts = np.zeros((K, M), dtype=int)
    for i, s in enumerate(samples):
        cid = labels[i]
        if cid >= 0:
            counts[cid, s['vid']] += 1

    # Try perfect POV mapping: cluster absent (0) in exactly one video => that name
    zero_mask = counts == 0
    cluster_zero_vid = [np.where(z)[0] for z in zero_mask]  # list of arrays per cluster
    if all(len(z) == 1 for z in cluster_zero_vid):
        chosen = [int(z[0]) for z in cluster_zero_vid]
        # Check uniqueness (each video assigned once)
        if len(set(chosen)) == min(K, M):
            cluster_to_name = {c['cluster']: names[chosen[idx]] for idx, c in enumerate(centers)}
            name_to_cluster = {v: k for k, v in cluster_to_name.items()}
            return name_to_cluster, cluster_to_name

    # Fallback: minimal-cost assignment (prefer minimal presence in own video)
    # Cost matrix C[k][j] = counts of cluster k in video j (we want to minimize total)
    C = counts.copy()
    # Brute-force minimal assignment (M <= 4)
    from itertools import permutations
    best = None
    # We can only assign up to min(K, M) names
    assignable = min(K, M)
    cluster_idxs = list(range(K))
    for perm in permutations(cluster_idxs, assignable):
        cost = sum(C[perm[j], j] for j in range(assignable))
        if (best is None) or (cost < best[0]):
            best = (cost, perm)
    if best is not None:
        _, perm = best
        cluster_to_name = {perm[j]: names[j] for j in range(len(perm))}
        name_to_cluster = {names[j]: perm[j] for j in range(len(perm))}
        return name_to_cluster, cluster_to_name

    return {}, {}

def annotate_and_export(app, video_paths, centers, meta, cluster_to_name, out_dir, sim_threshold=0.35):
    os.makedirs(out_dir, exist_ok=True)
    # Prepack centers
    center_vecs = [np.asarray(c['center'], dtype=np.float32) for c in centers]
    center_ids = [c['cluster'] for c in centers]

    for vid_idx, path in enumerate(video_paths):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"Warning: cannot open {path}, skipping.")
            continue
        fps = meta[vid_idx]['fps']; W = meta[vid_idx]['W']; H = meta[vid_idx]['H']
        base = os.path.splitext(os.path.basename(path))[0]
        out_video_path = os.path.join(out_dir, f"{base}_labeled.mp4")
        out_csv_path = os.path.join(out_dir, f"{base}_face_log.csv")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(out_video_path, fourcc, fps, (W, H))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Failed to open VideoWriter for {out_video_path}")
        csv_file = open(out_csv_path, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['video','frame','timestamp_s','name','cluster_id','similarity','x1','y1','x2','y2','area_px','rel_area_percent'])

        frame_idx = 0
        frame_area = W * H
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            faces = app.get(frame)
            annotations = []
            if faces:
                for f in faces:
                    emb = get_face_embedding(f)
                    if emb is None:
                        continue
                    emb = normalize(np.asarray(emb, dtype=np.float32))
                    # Find best center
                    best_sim, best_cid = -1.0, None
                    for cid, cen in zip(center_ids, center_vecs):
                        sim = float(np.dot(emb, cen))
                        if sim > best_sim:
                            best_sim, best_cid = sim, cid
                    name = cluster_to_name.get(best_cid)
                    label = name if (name is not None and best_sim >= sim_threshold) else (f"Person {best_cid+1}" if best_cid is not None else "Unknown")

                    x1, y1, x2, y2 = [int(max(0, v)) for v in f.bbox.astype(int)]
                    area = max(0, (x2 - x1)) * max(0, (y2 - y1))
                    rel_area = (area / frame_area) * 100.0
                    csv_writer.writerow([base, frame_idx, round(frame_idx / fps, 3), label, best_cid if best_cid is not None else -1, round(best_sim, 3), x1, y1, x2, y2, area, round(rel_area, 3)])
                    annotations.append((label, (x1, y1, x2, y2)))

            for label, (x1, y1, x2, y2) in annotations:
                color = (0, 255, 0) if not label.startswith("Person") and label != "Unknown" else (0, 200, 255) if label.startswith("Person") else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

            writer.write(frame)
            frame_idx += 1

        cap.release()
        writer.release()
        csv_file.close()
        print(f"Wrote: {out_video_path} and {out_csv_path}")

def main():
    p = argparse.ArgumentParser(description="Face labeling with consistent names from video filenames (GPU if available)")
    p.add_argument('inputs', nargs='+', help='Input MP4 paths (1–4). Basenames are used as person names (POV owner).')
    p.add_argument('--out_dir', default='outputs', help='Output directory')
    p.add_argument('--sample_every', type=int, default=3, help='Frame sampling step for clustering pass')
    p.add_argument('--min_det_score', type=float, default=0.4, help='Min detector score to keep a face sample')
    p.add_argument('--dbscan_eps', type=float, default=0.35, help='DBSCAN eps (cosine distance)')
    p.add_argument('--dbscan_min_samples', type=int, default=5, help='DBSCAN min_samples')
    p.add_argument('--assign_sim_threshold', type=float, default=0.35, help='Min cosine similarity to assign a detection to a cluster/name')
    p.add_argument('--cpu_only', action='store_true', help='Force CPU (debug/fallback)')
    args = p.parse_args()

    if not (1 <= len(args.inputs) <= 4):
        raise SystemExit("Provide 1–4 input videos.")
    for path in args.inputs:
        if not os.path.isfile(path):
            raise SystemExit(f"File not found: {path}")

    os.makedirs(args.out_dir, exist_ok=True)

    app, used_gpu = init_face_app(prefer_gpu=not args.cpu_only)
    print(f"Backend: {'GPU (CUDA)' if used_gpu else 'CPU'}")

    print("Pass 1: extracting embeddings...")
    samples, meta = extract_embeddings(app, args.inputs, sample_every=args.sample_every, min_det_score=args.min_det_score)
    print(f"Samples collected: {len(samples)}")

    print("Clustering identities...")
    labels, centers = cluster_identities(samples, eps=args.dbscan_eps, min_samples=args.dbscan_min_samples)
    print(f"Clusters found: {len(centers)}")

    print("Resolving cluster->name mapping from filenames...")
    name_to_cluster, cluster_to_name = assign_clusters_to_names(args.inputs, samples, labels, centers)
    if not cluster_to_name:
        print("Warning: could not confidently assign names to clusters. Will use Person 1..N labels.")

    # Save identity bank + mapping
    id_bank = {
        'centers': [{'cluster': c['cluster'], 'center': c['center'].tolist(), 'count': c['count']} for c in centers],
        'name_to_cluster': name_to_cluster,
        'cluster_to_name': cluster_to_name,
        'dbscan': {'eps': args.dbscan_eps, 'min_samples': args.dbscan_min_samples},
        'assign_sim_threshold': args.assign_sim_threshold
    }
    with open(os.path.join(args.out_dir, 'identity_bank.json'), 'w') as f:
        json.dump(id_bank, f, indent=2)
    print(f"Saved mapping: {os.path.join(args.out_dir, 'identity_bank.json')}")

    print("Pass 2: annotating...")
    annotate_and_export(app, args.inputs, centers, meta, cluster_to_name, args.out_dir, sim_threshold=args.assign_sim_threshold)
    print("Done.")

if __name__ == '__main__':
    main()