import os
os.environ.setdefault("INSIGHTFACE_ONNXRT_PROVIDERS", "CPUExecutionProvider")
os.environ.setdefault("ORT_LOG_SEVERITY_LEVEL", "4")  # suppress ORT CUDA warnings
import csv
import json
import argparse
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from insightface.app import FaceAnalysis

def normalize(v):
    n = np.linalg.norm(v) + 1e-8
    return v / n

def get_face_embedding(face):
    emb = getattr(face, 'normed_embedding', None)
    if emb is None:
        emb = getattr(face, 'embedding', None)
    return emb

def init_face_app(det_size=(640, 640)):
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=-1, det_size=det_size)  # CPU-only
    return app

def extract_embeddings(app, video_paths, sample_every=3, min_det_score=0.4, max_faces_per_frame=10):
    samples = []
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
            if not faces or len(faces) == 0:
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
                samples.append({'vid': vid_idx, 'frame': frame_idx, 'ts': ts, 'bbox': bbox, 'emb': emb, 'score': score})
            frame_idx += 1
        cap.release()
    return samples, meta

def cluster_identities(samples, eps=0.35, min_samples=5):
    if not samples:
        return {}, []
    X = np.stack([s['emb'] for s in samples], axis=0)
    db = DBSCAN(eps=eps, metric='cosine', min_samples=min_samples)
    labels = db.fit_predict(X)  # -1 is noise
    clusters = {}
    for lbl, emb in zip(labels, X):
        if lbl == -1:
            continue
        clusters.setdefault(lbl, []).append(emb)
    sorted_lbls = sorted(clusters.keys(), key=lambda l: len(clusters[l]), reverse=True)
    lbl_to_person = {lbl: i + 1 for i, lbl in enumerate(sorted_lbls)}
    centers = []
    for lbl in sorted_lbls:
        center = normalize(np.mean(np.stack(clusters[lbl], axis=0), axis=0))
        centers.append({'person_id': lbl_to_person[lbl], 'center': center, 'count': len(clusters[lbl])})
    return lbl_to_person, centers

def assign_to_center(emb, centers, sim_threshold=0.35):
    if not centers:
        return None, None
    best_sim, best_pid = -1.0, None
    for c in centers:
        sim = float(np.dot(emb, c['center']))
        if sim > best_sim:
            best_sim, best_pid = sim, c['person_id']
    if best_sim >= sim_threshold:
        return best_pid, best_sim
    return None, best_sim

def annotate_and_export(app, video_paths, centers, out_dir, meta, sim_threshold=0.35):
    os.makedirs(out_dir, exist_ok=True)
    for vid_idx, path in enumerate(video_paths):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"Warning: cannot open {path}, skipping.")
            continue
        fps = meta[vid_idx]['fps']
        W, H = meta[vid_idx]['W'], meta[vid_idx]['H']
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        base = os.path.splitext(os.path.basename(path))[0]
        out_video_path = os.path.join(out_dir, f"{base}_labeled.mp4")
        out_csv_path = os.path.join(out_dir, f"{base}_face_log.csv")
        writer = cv2.VideoWriter(out_video_path, fourcc, fps, (W, H))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Failed to open VideoWriter for {out_video_path}")
        csv_file = open(out_csv_path, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['video','frame','timestamp_s','label','person_id','similarity','x1','y1','x2','y2','area_px','rel_area_percent'])

        print(f"Annotating: {path} -> {out_video_path}")
        frame_idx = 0
        frame_area = W * H
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            faces = app.get(frame)
            annotations = []
            if faces and len(faces) > 0:
                for f in faces:
                    emb = get_face_embedding(f)
                    if emb is None:
                        continue
                    emb = normalize(np.asarray(emb, dtype=np.float32))
                    bbox = f.bbox.astype(int)

                    pid, sim = assign_to_center(emb, centers, sim_threshold=sim_threshold)

                    x1, y1, x2, y2 = [int(max(0, v)) for v in bbox]
                    area = max(0, (x2 - x1)) * max(0, (y2 - y1))
                    rel_area = (area / frame_area) * 100.0
                    label = f"Person {pid}" if pid is not None else "Unknown"
                    csv_writer.writerow([os.path.basename(path), frame_idx, round(frame_idx / fps, 3), label, pid if pid is not None else -1, round(sim if sim is not None else -1, 3), x1, y1, x2, y2, area, round(rel_area, 3)])
                    annotations.append((label, (x1, y1, x2, y2)))

            for label, (x1, y1, x2, y2) in annotations:
                color = (0, 255, 0) if label.startswith("Person") else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

            writer.write(frame)
            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"  {frame_idx}/{total_frames if total_frames>0 else '?'} frames")

        cap.release()
        writer.release()
        csv_file.close()
        print(f"Done: {out_video_path}, {out_csv_path}")

def main():
    p = argparse.ArgumentParser(description="CPU-only face labeling with consistent IDs across up to 4 videos")
    p.add_argument('inputs', nargs='+', help='Input MP4 paths (1–4)')
    p.add_argument('--out_dir', default='outputs', help='Directory for labeled videos and logs')
    p.add_argument('--sample_every', type=int, default=3, help='Frame sampling step for clustering pass (higher = faster)')
    p.add_argument('--min_det_score', type=float, default=0.4, help='Min detector score to keep a face sample during clustering')
    p.add_argument('--dbscan_eps', type=float, default=0.35, help='DBSCAN eps (cosine distance). Lower merges less, higher merges more.')
    p.add_argument('--dbscan_min_samples', type=int, default=5, help='DBSCAN min_samples')
    p.add_argument('--assign_sim_threshold', type=float, default=0.35, help='Min cosine similarity to assign to a person')
    args = p.parse_args()

    if len(args.inputs) == 0 or len(args.inputs) > 4:
        raise SystemExit("Provide 1–4 input videos.")

    for path in args.inputs:
        if not os.path.isfile(path):
            raise SystemExit(f"File not found: {path}")

    os.makedirs(args.out_dir, exist_ok=True)

    print("Initializing InsightFace (CPU-only)...")
    app = init_face_app()

    print("Pass 1: extracting embeddings and clustering identities...")
    samples, meta = extract_embeddings(app, args.inputs, sample_every=args.sample_every, min_det_score=args.min_det_score)
    _, centers = cluster_identities(samples, eps=args.dbscan_eps, min_samples=args.dbscan_min_samples)
    print(f"Identities found: {len(centers)}")

    # Save identity bank for transparency/reuse
    id_bank_path = os.path.join(args.out_dir, 'identity_bank.json')
    with open(id_bank_path, 'w') as f:
        json.dump({
            'centers': [{'person_id': c['person_id'], 'center': c['center'].tolist(), 'count': c['count']} for c in centers],
            'dbscan': {'eps': args.dbscan_eps, 'min_samples': args.dbscan_min_samples},
            'assign_sim_threshold': args.assign_sim_threshold
        }, f, indent=2)
    print(f"Saved identity bank: {id_bank_path}")

    # Convert centers to runtime format (numpy arrays)
    centers_np = [{'person_id': c['person_id'], 'center': np.asarray(c['center'], dtype=np.float32)} for c in centers]

    print("Pass 2: annotating videos...")
    annotate_and_export(app, args.inputs, centers_np, args.out_dir, meta, sim_threshold=args.assign_sim_threshold)
    print("All done.")

if __name__ == '__main__':
    main()