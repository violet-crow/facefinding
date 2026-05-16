import cv2
import numpy as np
import os
import argparse
from insightface.app import FaceAnalysis
from sklearn.cluster import DBSCAN
from collections import defaultdict

def time_to_sec(time_str):
    try:
        m, s = map(int, time_str.split(':'))
        return m * 60 + s
    except ValueError:
        print("Invalid format. Please use mm:ss.")
        return None

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default=r'P:\VHIL\Videos\POV')
    parser.add_argument('--ref_out', type=str, default='./known_faces', help="Where to save face crops")
    parser.add_argument('--rate', type=int, default=15, help="Sample every Nth frame for harvesting")
    return parser.parse_args()

def main():
    import onnxruntime as ort
    print("Available ORT Providers:", ort.get_available_providers())
    args = get_args()
    app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    while True:
        try:
            group_size = int(input("Enter group size (2-4): "))
            start_num = int(input("Enter first group member number (e.g., 1 or 100): "))
            break
        except ValueError:
            print("Please enter numeric values.")

    participants = []
    for i in range(group_size):
        member_id = start_num + i
        formatted_id = f"{member_id:03d}"
        filename = f"{formatted_id}_POV.mp4"
        filepath = os.path.join(args.dir, filename)

        if not os.path.exists(filepath):
            print(f"!!! Warning: {filename} not found in {args.dir}. Skipping member {formatted_id}.")
            continue

        sec = None
        while sec is None:
            ts = input(f"Enter 'ready' timestamp for {filename} (mm:ss): ")
            sec = time_to_sec(ts)

        participants.append({'id': formatted_id, 'path': filepath, 'start': sec})

    if not participants:
        print("No valid files found to process.")
        return

    all_embeddings = []
    all_crops = []
    metadata = [] 

    print("\n--- Harvesting Face Data (30s Window) ---")
    for p_idx, p in enumerate(participants):
        cap = cv2.VideoCapture(p['path'])
        fps = cap.get(cv2.CAP_PROP_FPS)
        start_frame = int(p['start'] * fps)
        end_frame = int((p['start'] + 30) * fps)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        curr_frame = start_frame
        
        print(f"Processing {p['id']}...")
        while curr_frame < end_frame:
            ret, frame = cap.read()
            if not ret: break
            
            if (curr_frame - start_frame) % args.rate == 0:
                faces = app.get(frame)
                for face in faces:
                    all_embeddings.append(face.normed_embedding)
                    
                    x1, y1, x2, y2 = map(int, face.bbox)
                    
                    # Add 50% padding for better human readability
                    pad_w = int((x2 - x1) * 0.5)
                    pad_h = int((y2 - y1) * 0.5)
                    y1_p, y2_p = max(0, y1 - pad_h), min(frame.shape[0], y2 + pad_h)
                    x1_p, x2_p = max(0, x1 - pad_w), min(frame.shape[1], x2 + pad_w)
                    
                    crop = frame[y1_p:y2_p, x1_p:x2_p]
                    if crop.size > 0:
                        all_crops.append(crop)
                        metadata.append(p_idx)
            
            curr_frame += 1
        cap.release()

    if not all_embeddings:
        print("No faces detected in the provided windows.")
        return

    print("--- Clustering and Identifying ---")
    clustering = DBSCAN(eps=0.5, min_samples=5, metric='cosine').fit(all_embeddings)
    
    cluster_counts = defaultdict(lambda: {i: 0 for i in range(len(participants))})
    for idx, cluster_id in enumerate(clustering.labels_):
        if cluster_id != -1:
            cluster_counts[cluster_id][metadata[idx]] += 1

    for cluster_id, counts in cluster_counts.items():
        wearer_idx = min(counts, key=counts.get)
        actual_id = participants[wearer_idx]['id']
        
        out_dir = os.path.join(args.ref_out, actual_id)
        os.makedirs(out_dir, exist_ok=True)
        
        saved_count = 0
        for idx, c_id in enumerate(clustering.labels_):
            if c_id == cluster_id and saved_count < 25:
                base_name = f"ref_{actual_id}_batch_{idx}"
                
                # Save BOTH the image (for you) and the raw embedding (for the Tracker)
                cv2.imwrite(os.path.join(out_dir, f"{base_name}.jpg"), all_crops[idx])
                np.save(os.path.join(out_dir, f"{base_name}.npy"), all_embeddings[idx])
                saved_count += 1
        
        print(f"Found Identity {actual_id}: Saved {saved_count} reference crops & data files.")

if __name__ == "__main__":
    main()