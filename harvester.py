import cv2
import numpy as np
import os
import argparse
import csv
from insightface.app import FaceAnalysis
from sklearn.cluster import DBSCAN
from collections import defaultdict

def time_to_sec(time_str):
    try:
        m, s = map(int, str(time_str).strip().split(':'))
        return m * 60 + s
    except (ValueError, AttributeError):
        print(f"Warning: Could not parse time '{time_str}'. Expected mm:ss format.")
        return None

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default=r'P:\VHIL\Facefinding + Faceharvester\harvest_manifest.csv')
    parser.add_argument('--dir', type=str, default=r'P:\VHIL\Videos\POV')
    parser.add_argument('--ref_out', type=str, default='./known_faces', help="Where to save face crops")
    return parser.parse_args()

def main():
    import onnxruntime as ort
    print("Available ORT Providers:", ort.get_available_providers())
    args = get_args()
    
    if not os.path.exists(args.csv):
        print(f"Error: CSV file not found at {args.csv}")
        return

    app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    # Read and group CSV data
    groups = defaultdict(list)
    with open(args.csv, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        required_headers = {'participant_num', 'group_num', 'start_time'}
        if not required_headers.issubset(set(reader.fieldnames)):
            print(f"Error: CSV must contain headers: {required_headers}")
            return
            
        for row in reader:
            groups[row['group_num']].append(row)

    for group_num, group_rows in groups.items():
        print(f"\n=====================================")
        print(f"Processing Group {group_num}")
        print(f"=====================================")
        
        participants = []
        for row in group_rows:
            formatted_id = f"{int(row['participant_num']):03d}"
            filename = f"{formatted_id}_POV.mp4"
            filepath = os.path.join(args.dir, filename)

            if not os.path.exists(filepath):
                print(f"!!! Warning: {filename} not found in {args.dir}. Skipping member {formatted_id}.")
                continue

            start_sec = time_to_sec(row['start_time'])
            if start_sec is None:
                continue

            participants.append({'id': formatted_id, 'path': filepath, 'start': start_sec})

        if not participants:
            print(f"No valid files/times found for Group {group_num}. Skipping.")
            continue

        all_embeddings = []
        all_crops = []
        metadata = [] 

        print(f"\n--- Harvesting Face Data (60s Window) ---")
        for p_idx, p in enumerate(participants):
            cap = cv2.VideoCapture(p['path'])
            if not cap.isOpened():
                print(f"Failed to open video: {p['path']}")
                continue

            fps = cap.get(cv2.CAP_PROP_FPS)
            start_frame = int(p['start'] * fps)
            end_frame = int((p['start'] + 60) * fps) 
            frame_step = max(1, int(fps / 2)) # Sample ~2 frames per second
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            curr_frame = start_frame
            
            print(f"Processing {p['id']} starting at {p['start']}s...")
            while curr_frame < end_frame:
                ret, frame = cap.read()
                if not ret: break
                
                if (curr_frame - start_frame) % frame_step == 0:
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
            print(f"No faces detected in the provided windows for Group {group_num}.")
            continue

        print("\n--- Clustering and Identifying ---")
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
                    
                    cv2.imwrite(os.path.join(out_dir, f"{base_name}.jpg"), all_crops[idx])
                    np.save(os.path.join(out_dir, f"{base_name}.npy"), all_embeddings[idx])
                    saved_count += 1
            
            print(f"Found Identity {actual_id}: Saved {saved_count} reference crops & data files.")

if __name__ == "__main__":
    main()