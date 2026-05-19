import cv2
import numpy as np
import os
import argparse
import csv
from sklearn.cluster import DBSCAN
from collections import defaultdict
import concurrent.futures

# Global variable to hold the model instance per worker process
global_app = None

def init_worker():
    """Initializes the InsightFace model once per worker process to avoid IPC overhead."""
    global global_app
    from insightface.app import FaceAnalysis
    
    # Optimize ONNXRuntime for high-VRAM cards
    providers = [
        ('CUDAExecutionProvider', {
            'device_id': 0,
            'arena_extend_strategy': 'kNextPowerOfTwo',
            'cudnn_conv_algo_search': 'EXHAUSTIVE',
            'do_copy_in_default_stream': True,
        })
    ]
    
    global_app = FaceAnalysis(name='buffalo_l', providers=providers)
    global_app.prepare(ctx_id=0, det_size=(640, 640))

def process_video_worker(args):
    """Worker function to process a single video."""
    p_idx, p, shift_sec = args
    global global_app
    
    cap = cv2.VideoCapture(p['path'])
    if not cap.isOpened():
        return [], [], []

    fps = cap.get(cv2.CAP_PROP_FPS)
    current_start_sec = p['original_start'] + shift_sec
    start_frame = int(current_start_sec * fps)
    end_frame = int((current_start_sec + 60) * fps) 
    frame_step = max(1, int(fps / 2)) 
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    curr_frame = start_frame
    
    local_embeddings = []
    local_crops = []
    local_metadata = []
    
    while curr_frame < end_frame:
        ret = cap.grab() # Fast forward without decoding
        if not ret: break
        
        if (curr_frame - start_frame) % frame_step == 0:
            ret, frame = cap.retrieve() # Only decode the frames we need
            if ret:
                faces = global_app.get(frame)
                for face in faces:
                    local_embeddings.append(face.normed_embedding)
                    
                    x1, y1, x2, y2 = map(int, face.bbox)
                    pad_w = int((x2 - x1) * 0.5)
                    pad_h = int((y2 - y1) * 0.5)
                    y1_p, y2_p = max(0, y1 - pad_h), min(frame.shape[0], y2 + pad_h)
                    x1_p, x2_p = max(0, x1 - pad_w), min(frame.shape[1], x2 + pad_w)
                    
                    crop = frame[y1_p:y2_p, x1_p:x2_p]
                    if crop.size > 0:
                        local_crops.append(crop)
                        local_metadata.append(p_idx)
        
        curr_frame += 1
    
    cap.release()
    return local_embeddings, local_crops, local_metadata

def time_to_sec(time_str):
    try:
        m, s = map(int, str(time_str).strip().split(':'))
        return m * 60 + s
    except (ValueError, AttributeError):
        print(f"Warning: Could not parse time '{time_str}'. Expected mm:ss format.")
        return None

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default=r'C:\Users\VHILAB Core\Desktop\Spatial Coherence\facefinding\harvest_manifest.csv', help="Path to the input CSV file")
    parser.add_argument('--dir', type=str, default=r'C:\Users\VHILAB Core\Desktop\Spatial Coherence\POV Videos', help="Path to the POV videos directory")
    parser.add_argument('--ref_out', type=str, default='./known_faces', help="Where to save face crops")
    return parser.parse_args()

def main():
    args = get_args()
    
    if not os.path.exists(args.csv):
        print(f"Error: CSV file not found at {args.csv}")
        return

    groups = defaultdict(list)
    with open(args.csv, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        required_headers = {'participant_num', 'group_num', 'start_time'}
        if not required_headers.issubset(set(reader.fieldnames)):
            print(f"Error: CSV must contain headers: {required_headers}")
            return
            
        for row in reader:
            groups[row['group_num']].append(row)

    # Initialize Process Pool. Max workers corresponds to average group size to avoid thrashing.
    # We use 4 here as your groups max out around 4. 
    with concurrent.futures.ProcessPoolExecutor(max_workers=4, initializer=init_worker) as executor:
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
                    print(f"!!! Warning: {filename} not found. Skipping member {formatted_id}.")
                    continue

                start_sec = time_to_sec(row['start_time'])
                if start_sec is None:
                    continue

                participants.append({'id': formatted_id, 'path': filepath, 'original_start': start_sec})

            if not participants:
                print(f"No valid files/times found for Group {group_num}. Skipping.")
                continue

            found_identities = set()
            max_attempts = 4
            shift_interval = 180

            for attempt in range(max_attempts):
                if len(found_identities) == len(participants):
                    break

                shift_sec = attempt * shift_interval
                print(f"\n--- Harvesting Data (Attempt {attempt + 1}/{max_attempts} | Time Shift: +{shift_sec}s) ---")
                
                all_embeddings = []
                all_crops = []
                metadata = [] 

                # Dispatch videos to parallel workers
                futures = []
                for p_idx, p in enumerate(participants):
                    print(f"  -> Dispatching {p['id']}_POV.mp4 from {p['original_start'] + shift_sec}s to {p['original_start'] + shift_sec + 60}s to worker...")
                    futures.append(executor.submit(process_video_worker, (p_idx, p, shift_sec)))
                
                # Gather results as they complete
                for future in concurrent.futures.as_completed(futures):
                    emb, crp, meta = future.result()
                    all_embeddings.extend(emb)
                    all_crops.extend(crp)
                    metadata.extend(meta)

                if not all_embeddings:
                    print("No faces detected in this window across all videos.")
                    continue

                print("Clustering and Identifying...")
                clustering = DBSCAN(eps=0.5, min_samples=5, metric='cosine').fit(all_embeddings)
                
                cluster_counts = defaultdict(lambda: {i: 0 for i in range(len(participants))})
                for idx, cluster_id in enumerate(clustering.labels_):
                    if cluster_id != -1:
                        cluster_counts[cluster_id][metadata[idx]] += 1

                for cluster_id, counts in cluster_counts.items():
                    wearer_idx = min(counts, key=counts.get)
                    actual_id = participants[wearer_idx]['id']
                    
                    if actual_id in found_identities:
                        continue
                    
                    out_dir = os.path.join(args.ref_out, actual_id)
                    os.makedirs(out_dir, exist_ok=True)
                    
                    saved_count = 0
                    for idx, c_id in enumerate(clustering.labels_):
                        if c_id == cluster_id and saved_count < 25:
                            base_name = f"ref_{actual_id}_batch_{idx}"
                            cv2.imwrite(os.path.join(out_dir, f"{base_name}.jpg"), all_crops[idx])
                            np.save(os.path.join(out_dir, f"{base_name}.npy"), all_embeddings[idx])
                            saved_count += 1
                    
                    print(f"Found Identity {actual_id}: Saved {saved_count} references.")
                    found_identities.add(actual_id)

            if len(found_identities) == len(participants):
                print(f"SUCCESS: All {len(participants)} identities found for Group {group_num}.")
            else:
                missing = [p['id'] for p in participants if p['id'] not in found_identities]
                print(f"WARNING: Group {group_num} finished all retries. Missing: {missing}. Moving to next group.")

if __name__ == "__main__":
    main()