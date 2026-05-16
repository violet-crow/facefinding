import cv2
import numpy as np
import pandas as pd
import argparse
import os
import math
import warnings
import onnxruntime as ort
from insightface.app import FaceAnalysis
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.filterwarnings("ignore", category=FutureWarning)

def time_to_sec(time_str):
    if pd.isna(time_str) or not str(time_str).strip(): return 0
    try:
        m, s = map(float, str(time_str).strip().split(':'))
        return m * 60 + s
    except ValueError:
        return 0

def format_time_ms(seconds):
    m = int(seconds // 60)
    s = seconds % 60
    return f"{m:02d}:{s:06.3f}"

def log_note(out_dir, message):
    with open(os.path.join(out_dir, "notes.txt"), "a") as f:
        f.write(message + "\n")

def detect_vision_pro_pointer(frame):
    """
    Detects the pointer using Adaptive Thresholding and Morphological Closing 
    to handle severe blurring, internal color gradients, and perspective distortion.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 1. Blur to merge internal color gradients and noise
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # 2. Adaptive thresholding with a larger block size (41) to capture the whole blob
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 41, 3)
    
    # 3. Morphological Closing to fill in holes and stitch fractured edges together
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Find contours on the newly closed, solid shapes
    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    best_center = None
    best_score = 0 

    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        if 20 < area < 3000:
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
                
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area == 0: 
                continue
            solidity = area / hull_area
            
            circularity = 4 * math.pi * (area / (perimeter * perimeter))
            
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h
            
            if solidity > 0.85 and 0.3 < aspect_ratio < 3.0:
                score = solidity + (circularity * 0.5)
                
                if score > best_score:
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        best_score = score
                        best_center = (cx, cy)

    return best_center

def process_single_video(video_task):
    filepath = video_task['filepath']
    base_name = video_task['base_name']
    start_sec = video_task['start_sec']
    end_sec = video_task['end_sec']
    master_embeddings = video_task['master_embeddings']
    target_ids = video_task['target_ids']
    out_dir = video_task['out_dir']
    draw_video = video_task['draw_video']

    app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    cap = cv2.VideoCapture(filepath)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    frame_interval = max(1, int(fps / 2))
    start_frame = int(start_sec * fps)
    end_frame = min(int(end_sec * fps), total_video_frames)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_idx = start_frame
    total_frames_to_process = max(1, end_frame - start_frame)
    
    print_interval = max(1, int(total_frames_to_process / 10))

    writer = None
    if draw_video:
        out_vid_path = os.path.join(out_dir, f"{base_name}_verification.mp4")
        writer = cv2.VideoWriter(out_vid_path, cv2.VideoWriter_fourcc(*'mp4v'), 2.0, (width, height))

    csv_data = []
    prev_gaze = None

    print(f"[{base_name}] Started processing.")

    while cap.isOpened() and frame_idx <= end_frame:
        ret, frame = cap.read()
        if not ret: break

        if (frame_idx - start_frame) % frame_interval != 0:
            frame_idx += 1
            continue

        frames_processed = frame_idx - start_frame
        if frames_processed % print_interval == 0 or frame_idx == end_frame:
            percent = (frames_processed / total_frames_to_process) * 100
            print(f"[{base_name}] Frame {frame_idx}/{end_frame} ({percent:.1f}%)")

        current_time_ms = format_time_ms(frame_idx / fps)
        
        gaze_coords = detect_vision_pro_pointer(frame)
        gaze_x, gaze_y = None, None
        delta_x, delta_y, euclidean_dist = None, None, None

        if gaze_coords:
            gaze_x, gaze_y = gaze_coords
            if prev_gaze:
                delta_x = gaze_x - prev_gaze[0]
                delta_y = gaze_y - prev_gaze[1]
                euclidean_dist = math.hypot(delta_x, delta_y)
            prev_gaze = (gaze_x, gaze_y)
            
            if draw_video:
                cv2.circle(frame, (gaze_x, gaze_y), 8, (255, 0, 0), -1)
                cv2.drawMarker(frame, (gaze_x, gaze_y), (255, 255, 0), cv2.MARKER_CROSS, 25, 2)
        else:
            prev_gaze = None 

        faces = app.get(frame)
        
        row_data = {
            'frame': frame_idx,
            'time': current_time_ms,
            'gaze_x': gaze_x,
            'gaze_y': gaze_y,
            'gaze_delta_x': delta_x,
            'gaze_delta_y': delta_y,
            'gaze_delta_euclidean': euclidean_dist,
            'faces_detected_count': len(faces),
            'gaze_on_any_face': False
        }

        all_columns = target_ids + ['UNKNOWN']
        for label_id in all_columns:
            row_data.update({
                f'{label_id}_detected': False,
                f'{label_id}_tl_x': '', f'{label_id}_tl_y': '',
                f'{label_id}_br_x': '', f'{label_id}_br_y': '',
                f'gaze_on_{label_id}': False
            })

        for face in faces:
            identity = "UNKNOWN"
            highest_sim = -1
            best_pid = None
            
            for pid, master_emb in master_embeddings.items():
                sim = np.dot(face.normed_embedding, master_emb)
                if sim > highest_sim:
                    highest_sim = sim
                    best_pid = pid

            if highest_sim > 0.38:
                identity = best_pid

            x1, y1, x2, y2 = map(int, face.bbox)
            
            gaze_hit = False
            if gaze_coords:
                gx, gy = gaze_coords
                if x1 <= gx <= x2 and y1 <= gy <= y2:
                    gaze_hit = True
                    row_data['gaze_on_any_face'] = True

            row_data.update({
                f'{identity}_detected': True,
                f'{identity}_tl_x': x1,
                f'{identity}_tl_y': y1,
                f'{identity}_br_x': x2,
                f'{identity}_br_y': y2,
                f'gaze_on_{identity}': gaze_hit
            })

            if draw_video:
                color = (0, 255, 0) if identity != "UNKNOWN" else (128, 128, 128)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label_str = f"{identity} ({highest_sim:.2f})" if identity != "UNKNOWN" else "UNKNOWN"
                cv2.putText(frame, label_str, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        csv_data.append(row_data)
        if writer:
            writer.write(frame)
        
        skip_frames = frame_interval - 1
        for _ in range(skip_frames):
            cap.grab()
            frame_idx += 1
            
        frame_idx += 1

    cap.release()
    if writer: writer.release()

    df_out = pd.DataFrame(csv_data)
    df_out.to_csv(os.path.join(out_dir, f"{base_name}_tracking.csv"), index=False)

    total_queried = len(df_out)
    # Count strictly frames where gaze was physically detected
    total_gaze_detected = df_out['gaze_x'].notna().sum()

    if total_queried > 0:
        avg_faces = df_out['faces_detected_count'].mean()
        
        summary_data = {
            'total_frames_queried': total_queried,
            'total_gaze_detected': total_gaze_detected,
            'avg_faces_per_frame': round(avg_faces, 2),
            'mean_gaze_delta_x': round(df_out['gaze_delta_x'].abs().mean(), 2),
            'mean_gaze_delta_y': round(df_out['gaze_delta_y'].abs().mean(), 2),
            'mean_gaze_euclidean': round(df_out['gaze_delta_euclidean'].mean(), 2)
        }

        # Calculate percentages based entirely on frames where the gaze circle existed
        if total_gaze_detected > 0:
            pct_gaze_any = (df_out['gaze_on_any_face'].sum() / total_gaze_detected) * 100
            summary_data['pct_gaze_on_any_face'] = round(pct_gaze_any, 2)
            
            for pid in all_columns:
                col_name = f'gaze_on_{pid}'
                if col_name in df_out.columns:
                    pct_gaze_specific = (df_out[col_name].sum() / total_gaze_detected) * 100
                    summary_data[f'pct_{col_name}'] = round(pct_gaze_specific, 2)
        else:
            summary_data['pct_gaze_on_any_face'] = 0.0
            for pid in all_columns:
                col_name = f'gaze_on_{pid}'
                if col_name in df_out.columns:
                    summary_data[f'pct_{col_name}'] = 0.0

        pd.DataFrame([summary_data]).to_csv(os.path.join(out_dir, f"{base_name}_summary.csv"), index=False)

    print(f"[{base_name}] Completed and saved.")
    return base_name

def get_args():
    parser = argparse.ArgumentParser(description="Multiprocessing Face & Gaze Tracker")
    parser.add_argument('manifest', nargs='?', type=str, default=r'C:\Users\VHILAB Core\Desktop\Spatial Coherence\facefinding\tracker_manifest.csv', help="Path to manifest.csv")
    parser.add_argument('-t', action='store_true', help="Test mode: Only process a specific group for a specific duration")
    parser.add_argument('-d', action='store_true', help="Output video with bounding boxes and gaze markers drawn onto frames")
    parser.add_argument('--dir', type=str, default=r'C:\Users\VHILAB Core\Desktop\Spatial Coherence\POV Videos')
    parser.add_argument('--out', type=str, default='./output')
    parser.add_argument('--known', type=str, default='./known_faces')
    parser.add_argument('--workers', type=int, default=6, help="Number of concurrent videos to process")
    return parser.parse_args()

def main():
    args = get_args()
    os.makedirs(args.out, exist_ok=True)
    
    with open(os.path.join(args.out, "notes.txt"), "w") as f:
        f.write("--- Processing Log ---\n")

    if not os.path.exists(args.manifest):
        print(f"Error: Could not find manifest file at {args.manifest}")
        return

    df_manifest = pd.read_csv(args.manifest, dtype={'participant_num': str, 'group_num': str})
    test_minutes = None

    if args.t:
        target_group = input("Enter the group_num to test: ").strip()
        df_manifest = df_manifest[df_manifest['group_num'] == target_group]
        
        if df_manifest.empty:
            print(f"Error: Group '{target_group}' not found in manifest.")
            return
            
        try:
            test_minutes = float(input("Enter minutes to process per video (e.g., 1 or 0.5): "))
        except ValueError:
            print("Error: Invalid number. Exiting.")
            return

        print(f"\n--- TEST MODE: Isolating Group {target_group} for {test_minutes} minute(s) per video ---")
    
    video_tasks = []
    grouped = df_manifest.groupby('group_num')
    
    for group_num, group_data in grouped:
        group_out_dir = os.path.join(args.out, f"Group_{group_num}")
        os.makedirs(group_out_dir, exist_ok=True)
        
        target_ids = group_data['participant_num'].tolist()
        
        master_embeddings = {}
        for pid in target_ids:
            ref_dir = os.path.join(args.known, pid)
            if not os.path.exists(ref_dir):
                log_note(args.out, f"Group {group_num}: Missing known_faces profile for participant {pid}.")
                continue
                
            person_embeddings = []
            for file_name in os.listdir(ref_dir):
                if file_name.lower().endswith(('.png', '.jpg')):
                    npy_path = os.path.join(ref_dir, f"{os.path.splitext(file_name)[0]}.npy")
                    if os.path.exists(npy_path):
                        person_embeddings.append(np.load(npy_path))
            
            if person_embeddings:
                master_embeddings[pid] = np.mean(person_embeddings, axis=0)
            else:
                log_note(args.out, f"Group {group_num}: No valid .npy files found in profile for {pid}.")

        for _, row in group_data.iterrows():
            pid = row['participant_num']
            filename = f"{pid}_POV.mp4"
            filepath = os.path.join(args.dir, filename)
            
            if not os.path.exists(filepath):
                log_note(args.out, f"Group {group_num}: Video missing. Skipping {filename}.")
                continue
                
            start_sec = time_to_sec(row['start_time'])
            end_sec = time_to_sec(row['end_time'])
            if end_sec == 0: end_sec = float('inf')

            if args.t and test_minutes is not None:
                end_sec = min(end_sec, start_sec + (test_minutes * 60))

            video_tasks.append({
                'filepath': filepath,
                'base_name': f"{pid}_POV",
                'start_sec': start_sec,
                'end_sec': end_sec,
                'master_embeddings': master_embeddings,
                'target_ids': target_ids,
                'out_dir': group_out_dir,
                'draw_video': args.d
            })

    if not video_tasks:
        print("No valid tasks found to process.")
        return

    print(f"Queue built: {len(video_tasks)} videos ready.")
    print(f"Starting {args.workers} workers...\n")

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_single_video, task) for task in video_tasks]
        
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                print(f"Worker generated an exception: {exc}")

    print("\nProcessing complete. Check notes.txt for warnings.")

if __name__ == "__main__":
    main()