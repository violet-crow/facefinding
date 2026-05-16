import cv2
import numpy as np
import pandas as pd
import argparse
import os
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
        out_vid_path = os.path.join(out_dir, f"{base_name}_face_verification.mp4")
        writer = cv2.VideoWriter(out_vid_path, cv2.VideoWriter_fourcc(*'mp4v'), 2.0, (width, height))

    csv_data = []

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
        
        faces = app.get(frame)
        
        row_data = {
            'frame': frame_idx,
            'time': current_time_ms,
            'faces_detected_count': len(faces)
        }

        all_columns = target_ids + ['UNKNOWN']
        for label_id in all_columns:
            row_data.update({
                f'{label_id}_detected': False,
                f'{label_id}_tl_x': '', f'{label_id}_tl_y': '',
                f'{label_id}_br_x': '', f'{label_id}_br_y': ''
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

            row_data.update({
                f'{identity}_detected': True,
                f'{identity}_tl_x': x1,
                f'{identity}_tl_y': y1,
                f'{identity}_br_x': x2,
                f'{identity}_br_y': y2
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
    df_out.to_csv(os.path.join(out_dir, f"{base_name}_face_tracking.csv"), index=False)

    total_queried = len(df_out)
    if total_queried > 0:
        avg_faces = df_out['faces_detected_count'].mean()
        
        summary_data = {
            'total_frames_queried': total_queried,
            'avg_faces_per_frame': round(avg_faces, 2)
        }
        pd.DataFrame([summary_data]).to_csv(os.path.join(out_dir, f"{base_name}_face_summary.csv"), index=False)

    print(f"[{base_name}] Completed and saved.")
    return base_name

def get_args():
    parser = argparse.ArgumentParser(description="Multiprocessing Face Tracker (No Gaze)")
    parser.add_argument('manifest', nargs='?', type=str, default=r'C:\Users\VHILAB Core\Desktop\Spatial Coherence\facefinding\tracker_manifest.csv', help="Path to manifest.csv")
    parser.add_argument('-t', action='store_true', help="Test mode")
    parser.add_argument('-d', action='store_true', help="Output drawn verification video")
    parser.add_argument('--dir', type=str, default=r'C:\Users\VHILAB Core\Desktop\Spatial Coherence\POV Videos')
    parser.add_argument('--out', type=str, default='./output')
    parser.add_argument('--known', type=str, default='./known_faces')
    parser.add_argument('--workers', type=int, default=6, help="Workers")
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
                log_note(args.out, f"Group {group_num}: Missing profile for {pid}.")
                continue
                
            person_embeddings = []
            for file_name in os.listdir(ref_dir):
                if file_name.lower().endswith(('.png', '.jpg')):
                    npy_path = os.path.join(ref_dir, f"{os.path.splitext(file_name)[0]}.npy")
                    if os.path.exists(npy_path):
                        person_embeddings.append(np.load(npy_path))
            
            if person_embeddings:
                master_embeddings[pid] = np.mean(person_embeddings, axis=0)

        for _, row in group_data.iterrows():
            pid = row['participant_num']
            filename = f"{pid}_POV.mp4"
            filepath = os.path.join(args.dir, filename)
            
            if not os.path.exists(filepath):
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
        print("No valid tasks found.")
        return

    print(f"Queue built: {len(video_tasks)} videos ready.")
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_single_video, task) for task in video_tasks]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                print(f"Exception: {exc}")

    print("\nProcessing complete.")

if __name__ == "__main__":
    main()