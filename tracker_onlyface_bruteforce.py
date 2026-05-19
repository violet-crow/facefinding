import cv2
import pandas as pd
import argparse
import os
import warnings
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

def process_single_video(video_task):
    filepath = video_task['filepath']
    base_name = video_task['base_name']
    start_sec = video_task['start_sec']
    end_sec = video_task['end_sec']
    out_dir = video_task['out_dir']
    draw_video = video_task['draw_video']

    app = FaceAnalysis(name='buffalo_l', allowed_modules=['detection'], providers=['CUDAExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    cap = cv2.VideoCapture(filepath)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    frame_interval = max(1, round(fps / 2)) 
    start_frame = int(start_sec * fps)
    end_frame = min(int(end_sec * fps), total_video_frames)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_idx = start_frame
    total_frames_to_process = max(1, end_frame - start_frame)
    print_interval = max(1, int(total_frames_to_process / 10))

    writer = None
    if draw_video:
        out_vid_path = os.path.join(out_dir, f"{base_name}_face_detection.mp4")
        writer = cv2.VideoWriter(out_vid_path, cv2.VideoWriter_fourcc(*'mp4v'), 2.0, (width, height))

    csv_data = []
    print(f"[{base_name}] Started processing.")

    # --- BRUTE FORCE VARIABLES ---
    consecutive_failures = 0
    max_failures = 1000  # Gives up after ~30 seconds of pure dead frames

    while cap.isOpened() and frame_idx <= end_frame:
        ret, frame = cap.read()
        
        # --- BRUTE FORCE LOGIC ---
        if not ret: 
            consecutive_failures += 1
            if consecutive_failures > max_failures:
                print(f"[{base_name}] Reached true EOF or unrecoverable corruption at frame {frame_idx}.")
                break
            
            # Manually force the playhead forward past the bad frame
            frame_idx += 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            continue
            
        consecutive_failures = 0 # Reset strikes on a successful read
        # -------------------------

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

        for i, face in enumerate(faces):
            x1, y1, x2, y2 = map(int, face.bbox)
            row_data.update({
                f'face_{i}_tl_x': x1,
                f'face_{i}_tl_y': y1,
                f'face_{i}_br_x': x2,
                f'face_{i}_br_y': y2
            })

            if draw_video:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Face {i}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

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
    if total_queried > 0:
        total_faces = int(df_out['faces_detected_count'].sum())
        mean_faces = df_out['faces_detected_count'].mean()
        
        summary_data = {
            'total_frames_processed': total_queried,
            'total_faces_detected': total_faces,
            'mean_faces_per_frame': round(mean_faces, 2)
        }
        pd.DataFrame([summary_data]).to_csv(os.path.join(out_dir, f"{base_name}_summary.csv"), index=False)

    print(f"[{base_name}] Completed and saved.")
    return base_name

def get_args():
    parser = argparse.ArgumentParser(description="Multiprocessing Face Detection Pipeline (Brute Force Mode)")
    parser.add_argument('manifest', nargs='?', type=str, default=r'tracker_manifest.csv', help="Path to manifest.csv")
    parser.add_argument('-t', action='store_true', help="Test mode")
    parser.add_argument('-d', action='store_true', help="Output drawn bounding box video")
    parser.add_argument('--dir', type=str, required=True, help="Directory containing source videos")
    parser.add_argument('--out', type=str, default='./output')
    parser.add_argument('--workers', type=int, default=6, help="Worker pool count")
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