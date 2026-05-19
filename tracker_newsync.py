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

    # Restricted to 'detection' only to save VRAM and processing time
    app = FaceAnalysis(name='buffalo_l', allowed_modules=['detection'], providers=['CUDAExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    cap = cv2.VideoCapture(filepath)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    start_msec = start_sec * 1000.0
    end_msec = end_sec * 1000.0
    target_msec = start_msec

    cap.set(cv2.CAP_PROP_POS_MSEC, start_msec)

    writer = None
    if draw_video:
        out_vid_path = os.path.join(out_dir, f"{base_name}_face_detection.mp4")
        writer = cv2.VideoWriter(out_vid_path, cv2.VideoWriter_fourcc(*'mp4v'), 2.0, (width, height))

    csv_data = []
    frames_processed_count = 0
    print(f"[{base_name}] Started processing.")

    while cap.isOpened():
        current_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
        
        if current_msec > end_msec:
            break

        if current_msec >= target_msec:
            ret, frame = cap.read()
            if not ret: 
                break
                
            frames_processed_count += 1
            current_time_str = format_time_ms(current_msec / 1000.0)
            
            if frames_processed_count % 50 == 0:
                print(f"[{base_name}] Processed frame at {current_time_str}")

            faces = app.get(frame)
            
            row_data = {
                'frame': int(cap.get(cv2.CAP_PROP_POS_FRAMES)),
                'time': current_time_str,
                'faces_detected_count': len(faces)
            }

            # Dynamically append coordinate columns for each detected face
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
            
            target_msec += 500.0
            
        else:
            ret = cap.grab()
            if not ret: 
                break

    cap.release()
    if writer: writer.release()

    # Tracking output
    df_out = pd.DataFrame(csv_data)
    df_out.to_csv(os.path.join(out_dir, f"{base_name}_tracking.csv"), index=False)

    # Summary output
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
    parser = argparse.ArgumentParser(description="Multiprocessing Face Detection Pipeline")
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