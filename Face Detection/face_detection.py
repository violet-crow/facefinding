import cv2
import pandas as pd
import os
import warnings
from insightface.app import FaceAnalysis
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.filterwarnings("ignore", category=FutureWarning)

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MANIFEST_PATH = os.path.join(BASE_DIR, 'tracker_manifest.csv')
VIDEO_DIR = r"P:\VHIL\Videos\POV"
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
NUM_WORKERS = 6

# --- Crop Settings ---
CROP_Y_START = 172
CROP_Y_END = 1516
# ---------------------

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

    app = FaceAnalysis(name='buffalo_l', allowed_modules=['detection'], providers=['CUDAExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    cap = cv2.VideoCapture(filepath)
    
    start_msec = start_sec * 1000.0
    end_msec = end_sec * 1000.0
    target_msec = start_msec

    csv_data = []
    frames_processed_count = 0
    
    print(f"[{base_name}] Started. Fast-forwarding to {format_time_ms(start_sec)}...")

    while cap.isOpened():
        ret = cap.grab()
        if not ret: 
            break

        current_msec = cap.get(cv2.CAP_PROP_POS_MSEC)

        if current_msec > end_msec:
            break

        if current_msec >= target_msec:
            ret, frame = cap.retrieve()
            if not ret: 
                break
            
            frame_cropped = frame[CROP_Y_START:CROP_Y_END, :]
            expected_time_str = format_time_ms(target_msec / 1000.0)
            faces = app.get(frame_cropped)
            current_frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            
            row_data = {
                'frame': current_frame_num,
                'time': expected_time_str,
                'actual_video_time': format_time_ms(current_msec / 1000.0),
                'faces_detected_count': len(faces)
            }

            for i, face in enumerate(faces):
                x1, y1, x2, y2 = map(int, face.bbox)
                row_data.update({
                    f'face_{i}_tl_x': x1,
                    f'face_{i}_tl_y': y1 + CROP_Y_START,
                    f'face_{i}_br_x': x2,
                    f'face_{i}_br_y': y2 + CROP_Y_START
                })

            csv_data.append(row_data)
            frames_processed_count += 1
            
            if frames_processed_count % 10 == 0:
                print(f"[{base_name}] Processing frame {current_frame_num} | Time: {expected_time_str}")

            target_msec += 500.0

    cap.release()

    df_out = pd.DataFrame(csv_data)
    df_out.to_csv(os.path.join(out_dir, f"{base_name}_tracking.csv"), index=False)

    total_queried = len(df_out)
    if total_queried > 0:
        summary_data = {
            'total_frames_processed': total_queried,
            'total_faces_detected': int(df_out['faces_detected_count'].sum()),
            'mean_faces_per_frame': round(df_out['faces_detected_count'].mean(), 2)
        }
        pd.DataFrame([summary_data]).to_csv(os.path.join(out_dir, f"{base_name}_summary.csv"), index=False)

    print(f"[{base_name}] Completed and saved.")
    return base_name

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(MANIFEST_PATH):
        print(f"Error: Manifest not found at {MANIFEST_PATH}")
        return

    df_manifest = pd.read_csv(MANIFEST_PATH, dtype={'participant_num': str, 'group_num': str})
    
    video_tasks = []
    grouped = df_manifest.groupby('group_num')
    
    for group_num, group_data in grouped:
        group_out_dir = os.path.join(OUTPUT_DIR, f"Group_{group_num}")
        os.makedirs(group_out_dir, exist_ok=True)
        
        for _, row in group_data.iterrows():
            pid = row['participant_num']
            filename = f"{pid}_POV.mp4"
            filepath = os.path.join(VIDEO_DIR, filename)
            
            if not os.path.exists(filepath):
                print(f"File not found: {filepath}")
                continue
                
            start_sec = time_to_sec(row['start_time'])
            end_sec = time_to_sec(row['end_time'])
            if end_sec == 0: end_sec = float('inf')

            video_tasks.append({
                'filepath': filepath,
                'base_name': f"{pid}_POV",
                'start_sec': start_sec,
                'end_sec': end_sec,
                'out_dir': group_out_dir
            })

    if not video_tasks:
        print("No valid tasks found.")
        return

    print(f"Queue built: {len(video_tasks)} videos ready.")
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(process_single_video, task) for task in video_tasks]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                print(f"Exception: {exc}")

if __name__ == "__main__":
    main()