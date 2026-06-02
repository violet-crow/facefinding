import os
import glob
import subprocess
import cv2
import pandas as pd

CSV_DIR = "./output/"
VIDEO_DIR = r"P:\VHIL\Videos\POV"
OUTPUT_DIR = "./output/spot_checks"
FRAMES_TO_SAMPLE = 5
BROKEN_LOG = "broken_videos.txt"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def parse_time_to_seconds(time_str):
    minutes, seconds = str(time_str).split(':')
    return int(minutes) * 60 + float(seconds)

print(f"Scanning for CSV files in {CSV_DIR}...")
csv_paths = glob.glob(os.path.join(CSV_DIR, "**", "*_POV_tracking.csv"), recursive=True)
print(f"Found {len(csv_paths)} CSV files. Starting extraction...\n")

for csv_path in csv_paths:
    filename = os.path.basename(csv_path)
    subject_id = filename.split('_')[0]
    
    print(f"--- Processing Subject ID: {subject_id} ---")
    
    existing_images = glob.glob(os.path.join(OUTPUT_DIR, f"{subject_id}_frame_*.jpg"))
    num_existing = len(existing_images)
    
    if num_existing >= FRAMES_TO_SAMPLE:
        print(f"Found {num_existing} existing spot checks. Skipping.")
        print(f"Finished Subject ID: {subject_id}\n")
        continue
    elif num_existing > 0:
        print(f"Found {num_existing} spot checks (incomplete). Deleting and resampling.")
        for img in existing_images:
            try:
                os.remove(img)
            except OSError as e:
                print(f"[ERROR] Failed to delete {img}: {e}")

    print(f"CSV: {filename}")
    
    video_path = os.path.join(VIDEO_DIR, f"{subject_id}_POV.mp4")

    if not os.path.exists(video_path):
        print(f"[ERROR] Missing video for ID {subject_id}: {video_path}")
        print("Skipping to next subject...\n")
        continue

    print(f"Video found: {video_path}")
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[ERROR] Failed to read CSV {filename}: {e}")
        continue
        
    shuffled_df = df.sample(frac=1).reset_index(drop=True)
    
    print(f"Attempting to extract {FRAMES_TO_SAMPLE} valid frames...")
    
    successful_extractions = 0
    video_broken = False

    for _, row in shuffled_df.iterrows():
        if successful_extractions >= FRAMES_TO_SAMPLE:
            break

        try:
            frame_idx = int(row['frame'])
            faces_count = int(row['faces_detected_count'])
            timestamp = parse_time_to_seconds(row['time'])
        except KeyError as e:
            print(f"[ERROR] Missing expected column in CSV: {e}")
            break
        except ValueError as e:
            print(f"[ERROR] Data format issue in row: {e}")
            continue
            
        print(f" -> Trying frame {frame_idx:06d} at {timestamp:.2f}s...")
        
        raw_extract_path = os.path.join(OUTPUT_DIR, f"temp_{subject_id}_{frame_idx}.jpg")
        final_out_path = os.path.join(OUTPUT_DIR, f"{subject_id}_frame_{frame_idx:06d}.jpg")

        ffmpeg_cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-ss", str(timestamp), 
            "-i", video_path, 
            "-frames:v", "1", 
            "-q:v", "2", 
            raw_extract_path
        ]
        
        try:
            subprocess.run(ffmpeg_cmd, check=True, timeout=15)
        except subprocess.TimeoutExpired:
            print(f"    [WARNING] FFmpeg timed out on frame {frame_idx}.")
            video_broken = True
            break
        except subprocess.CalledProcessError as e:
            print(f"    [ERROR] FFmpeg failed for frame {frame_idx}: {e}. Pulling replacement frame...")
            continue

        if not os.path.exists(raw_extract_path):
            print(f"    [ERROR] FFmpeg output missing for frame {frame_idx}. Pulling replacement frame...")
            continue

        frame = cv2.imread(raw_extract_path)

        cv2.putText(frame, f"Faces detected: {faces_count}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        for i in range(4):
            tl_x_col = f"face_{i}_tl_x"
            if tl_x_col in row and pd.notna(row[tl_x_col]):
                tl = (int(row[tl_x_col]), int(row[f"face_{i}_tl_y"]))
                br = (int(row[f"face_{i}_br_x"]), int(row[f"face_{i}_br_y"]))
                cv2.rectangle(frame, tl, br, (0, 255, 0), 2)

        cv2.imwrite(final_out_path, frame)
        os.remove(raw_extract_path)
        
        successful_extractions += 1
        print(f"    [SUCCESS] Extracted {successful_extractions}/{FRAMES_TO_SAMPLE}.")

    if video_broken:
        print(f"[ERROR] Video {subject_id} timed out. Logging to {BROKEN_LOG} and skipping.")
        with open(BROKEN_LOG, "a") as f:
            f.write(f"{video_path}\n")
        continue

    if successful_extractions < FRAMES_TO_SAMPLE:
        print(f"[WARNING] Could only extract {successful_extractions}/{FRAMES_TO_SAMPLE} valid frames for {subject_id}.")

    print(f"Finished Subject ID: {subject_id}\n")

print("All spot check generation complete.")