import csv
import os
import re
import subprocess
import argparse
from pathlib import Path

INPUT_DIR = r"P:\VHIL\Videos\facefinding\Face Detection\output"
VIDEO_DIR = r"P:\VHIL\Videos\POV"
OUTPUT_DIR = r"P:\VHIL\Videos\facefinding\Self View Removal\Small Faces Removed"
SPOT_CHECK_DIR = r"P:\VHIL\Videos\facefinding\Self View Removal\Detected Small Faces"
LOG_FILE = "detected_small_faces.csv"

def parse_time_to_seconds(time_str):
    parts = str(time_str).split(':')
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    elif len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    else:
        return float(parts[0])

def process_tracking_data():
    print("Step 1: Processing tracking data...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    with open(LOG_FILE, 'w', newline='') as log_f:
        log_writer = csv.writer(log_f)
        log_writer.writerow(['video_name', 'frame', 'time', 'tl_x', 'tl_y', 'br_x', 'br_y'])
        
        for filepath in Path(INPUT_DIR).rglob('*_POV_tracking.csv'):
            filename = filepath.name
            video_id = filename.split('_')[0]
            video_name = f"{video_id}_POV.mp4"
            
            out_filename = filename
            out_filepath = Path(OUTPUT_DIR) / out_filename
            
            with open(filepath, 'r', newline='') as in_f, open(out_filepath, 'w', newline='') as out_f:
                reader = csv.DictReader(in_f)
                fieldnames = reader.fieldnames
                
                available_faces = set()
                if fieldnames:
                    for field in fieldnames:
                        match = re.match(r"face_(\d+)_", field)
                        if match:
                            available_faces.add(int(match.group(1)))
                
                available_faces = sorted(list(available_faces))
                writer = csv.DictWriter(out_f, fieldnames=fieldnames)
                writer.writeheader()
                
                for row in reader:
                    try:
                        face_count = int(row.get('faces_detected_count', 0))
                    except ValueError:
                        face_count = 0
                        
                    if face_count > 1:
                        valid_faces = []
                        for i in available_faces:
                            prefix = f"face_{i}_"
                            try:
                                tl_x_str = row.get(f"{prefix}tl_x", "")
                                if tl_x_str == '': 
                                    continue
                                tl_x = float(tl_x_str)
                                tl_y = float(row[f"{prefix}tl_y"])
                                br_x = float(row[f"{prefix}br_x"])
                                br_y = float(row[f"{prefix}br_y"])
                                area = abs((br_x - tl_x) * (br_y - tl_y))
                                valid_faces.append((i, area, tl_x, tl_y, br_x, br_y))
                            except (ValueError, TypeError, KeyError):
                                continue 
                        
                        if len(valid_faces) > 1:
                            small_face_candidates = []
                            for face in valid_faces:
                                face_id, face_area = face[0], face[1]
                                # Updated thresholds based on prompt
                                if face_area < 8500:
                                    other_areas = [f[1] for f in valid_faces if f[0] != face_id]
                                    avg_other_area = sum(other_areas) / len(other_areas)
                                    if face_area <= (0.34 * avg_other_area):
                                        small_face_candidates.append(face)
                            
                            if len(small_face_candidates) == 1:
                                face_to_remove = small_face_candidates[0]
                                f_id, f_area, tl_x, tl_y, br_x, br_y = face_to_remove
                                
                                log_writer.writerow([video_name, row['frame'], row['time'], tl_x, tl_y, br_x, br_y])
                                
                                row['faces_detected_count'] = str(face_count - 1)
                                valid_faces = [f for f in valid_faces if f[0] != f_id]
                                
                                for idx, i in enumerate(available_faces):
                                    prefix = f"face_{i}_"
                                    if idx < len(valid_faces):
                                        f_data = valid_faces[idx]
                                        row[f"{prefix}tl_x"] = f_data[2]
                                        row[f"{prefix}tl_y"] = f_data[3]
                                        row[f"{prefix}br_x"] = f_data[4]
                                        row[f"{prefix}br_y"] = f_data[5]
                                    else:
                                        row[f"{prefix}tl_x"] = ''
                                        row[f"{prefix}tl_y"] = ''
                                        row[f"{prefix}br_x"] = ''
                                        row[f"{prefix}br_y"] = ''
                    
                    writer.writerow(row)
    print("Data processing complete.")

def generate_spot_checks():
    print("\nStep 2: Generating spot check frames with FFmpeg...")
    os.makedirs(SPOT_CHECK_DIR, exist_ok=True)
    
    if not os.path.exists(LOG_FILE):
        print(f"[ERROR] Log file {LOG_FILE} not found. Run Step 1 first.")
        return

    with open(LOG_FILE, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_name = row['video_name']
            subject_id = video_name.split('_')[0]
            video_path = Path(VIDEO_DIR) / video_name
            
            if not video_path.exists():
                print(f"[ERROR] Missing video: {video_path}")
                continue
                
            frame_idx = row['frame']
            timestamp = parse_time_to_seconds(row['time'])
            
            clean_time = row['time'].replace(':', '_').replace('.', '_')
            final_out_path = Path(SPOT_CHECK_DIR) / f"{subject_id}_{clean_time}_spot.jpg"
            
            if final_out_path.exists():
                print(f" -> Skipping frame {frame_idx} from {video_name} (already exists).")
                continue
            
            tl_x = int(float(row['tl_x']))
            tl_y = int(float(row['tl_y']))
            br_x = int(float(row['br_x']))
            br_y = int(float(row['br_y']))
            
            box_width = br_x - tl_x
            box_height = br_y - tl_y
            
            drawbox_filter = f"drawbox=x={tl_x}:y={tl_y}:w={box_width}:h={box_height}:color=green:t=3"
            
            ffmpeg_cmd = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-ss", str(timestamp), 
                "-i", str(video_path), 
                "-vf", drawbox_filter,
                "-frames:v", "1", 
                "-q:v", "2", 
                str(final_out_path)
            ]
            
            print(f" -> Extracting frame {frame_idx} from {video_name}...")
            try:
                subprocess.run(ffmpeg_cmd, check=True, timeout=15)
            except subprocess.TimeoutExpired:
                print(f"    [WARNING] FFmpeg timed out on frame {frame_idx}.")
                continue
            except subprocess.CalledProcessError as e:
                print(f"    [ERROR] FFmpeg failed for frame {frame_idx}: {e}")
                continue

    print("Spot check generation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', action='store_true', help="Skip spot check frame generation")
    args = parser.parse_args()

    process_tracking_data()
    
    if not args.s:
        generate_spot_checks()
    else:
        print("\nStep 2: Skipped spot check generation due to -s flag.")