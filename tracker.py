import cv2
import numpy as np
import pandas as pd
import argparse
import os
import csv
import onnxruntime as ort
from insightface.app import FaceAnalysis

def time_to_sec(time_str):
    try:
        m, s = map(int, time_str.split(':'))
        return m * 60 + s
    except ValueError:
        print("Invalid format. Please use mm:ss.")
        return None

def get_args():
    parser = argparse.ArgumentParser(description="Single-Pass CUDA Face Tracker")
    parser.add_argument('-d', action='store_true', help="Output video with bounding boxes")
    parser.add_argument('-t', action='store_true', help="Enable Test Mode (Overrides start/end times globally)")
    parser.add_argument('-r', '--rate', type=int, default=30, help="Sample rate (process 1 in every R frames)")
    parser.add_argument('--dir', type=str, default=r'P:\VHIL\Videos\POV', help="Target directory")
    parser.add_argument('--out', type=str, default='./output', help="Output directory")
    parser.add_argument('--known', type=str, default='./known_faces', help="Directory containing reference face folders")
    return parser.parse_args()

def main():
    print("Available ORT Providers:", ort.get_available_providers())
    
    args = get_args()

    group_num = input("Enter the Group Number (for output folder naming): ")
    
    while True:
        try:
            group_size = int(input("Enter group size (2-4): "))
            start_num = int(input("Enter first group member number (e.g., 1 or 100): "))
            break
        except ValueError:
            print("Please enter numeric values.")

    participants = []
    target_ids = []
    
    for i in range(group_size):
        member_id = start_num + i
        formatted_id = f"{member_id:03d}"
        filename = f"{formatted_id}_POV.mp4"
        filepath = os.path.join(args.dir, filename)

        if not os.path.exists(filepath):
            print(f"!!! Error: Could not find {filepath}. Please verify files exist before tracking.")
            return

        sec_start = None
        while sec_start is None:
            ts_start = input(f"Enter 'ready' (start) timestamp for {filename} (mm:ss): ")
            sec_start = time_to_sec(ts_start)

        sec_end = None
        while sec_end is None:
            ts_end = input(f"Enter end timestamp for {filename} (mm:ss): ")
            sec_end = time_to_sec(ts_end)

        participants.append({
            'id': formatted_id, 
            'path': filepath, 
            'start_sec': sec_start,
            'end_sec': sec_end
        })
        target_ids.append(formatted_id)

    # Handle Test Mode (-t flag)
    test_start_sec, test_end_sec = None, None
    if args.t:
        print("\n--- TEST MODE ACTIVE (-t) ---")
        print("These global timestamps will override the individual file times for this run.")
        while test_start_sec is None:
            test_start_sec = time_to_sec(input("Enter global TEST start time (mm:ss): "))
        while test_end_sec is None:
            test_end_sec = time_to_sec(input("Enter global TEST end time (mm:ss): "))

    group_out_dir = os.path.join(args.out, f"Group_{group_num}")
    os.makedirs(group_out_dir, exist_ok=True)

    app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    print("\n--- Loading Reference Profiles ---")
    master_embeddings = {}
    
    for pid in target_ids:
        ref_dir = os.path.join(args.known, pid)
        if not os.path.exists(ref_dir):
            print(f"!!! Warning: No reference folder found for {pid} at {ref_dir}.")
            continue
            
        person_embeddings = []
        for file_name in os.listdir(ref_dir):
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                base_name = os.path.splitext(file_name)[0]
                npy_path = os.path.join(ref_dir, f"{base_name}.npy")
                
                if os.path.exists(npy_path):
                    person_embeddings.append(np.load(npy_path))
        
        if person_embeddings:
            master_embeddings[pid] = np.mean(person_embeddings, axis=0)
            print(f"Loaded Master Profile for {pid} (Built from {len(person_embeddings)} verified images)")
        else:
            print(f"!!! Warning: No valid data found in {ref_dir}.")

    if not master_embeddings:
        print("No master profiles loaded. Exiting.")
        return

    print("\n--- Processing Videos ---")
    for p in participants:
        file_path = p['path']
        
        # Apply Test Mode overrides if applicable
        effective_start = test_start_sec if args.t else p['start_sec']
        effective_end = test_end_sec if args.t else p['end_sec']
        
        cap = cv2.VideoCapture(file_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_area = width * height
        base_name = os.path.basename(file_path).replace('.mp4', '')

        print(f"\nTracking {base_name} from {int(effective_start // 60):02d}:{int(effective_start % 60):02d} to {int(effective_end // 60):02d}:{int(effective_end % 60):02d}...")

        writer = None
        if args.d:
            out_vid_path = os.path.join(group_out_dir, f"{base_name}_labeled.mp4")
            writer = cv2.VideoWriter(out_vid_path, cv2.VideoWriter_fourcc(*'mp4v'), fps / args.rate, (width, height))

        csv_data = []
        face_counts = []
        
        # Fast-forward video to the targeted start timestamp
        start_frame = int(effective_start * fps)
        end_frame = min(int(effective_end * fps), total_video_frames)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_idx = start_frame

        # Guard against zero-length processing windows
        total_frames_to_process = max(1, end_frame - start_frame)

        while cap.isOpened() and frame_idx <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break
                
            if (frame_idx - start_frame) % args.rate == 0:
                # Update progress bar relative to the selected window
                frames_processed = frame_idx - start_frame
                percent = (frames_processed / total_frames_to_process) * 100
                print(f"\rProcessing frame {frame_idx}/{end_frame} ({percent:.1f}%)", end="", flush=True)

                faces = app.get(frame)
                face_counts.append(len(faces))
                
                current_sec = frame_idx / fps
                mm_ss = f"{int(current_sec // 60):02d}:{int(current_sec % 60):02d}"
                row = {'frame': frame_idx, 'time': mm_ss}
                
                for label in target_ids:
                    row.update({f'{label}_in_frame': False, f'{label}_x1': '', f'{label}_y1': '', 
                                f'{label}_x2': '', f'{label}_y2': '', f'{label}_pct': ''})

                for face in faces:
                    identity = None
                    highest_sim = -1

                    for pid, master_emb in master_embeddings.items():
                        sim = np.dot(face.normed_embedding, master_emb)
                        if sim > 0.5 and sim > highest_sim: 
                            highest_sim = sim
                            identity = pid
                    
                    if identity:
                        x1, y1, x2, y2 = map(int, face.bbox)
                        face_area = (x2 - x1) * (y2 - y1)
                        pct_area = (face_area / total_area) * 100

                        row.update({
                            f'{identity}_in_frame': True,
                            f'{identity}_x1': x1, f'{identity}_y1': y1,
                            f'{identity}_x2': x2, f'{identity}_y2': y2,
                            f'{identity}_pct': round(pct_area, 2)
                        })

                        if args.d:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, identity, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                csv_data.append(row)
                if args.d:
                    writer.write(frame)

            frame_idx += 1

        print() # Break the carriage return line so the next print starts fresh
        cap.release()
        if writer: writer.release()

        df = pd.DataFrame(csv_data)
        df.to_csv(os.path.join(group_out_dir, f"{base_name}_tracking.csv"), index=False)

        avg_faces = sum(face_counts) / len(face_counts) if face_counts else 0
        with open(os.path.join(group_out_dir, f"{base_name}_summary.csv"), 'w', newline='') as f:
            csv.writer(f).writerow(['Average_Faces_Per_Sampled_Frame', avg_faces])

    print(f"\n--- Group {group_num} Processing Complete ---")

if __name__ == "__main__":
    main()