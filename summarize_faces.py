import csv
from pathlib import Path

def aggregate_summaries(root_folder, output_filename="all_summaries.csv"):
    root_path = Path(root_folder)
    headers = [
        "participant_num", 
        "total_frames_processed", 
        "total_faces_detected", 
        "mean_faces_per_frame"
    ]
    
    with open(output_filename, mode='w', newline='', encoding='utf-8') as out_file:
        writer = csv.writer(out_file)
        writer.writerow(headers)
        
        # rglob recursively finds all matching files in subdirectories
        for file_path in root_path.rglob('*_summary.csv'):
            participant_num = file_path.name[:3]
            
            with open(file_path, mode='r', newline='', encoding='utf-8') as in_file:
                reader = csv.reader(in_file)
                try:
                    # Skip the first row (headers A1:C1)
                    next(reader)
                    # Extract the second row (data A2:C2)
                    data_row = next(reader)
                    
                    if len(data_row) >= 3:
                        writer.writerow([
                            participant_num, 
                            data_row[0], # A2
                            data_row[1], # B2
                            data_row[2]  # C2
                        ])
                    else:
                        print(f"Warning: {file_path.name} does not have at least 3 columns in row 2.")
                except StopIteration:
                    print(f"Warning: {file_path.name} is empty or missing row 2 data.")

if __name__ == "__main__":
    # Replace '.' with the path to your target directory
    target_directory = r"P:\VHIL\Videos\facefinding\output" 
    aggregate_summaries(target_directory)
    print(f"Aggregation complete. Output saved to all_summaries.csv")