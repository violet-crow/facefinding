import pandas as pd
from pathlib import Path

# --- CONFIGURATION ---
# Replace with the path to your main directory containing the subdirectories
TARGET_DIR = r'P:\VHIL\Videos\facefinding\output' 
# The path where the summary CSV will be saved (this will overwrite any existing file)
SUMMARY_CSV = 'summarize_faces_in_each_frame.csv' 
# ---------------------

def create_summary():
    results = []
    
    # 1) & 2) Recursively explore all subdirectories and find files ending in _tracking.csv
    # The rglob method handles the deep searching automatically.
    for filepath in Path(TARGET_DIR).rglob('*_tracking.csv'):
        filename = filepath.name
        
        # Extract the participant number (everything before the first underscore)
        participant_num = filename.split('_')[0]
        
        try:
            df = pd.read_csv(filepath)
            
            # 4b) Total frames queried (equal to the number of non-header rows)
            total_frames = len(df)
            
            # 4a) Identify the 3rd column (index 2, since it is 0-indexed)
            faces_col = df.columns[2]
            
            # Fill missing values with a dummy number, convert to integer, then to string. 
            # This ensures pandas doesn't accidentally count '1.0' and '1' as two separate things.
            counts = df[faces_col].fillna(-1).astype(int).astype(str).value_counts()
            
            # Extract counts for 0, 1, 2, 3, 4 (defaulting to 0 if the number didn't appear in the file)
            results.append({
                'participant_num': participant_num,
                'count_0': counts.get('0', 0),
                'count_1': counts.get('1', 0),
                'count_2': counts.get('2', 0),
                'count_3': counts.get('3', 0),
                'count_4': counts.get('4', 0),
                'total_frames': total_frames
            })
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    # 3) Create the summary CSV (overwrites by default)
    if results:
        summary_df = pd.DataFrame(results)
        
        # Sort by participant number before saving so the file is organized sequentially
        summary_df = summary_df.sort_values('participant_num')
        summary_df.to_csv(SUMMARY_CSV, index=False)
        
        print(f"Successfully processed {len(results)} files. Summary saved to {SUMMARY_CSV}")
    else:
        print(f"No *_tracking.csv files were found in {TARGET_DIR}.")

if __name__ == '__main__':
    create_summary()