import pandas as pd
from pathlib import Path

# --- CONFIGURATION ---
TARGET_DIR = r'P:\VHIL\Videos\facefinding\Self View Removal\Small Faces Removed' 
SUMMARY_CSV = 'summarize_faces_in_each_frame_nosmalls.csv' 
# ---------------------

def create_summary():
    results = []
    
    for filepath in Path(TARGET_DIR).rglob('*_tracking.csv'):
        filename = filepath.name
        participant_num = filename.split('_')[0]
        
        try:
            df = pd.read_csv(filepath)
            
            # Clean column names in case of trailing whitespaces
            df.columns = df.columns.str.strip()
            
            total_frames = len(df)
            
            # Directly target the named column
            faces_col = 'faces_detected_count'
            
            counts = df[faces_col].fillna(-1).astype(int).astype(str).value_counts()
            
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

    if results:
        summary_df = pd.DataFrame(results)
        
        # Convert to numeric to ensure correct sequential sorting (1, 2, 10 instead of 1, 10, 2)
        try:
            summary_df['participant_num'] = pd.to_numeric(summary_df['participant_num'])
        except ValueError:
            pass
            
        summary_df = summary_df.sort_values('participant_num')
        
        summary_df.to_csv(SUMMARY_CSV, index=False)
        print(f"Successfully processed {len(results)} files. Summary saved to {SUMMARY_CSV}")
    else:
        print(f"No *_tracking.csv files were found in {TARGET_DIR}.")

if __name__ == '__main__':
    create_summary()