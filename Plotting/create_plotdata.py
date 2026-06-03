import os
import re
import pandas as pd

base_dir = r"P:\VHIL\Videos\facefinding\Self View Removal\process_nosmalls"
manifest_path = "plot_manifest.csv"

def time_to_seconds(t_str):
    if pd.isna(t_str):
        return 0.0
    if isinstance(t_str, (int, float)):
        return float(t_str)
    
    t_str = str(t_str).strip()
    parts = t_str.split(':')
    
    if len(parts) == 1:
        return float(parts[0])
    elif len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    elif len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    return 0.0

manifest = pd.read_csv(manifest_path)

for group, group_df in manifest.groupby('group'):
    series_list = []
    
    for _, row in group_df.iterrows():
        filename = row['filename']
        sync_time_str = row['sync_time']
        
        file_path = os.path.join(base_dir, filename)
        
        if not os.path.isfile(file_path):
            print(f"Missing file: {filename}")
            continue
            
        match = re.search(r'(\d{3})_POV_tracking\.csv', filename)
        if not match:
            print(f"Skipping {filename}: Does not match XXX_POV_tracking.csv format.")
            continue
            
        file_id = match.group(1)
        
        df = pd.read_csv(file_path)
        if 'time' not in df.columns or 'faces_detected_count' not in df.columns:
            print(f"Missing 'time' or 'faces_detected_count' in {filename}")
            continue
            
        sync_sec = time_to_seconds(sync_time_str)
        df['time_sec'] = df['time'].apply(time_to_seconds)
        
        df['relative_time'] = (df['time_sec'] - sync_sec).round(2)
        
        series = df.set_index('relative_time')['faces_detected_count'].rename(f"faces_{file_id}")
        series = series[~series.index.duplicated(keep='first')]
        
        series_list.append(series)
        
    if series_list:
        merged_df = pd.concat(series_list, axis=1)
        merged_df.sort_index(inplace=True)
        
        merged_df.insert(0, 'time', merged_df.index)
        
        output_path = f"{group}_facesplot.csv"
        merged_df.to_csv(output_path, index=False)