import os
import re
import pandas as pd

base_dir = r"P:\VHIL\Videos\facefinding\Self View Removal\Small Faces Removed"
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
    
    # Pre-scan for group-level metadata
    task_start_sec = None
    condition_str = ""
    
    for _, row in group_df.iterrows():
        if 'condition' in row and pd.notna(row['condition']):
            condition_str = str(row['condition'])
        if 'task_time' in row and pd.notna(row['task_time']):
            task_start_sec = time_to_seconds(row['task_time']) - time_to_seconds(row['sync_time'])

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
        
        # Calculate Numeric Average
        raw_mean_num = merged_df.mean(axis=1)
        raw_mean_num.index = pd.to_timedelta(raw_mean_num.index, unit='s')
        smoothed_num = raw_mean_num.rolling('5s', min_periods=1).mean()
        merged_df['average_numeric'] = smoothed_num.values.round(2)
        
        # Calculate Binary Average
        binary_df = merged_df.copy()
        binary_df[binary_df > 0] = 1
        raw_mean_bin = binary_df.mean(axis=1)
        raw_mean_bin.index = pd.to_timedelta(raw_mean_bin.index, unit='s')
        smoothed_bin = raw_mean_bin.rolling('5s', min_periods=1).mean()
        merged_df['average_binary'] = smoothed_bin.values.round(2)
        
        # Append group metadata
        if task_start_sec is not None:
            merged_df['task_start_sec'] = task_start_sec
            merged_df['task_end_sec'] = task_start_sec + (13 * 60) # + 13 minutes
        if condition_str:
            merged_df['condition'] = condition_str
        
        merged_df.insert(0, 'time', merged_df.index)
        
        output_path = f"{group}_facesplot.csv"
        merged_df.to_csv(output_path, index=False)