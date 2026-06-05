import pandas as pd
import os

# Define the directory containing the source tracking CSVs
SOURCE_DIR = r"P:\VHIL\Videos\facefinding\Self View Removal\Small Faces Removed" 
MANIFEST_FILE = "plot_manifest.csv"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else "."

def parse_time_to_seconds(t):
    if pd.isna(t):
        return float('nan')
    t_str = str(t).strip()
    parts = t_str.split(':')
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    try:
        return float(t_str)
    except ValueError:
        return float('nan')

def main():
    manifest_path = os.path.join(SCRIPT_DIR, MANIFEST_FILE)
    manifest = pd.read_csv(manifest_path)

    manifest['sync_time_sec'] = manifest['sync_time'].apply(parse_time_to_seconds)
    manifest['task_time_sec'] = manifest['task_time'].apply(parse_time_to_seconds)

    group_offsets = {}
    for group, group_df in manifest.groupby('group'):
        valid_tasks = group_df.dropna(subset=['task_time_sec'])
        if valid_tasks.empty:
            continue
        
        ref_row = valid_tasks.iloc[0]
        group_offsets[group] = ref_row['task_time_sec'] - ref_row['sync_time_sec']

    manifest['calc_task_time'] = manifest.apply(
        lambda row: row['sync_time_sec'] + group_offsets[row['group']] 
        if row['group'] in group_offsets else float('nan'), 
        axis=1
    )

    valid_files = manifest.dropna(subset=['calc_task_time'])

    for _, row in valid_files.iterrows():
        filename = row['filename']
        source_path = os.path.join(SOURCE_DIR, filename)
        
        if not os.path.exists(source_path):
            continue

        df = pd.read_csv(source_path)
        
        if 'time' not in df.columns:
            continue

        df['time_sec'] = df['time'].apply(parse_time_to_seconds)

        start_limit = row['calc_task_time']
        end_limit = start_limit + (13 * 60 + 5)

        trimmed_df = df[(df['time_sec'] >= start_limit) & (df['time_sec'] <= end_limit)].copy()
        
        if 'time_sec' in trimmed_df.columns and 'time' in df.columns:
            trimmed_df = trimmed_df.drop(columns=['time_sec'])

        base_name, ext = os.path.splitext(filename)
        out_filename = f"{base_name}_trimmed{ext}"
        out_path = os.path.join(SCRIPT_DIR, out_filename)

        trimmed_df.to_csv(out_path, index=False)

if __name__ == "__main__":
    main()