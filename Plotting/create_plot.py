import argparse
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def generate_plot(csv_filename, use_numeric):
    if not os.path.isfile(csv_filename):
        print(f"Error: File '{csv_filename}' not found.")
        return

    df = pd.read_csv(csv_filename)
    
    if 'time' not in df.columns:
        print(f"Error: 'time' column not found in {csv_filename}.")
        return

    plt.figure(figsize=(30, 12))
    
    face_cols = [col for col in df.columns if col.startswith('faces_')]
    
    max_faces_detected = int(df[face_cols].max().max()) if not df[face_cols].empty else 1
    max_faces_detected = max(1, max_faces_detected)
    
    for i, col in enumerate(face_cols):
        if not use_numeric:
            df.loc[df[col] > 0, col] = 1
            
        participant_id = col.split('_')[1]
        offset = i * 0.03
        plt.plot(df['time'], df[col] + offset, label=participant_id, alpha=0.8, linewidth=1.5)

    avg_col = 'average_numeric' if use_numeric else 'average_binary'
    if avg_col in df.columns:
        plt.plot(df['time'], df[avg_col], label='Average', color='black', linewidth=2, linestyle='--')
    elif 'average_faces' in df.columns:
        plt.plot(df['time'], df['average_faces'], label='Average (Legacy)', color='black', linewidth=2, linestyle='--')

    # Draw vertical lines and shade background for task phase
    if 'task_start_sec' in df.columns and pd.notna(df['task_start_sec'].iloc[0]):
        t_start = df['task_start_sec'].iloc[0]
        t_end = df['task_end_sec'].iloc[0]
        
        # Shade the region between t_start and t_end
        plt.axvspan(t_start, t_end, color='lightgrey', alpha=0.3, label='Task Phase Region')
        
        # Draw the red boundary lines
        plt.axvline(x=t_start, color='red', linestyle='--', linewidth=2, label='Task Phase Start/End', alpha=0.6)
        plt.axvline(x=t_end, color='red', linestyle='--', linewidth=2, alpha=0.6)

    max_offset = len(face_cols) * 0.03

    # Y-axis scaling
    if use_numeric:
        plt.ylim(-0.1, max_faces_detected + max_offset + 0.1)
        plt.yticks(list(range(0, max_faces_detected + 1)))
        plt.ylabel('Faces Detected (Count)')
    else:
        plt.ylim(-0.1, 1.2 + max_offset)
        plt.yticks([0, 1])
        plt.ylabel('Faces Detected (1 = Yes, 0 = No)')
        
    # X-axis scaling and formatting
    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(30))
    
    def time_formatter(x, pos):
        sign = "-" if x < 0 else ""
        x = abs(int(x))
        m, s = divmod(x, 60)
        return f"{sign}{m}:{s:02d}"
        
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(time_formatter))
    plt.xlabel('Time (mm:ss relative to sync)')
    
    # Positioning the legend and text
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=7, title="ID")
    
    if 'condition' in df.columns and pd.notna(df['condition'].iloc[0]):
        condition_text = str(df['condition'].iloc[0])
        plt.text(0.5, -0.22, f"Condition: {condition_text}", transform=ax.transAxes, 
                 ha='center', va='top', fontsize=24, fontweight='bold')
    
    suffix = "_number.png" if use_numeric else "_binary.png"
    output_filename = csv_filename.replace('.csv', suffix)
    
    plt.savefig(output_filename, bbox_inches='tight')
    plt.close()
    
    print(f"Generated: {output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate plots for face tracking data.")
    parser.add_argument("target", help="CSV filename or 'all' to process all files")
    parser.add_argument("-n", "--numeric", action="store_true", help="Plot raw face count instead of binary")
    
    args = parser.parse_args()

    if args.target.lower() == 'all':
        files = glob.glob("*_facesplot.csv")
        if not files:
            print("No '_facesplot.csv' files found in the current directory.")
        else:
            for file in files:
                generate_plot(file, args.numeric)
    else:
        generate_plot(args.target, args.numeric)