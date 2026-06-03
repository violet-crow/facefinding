import sys
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

def generate_plot(csv_filename):
    if not os.path.isfile(csv_filename):
        print(f"Error: File '{csv_filename}' not found.")
        return

    df = pd.read_csv(csv_filename)
    
    if 'time' not in df.columns:
        print(f"Error: 'time' column not found in {csv_filename}.")
        return

    plt.figure(figsize=(10, 6))
    
    for col in df.columns:
        if col.startswith('faces_'):
            participant_id = col.split('_')[1]
            plt.plot(df['time'], df[col], label=participant_id)

    plt.ylim(0, 4)
    plt.yticks([0, 1, 2, 3, 4])
    plt.ylabel('Faces')
    plt.xlabel('Time (seconds relative to sync)')
    
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5, title="ID")
    
    output_filename = csv_filename.replace('.csv', '.png')
    plt.savefig(output_filename, bbox_inches='tight')
    plt.close()
    
    print(f"Generated: {output_filename}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python create_plot.py [filename | all]")
        sys.exit(1)

    target = sys.argv[1]

    if target.lower() == 'all':
        files = glob.glob("*_facesplot.csv")
        if not files:
            print("No '_facesplot.csv' files found in the current directory.")
            sys.exit(0)
            
        for file in files:
            generate_plot(file)
    else:
        generate_plot(target)