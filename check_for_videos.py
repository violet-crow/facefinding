import os
import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser(description="Video File Checker")
    parser.add_argument('manifest', nargs='?', type=str, default='tracker_manifest.csv', help="Path to manifest.csv")
    parser.add_argument('--dir', type=str, required=True, help="Directory containing source videos")
    parser.add_argument('--out', type=str, default='video_audit_report.txt', help="Output text file path")
    args = parser.parse_args()

    if not os.path.exists(args.manifest):
        print(f"Error: Manifest file not found at {args.manifest}")
        return

    # Read manifest, strictly treating participant_num as a string to preserve leading zeros
    df = pd.read_csv(args.manifest, dtype={'participant_num': str})
    
    # Isolate unique participants to avoid redundant checks
    participants = df['participant_num'].unique()
    
    found_count = 0
    missing_count = 0

    print(f"Checking {len(participants)} expected video files...\n")

    with open(args.out, 'w') as f:
        f.write("--- Video File Audit Report ---\n\n")
        
        for pid in participants:
            filename = f"{pid}_POV.mp4"
            filepath = os.path.join(args.dir, filename)
            
            if os.path.exists(filepath):
                status = "FOUND"
                found_count += 1
            else:
                status = "NOT FOUND <---"
                missing_count += 1
                
            line = f"{filename}: {status}"
            print(line)
            f.write(line + "\n")
            
        summary = (
            f"\n--- Summary ---\n"
            f"Total Expected: {len(participants)}\n"
            f"Found: {found_count}\n"
            f"Missing: {missing_count}\n"
        )
        print(summary)
        f.write(summary)

    print(f"Report saved to {os.path.abspath(args.out)}")

if __name__ == "__main__":
    main()