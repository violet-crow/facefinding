import os, sys
# Add CUDA DLL dirs before imports
if os.name == "nt":
    def _add_cuda_bins(prefix):
        sp = os.path.join(prefix, "Lib", "site-packages")
        for rel in [
            r"nvidia\cuda_runtime\bin",
            r"nvidia\cublas\bin",
            r"nvidia\cudnn\bin",
            r"nvidia\cufft\bin",
            r"nvidia\curand\bin",
            r"nvidia\cusolver\bin",
            r"nvidia\cusparse\bin",
        ]:
            p = os.path.join(sp, rel)
            if os.path.isdir(p):
                os.add_dll_directory(p)
    _add_cuda_bins(sys.base_prefix)
    if sys.prefix != sys.base_prefix:
        _add_cuda_bins(sys.prefix)

import PySimpleGUI as sg
import json, numpy as np
from multi_face_label import init_face_app, extract_embeddings, cluster_identities, annotate_and_export, _has_cuda_provider

def run_pipeline(window, paths, out_dir, use_gpu, sample_every, min_det_score, dbscan_eps, dbscan_min_samples, assign_sim_threshold):
    os.makedirs(out_dir, exist_ok=True)
    window['-STATUS-'].update('Initializing…')
    window['-PB-'].update(current_count=0)

    app, used_gpu = init_face_app(gpu=use_gpu)
    window['-STATUS-'].update(f"Backend: {'GPU (CUDA)' if used_gpu else 'CPU'}")

    window['-STATUS-'].update('Clustering identities (pass 1)…')
    window['-PB-'].update(current_count=10)
    samples, meta = extract_embeddings(app, paths, sample_every=sample_every, min_det_score=min_det_score)
    _, centers = cluster_identities(samples, eps=dbscan_eps, min_samples=dbscan_min_samples)

    id_bank = {
        'centers': [{'person_id': c['person_id'], 'center': c['center'].tolist(), 'count': c['count']} for c in centers],
        'dbscan': {'eps': dbscan_eps, 'min_samples': dbscan_min_samples},
        'assign_sim_threshold': assign_sim_threshold
    }
    with open(os.path.join(out_dir, 'identity_bank.json'), 'w') as f:
        json.dump(id_bank, f, indent=2)

    centers_np = [{'person_id': c['person_id'], 'center': np.asarray(c['center'], dtype=np.float32)} for c in id_bank['centers']]

    n = max(1, len(paths))
    window['-PB-'].update(current_count=20)

    def on_progress(vid_idx, frame_idx, total_frames):
        base = 20 + (vid_idx * (80.0 / n))
        frac = (frame_idx / max(1, total_frames)) * (80.0 / n)
        pct = min(100, int(base + frac))
        window['-PB-'].update(current_count=pct)
        window['-STATUS-'].update(f"Annotating {os.path.basename(paths[vid_idx])} ({frame_idx}/{total_frames})")

    window['-STATUS-'].update('Annotating videos (pass 2)…')
    annotate_and_export(app, paths, centers_np, out_dir, meta, sim_threshold=assign_sim_threshold, on_progress=on_progress)
    window['-PB-'].update(current_count=100)
    window['-STATUS-'].update('Done.')

    outputs = []
    for p in paths:
        base = os.path.splitext(os.path.basename(p))[0]
        outputs.append(os.path.join(out_dir, f"{base}_labeled.mp4"))
        outputs.append(os.path.join(out_dir, f"{base}_face_log.csv"))
    window['-OUTLIST-'].update(outputs)

layout = [
    [sg.Text("Select 1–4 MP4 files"), sg.Input(key="-FILES-"), sg.FilesBrowse(file_types=(("MP4", "*.mp4"),))],
    [sg.Text("Output folder"), sg.Input(key="-OUT-"), sg.FolderBrowse()],
    [sg.Checkbox("Use GPU (CUDA if available)", default=True, key="-GPU-")],
    [sg.Text("sample_every"), sg.Spin(values=list(range(1,21)), initial_value=3, key="-SAMPLE-"),
     sg.Text("min_det_score"), sg.Spin(values=[x/100 for x in range(0,101,5)], initial_value=0.4, key="-DETS-")],
    [sg.Text("dbscan_eps"), sg.Spin(values=[x/100 for x in range(10,51)], initial_value=0.35, key="-EPS-"),
     sg.Text("dbscan_min_samples"), sg.Spin(values=list(range(3,21)), initial_value=5, key="-MIN-")],
    [sg.Text("assign_sim_threshold"), sg.Spin(values=[x/100 for x in range(20,81,5)], initial_value=0.35, key="-TH-")],
    [sg.Button("Run"), sg.Button("Exit")],
    [sg.ProgressBar(100, orientation='h', size=(40, 20), key='-PB-')],
    [sg.Text("", key='-STATUS-')],
    [sg.Listbox(values=[], size=(80, 6), key="-OUTLIST-")]
]

window = sg.Window("Face Labeler", layout)

while True:
    event, values = window.read()
    if event in (sg.WIN_CLOSED, "Exit"):
        break
    if event == "Run":
        files = values["-FILES-"].split(";") if values["-FILES-"] else []
        files = [f for f in files if f.strip()]
        if not files or len(files) > 4:
            sg.popup("Select 1–4 MP4 files.")
            continue
        out_dir = values["-OUT-"] or "outputs"
        try:
            run_pipeline(
                window, files, out_dir,
                use_gpu=values["-GPU-"],
                sample_every=int(values["-SAMPLE-"]),
                min_det_score=float(values["-DETS-"]),
                dbscan_eps=float(values["-EPS-"]),
                dbscan_min_samples=int(values["-MIN-"]),
                assign_sim_threshold=float(values["-TH-"])
            )
        except Exception as e:
            sg.popup_error(f"Error: {e}")

window.close()
