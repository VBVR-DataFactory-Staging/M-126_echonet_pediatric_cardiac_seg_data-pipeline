"""M-126: EchoNet-Pediatric LV segmentation on pediatric echo video.

Layout:
    _extracted/M-126_EchoNetPediatric/echonetpediatric/pediatric_echo_avi/pediatric_echo_avi/
        {A4C,PSAX}/Videos/<case>.avi
        {A4C,PSAX}/FileList.csv
        {A4C,PSAX}/VolumeTracings.csv   (frame,x1,y1,x2,y2)

Case A (real video): extract the ES/ED frames with LV polygon traces, overlay trace.
Output: short video of the LV trace walkthrough.
"""
from __future__ import annotations
from pathlib import Path
import cv2, numpy as np, csv
from collections import defaultdict
from common import DATA_ROOT, write_task, COLORS, fit_square, overlay_mask

PID = "M-126"; TASK_NAME = "echonet_pediatric_cardiac_seg"; FPS = 10

PROMPT = ("This is a pediatric echocardiogram (A4C or PSAX view). "
          "Segment the left ventricle endocardial border (red contour) "
          "and highlight the walk-through from end-diastole to end-systole.")

def load_tracings(csv_path: Path):
    """Return {video_name: {frame_idx: [(x1,y1,x2,y2), ...]}}."""
    traces = defaultdict(lambda: defaultdict(list))
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = row.get("FileName") or row.get("VideoName") or ""
            frame = int(float(row.get("Frame", 0)))
            x1, y1 = float(row.get("X1", 0)), float(row.get("Y1", 0))
            x2, y2 = float(row.get("X2", 0)), float(row.get("Y2", 0))
            traces[vid][frame].append((x1, y1, x2, y2))
    return traces

def trace_to_mask(lines, shape):
    """Convert list of (x1,y1,x2,y2) line segments to a binary LV mask (fill polygon)."""
    mask = np.zeros(shape[:2], dtype=np.uint8)
    if not lines: return mask
    # Use line endpoints as polygon vertices (start points)
    pts = np.array([[l[0], l[1]] for l in lines] + [[l[2], l[3]] for l in reversed(lines)], dtype=np.int32)
    cv2.fillPoly(mask, [pts], 1)
    return mask

def process_video(vid_path: Path, traces_for_vid: dict, idx: int):
    cap = cv2.VideoCapture(str(vid_path))
    if not cap.isOpened(): return None
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_to_use = sorted(traces_for_vid.keys())
    if not frames_to_use: return None
    first_frames, last_frames, gt_frames, flags = [], [], [], []
    for target_f in frames_to_use:
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_f)
        ok, img = cap.read()
        if not ok: continue
        mask = trace_to_mask(traces_for_vid[target_f], img.shape)
        img_r = fit_square(img, 512)
        mask_r = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
        annot = overlay_mask(img_r, mask_r, color=COLORS["red"], alpha=0.45)
        first_frames.append(img_r)
        last_frames.append(annot)
        has = bool(mask_r.sum() > 0)
        flags.append(has)
        if has: gt_frames.append(annot)
    cap.release()
    if not gt_frames: gt_frames = last_frames[:2]
    if not first_frames: return None
    pick = 0
    meta = {"task": "EchoNet-Pediatric LV segmentation", "dataset": "EchoNet-Pediatric",
            "case_id": vid_path.stem, "modality": "pediatric echocardiogram video",
            "view": vid_path.parent.parent.name,  # A4C or PSAX
            "classes": ["LV_endocardium"], "colors": {"LV_endocardium": "red"},
            "fps": FPS, "frames_per_video": len(first_frames),
            "case_type": "A_real_video_trace",
            "num_traced_frames": len(traces_for_vid)}
    return write_task(PID, TASK_NAME, idx, first_frames[pick], last_frames[pick],
                      first_frames, last_frames, gt_frames, PROMPT, meta, FPS)

def main():
    root = DATA_ROOT / "_extracted" / "M-126_EchoNetPediatric" / "echonetpediatric" / "pediatric_echo_avi" / "pediatric_echo_avi"
    all_pairs = []
    for view in ["A4C", "PSAX"]:
        vd = root / view
        if not vd.exists(): continue
        traces_csv = vd / "VolumeTracings.csv"
        if not traces_csv.exists(): continue
        traces = load_tracings(traces_csv)
        for vid in sorted((vd / "Videos").glob("*.avi")):
            if vid.name in traces:
                all_pairs.append((vid, traces[vid.name]))
    print(f"  {len(all_pairs)} EchoNet-Pediatric traced videos")
    for i, (vid, tr) in enumerate(all_pairs):
        d = process_video(vid, tr, i)
        if d: print(f"  wrote {d}")

if __name__ == "__main__":
    main()
