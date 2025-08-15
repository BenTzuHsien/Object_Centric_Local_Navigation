import os, cv2, torch, numpy as np, sys, subprocess
from pathlib import Path
from PIL import Image
from backbones.grounded_sam2 import GroundedSAM2

# ── CONFIG ────────────────────────────────────────────────────────────────
traj_num    = "000"

TRAJECTORY  = Path(f"/data/shared_data/SPOT_Real_World_Dataset/map1/{traj_num}")
BASE_DIR    = Path(f"/home/mahmu059/GroundedSAM/samurai_res/map1_traj_{traj_num}")
FRAMES_DIR  = BASE_DIR / f"samurai_map1_traj_{traj_num}" / "frames"   
FRAMES_DIR.mkdir(parents=True, exist_ok=True)

PROMPT       = "green chair."
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
OUT_BBOX_TXT = BASE_DIR / "first_bbox.txt"

DEMO_PY     = Path("/home/mahmu059/GroundedSAM/samurai/scripts/demo.py")
SAMURAI_DIR = DEMO_PY.parent
OUT_TRACK   = BASE_DIR / "video_out_track"
OUT_TRACK.mkdir(parents=True, exist_ok=True)

# ── LOAD MODEL ────────────────────────────────────────────────────────────
model = GroundedSAM2().to(DEVICE).eval()

# ── HELPERS ───────────────────────────────────────────────────────────────
def read_four_images(folder: Path):
    paths = sorted(folder.glob("*.jpg"), key=lambda p: int(p.stem))
    if len(paths) < 4:
        raise FileNotFoundError(f"Needs 4 images in {folder} (found {len(paths)}).")
    imgs = [cv2.imread(str(p)) for p in paths[:4]]
    if not all(im is not None for im in imgs):
        raise FileNotFoundError(f"Could not read one or more images in {folder}.")
    h, w = imgs[0].shape[:2]
    imgs = [cv2.resize(im, (w, h)) for im in imgs]
    return np.hstack(imgs)

# ── BUILD COMBINED IMAGES FOR WHOLE TRAJECTORY ────────────────────────────
step_dirs = sorted([d for d in TRAJECTORY.iterdir() if d.is_dir()],
                   key=lambda p: int(p.name))  # numeric sort: 00,01,02...
if not step_dirs:
    raise FileNotFoundError(f"No step folders found under {TRAJECTORY}")

for idx, step in enumerate(step_dirs):
    combined  = read_four_images(step)
    out_path  = FRAMES_DIR / f"{idx:03d}.jpg"
    cv2.imwrite(str(out_path), combined)
print(f"Saved {len(step_dirs)} combined frames to: {FRAMES_DIR}")

# ── RUN GSAM2 ON FIRST COMBINED & WRITE first_bbox.txt ────────────────────
first_combined_path = FRAMES_DIR / "000.jpg"
first_bgr = cv2.imread(str(first_combined_path))
if first_bgr is None:
    raise FileNotFoundError(f"Missing first combined: {first_combined_path}")

first_pil = Image.fromarray(cv2.cvtColor(first_bgr, cv2.COLOR_BGR2RGB))
with torch.no_grad():
    feats, mask, xywh_txt, xyxy_px = model(first_pil, PROMPT, return_mask=True, fully_masked=True)

if xywh_txt is None or xyxy_px is None:
    raise RuntimeError("GroundedSAM2 did not return a bbox. Try lowering BOX_THRESHOLD/TEXT_THRESHOLD or check the prompt/images.")

# Save bbox (x y w h) in combined coords
x, y, w_box, h_box = map(int, xywh_txt)
with open(OUT_BBOX_TXT, "w") as f:
    f.write(f"{x},{y},{w_box},{h_box}\n")
print(f"Wrote init bbox: {OUT_BBOX_TXT}  (x,y,w,h on combined)")

# quick vis of first combined 
vis = first_bgr.copy()
if mask is not None:
    m = (mask > 0).astype(np.uint8)
    overlay = vis.copy()
    overlay[m == 1] = (0, 255, 255)
    vis = cv2.addWeighted(overlay, 0.4, vis, 0.6, 0)
x1, y1, x2, y2 = map(int, xyxy_px)
cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
vis_path = BASE_DIR / "first_combined_vis.jpg"
cv2.imwrite(str(vis_path), vis)
print(f"Saved first combined vis: {vis_path}")

# ── RUN SAMURAI ON THE COMBINED SEQUENCE ──────────────────────────────────
assert DEMO_PY.exists(), f"samurai demo.py not found at {DEMO_PY}"
assert FRAMES_DIR.exists(), f"combined frames folder missing: {FRAMES_DIR}"
assert OUT_BBOX_TXT.exists(), f"first_bbox.txt missing: {OUT_BBOX_TXT}"

cmd = [
    sys.executable, str(DEMO_PY),
    "--video_path", str(FRAMES_DIR),
    "--txt_path",   str(OUT_BBOX_TXT),
    "--video_output_path", str(OUT_TRACK / "samurai_out2.mp4")
    ]
print("Running SAMURAI:", " ".join(cmd))
subprocess.run(cmd, cwd=str(SAMURAI_DIR), check=True)
print("SAMURAI finished, video is saved at:", OUT_TRACK / "samurai_out2.mp4")
