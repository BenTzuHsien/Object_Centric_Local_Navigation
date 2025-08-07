import os, csv, cv2, torch, numpy as np
from PIL import Image
from pathlib import Path
from backbones.grounded_sam2 import GroundedSAM2

# Specify the model to evaluate 
MODEL = 'gsam_mlp5_bi_map3'
PROMPT_TXT = 'green chair.'

# Sucess metric
# If the center shift is less than this threshold, we consider it a success
SHIFT_THRESHOLD = 110


# Paths 
DATA_ROOT = Path('/data/shared_data/SPOT_Real_World_Dataset/rollout')
ROLLOUT_DIR = DATA_ROOT / MODEL
GOAL_DIR = ROLLOUT_DIR / 'Goal_Images'

# Where to save viz and results 
VIS_DIR = Path.cwd() / f'map2_rollout_{MODEL}_center_shift'
CSV_PATH = VIS_DIR / f'{MODEL}_center_shift.csv'
FINAL_DISTRIBUTION = VIS_DIR / f'{MODEL.lower()}_com_distribution.jpg'
os.makedirs(VIS_DIR, exist_ok=True)

# Step directory utility
def step_dir(folder):
    subs = sorted(d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d)))
    return os.path.join(folder, subs[-1]) if subs else None


# Viz functions

# viz constant
ALPHA = 0.45

def pil_to_bgr(img):
    return np.array(img)[..., ::-1]

def overlay(bgr, mask, col, alpha=ALPHA):
    out = bgr.copy()
    out[mask] = col
    return cv2.addWeighted(out, alpha, bgr, 1 - alpha, 0)


# CoM functions
def center_of_mass(mask):
    yx = np.argwhere(mask)
    return yx.mean(axis=0) if len(yx) else None

def center_shift(mask1, mask2):
    c1 = center_of_mass(mask1)
    c2 = center_of_mass(mask2)
    return np.linalg.norm(c1 - c2) if c1 is not None and c2 is not None else float('inf'), c1, c2


# Model intitialization
DEVICE = 'cuda:0'
# Load the GroundedSAM model
gsam = GroundedSAM2().to(DEVICE).eval()


# Compute goal mask and center once
goal_img = Image.open(os.path.join(GOAL_DIR, '0.jpg')).convert('RGB')
_, g_mask = gsam(goal_img, PROMPT_TXT, return_mask=True)
g_bool, g_bgr = g_mask.astype(bool), pil_to_bgr(goal_img)
goal_center = center_of_mass(g_bool)


# Copy canvas for rollout center aggregation
distribution_canvas = g_bgr.copy()
rollout_centers = []

# List to track skipped cases with no valid masks
skipped_cases = []


# Success tracking
total_cases = 0
success_count = 0


# Write CSV header
with open(CSV_PATH, 'w', newline='') as f:
    csv.writer(f).writerow(['case', 'center_shift_px', 'success'])

# Rollout evaluation
with torch.no_grad(), open(CSV_PATH, 'a', newline='') as fcsv:
    writer = csv.writer(fcsv)

    for case in sorted(d for d in os.listdir(ROLLOUT_DIR)
                   if os.path.isdir(os.path.join(ROLLOUT_DIR, d))
                   and d != 'Goal_Images'):


        sd = step_dir(os.path.join(ROLLOUT_DIR, case))
        if sd is None:
            print(f'skip {case} (no step folders)')
            continue

        roll_path = os.path.join(sd, '0.jpg')
        roll_img = Image.open(roll_path).convert('RGB')

        _, r_mask = gsam(roll_img, PROMPT_TXT, return_mask=True)

        # For empty or invalid masks, we skip the case
        if r_mask is None:
            print(f'{case}: Skipped (no valid mask returned)')
            skipped_cases.append(case)
            continue

        r_bool, r_bgr = r_mask.astype(bool), pil_to_bgr(roll_img)

        # Compute center shift
        shift, c1, c2 = center_shift(g_bool, r_bool)
        success = shift <= SHIFT_THRESHOLD
        total_cases += 1  


        # Success tracking
        if success:
            success_count += 1

        # Print and log results
        print(f'{case}: center shift = {shift:.2f}px ,  Success = {success}')
        writer.writerow([case, f'{shift:.2f}', success])

        # --- Visualization ---
        g_overlay = overlay(g_bgr, g_bool, (0, 255, 0))
        r_overlay = overlay(r_bgr, r_bool, (0, 255, 0))

        if c1 is not None:
            pt1 = (int(c1[1]), int(c1[0]))
            # Black outline
            cv2.circle(g_overlay, pt1, 8, (0, 0, 0), -1, lineType=cv2.LINE_AA)
            # Yellow inner circle
            cv2.circle(g_overlay, pt1, 5, (0, 255, 255), -1, lineType=cv2.LINE_AA)
            # Label
            cv2.putText(g_overlay, "Goal CoM", (pt1[0] + 10, pt1[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)

        if c2 is not None:
            pt2 = (int(c2[1]), int(c2[0]))

            # Black outline
            cv2.circle(r_overlay, pt2, 8, (0, 0, 0), -1, lineType=cv2.LINE_AA)
            # Yellow inner circle
            cv2.circle(r_overlay, pt2, 5, (0, 255, 255), -1, lineType=cv2.LINE_AA)
            # Label
            cv2.putText(r_overlay, "Rollout CoM", (pt2[0] + 10, pt2[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)

            # Add red dot to final distribution image
            rollout_centers.append(pt2)
            cv2.circle(distribution_canvas, pt2, 5, (0, 0, 255), -1)


        # Save side-by-side image
        combined = cv2.hconcat([g_overlay, r_overlay])
        label = f'Shift: {shift:.1f}px , {"Success" if success else "Fail"}'
        cv2.putText(combined, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(VIS_DIR / f'{case}_center_shift.jpg'), combined)


success_rate = (success_count / total_cases) * 100 if total_cases > 0 else 0
print(f'\nSkipped {len(skipped_cases)} cases with no valid masks: {skipped_cases}')
print(f'\nTotal Cases: {total_cases}, Successes: {success_count} , Success Rate: {success_rate:.2f}%')


# Final visualization of distribution 
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import matplotlib.font_manager as fm

# Convert image
canvas_rgb = cv2.cvtColor(distribution_canvas, cv2.COLOR_BGR2RGB)
fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(canvas_rgb)
ax.axis('off')

# Plot rollout centers (in red with black edge)
rollout_np = np.array(rollout_centers)
ax.scatter(
    rollout_np[:, 0], rollout_np[:, 1],
    c='red', s=40, label='Rollout CoM',
    edgecolors='black', linewidths=0.5, zorder=4
)

# Plot goal CoM
if goal_center is not None:
    ax.scatter([goal_center[1]], [goal_center[0]],
               c='limegreen', s=80, label='Goal CoM',
               edgecolors='black', linewidths=1.2, zorder=5)


caption = f"Success Rate: {success_rate:.1f}% ({success_count}/{total_cases})"
ax.text(
    0.01, 0.99, caption,
    transform=ax.transAxes,
    fontsize=15, color='white',
    verticalalignment='top',
    bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.6)
)


# Save result
fig.tight_layout(pad=1)
plt.savefig(str(FINAL_DISTRIBUTION).replace('.jpg', '.png'), dpi=200, bbox_inches='tight')
plt.show()
