"""Depth-Anything-V2 (Base) depth for each plate -> depth/{k}.npy (0 far .. 1 near), paper border flattened."""
import sys, os, numpy as np, cv2, torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from PIL import Image
ROOT = os.environ.get('MT_ROOT', '/home/user/ww1')
torch.set_num_threads(os.cpu_count())
name = 'depth-anything/Depth-Anything-V2-Base-hf'
proc = AutoImageProcessor.from_pretrained(name)
model = AutoModelForDepthEstimation.from_pretrained(name).eval()
os.makedirs(f'{ROOT}/depth', exist_ok=True)
for k in sys.argv[1:]:
    img = cv2.imread(f'{ROOT}/img/{k}.png')
    H, W = img.shape[:2]
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    with torch.no_grad():
        inp = proc(images=pil, return_tensors='pt')
        out = model(**inp).predicted_depth
    d = torch.nn.functional.interpolate(out[None], size=(H, W), mode='bicubic', align_corners=False)[0, 0].numpy()
    lo, hi = np.percentile(d, 1), np.percentile(d, 99.5)
    d = np.clip((d - lo) / (hi - lo + 1e-6), 0, 1).astype(np.float32)
    pm = cv2.imread(f'{ROOT}/plates/{k}_paper.png', 0)
    if pm is not None:
        pm = pm.astype(np.float32) / 255
        inner = d[pm < 0.5]
        mid = float(np.median(inner)) if inner.size else 0.5
        d = d * (1 - pm) + mid * pm
    d = cv2.bilateralFilter(d, 9, 0.08, 9)
    d = cv2.GaussianBlur(d, (0, 0), 2.5)
    np.save(f'{ROOT}/depth/{k}.npy', d.astype(np.float16))
    cv2.imwrite(f'{ROOT}/depth/{k}.png', (d * 255).astype(np.uint8))
    print('depth', k, flush=True)
