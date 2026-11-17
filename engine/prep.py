"""Tone plates to warm charcoal BW at native res, keep warm colour (flames),
remove artist marks in the paper border, and build a 'paper' mask (border)."""
import cv2, numpy as np, json, sys, os
ROOT = os.environ.get('MT_ROOT', '/home/user/ww1')
PAPER = np.array([214, 226, 236], np.float32)  # BGR warm paper
INK = np.array([16, 17, 20], np.float32)
# rectangles (fractions l,t,r,b) where artist marks / stray text live
MARKS = {
    'P01_carriage': [[0.05, 0.946, 0.15, 0.997], [0.52, 0.975, 0.97, 1.0]],
    'P03_hq': [[0.03, 0.975, 0.2, 1.0], [0.56, 0.975, 0.92, 1.0]],
    'P05_meuse': [[0.6, 0.975, 0.97, 1.0]],
    'P07_village': [[0.62, 0.975, 0.97, 1.0]],
    'P09_mgnest': [[0.62, 0.975, 0.9, 1.0]],
    'P10_runner': [[0.70, 0.97, 0.95, 1.0]],
    'P12_crosses': [[0.60, 0.950, 0.77, 0.99]],
}

def tone(img):
    f = img.astype(np.float32) / 255
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    g = np.clip((g - 0.03) / 0.92, 0, 1)
    base = INK + (PAPER - INK) * g[..., None]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    sat = np.clip((hsv[..., 1] / 255 - 0.35) / 0.3, 0, 1) * np.clip((hsv[..., 2] / 255 - 0.3) / 0.3, 0, 1)
    sat = cv2.GaussianBlur(sat, (0, 0), 2)[..., None]
    out = base * (1 - sat) + f * 255 * sat
    return np.clip(out, 0, 255).astype(np.uint8), (sat[..., 0] * 255).astype(np.uint8)

def remove_marks(img, rects):
    H, W = img.shape[:2]
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    m = np.zeros((H, W), np.uint8)
    for l, t, r, b in rects:
        x0, y0, x1, y1 = int(l * W), int(t * H), int(r * W), int(b * H)
        m[y0:y1, x0:x1] = 255
    dark = ((g < 200) & (m > 0)).astype(np.uint8) * 255
    dark = cv2.dilate(dark, np.ones((9, 9), np.uint8))
    return cv2.inpaint(img, dark, 7, cv2.INPAINT_TELEA)

def paper_mask(img):
    """1 where the pixel is blank paper connected to the outer border."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lite = (cv2.GaussianBlur(g, (0, 0), 3) > 222).astype(np.uint8)
    H, W = g.shape
    n, lab = cv2.connectedComponents(lite, connectivity=4)
    edge = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    m = np.isin(lab, list(edge)).astype(np.float32)
    m = cv2.erode(m, np.ones((7, 7), np.uint8))
    return cv2.GaussianBlur(m, (0, 0), 6)

if __name__ == '__main__':
    os.makedirs(f'{ROOT}/plates', exist_ok=True)
    for k in sys.argv[1:]:
        img = cv2.imread(f'{ROOT}/img/{k}.png')
        if k in MARKS:
            img = remove_marks(img, MARKS[k])
        toned, sat = tone(img)
        cv2.imwrite(f'{ROOT}/plates/{k}.png', toned)
        if sat.max() > 40:
            cv2.imwrite(f'{ROOT}/plates/{k}_warm.png', sat)
        cv2.imwrite(f'{ROOT}/plates/{k}_paper.png', (paper_mask(img) * 255).astype(np.uint8))
        print(k, img.shape, 'warm' if sat.max() > 40 else '', flush=True)
