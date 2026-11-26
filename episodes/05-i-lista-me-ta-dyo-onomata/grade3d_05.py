"""Ep05: the 2D finish for the Blender clips, in the channel's black-and-white ink look.
Everything becomes cream paper + ink (the tones measured on the ep05 plates: paper RGB 255/247/232, ink 10/8/7),
except red-orange (the fire): that keeps its colour. Then the same finish as engine/grade_3d.py
(lamp pool, warm grade, film).  Needs engine/grade_3d.py (PR «Υβριδική μέθοδος 2D + 3D»).

usage:
  python3 grade3d_05.py in.mp4 out.mp4 [--fps 30] [--gain G] [--no-pool]
  python3 grade3d_05.py in.mp4 - --still N out.png
"""
import sys, os, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'engine'))
import numpy as np, cv2
from grade_3d import lamp_pool, finish, frames, fps_of
from eng import Film, W, H

PAPER = np.float32([232, 247, 255])      # BGR, plates' paper
INK = np.float32([7, 8, 10])             # BGR, plates' ink

def mono_keep_fire(img):
    """img BGR float 0..255 -> cream/ink monochrome, fire (red-orange, saturated) kept in colour."""
    u8 = np.clip(img, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(u8, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = hsv[..., 0], hsv[..., 1] / 255, hsv[..., 2] / 255
    hue_w = np.clip(1 - np.minimum(np.abs(h - 8), 180 - np.abs(h - 8)) / 18, 0, 1)    # red to orange (OpenCV hue 170..26)
    m = hue_w * np.clip((s - 0.30) / 0.25, 0, 1) * np.clip((v - 0.20) / 0.25, 0, 1)
    m = cv2.GaussianBlur(m, (0, 0), 2.0)[..., None]
    lum = (0.114 * img[..., 0] + 0.587 * img[..., 1] + 0.299 * img[..., 2])[..., None] / 255
    mono = INK + np.clip(lum, 0, 1) ** 1.05 * (PAPER - INK)
    return mono * (1 - m) + img * m

def main():
    src, dst = sys.argv[1], sys.argv[2]
    arg = lambda k, d: float(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d
    still = int(arg('--still', 0)) or None
    fps = arg('--fps', None) or fps_of(src)
    pool = (lamp_pool() if '--no-pool' not in sys.argv else np.ones((H, W, 1), np.float32)) * arg('--gain', 1.0)
    film = Film(seed=5); enc = None
    if still is None:
        enc = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}',
                                '-r', f'{fps}', '-i', '-', '-c:v', 'libx264', '-crf', '16', '-preset', 'slow',
                                '-pix_fmt', 'yuv420p', dst], stdin=subprocess.PIPE)
    for fi, img in enumerate(frames(src)):
        if still is not None and fi + 1 != still: continue
        out = finish(mono_keep_fire(img * pool), fi, fps, film)
        if still is not None:
            cv2.imwrite(sys.argv[sys.argv.index('--still') + 2], out); return
        enc.stdin.write(out.tobytes())
    if enc: enc.stdin.close(); enc.wait()

if __name__ == '__main__':
    main()
