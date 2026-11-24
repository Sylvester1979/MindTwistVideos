"""The 2D 'skin' for Blender renders, so 3D shots cut invisibly into 2D episodes.
Same as the 2D finish: warm grade (warmth 0.6), Film (grain, flicker, weave, dust, vignette), plus the lamp pool
of the 2D desk. Also fixes what the user's machine may get wrong: frame rate (--fps) and exposure (--gain).

usage:
  python3 grade_3d.py in.mp4 out.mp4 [--fps 30] [--gain 1.25]
  python3 grade_3d.py in.mp4 - --still 135 out.png          (one frame, for checks)
Exposure check: paper on a card should read ≈ RGB 187/173/153 after grading (2D original, 1A1P-01).
Measured on 1A1P-01 v2 (EEVEE, Standard view): --gain 1.25 gave 179/163/146."""
import sys, os, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2
from eng import Film, W, H

def lamp_pool():
    """the light pool of the 2D desk (cards05.desk), softened because here it also falls on the cards; centre = 1."""
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    spot = np.exp(-(((x - W * 0.48) / (W * 0.7)) ** 2 + ((y - H * 0.45) / (H * 0.5)) ** 2))
    return (0.62 + 0.38 * spot)[..., None]

def finish(img, fi, fps, film, warmth=0.6, grain=0.85):
    """identical to lib05.grade(): warm split, then film."""
    out = img * (1 + warmth * np.float32([-0.04, 0.0, 0.05]))
    out = film.apply(out, fi, fi / fps, strength=grain, weave=True)
    return np.clip(out, 0, 255).astype(np.uint8)

def frames(src):
    p = subprocess.Popen(['ffmpeg', '-v', 'quiet', '-i', src, '-vf', f'scale={W}:{H}:flags=lanczos',
                          '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-'], stdout=subprocess.PIPE)
    n = W * H * 3
    while True:
        b = p.stdout.read(n)
        if len(b) < n: break
        yield np.frombuffer(b, np.uint8).reshape(H, W, 3).astype(np.float32)
    p.stdout.close(); p.wait()

def fps_of(src):
    r = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=r_frame_rate',
                        '-of', 'csv=p=0', src], capture_output=True, text=True).stdout.strip()
    a, b = r.split('/'); return float(a) / float(b)

def main():
    src, dst = sys.argv[1], sys.argv[2]
    arg = lambda k, d: float(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d
    still = int(arg('--still', 0)) or None
    fps = arg('--fps', None) or fps_of(src)          # --fps 30: the Blender frames are 30 fps frames, whatever the file says
    pool = lamp_pool() * arg('--gain', 1.0)
    film = Film(seed=5); enc = None
    if still is None:
        enc = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}',
                                '-r', f'{fps}', '-i', '-', '-c:v', 'libx264', '-crf', '16', '-preset', 'slow',
                                '-pix_fmt', 'yuv420p', dst], stdin=subprocess.PIPE)
    for fi, img in enumerate(frames(src)):
        if still is not None and fi + 1 != still: continue
        out = finish(img * pool, fi, fps, film)
        if still is not None:
            cv2.imwrite(sys.argv[sys.argv.index('--still') + 2], out); return
        enc.stdin.write(out.tobytes())
    if enc: enc.stdin.close(); enc.wait()

if __name__ == '__main__':
    main()
