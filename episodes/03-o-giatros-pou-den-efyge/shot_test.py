"""Engine-only motion test on one plate: slow push-in with depth parallax, motes, light film.
usage: MT_ROOT=/home/user/ep03 python3 shot_test.py U2_window out.mp4 [secs]
"""
import sys, os, subprocess, numpy as np, cv2
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'engine'))
from eng import Plate, Motes, Film, W, H, FPS, ease, lerp

key, out = sys.argv[1], sys.argv[2]
secs = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0
P = Plate(key)
MO, FI = Motes(n=55, seed=4), Film(seed=8)
# window region (plate px) for U2: centre of the frame's upper third
c0 = (P.pw / 2, P.ph / 2, 1.0)
c1 = (P.pw * 0.52, P.ph * 0.36, 1.32)
tmp = out[:-4] + '.part.mp4'
ff = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}',
                       '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-crf', '18', '-preset', 'veryfast',
                       '-pix_fmt', 'yuv420p', tmp], stdin=subprocess.PIPE)
n = int(secs * FPS)
for fi in range(n):
    t = fi / FPS
    u = ease(fi / (n - 1))
    cam = (lerp(c0[0], c1[0], u), lerp(c0[1], c1[1], u), lerp(c0[2], c1[2], u))
    img = P.render(t, fi, cam, par=(lerp(0, 10, u), 0, lerp(0, 70, u)), focus=0.55, boil=0.0, dof=0.0)
    img = MO.apply(img, t, 0.55)
    img = FI.apply(img, fi, t, strength=0.45, weave=False, flicker=0.4, grain=0.8)
    ff.stdin.write(np.clip(img, 0, 255).astype(np.uint8).tobytes())
ff.stdin.close()
if ff.wait() == 0:
    os.replace(tmp, out)
    print('ok', out)
