"""Ep05 render. usage:
  render05.py stills t1 t2 ...        -> $MT_ROOT/test/s_<t>.png
  render05.py video t0 t1 out.mp4     -> frames [t0,t1) at 30 fps (x264 veryfast crf 18)"""
import os, sys, subprocess, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import cv2, numpy as np
from lib05 import ROOT, W, H
import shots05
FPS = 30

if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'stills':
        for a in sys.argv[2:]:
            t = float(a); t0 = time.time()
            img = shots05.frame(t, int(t * FPS))
            cv2.imwrite(f'{ROOT}/test/s_{t:06.2f}.png', img)
            print(f'{t:7.2f}  {time.time() - t0:.2f}s', flush=True)
    elif mode == 'video':
        t0, t1, out = float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
        p = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}',
                              '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '18',
                              '-pix_fmt', 'yuv420p', out], stdin=subprocess.PIPE)
        f0, f1 = int(round(t0 * FPS)), int(round(t1 * FPS))
        st = time.time()
        for fi in range(f0, f1):
            p.stdin.write(shots05.frame(fi / FPS, fi).tobytes())
            if (fi - f0) % 90 == 0: print(f'{fi / FPS:7.2f}  {(time.time() - st) / max(1, fi - f0 + 1):.2f}s/f', flush=True)
        p.stdin.close(); p.wait()
