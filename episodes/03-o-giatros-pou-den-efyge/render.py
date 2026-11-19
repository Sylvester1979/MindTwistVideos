"""Episode 3 renderer: «Ο γιατρός που δεν έφυγε». Data-driven: timeline.json (shots + subs) + MOVES below.
   python3 render.py stills t1 t2 ...      -> test/still_<t>.jpg
   python3 render.py video f0 f1 out.mp4    -> frames [f0,f1)
Plates: $MT_ROOT/plates/<k>.png (graded by look.py), depth in $MT_ROOT/depth/<k>.npy.
AI regions: $MT_ROOT/ai/<k>.mp4 + AI[k] box -> hybrid plate per frame (see hybrid.py)."""
import sys, os, json, subprocess, functools
import numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'engine')); sys.path.insert(0, HERE)
from eng import *
from eng import _ink_noise
from look import grade

TL = json.load(open(f'{ROOT}/timeline.json'))
T, SHOTS, SUBS = TL['T'], TL['shots'], TL['subs']
XF = 0.6                       # crossfade between shots (s)
FILM, MOTES = Film(seed=8), Motes(n=50, seed=4)
INKC = np.array([44, 52, 62], np.float32)     # warm dark ink (BGR)
CREAMC = np.array([222, 236, 244], np.float32)

# camera: (cx, cy, zoom) in plate fractions, start -> end; par = (px, py, pz) at the end
D = dict(par=(6, 0, 60), motes=0.0, flick=0.0)
MOVES = {
    'A01_street':  dict(c=((0.50, 0.44, 1.16), (0.50, 0.56, 1.16)), par=(0, 8, 30)),
    'A02_window':  dict(c=((0.50, 0.50, 1.04), (0.50, 0.43, 1.30)), motes=0.4),
    'A03_guard':   dict(c=((0.50, 0.50, 1.05), (0.50, 0.66, 1.30))),
    'A04_diary':   dict(c=((0.50, 0.50, 1.05), (0.54, 0.66, 1.45)), flick=1.0, motes=0.5),
    'B01_portrait':dict(c=((0.50, 0.52, 1.05), (0.45, 0.38, 1.28)), motes=0.4),
    'B02_orphans': dict(c=((0.42, 0.52, 1.22), (0.58, 0.48, 1.22)), par=(14, 0, 30), motes=0.4),
    'B03_wall':    dict(c=((0.50, 0.45, 1.05), (0.45, 0.70, 1.30))),
    'B04_refuses': dict(c=((0.50, 0.52, 1.05), (0.50, 0.55, 1.25))),
    'B05_stefa':   dict(c=((0.50, 0.55, 1.05), (0.50, 0.35, 1.30)), motes=0.4),
    'C01_play':    dict(c=((0.50, 0.55, 1.05), (0.50, 0.42, 1.35)), motes=0.5),
    'C02_sickboy': dict(c=((0.50, 0.52, 1.05), (0.38, 0.55, 1.30)), motes=0.4),
    'C03_watches': dict(c=((0.50, 0.50, 1.05), (0.55, 0.35, 1.30)), motes=0.4),
    'C04_rails':   dict(c=((0.50, 0.58, 1.05), (0.50, 0.52, 1.38)), par=(0, 0, 130)),
    'D01_gate':    dict(c=((0.50, 0.58, 1.05), (0.50, 0.52, 1.28))),
    'D02_dress':   dict(c=((0.44, 0.50, 1.20), (0.56, 0.50, 1.20)), par=(12, 0, 30), motes=0.4),
    'D03_bundle':  dict(c=((0.50, 0.45, 1.05), (0.50, 0.58, 1.28))),
    'E01_march':   dict(c=((0.48, 0.55, 1.35), (0.50, 0.50, 1.05)), par=(0, 0, -60)),
    'E02_column':  dict(c=((0.50, 0.60, 1.20), (0.50, 0.42, 1.20)), par=(0, 10, 30)),
    'F01_platform':dict(c=((0.50, 0.50, 1.05), (0.60, 0.33, 1.50))),
    'F02_wagon':   dict(c=((0.50, 0.50, 1.05), (0.46, 0.52, 1.28))),
    'F03_empty':   dict(c=((0.50, 0.55, 1.04), (0.50, 0.52, 1.16)), motes=0.5),
    'G02_hidden':  dict(c=((0.50, 0.50, 1.05), (0.50, 0.50, 1.32))),
    'G03_page':    dict(c=((0.50, 0.50, 1.05), (0.60, 0.42, 1.35)), motes=0.4),
    'H01_flowers': dict(c=((0.50, 0.62, 1.40), (0.50, 0.50, 1.04)), par=(0, 0, -50), motes=0.4),
}
# hybrid AI regions: plate-fraction box (l, t, r, b); clip in $ROOT/ai/<k>.mp4
AI = {}   # optional hybrid AI regions, e.g. 'A02_window': (l, t, r, b); clip in $ROOT/ai/<k>.mp4
# text cards: shot -> (lines, kind, size, y, t_in, t_out) times relative to shot start/end
CARDS = {
    'A01_street': (['4 Αυγούστου 1942', 'Βαρσοβία'], 'serif', 70, 0.17, 0.4, -0.5),
    'B01_portrait': (['Γιάνους Κόρτσακ'], 'serif', 76, 0.15, 0.3, -0.4),
    'C04_rails': (['22 Ιουλίου 1942', 'Τα πρώτα τρένα για την Τρεμπλίνκα'], 'serif', 58, 0.15, 0.4, -0.5),
}


@functools.lru_cache(None)
def plate(k):
    return Plate(k)


@functools.lru_cache(None)
def ai_clip(k):
    """graded, aligned, colour-matched AI frames + feathered mask at plate res (see hybrid.py)."""
    p = f'{ROOT}/ai/{k}.mp4'
    if k not in AI or not os.path.exists(p):
        return None
    raw = cv2.imread(f'{ROOT}/img/{k}.png').astype(np.float32); ph, pw = raw.shape[:2]
    cap = cv2.VideoCapture(p); fr = []
    while True:
        ok, f = cap.read()
        if not ok: break
        fr.append(cv2.resize(f, (pw, ph), interpolation=cv2.INTER_CUBIC).astype(np.float32))
    g = lambda a: cv2.resize(cv2.cvtColor(np.clip(a, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY), None, fx=.25, fy=.25).astype(np.float32)
    Wm = np.eye(2, 3, dtype=np.float32)
    try:
        _, Wm = cv2.findTransformECC(g(raw), g(fr[0]), Wm, cv2.MOTION_AFFINE,
                                     (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6), None, 5)
        Wm[:, 2] *= 4
    except cv2.error:
        pass
    fr = [cv2.warpAffine(f, Wm, (pw, ph), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT) for f in fr]
    l, t, r, b = AI[k]
    m = np.zeros((ph, pw), np.float32); m[int(t * ph):int(b * ph), int(l * pw):int(r * pw)] = 1
    m = cv2.GaussianBlur(m, (0, 0), 0.025 * pw)[..., None]
    sel = m[..., 0] > 0.5
    mu_s, sd_s = raw[sel].mean(0), raw[sel].std(0) + 1e-3
    mu_c, sd_c = fr[0][sel].mean(0), fr[0][sel].std(0) + 1e-3
    fr = [grade((f - mu_c) / sd_c * sd_s + mu_s).astype(np.float16) for f in fr]
    return fr, m


def shot_frame(s, t, fi):
    k = s['k']
    if k == 'G01_count':
        return count_frame(s, t, fi)
    mv = {**D, **MOVES.get(k, {})}
    (a0, b0, z0), (a1, b1, z1) = mv.get('c', ((0.5, 0.5, 1.05), (0.5, 0.5, 1.25)))
    dur = s['t1'] - s['t0'] + XF
    u = ease(clamp01((t - s['t0'] + XF / 2) / dur))
    P = plate(k)
    clip = ai_clip(k)
    if clip is not None:
        fr, m = clip
        if not hasattr(P, 'still'): P.still = P.img.copy()
        x = clamp01((t - s['t0']) / max(0.1, s['t1'] - s['t0'])) * (len(fr) - 1)
        i0 = int(x); i1 = min(len(fr) - 1, i0 + 1); a = x - i0
        cf = fr[i0].astype(np.float32) * (1 - a) + fr[i1].astype(np.float32) * a
        P.img = P.still * (1 - m) + cf * m
    cam = (lerp(a0, a1, u) * P.pw, lerp(b0, b1, u) * P.ph, lerp(z0, z1, u))
    px, py, pz = mv['par']
    img = P.render(t, fi, cam, par=(px * u, py * u, pz * u), focus=0.55, boil=0.0)
    if mv['flick']:
        fl = 1 + 0.035 * mv['flick'] * (np.sin(t * 9.1) + 0.6 * np.sin(t * 15.7 + 1))
        img = img * fl
    if mv['motes']:
        img = MOTES.apply(img, t, mv['motes'])
    if k in CARDS:
        img = card(img, k, s, t)
    if k == 'A04_diary':
        img = quote(img, QUOTES[k][0], s, t, 0.30, *QUOTES[k][1:])
    if k == 'E02_column':
        img = quote(img, QUOTES[k][0], s, t, 0.26, *QUOTES[k][1:])
    return img


QUOTES = {   # shot -> (lines, t_in rel. start, t_out rel. end if <0)
    'A04_diary': (['«Το φαλακρό μου κεφάλι', 'στο παράθυρο.', 'Τι υπέροχος στόχος.', 'Γιατί στέκεται', 'και κοιτάζει ήρεμα;»'], 1.8, -0.8),
    'E02_column': (['«Δεν ήταν πορεία', 'προς τα βαγόνια.', 'Ήταν μια σιωπηλή', 'διαμαρτυρία.»'], 2.5, -0.6),
}
QWIN = [(s['t0'] + QUOTES[s['k']][1], s['t1'] + QUOTES[s['k']][2]) for s in SHOTS if s['k'] in QUOTES]


def card(img, k, s, t):
    lines, kind, size, y, ti, to = CARDS[k]
    t_out = s['t1'] + to if to < 0 else s['t0'] + to
    a = clamp01((t - s['t0'] - ti) / 0.6) * clamp01((t_out - t) / 0.6)
    if a <= 0: return img
    al = text_layer(lines, kind, size, 560, 1.25)
    cy = int(H * y)
    img = band(img, cy - al.shape[0] // 2 - 6, cy + al.shape[0] // 2 + 6, 0.45 * a, 90)
    return paste_alpha(img, al, W / 2, cy, (236, 242, 246), a, shadow=1.0, rough=True)


def paper_text(img, al, cy, a, pad=(40, 40), op=0.9):
    """ink text on a torn cream paper strip (paper-craft title card)."""
    px, py = pad
    h, w = al.shape[0] + 2 * py, al.shape[1] + 2 * px
    x0, y0 = W // 2 - w // 2, cy - h // 2
    n = _ink_noise(5)
    edge = np.ones((h, w), np.float32)
    rough = (n[:h, :w] - 0.5) * 18
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.minimum(np.minimum(xx, w - 1 - xx), np.minimum(yy, h - 1 - yy)) + rough
    edge = np.clip(d / 3.0, 0, 1)
    sub = img[y0:y0 + h, x0:x0 + w]
    sh = cv2.GaussianBlur(edge, (0, 0), 7)[..., None]
    sub = sub * (1 - 0.35 * a * np.roll(np.roll(sh, 6, 0), 4, 1))
    paper = CREAMC * (0.93 + 0.07 * n[:h, :w])[..., None]
    m = (edge * op * a)[..., None]
    img[y0:y0 + h, x0:x0 + w] = sub * (1 - m) + paper * m
    return paste_alpha(img, al, W / 2, cy, INKC, a, rough=True)


def quote(img, lines, s, t, y, ti, to):
    t_in = s['t0'] + ti
    t_out = s['t1'] + to if to < 0 else s['t0'] + to
    a = clamp01((t - t_in) / 0.8) * clamp01((t_out - t) / 0.8)
    if a <= 0: return img
    return paper_text(img, text_layer(lines, 'serif', 72, 520, 1.22), int(H * y), a, pad=(44, 40), op=0.92)


# ---------------------------------------------------------------- ~4.000 children
@functools.lru_cache(None)
def count_field(n=4000, seed=3):
    r = np.random.default_rng(seed)
    cols, rows = 50, 82
    pts = [((i + 0.5 + (j % 2) * 0.5) * W / cols, (j + 0.5) * H / rows) for j in range(rows) for i in range(cols)]
    pts = np.array(pts, np.float32)[:n]
    pts += r.normal(0, 2.0, pts.shape).astype(np.float32)
    order = r.permutation(len(pts))
    return pts[order], 0.75 + r.random(len(pts)) * 0.4


@functools.lru_cache(None)
def count_bg():
    n = _ink_noise(9)
    return INK * 1.6 * (0.8 + 0.4 * n)[..., None] * vignette()


def count_frame(s, t, fi):
    p = clamp01((t - s['t0'] - 0.3) / max(1.0, (s['t1'] - s['t0']) * 0.75))
    out = count_bg().copy()
    pts, sc = count_field()
    k = int(len(pts) * ease2(p))
    lay = np.zeros((H, W), np.float32)
    for (x, y), q in zip(pts[:k], sc[:k]):
        r = 3.2 * q
        cv2.circle(lay, (int(x), int(y - 5 * q)), max(1, int(round(r))), 1.0, -1, cv2.LINE_AA)
        cv2.ellipse(lay, (int(x), int(y + 4 * q)), (max(1, int(4 * q)), max(1, int(6 * q))), 0, 0, 360, 1.0, -1, cv2.LINE_AA)
    lay = cv2.GaussianBlur(lay, (0, 0), 0.7) * 0.8
    out = out * (1 - lay[..., None]) + PAPER * 0.9 * lay[..., None]
    num = int(round(4000 * ease2(p)))
    s_ = f'{num:,}'.replace(',', '.')
    al = text_layer(['≈ ' + s_], 'serif', 210, 600)
    y0, y1 = int(H * 0.22) - 150, int(H * 0.22) + 200
    out[y0:y1] = out[y0:y1] * 0.15 + count_bg()[y0:y1] * 0.85
    out = paste_alpha(out, al, W / 2, H * 0.22, RED * 1.15, 1.0, shadow=0.8, rough=True)
    a2 = clamp01((p - 0.35) / 0.3)
    if a2 > 0:
        al2 = text_layer(['παιδιά από τα ορφανοτροφεία', 'της Βαρσοβίας'], 'serif', 56, 500, 1.2)
        out = paste_alpha(out, al2, W / 2, H * 0.22 + 135, PAPER, a2, shadow=0.8, rough=True)
    return out


# ---------------------------------------------------------------- subtitles
SUBY = 1610


@functools.lru_cache(None)
def sub_alpha(text):
    size = 58
    while True:
        al = text_layer([text], 'serif', size, 560)
        if al.shape[1] < W - 80 or size <= 40: return al
        size -= 2


def subtitles(img, t):
    for s in SUBS:
        if any(a - 0.3 <= s['t0'] <= b for a, b in QWIN):
            continue   # the quote card already shows these words
        if s['t0'] - 0.05 <= t <= s['t1'] + 0.1:
            a = clamp01((t - s['t0'] + 0.05) / 0.1) * clamp01((s['t1'] + 0.1 - t) / 0.1)
            img = band(img, SUBY - 48, SUBY + 48, 0.45 * a, 70)
            img = paste_alpha(img, sub_alpha(s['text']), W / 2, SUBY, (236, 242, 246), a, shadow=1.0, rough=False)
    return img


# ---------------------------------------------------------------- frame
def frame(t, fi):
    cur = [s for s in SHOTS if s['t0'] - XF / 2 <= t < s['t1'] + XF / 2]
    if not cur:
        cur = [SHOTS[-1]]
    if len(cur) == 1:
        img = shot_frame(cur[0], t, fi)
    else:
        a, b = cur[0], cur[1]
        u = ease(clamp01((t - (b['t0'] - XF / 2)) / XF))
        img = crossfade(shot_frame(a, t, fi), shot_frame(b, t, fi), u)
    # fade in / out
    img = img * clamp01(t / 0.6) * clamp01((T - t) / 1.2)
    img = FILM.apply(img, fi, t, strength=0.9)          # episode-2 film look
    if os.environ.get('NOSUBS') is None:
        img = subtitles(img, t)
    return np.clip(img, 0, 255).astype(np.uint8)


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'stills':
        os.makedirs(f'{ROOT}/test', exist_ok=True)
        for a in sys.argv[2:]:
            t = float(a)
            cv2.imwrite(f'{ROOT}/test/still_{a}.jpg', frame(t, int(t * FPS)), [cv2.IMWRITE_JPEG_QUALITY, 88])
            print('still', a, flush=True)
    elif mode == 'video':
        f0, f1, out = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
        tmp = out.replace('.mp4', '.part.mp4')
        p = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                              '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '18', '-tune', 'grain', '-pix_fmt', 'yuv420p', tmp], stdin=subprocess.PIPE)
        for fi in range(f0, f1):
            p.stdin.write(frame(fi / FPS, fi).tobytes())
            if fi % 60 == 0: print(out, fi, flush=True)
        p.stdin.close(); p.wait()
        if p.returncode == 0: os.rename(tmp, out)
