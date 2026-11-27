"""Ep05 thumbnail: the user's illustration (nano-banana, 26/11: the Metropolitan and the mayor in the commander's office)
with the title set in code, in the video's serif (EB Garamond), in the dark ceiling above them.
usage: python3 thumb05.py <illustration.jpg> <outdir>  -> thumb_<colour>.jpg; the user (26/11) asked for red on «Η ΛΙΣΤΑ ΠΟΥ ΔΕΝ
ΠΑΡΑΔΟΘΗΚΕ ΠΟΤΕ»: thumb_paper_red / thumb_amber_red (first line in paper or amber, the rest red, ep. 4's red pencil)"""
import sys, os
from PIL import Image, ImageDraw, ImageFont, ImageFilter

SRC, OUT = sys.argv[1], sys.argv[2]
FONT = os.environ.get('SERIF', os.path.expanduser('~/.fonts/EBGaramond.ttf'))
W, H = 1080, 1920
MAXW = 900                                                     # the drawing's frame is ~40 px in from each side
LINES = [('ΖΑΚΥΝΘΟΣ 1943', 150, 640, 0.06), ('Η ΛΙΣΤΑ ΠΟΥ ΔΕΝ', 92, 560, 0.04), ('ΠΑΡΑΔΟΘΗΚΕ ΠΟΤΕ', 92, 560, 0.04)]
PAPER, AMBER, RED = (236, 226, 214), (240, 196, 120), (178, 44, 42)   # the plates' paper · the desk lamp · ep. 4's red pencil
COLOURS = {'paper_red': (PAPER, RED, RED), 'amber_red': (AMBER, RED, RED)}   # one colour per line


def font(size, wght):
    f = ImageFont.truetype(FONT, size)
    try: f.set_variation_by_axes([wght])
    except Exception: pass
    return f


def base():
    im = Image.open(SRC).convert('RGB')
    s = W / im.width
    im = im.resize((W, round(im.height * s)), Image.LANCZOS)
    top = (im.height - H) // 2
    return im.crop((0, top, W, top + H))


def title(im, colours):
    out = im
    lays = [Image.new('L', (W, H), 0) for _ in LINES]
    y = 150
    for i, (txt, size, wght, track) in enumerate(LINES):
        d = ImageDraw.Draw(lays[i])
        while True:                                        # fit inside the illustration's drawn border
            f = font(size, wght); tr = size * track
            w = sum(f.getlength(c) for c in txt) + tr * (len(txt) - 1)
            if w <= MAXW: break
            size -= 2
        x = (W - w) / 2
        for c in txt:
            d.text((x, y), c, font=f, fill=255); x += f.getlength(c) + tr
        y += int(size * (1.22 if i == 0 else 1.08)) + (28 if i == 0 else 0)
    for lay in lays:
        shadow = lay.filter(ImageFilter.GaussianBlur(10))
        out = Image.composite(Image.new('RGB', (W, H), (8, 6, 5)), out, shadow.point(lambda v: int(v * 0.85)))
    for lay, col in zip(lays, colours):
        out = Image.composite(Image.new('RGB', (W, H), col), out, lay)
    return out


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    b = base()
    for name, col in COLOURS.items():
        title(b, col).save(f'{OUT}/thumb_{name}.jpg', quality=93)
        print(f'{OUT}/thumb_{name}.jpg')
