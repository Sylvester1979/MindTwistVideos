#  Mind Twist GR · επ. 05 «Ζάκυνθος 1943: η λίστα που δεν παραδόθηκε ποτέ» · thumbnail, φωτορεαλιστικό (Cycles)
#  Blender 5.2 LTS. Μία εικόνα 1080×1920.
#
#  Η σκηνή: νύχτα, το γραφείο του διοικητή. Σε έναν φάκελο από χαρτόνι με κόκκινη σφραγίδα «ΖΑΚΥΝΘΟΣ 1943»
#  ακουμπά το χαρτί με τα δύο ονόματα, όπως τα δίνει η πηγή (Δ. Βίτσος):
#  «Ο Ζακύνθου Χρυσόστομος» και «Δήμαρχος Ζακυνθίων Λουκάς Καρρέρ», γραμμένα με πένα. Μια πένα της δεκαετίας
#  του '40 (χωρίς μάρκα), το καπάκι της δίπλα, ένα μελανοδοχείο,
#  το ζεστό φως της λάμπας από πάνω αριστερά, το κρύο φως του παραθύρου από δεξιά.
#
#  Χτίζει τη σκηνή από το μηδέν και τη σώζει ως ep05_thumb.blend, με τις υφές πακεταρισμένες μέσα.
#  usage: blender -b -P mt_ep05_thumb.py -- --tex <φάκελος υφών> [--out <file.blend>] [--preview <png>]
import bpy, bmesh, math, os, sys
from mathutils import Vector

argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
def arg(name, default=None):
    return argv[argv.index(name) + 1] if name in argv else default
TEX = arg('--tex', os.path.join(os.path.dirname(bpy.data.filepath or __file__), 'tex'))
OUT = arg('--out', os.path.join(os.path.dirname(TEX), 'ep05_thumb.blend'))
PREVIEW = arg('--preview')

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.name = 'ep05_thumb'


# ---------------------------------------------------------------- helpers
def obj(name, mesh):
    o = bpy.data.objects.new(name, mesh); scene.collection.objects.link(o); return o

def nodes_of(mat):
    mat.use_nodes = True; nt = mat.node_tree
    for n in list(nt.nodes): nt.nodes.remove(n)
    out = nt.nodes.new('ShaderNodeOutputMaterial'); out.location = (600, 0)
    bsdf = nt.nodes.new('ShaderNodeBsdfPrincipled'); bsdf.location = (300, 0)
    nt.links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
    return nt, bsdf

def setin(node, name, value):
    if name in node.inputs: node.inputs[name].default_value = value

def image(nt, path, colour=True, loc=(-500, 0)):
    n = nt.nodes.new('ShaderNodeTexImage'); n.location = loc
    n.image = bpy.data.images.load(path, check_existing=True)
    n.image.colorspace_settings.name = 'sRGB' if colour else 'Non-Color'
    n.interpolation = 'Cubic'
    return n

def bump(nt, height_socket, strength, distance, loc=(0, -350)):
    b = nt.nodes.new('ShaderNodeBump'); b.location = loc
    b.inputs['Strength'].default_value = strength; b.inputs['Distance'].default_value = distance
    nt.links.new(height_socket, b.inputs['Height'])
    return b

def plane(name, sx, sy, nx=1, ny=1):
    """a grid plane sx×sy (m), centred, UVs 0..1."""
    me = bpy.data.meshes.new(name); bm = bmesh.new()
    bm.loops.layers.uv.new('UVMap')                      # calc_uvs only fills a UV layer that already exists
    bmesh.ops.create_grid(bm, x_segments=nx, y_segments=ny, size=0.5, calc_uvs=True)
    for v in bm.verts: v.co.x *= sx; v.co.y *= sy
    bm.to_mesh(me); bm.free()
    return obj(name, me)

def lathe(name, profile, segs=64):
    """profile [(r, z), ...] turned around Z."""
    me = bpy.data.meshes.new(name); bm = bmesh.new(); rings = []
    for r, z in profile:
        ring = []
        for i in range(segs):
            a = 2 * math.pi * i / segs
            ring.append(bm.verts.new((r * math.cos(a), r * math.sin(a), z)))
        rings.append(ring)
    for a_, b_ in zip(rings, rings[1:]):
        for i in range(segs):
            bm.faces.new((a_[i], a_[(i + 1) % segs], b_[(i + 1) % segs], b_[i]))
    bmesh.ops.contextual_create(bm, geom=rings[-1]); bmesh.ops.contextual_create(bm, geom=rings[0])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(me); bm.free()
    o = obj(name, me)
    for p in me.polygons: p.use_smooth = True
    return o


# ---------------------------------------------------------------- the desk: dark varnished walnut (procedural)
PBR = arg('--pbr', os.path.join(os.path.dirname(TEX), 'pbr'))
def pbr_set(nt, b, files, scale=(1.0, 1.0), coords='UV', normal_strength=1.0, colour=True, x=-1100):
    """scanned PBR maps (Poly Haven / ambientCG, CC0, 2K): files = dict(col=, rough=, nor=, metal=)."""
    tc = nt.nodes.new('ShaderNodeTexCoord'); tc.location = (x - 400, 0)
    mp = nt.nodes.new('ShaderNodeMapping'); mp.location = (x - 200, 0); mp.inputs['Scale'].default_value = (scale[0], scale[1], 1.0)
    nt.links.new(tc.outputs[coords], mp.inputs['Vector'])
    out = {}
    for i, (k, f) in enumerate(files.items()):
        n = nt.nodes.new('ShaderNodeTexImage'); n.location = (x, 300 - 280 * i)
        n.image = bpy.data.images.load(os.path.join(PBR, f), check_existing=True)
        n.image.colorspace_settings.name = 'sRGB' if k == 'col' else 'Non-Color'
        if coords == 'Object': n.projection = 'BOX'; n.projection_blend = 0.3
        nt.links.new(mp.outputs['Vector'], n.inputs['Vector']); out[k] = n
    if 'nor' in out:
        nm = nt.nodes.new('ShaderNodeNormalMap'); nm.location = (x + 300, -500); nm.inputs['Strength'].default_value = normal_strength
        nt.links.new(out['nor'].outputs['Color'], nm.inputs['Color']); out['normal'] = nm
    return out

desk = plane('Desk', 1.6, 1.1)
m = bpy.data.materials.new('LacqueredCherry'); nt, b = nodes_of(m)
w = pbr_set(nt, b, dict(col='lacquered_cherry_wood/lcw_diff_2k.jpg', rough='lacquered_cherry_wood/lcw_rough_2k.jpg',
                        nor='lacquered_cherry_wood/lcw_nor_gl_2k.jpg'), scale=(1.6, 1.1))
nt.links.new(w['col'].outputs['Color'], b.inputs['Base Color'])
nt.links.new(w['rough'].outputs['Color'], b.inputs['Roughness'])
nt.links.new(w['normal'].outputs['Normal'], b.inputs['Normal'])
setin(b, 'Coat Weight', 0.08); setin(b, 'Coat Roughness', 0.3)
desk.data.materials.append(m)
# (the procedural walnut of the first version was replaced by the scanned wood, 26/11)

# ---------------------------------------------------------------- the folder: two manila covers, the top one stamped
FW, FH = 0.240, 0.300
FOLD_LOC, FOLD_ROT = (0.004, 0.055), -3.0
fm = bpy.data.materials.new('Manila'); nt, b = nodes_of(fm)
ic = image(nt, os.path.join(TEX, 'folder_col.png'), True, (-600, 100))
ib = image(nt, os.path.join(TEX, 'folder_bump.png'), False, (-600, -300))
nt.links.new(ic.outputs['Color'], b.inputs['Base Color'])
setin(b, 'Sheen Weight', 0.25); setin(b, 'Sheen Roughness', 0.6)
cb = pbr_set(nt, b, dict(rough='Cardboard002/Cardboard002_2K-JPG_Roughness.jpg', nor='Cardboard002/Cardboard002_2K-JPG_NormalGL.jpg'),
             scale=(0.96, 1.20), normal_strength=0.12, x=-1500)                      # manila has no ribs: only a hint of tooth
rmap = nt.nodes.new('ShaderNodeMapRange'); rmap.location = (-200, -100)
rmap.inputs['To Min'].default_value = 0.62; rmap.inputs['To Max'].default_value = 0.88
nt.links.new(cb['rough'].outputs['Color'], rmap.inputs['Value']); nt.links.new(rmap.outputs['Result'], b.inputs['Roughness'])
fbmp = bump(nt, ib.outputs['Color'], 0.35, 0.0006)
nt.links.new(cb['normal'].outputs['Normal'], fbmp.inputs['Normal'])
nt.links.new(fbmp.outputs['Normal'], b.inputs['Normal'])
back = bpy.data.materials.new('ManilaBack'); nt2, b2 = nodes_of(back)
setin(b2, 'Base Color', (0.52, 0.36, 0.17, 1)); setin(b2, 'Roughness', 0.8)
for i, (dz, dxy, mat) in enumerate(((0.0004, (0.0035, -0.0025), back), (0.0018, (0.0, 0.0), fm))):
    f = plane(f'Folder{i}', FW, FH)
    sol = f.modifiers.new('thick', 'SOLIDIFY'); sol.thickness = 0.0012; sol.offset = -1
    bev = f.modifiers.new('edge', 'BEVEL'); bev.width = 0.0004; bev.segments = 2
    f.location = (FOLD_LOC[0] + dxy[0], FOLD_LOC[1] + dxy[1], dz)
    f.rotation_euler = (0, 0, math.radians(FOLD_ROT + (0.8 if i == 0 else 0)))
    f.data.materials.append(mat)

# ---------------------------------------------------------------- the sheet: A4, a soft fold across the middle, corners lifting
SW, SH = 0.210, 0.297
SHEET_LOC, SHEET_ROT = (0.006, -0.0235), 2.5
sheet = plane('Sheet', SW, SH, 84, 118)
for v in sheet.data.vertices:
    x, y = v.co.x, v.co.y
    z = 0.00055 * math.exp(-(y / 0.0035) ** 2)                                   # the fold's ridge
    z += 0.00025 * math.exp(-(y / 0.012) ** 2)
    ex = max(0.0, abs(x) - 0.088) / 0.017; ey = max(0.0, abs(y) - 0.130) / 0.018
    z += 0.0035 * (ex ** 2) * (0.6 + 0.4 * (y < 0)) + 0.0022 * (ey ** 2)        # edges curling up a little
    z += 0.00035 * math.sin(x * 41 + 0.6) * math.sin(y * 23 + 1.3)               # handling waviness
    v.co.z = z
for p in sheet.data.polygons: p.use_smooth = True
sheet.location = (SHEET_LOC[0], SHEET_LOC[1], 0.0032)
sheet.rotation_euler = (0, 0, math.radians(SHEET_ROT))
sm = bpy.data.materials.new('Paper'); nt, b = nodes_of(sm)
pc = image(nt, os.path.join(TEX, 'paper_col.png'), True, (-600, 100))
pb = image(nt, os.path.join(TEX, 'paper_bump.png'), False, (-600, -300))
nt.links.new(pc.outputs['Color'], b.inputs['Base Color'])
setin(b, 'Sheen Weight', 0.3); setin(b, 'Sheen Roughness', 0.5)
setin(b, 'Subsurface Weight', 0.06); setin(b, 'Subsurface Scale', 0.002)
pp = pbr_set(nt, b, dict(rough='Paper006/Paper006_2K-JPG_Roughness.jpg', nor='Paper006/Paper006_2K-JPG_NormalGL.jpg'),
             scale=(0.42, 0.59), normal_strength=0.5, x=-1500)
prr = nt.nodes.new('ShaderNodeMapRange'); prr.location = (-200, -100)
prr.inputs['To Min'].default_value = 0.55; prr.inputs['To Max'].default_value = 0.78
nt.links.new(pp['rough'].outputs['Color'], prr.inputs['Value']); nt.links.new(prr.outputs['Result'], b.inputs['Roughness'])
pbmp = bump(nt, pb.outputs['Color'], 0.45, 0.0004)
nt.links.new(pp['normal'].outputs['Normal'], pbmp.inputs['Normal'])
nt.links.new(pbmp.outputs['Normal'], b.inputs['Normal'])
sheet.data.materials.append(sm)

# ---------------------------------------------------------------- the fountain pen: a generic 1940s lever-filler
#  black celluloid, gold-filled furniture, 14k gold nib. No maker's marks anywhere (on purpose).
#  Built in its own frame: Z runs from the nib's tip (z = 0) to the end of the barrel, the nib's top faces +Y.
from mathutils import Matrix
import numpy as _np

def skin(name, rings, cap0=True, cap1=True, smooth=True):
    """faces between consecutive rings (lists of points, same length, closed loops)."""
    me = bpy.data.meshes.new(name); bm = bmesh.new()
    R = [[bm.verts.new(p) for p in ring] for ring in rings]
    n = len(rings[0])
    for a_, b_ in zip(R, R[1:]):
        for i in range(n):
            bm.faces.new((a_[i], a_[(i + 1) % n], b_[(i + 1) % n], b_[i]))
    if cap0: bm.faces.new(list(reversed(R[0])))
    if cap1: bm.faces.new(R[-1])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(me); bm.free()
    o = obj(name, me)
    for p in me.polygons: p.use_smooth = smooth
    return o

def loft(name, secs, segs=96, **kw):
    """secs: [(z, cx, cy, rx, ry)] -> elliptical rings along Z."""
    rings = []
    for z, cx, cy, rx, ry in secs:
        rings.append([(cx + rx * math.cos(2 * math.pi * k / segs), cy + ry * math.sin(2 * math.pi * k / segs), z)
                      for k in range(segs)])
    return skin(name, rings, **kw)

def turn(name, prof, segs=96, **kw):
    """lathe from [(r, z)]."""
    return loft(name, [(z, 0, 0, r, r) for r, z in prof], segs, **kw)

def interp(pts, z):
    zs = [p[1] for p in pts]; rs = [p[0] for p in pts]
    return float(_np.interp(z, zs, rs))

def sweep(name, path, width, thick, segs=12):
    """a strip (rounded rectangle section) along path = [(P, N)] (point, outward normal), for clip and lever."""
    rings = []
    for k, (P, Nn) in enumerate(path):
        P = Vector(P); Nn = Vector(Nn).normalized()
        T = (Vector(path[min(k + 1, len(path) - 1)][0]) - Vector(path[max(k - 1, 0)][0])).normalized()
        B = T.cross(Nn).normalized(); Nn = B.cross(T).normalized()
        w = width(k / (len(path) - 1)) if callable(width) else width
        ring = []
        for q in range(segs):
            a = 2 * math.pi * q / segs
            c, s_ = math.cos(a), math.sin(a)
            ex = math.copysign(abs(c) ** 0.45, c); ey = math.copysign(abs(s_) ** 0.45, s_)   # superellipse: flat faces, soft edges
            ring.append(P + B * (ex * w / 2) + Nn * (ey * thick / 2))
        rings.append(ring)
    return skin(name, rings)

# -- materials
def celluloid(name, colour, rough_lo=0.02, rough_hi=0.07):
    m = bpy.data.materials.new(name); nt, b = nodes_of(m)
    setin(b, 'Base Color', colour); setin(b, 'Roughness', 0.30); setin(b, 'IOR', 1.49)
    setin(b, 'Coat Weight', 1.0); setin(b, 'Coat IOR', 1.5)
    pl = pbr_set(nt, b, dict(rough='Plastic006/Plastic006_2K-JPG_Roughness.jpg', nor='Plastic006/Plastic006_2K-JPG_NormalGL.jpg'),
                 scale=(9, 9), coords='Object', normal_strength=0.12)
    mr = nt.nodes.new('ShaderNodeMapRange'); mr.location = (-200, -150)
    mr.inputs['To Min'].default_value = rough_lo; mr.inputs['To Max'].default_value = rough_hi
    nt.links.new(pl['rough'].outputs['Color'], mr.inputs['Value']); nt.links.new(mr.outputs['Result'], b.inputs['Coat Roughness'])
    if 'Coat Normal' in b.inputs: nt.links.new(pl['normal'].outputs['Normal'], b.inputs['Coat Normal'])   # hairline scratches in the gloss
    return m

def goldmat(name, r_lo, r_hi, tint=(1.0, 0.77, 0.40)):
    m = bpy.data.materials.new(name); nt, b = nodes_of(m)
    setin(b, 'Metallic', 1.0)
    gm = pbr_set(nt, b, dict(col='Metal048B/Metal048B_2K-JPG_Color.jpg', rough='Metal048B/Metal048B_2K-JPG_Roughness.jpg',
                             nor='Metal048B/Metal048B_2K-JPG_NormalGL.jpg'), scale=(25, 25), coords='Object', normal_strength=0.15)
    mix = nt.nodes.new('ShaderNodeMix'); mix.data_type = 'RGBA'; mix.blend_type = 'MULTIPLY'; mix.location = (-150, 200)
    mix.inputs['Factor'].default_value = 0.35
    nt.links.new(gm['col'].outputs['Color'], mix.inputs['B']); mix.inputs['A'].default_value = (*tint, 1)
    nt.links.new(mix.outputs['Result'], b.inputs['Base Color'])
    mr = nt.nodes.new('ShaderNodeMapRange'); mr.location = (-150, -150)
    mr.inputs['To Min'].default_value = r_lo; mr.inputs['To Max'].default_value = r_hi
    nt.links.new(gm['rough'].outputs['Color'], mr.inputs['Value']); nt.links.new(mr.outputs['Result'], b.inputs['Roughness'])
    nt.links.new(gm['normal'].outputs['Normal'], b.inputs['Normal'])
    return m

CELL = celluloid('PenCelluloid', (0.010, 0.010, 0.011, 1))
GOLD = goldmat('GoldFilled', 0.14, 0.32)                       # worn gold-filled trim
NIBG = goldmat('NibGold14k', 0.07, 0.16, (1.0, 0.80, 0.47))    # the nib, polished by use
EBON = bpy.data.materials.new('Ebonite'); nt, b = nodes_of(EBON)
setin(b, 'Base Color', (0.022, 0.016, 0.012, 1)); setin(b, 'Roughness', 0.42)
TIPM = bpy.data.materials.new('Tipping'); nt, b = nodes_of(TIPM)
setin(b, 'Base Color', (0.62, 0.62, 0.64, 1)); setin(b, 'Metallic', 1.0); setin(b, 'Roughness', 0.12)
SLOT = bpy.data.materials.new('LeverSlot'); nt, b = nodes_of(SLOT)
setin(b, 'Base Color', (0.003, 0.003, 0.003, 1)); setin(b, 'Roughness', 0.6)

pen = bpy.data.objects.new('Pen', None); scene.collection.objects.link(pen); pen.empty_display_size = 0.02
def part(o, mat):
    o.parent = pen; o.data.materials.append(mat); return o

# -- the nib: a curved gold shell, tines, slit, breather hole, tipping
NIB_L, NIB_EXP = 0.024, 0.0170                                  # length, exposed length (the rest is in the section)
HW = [(0.00025, 0.0), (0.00042, 0.00025), (0.00058, 0.0006), (0.00080, 0.002), (0.0015, 0.005), (0.0025, 0.009),
      (0.00325, 0.0125), (0.00350, 0.0145), (0.00345, 0.0155), (0.00335, 0.017), (0.00335, NIB_L)]
def spine(z):                                                   # height of the nib's centre line above the axis
    return 0.00042 + 0.0031 * min(z / NIB_EXP, 1.0) ** 0.85 + (0.0001 * (z - NIB_EXP) / 0.007 if z > NIB_EXP else 0.0)
def curv_r(z):                                                  # radius of the cross curve: near flat at the tip
    t = max(0.0, 1.0 - z / NIB_EXP)
    return 0.0036 + 0.009 * t * t
SLIT_HALF, HOLE_Z, HOLE_R = 0.000045, 0.0074, 0.00052
nb = bmesh.new(); NZ, NX = 300, 140
grid = []
for i in range(NZ + 1):
    z = NIB_L * (i / NZ) ** 1.15                                # denser at the tip
    hw = interp(HW, z); R = curv_r(z); y0 = spine(z); row = []
    for j in range(NX + 1):
        s_ = -1 + 2 * j / NX
        s_ = math.copysign(abs(s_) ** 1.0, s_)
        xf = s_ * hw; ph = xf / R
        row.append(nb.verts.new((R * math.sin(ph), y0 - R * (1 - math.cos(ph)), z)))
    grid.append(row)
for i in range(NZ):
    for j in range(NX):
        vs = (grid[i][j], grid[i + 1][j], grid[i + 1][j + 1], grid[i][j + 1])
        c = sum((v.co for v in vs), Vector()) / 4
        zc = c.z; hw = interp(HW, zc); xf = curv_r(zc) * math.asin(max(-1, min(1, c.x / curv_r(zc))))
        if zc < HOLE_Z and abs(xf) < SLIT_HALF + 0.000004 * (zc / HOLE_Z): continue      # the slit
        if (xf ** 2 + ((zc - HOLE_Z) * 1.0) ** 2) < HOLE_R ** 2: continue               # the breather hole
        nb.faces.new(vs)
bmesh.ops.delete(nb, geom=[v for v in nb.verts if not v.link_faces], context='VERTS')
me = bpy.data.meshes.new('Nib'); nb.to_mesh(me); nb.free()
nib = part(obj('Nib', me), NIBG)
for p in me.polygons: p.use_smooth = True
so = nib.modifiers.new('thickness', 'SOLIDIFY'); so.thickness = 0.00026; so.offset = -1; so.use_even_offset = True
tip = bpy.data.meshes.new('Tipping'); bm = bmesh.new(); bmesh.ops.create_uvsphere(bm, u_segments=32, v_segments=16, radius=1.0)
bm.to_mesh(tip); bm.free()
tipo = part(obj('Tipping', tip), TIPM); tipo.scale = (0.00052, 0.00034, 0.00042); tipo.location = (0, spine(0) - 0.00022, 0.00028)
for p in tip.polygons: p.use_smooth = True

# -- the feed: black ebonite under the nib, rounded nose, comb fins along its underside
fs = []
for z in _np.linspace(0.0022, 0.031, 40):
    t = (z - 0.0022) / 0.0128
    rx = 0.0006 + 0.0019 * min(t, 1) ** 0.7 + (0.0003 * (z - 0.015) / 0.016 if z > 0.015 else 0)
    ry = 0.00035 + 0.0015 * min(t, 1) ** 0.7 + (0.0003 * (z - 0.015) / 0.016 if z > 0.015 else 0)
    fs.append((z, 0.0, spine(z) - 0.00027 - ry, rx, ry))
fs = [(0.0019, 0, fs[0][2], 0.0003, 0.0002)] + fs
part(loft('Feed', fs, 48), EBON)
for k in range(7):                                             # the fins
    z = 0.0090 + k * 0.00095; ry = 0.0012
    fin = loft(f'FeedFin{k}', [(z, 0, spine(z) - 0.0011 - ry * 0.9, 0.0016, ry), (z + 0.00045, 0, spine(z) - 0.0011 - ry * 0.9, 0.0016, ry)], 32)
    part(fin, EBON)

# -- the section: black, tapering to a small flared lip, recessed where nib and feed go in
SEC = [(0.0030, 0.0215), (0.0030, 0.0166), (0.0036, 0.0160), (0.0040, 0.0166), (0.0040, 0.0176), (0.0037, 0.0190),
       (0.0036, 0.0215), (0.0038, 0.0260), (0.0043, 0.0315), (0.0044, 0.0330), (0.0047, 0.0332), (0.0047, 0.0345)]
part(turn('Section', SEC), CELL)
# -- the barrel: streamlined, torpedo end
BAR = [(0.0047, 0.0343), (0.0055, 0.0344), (0.00575, 0.0352), (0.00585, 0.040), (0.0060, 0.060), (0.0060, 0.078),
       (0.00585, 0.092), (0.0055, 0.102), (0.0048, 0.110), (0.0039, 0.1155), (0.0029, 0.1195), (0.0019, 0.1222),
       (0.0010, 0.1238), (0.0, 0.1244)]
part(turn('Barrel', BAR, 128), CELL)
# -- the lever (gold, lying in its slot on the side that faces the camera) and its slot
LEV_A = math.radians(-58)                                      # angle from the nib's top (+Y) round towards the camera
lev_dir = Vector((math.sin(LEV_A), math.cos(LEV_A), 0))
def onbarrel(z, off):
    r = interp(BAR, z) + off
    return (lev_dir * r + Vector((0, 0, z)), lev_dir)
part(sweep('LeverSlot', [onbarrel(z, 0.00002) for z in _np.linspace(0.0585, 0.0862, 40)], 0.0019, 0.00006), SLOT)
part(sweep('Lever', [onbarrel(z, 0.00022) for z in _np.linspace(0.0595, 0.0850, 50)],
           lambda t: 0.00125 - 0.0002 * t, 0.00042), GOLD)
tab_me = bpy.data.meshes.new('LeverTab'); bm = bmesh.new(); bmesh.ops.create_uvsphere(bm, u_segments=32, v_segments=16, radius=1.0)
bm.to_mesh(tab_me); bm.free()
for p in tab_me.polygons: p.use_smooth = True
tab = part(obj('LeverTab', tab_me), GOLD)
P, _ = onbarrel(0.0852, 0.00045); tab.location = P
tab.rotation_euler = (0, 0, math.atan2(-lev_dir.x, lev_dir.y)); tab.scale = (0.00125, 0.00055, 0.0017)

# -- place the pen: lying on the sheet below the names, nib up and pointing to the names, resting on the barrel's
#    widest point and the section's lip (so it tilts a little towards the nib)
PEN_TIP = Vector((-0.026, -0.036))                             # where the nib's tip is, on the desk plane (m)
PEN_ANG = math.radians(-34)                                    # direction from the tip back along the barrel
SURF = 0.0032 + 0.00005
tilt = math.atan((0.0060 - 0.0040) / (0.070 - 0.0172))
zdir = Vector((math.cos(PEN_ANG) * math.cos(tilt), math.sin(PEN_ANG) * math.cos(tilt), math.sin(tilt)))
ydir = (Vector((0, 0, 1)) - zdir * zdir.z).normalized(); xdir = ydir.cross(zdir)
h_tip = SURF + 0.0040 / math.cos(tilt) - 0.0172 * math.sin(tilt)    # axis height at the tip so the lip touches
pen.matrix_world = Matrix.Translation(Vector((PEN_TIP.x, PEN_TIP.y, h_tip))) @ Matrix((xdir, ydir, zdir)).transposed().to_4x4()

# -- the cap, lying beside it: bands, ball-ended clip, a black jewel on top
cap = bpy.data.objects.new('Cap', None); scene.collection.objects.link(cap); cap.empty_display_size = 0.02
def cpart(o, mat):
    o.parent = cap; o.data.materials.append(mat); return o
CAPO = [(0.00605, 0.0), (0.0064, 0.0003), (0.0065, 0.0012), (0.0065, 0.045), (0.0062, 0.052), (0.0054, 0.0575),
        (0.0043, 0.0612), (0.0034, 0.0630)]
cpart(turn('CapShell', [(0.0057, 0.050), (0.0057, 0.0006), (0.0059, 0.0)] + CAPO, 128), CELL)
cpart(turn('CapBandWide', [(0.00648, 0.0032), (0.00662, 0.0034), (0.00664, 0.0040), (0.00660, 0.0044), (0.00664, 0.0048),
                            (0.00662, 0.0074), (0.00648, 0.0076)], 128), GOLD)
cpart(turn('CapBandThin', [(0.00648, 0.0092), (0.00660, 0.0093), (0.00660, 0.0103), (0.00648, 0.0104)], 128), GOLD)
cpart(turn('CapRing', [(0.0033, 0.0628), (0.00345, 0.0632), (0.00345, 0.0640), (0.0031, 0.0644), (0.0, 0.0644)], 96), GOLD)
cpart(turn('CapJewel', [(0.0026, 0.0643), (0.0027, 0.0652), (0.0022, 0.0664), (0.0012, 0.0671), (0.0, 0.0673)], 96), CELL)
# the clip runs from under the ring, over the shoulder, down the side, ending in a ball that touches the cap
cl = []
for z in _np.linspace(0.0636, 0.0205, 70):
    r = interp(CAPO, min(z, 0.0630)) if z <= 0.0630 else 0.0034
    t = (0.0636 - z) / (0.0636 - 0.0205)
    stand = 0.00045 + 0.00055 * math.sin(min(t, 1) * math.pi * 0.85) ** 0.8
    cl.append(((0, r + stand, z), (0, 1, -0.0)))
# follow the shoulder: outward normal of the profile
for k, (P, Nn) in enumerate(cl):
    z = P[2]; dz = 0.0004
    dr = (interp(CAPO, min(z + dz, 0.063)) - interp(CAPO, min(z - dz, 0.063))) / (2 * dz)
    cl[k] = (P, (0, 1, -dr))
cpart(sweep('Clip', cl, lambda t: 0.0036 - 0.0010 * t, 0.00085, 16), GOLD)
ball_me = bpy.data.meshes.new('ClipBall'); bm = bmesh.new(); bmesh.ops.create_uvsphere(bm, u_segments=32, v_segments=16, radius=0.0017)
bm.to_mesh(ball_me); bm.free()
for p in ball_me.polygons: p.use_smooth = True
ball = cpart(obj('ClipBall', ball_me), GOLD); ball.location = (0, 0.0065 + 0.0011, 0.0200); ball.scale = (1.0, 0.9, 1.15)
CAP_OPEN = Vector((0.010, -0.112))                             # the open end, near the pen
CAP_ANG = math.radians(170)                                    # direction from the open end to the top
CLIP_A = math.radians(-70)                                     # clip turned towards the camera, the cap resting on its side
czd = Vector((math.cos(CAP_ANG), math.sin(CAP_ANG), 0)); up = Vector((0, 0, 1))
side = up.cross(czd)                                           # horizontal, perpendicular to the cap
cyd = (up * math.cos(CLIP_A) - side * math.sin(CLIP_A)).normalized(); cxd = cyd.cross(czd)
cap.matrix_world = Matrix.Translation(Vector((CAP_OPEN.x, CAP_OPEN.y, SURF + 0.0065))) @ Matrix((cxd, cyd, czd)).transposed().to_4x4()

# ---------------------------------------------------------------- the inkwell, half out of frame at the top left
glass = bpy.data.materials.new('Glass'); nt, b = nodes_of(glass)
setin(b, 'Base Color', (0.9, 0.95, 0.93, 1)); setin(b, 'Roughness', 0.03); setin(b, 'Transmission Weight', 1.0); setin(b, 'IOR', 1.5)
inkm = bpy.data.materials.new('Ink'); nt, b = nodes_of(inkm)
setin(b, 'Base Color', (0.004, 0.006, 0.02, 1)); setin(b, 'Roughness', 0.05)
well = lathe('Inkwell', [(0.0, 0.0), (0.034, 0.0), (0.036, 0.004), (0.036, 0.030), (0.030, 0.038), (0.014, 0.042),
                         (0.013, 0.050), (0.0, 0.050)])
well.data.materials.append(glass); well.location = (-0.30, 0.34, 0)                 # out of frame: it shows in reflections only
ink_in = lathe('InkInside', [(0.0, 0.002), (0.032, 0.002), (0.032, 0.022), (0.0, 0.022)]); ink_in.location = well.location
ink_in.data.materials.append(inkm)

# ---------------------------------------------------------------- light: the desk lamp (warm), the window (cool), a dark room
def light(name, kind, loc, energy, colour, **kw):
    d = bpy.data.lights.new(name, kind); d.energy = energy; d.color = colour
    for k, v in kw.items(): setattr(d, k, v)
    o = bpy.data.objects.new(name, d); scene.collection.objects.link(o); o.location = loc
    return o

def aim(o, target):
    o.rotation_euler = (Vector(target) - o.location).to_track_quat('-Z', 'Y').to_euler()

lamp = light('Lamp', 'SPOT', (-0.30, 0.21, 0.30), 6.5, (1.0, 0.70, 0.42),          # low enough to rake the paper's texture
             spot_size=math.radians(50), spot_blend=0.85, shadow_soft_size=0.03)
aim(lamp, (0.0, 0.055, 0.0))
win = light('Window', 'AREA', (0.55, -0.20, 0.55), 0.45, (0.62, 0.74, 1.0), size=0.6)
aim(win, (0.0, 0.03, 0.0))
bounce = light('Bounce', 'AREA', (-0.1, -0.45, 0.25), 0.08, (1.0, 0.85, 0.7), size=0.5)
spill = light('DeskSpill', 'AREA', (-0.20, 0.42, 0.55), 0.9, (1.0, 0.70, 0.42), size=0.30)   # so the wood shows (the user, 26/11)
aim(spill, (0.02, 0.27, 0.0))
aim(bounce, (0.0, 0.05, 0.0))
world = bpy.data.worlds.new('Room'); scene.world = world
try: world.use_nodes = True
except Exception: pass
bg = world.node_tree.nodes.get('Background')
if bg: bg.inputs['Color'].default_value = (0.0025, 0.0022, 0.0020, 1); bg.inputs['Strength'].default_value = 1.0

# ---------------------------------------------------------------- camera: 50 mm, slightly tilted, focused on the names
cam_d = bpy.data.cameras.new('Camera'); cam_d.lens = 50; cam_d.sensor_fit = 'VERTICAL'; cam_d.sensor_height = 36
cam = bpy.data.objects.new('Camera', cam_d); scene.collection.objects.link(cam); scene.camera = cam
TARGET = Vector((0.004, 0.060, 0.0)); TILT = math.radians(17); DIST = 0.52
cam.location = TARGET + Vector((0.0, -math.sin(TILT), math.cos(TILT))) * DIST
cam.rotation_euler = (TARGET - cam.location).to_track_quat('-Z', 'Y').to_euler()
FOCUS = (0.004, 0.040, 0.004)                                                    # between the two names
cam_d.dof.use_dof = True; cam_d.dof.aperture_fstop = 8.0
cam_d.dof.focus_distance = (Vector(FOCUS) - cam.location).length

# ---------------------------------------------------------------- render: Cycles, 1080×1920, AgX
r = scene.render
r.engine = 'CYCLES'; r.resolution_x, r.resolution_y, r.resolution_percentage = 1080, 1920, 100
r.film_transparent = False
cy = scene.cycles
cy.device = 'GPU'                                  # uses the GPU set in Preferences ▸ System (CUDA/OptiX), otherwise CPU
cy.samples = 512; cy.use_adaptive_sampling = True; cy.adaptive_threshold = 0.008
cy.use_denoising = True
try: cy.denoiser = 'OPENIMAGEDENOISE'
except Exception: pass
cy.max_bounces = 10; cy.diffuse_bounces = 4; cy.glossy_bounces = 4; cy.transmission_bounces = 8
cy.caustics_reflective = False; cy.caustics_refractive = False
cy.blur_glossy = 1.0
try: cy.tile_size = 512                                # friendlier to a 4 GB card
except Exception: pass
vs = scene.view_settings
vs.view_transform = 'AgX'
for look in ('AgX - Medium High Contrast', 'Medium High Contrast'):
    try: vs.look = look; break
    except Exception: pass
vs.exposure = 0.0
r.image_settings.file_format = 'PNG'; r.image_settings.color_depth = '16'
r.filepath = '//ep05_thumb_render.png'

if PREVIEW:                                          # quick check in the sandbox (CPU)
    cy.device = 'CPU'; cy.samples = int(arg('--samples', 48)); r.resolution_percentage = int(arg('--pct', 50))
    r.filepath = PREVIEW
    bpy.ops.render.render(write_still=True)
    cy.device = 'GPU'; cy.samples = 512; r.resolution_percentage = 100; r.filepath = '//ep05_thumb_render.png'

bpy.ops.file.pack_all()
bpy.ops.wm.save_as_mainfile(filepath=OUT)
print('saved', OUT)
