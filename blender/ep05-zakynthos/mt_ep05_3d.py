# =====================================================================================
#  Mind Twist GR · επ. 05 «Η λίστα με τα δύο ονόματα» · τα τρία κλιπ 3D (1080×1920, 30 fps, EEVEE)
#    B  Η λίστα καίγεται      7,0″  (210 καρέ)   σκηνικό Β: γραφείο Μητροπολίτη, δίσκος, κερί
#    C  Το χαρτί              12,0″ (360 καρέ)   σκηνικό Α: γραφείο διοικητή, πηλήκιο, λάμπα
#    D  Φινάλε (στάχτη)       8,0″  (240 καρέ)   σκηνικό Β, η στάχτη και το χαρτί με τα δύο ονόματα
#  Blender 4.2 LTS ή νεότερο (και 5.x). Όλα σε κώδικα. Τα textures (textures/) τα φτιάχνει το make_textures_ep05.py.
#  Το χρώμα δεν έχει σημασία: το φινίρισμα (grade3d_05.py) τα κάνει μελάνι και χαρτί και κρατά μόνο τη φωτιά.
#
#  ΧΡΗΣΗ (από τερματικό, μέσα στον φάκελο):
#     blender -b -P mt_ep05_3d.py -- --clip C --stills     (καρέ-δείγματα, γρήγορο)
#     blender -b -P mt_ep05_3d.py -- --clip C --render     (όλο το κλιπ)   ·  ίδια για B και D
#  Μέσα στο Blender: άλλαξε το CLIP παρακάτω → Run Script → Ctrl+F12.
#  Αποτελέσματα: render/ep05_3dX_…mp4  (και render/stills_X_*.png)
# =====================================================================================
import bpy, math, os, sys

CLIP     = 'C'        # 'B', 'C' ή 'D' (το --clip στο τερματικό έχει προτεραιότητα)
ENGINE   = 'EEVEE'
PREVIEW  = False      # True = μισή ανάλυση για γρήγορο έλεγχο. Το τελικό ΠΑΝΤΑ με False.
EXPOSURE = 0.3
STYLE    = 'ink'      # 'ink' = κινούμενο σχέδιο μελανιού (σαν τις πλάκες) · 'real' = το ρεαλιστικό της v3
TEX_DIR  = ''
FPS = 30

argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
if '--clip' in argv: CLIP = argv[argv.index('--clip') + 1].upper()
def _arg(name, cast, default=None):                       # δοκιμές: --pct 25 --samples 16 --frames 1,90 --outdir d
    return cast(argv[argv.index(name) + 1]) if name in argv else default
STYLE = _arg('--style', str, STYLE); INK_GAIN = _arg('--inkgain', float, {'B': 3.0, 'C': 4.5, 'D': 5.0}[CLIP])
LENGTH = {'B': 210, 'C': 418, 'D': 240}[CLIP]
# C was 360 frames (12″) and was slowed down in the edit; the user (26/11) chose normal speed: the same motion over
# 418 frames (13.93″, the length of its shot), every key of C at CF(its old frame)
def CF(f): return 1 + (f - 1) * (LENGTH - 1) / 359 if CLIP == 'C' else f
NAME = {'B': 'ep05_3dB_burn', 'C': 'ep05_3dC_paper', 'D': 'ep05_3dD_finale'}[CLIP]

# ---------------------------------------------------------------- βοηθητικά (ίδια με το πρότυπο 1a1p-01)
def find_tex_dir():
    c = []
    if TEX_DIR: c.append(TEX_DIR)
    try: c.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'textures'))
    except NameError: pass
    for t in bpy.data.texts:
        if t.filepath: c.append(os.path.join(os.path.dirname(bpy.path.abspath(t.filepath)), 'textures'))
    if bpy.data.filepath: c.append(os.path.join(os.path.dirname(bpy.data.filepath), 'textures'))
    for d in c:
        if os.path.isfile(os.path.join(d, 'names.png')): return d
    raise RuntimeError('Δεν βρέθηκε ο φάκελος textures/ (τρέξε πρώτα το make_textures_ep05.py).')

TEX = find_tex_dir()
OUTDIR = os.path.join(os.path.dirname(TEX), 'render')
if '--outdir' in argv: OUTDIR = argv[argv.index('--outdir') + 1]
scene = bpy.context.scene
_KEYS = []

def key(target, attr, frame, value, interp='BEZIER'):
    setattr(target, attr, value)
    target.keyframe_insert(attr, frame=frame)
    _KEYS.append((target.id_data, target.path_from_id(attr), frame, interp))

def fcurves_of(idb):
    ad = idb.animation_data
    if ad is None or ad.action is None: return []
    act = ad.action
    fcs = getattr(act, 'fcurves', None)
    if fcs is not None and len(fcs): return list(fcs)
    out = []
    for layer in getattr(act, 'layers', []):
        for strip in layer.strips:
            for bag in getattr(strip, 'channelbags', []): out += list(bag.fcurves)
    return out

def apply_interp():
    groups = {}
    for idb, path, f, ip in _KEYS:
        groups.setdefault((type(idb).__name__, idb.name_full), (idb, {}))[1][(path, f)] = ip
    for idb, table in groups.values():
        for fc in fcurves_of(idb):
            for kp in fc.keyframe_points:
                ip = table.get((fc.data_path, int(round(kp.co[0]))))
                if ip and ip != 'BEZIER': kp.interpolation = ip
            fc.update()

def try_set(obj, attr, val):
    try: setattr(obj, attr, val); return True
    except (AttributeError, TypeError, ValueError): return False

def wipe_scene():
    for ob in list(bpy.data.objects): bpy.data.objects.remove(ob, do_unlink=True)
    for coll in (bpy.data.meshes, bpy.data.materials, bpy.data.lights, bpy.data.cameras,
                 bpy.data.images, bpy.data.actions, bpy.data.worlds, bpy.data.shape_keys):
        for d in list(coll):
            if d.users == 0:
                try: coll.remove(d)
                except (RuntimeError, ReferenceError, AttributeError): pass

def link_obj(ob):
    scene.collection.objects.link(ob); return ob

def empty(name, loc=(0, 0, 0)):
    e = bpy.data.objects.new(name, None); e.location = loc; e.empty_display_size = 0.02
    return link_obj(e)

def mesh_obj(name, verts, faces, uvs=None, mats=(), face_mat=None, smooth=False):
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    if uvs is not None:
        uvl = me.uv_layers.new(name='UVMap')
        for poly in me.polygons:
            for li in poly.loop_indices:
                uvl.data[li].uv = uvs[me.loops[li].vertex_index]
    for m in mats: me.materials.append(m)
    if face_mat is not None:
        for p, i in zip(me.polygons, face_mat): p.material_index = i
    if smooth:
        try: me.shade_smooth()
        except AttributeError:
            for p in me.polygons: p.use_smooth = True
    me.validate(); me.update()
    return link_obj(bpy.data.objects.new(name, me))

def lathe(name, prof, mats, mat_of=lambda i, z: 0, seg=56, scale=(1, 1)):
    verts, faces, fm, rings = [], [], [], []
    for z, r in prof:
        if r < 1e-7: rings.append([len(verts)]); verts.append((0.0, 0.0, z))
        else:
            rings.append(list(range(len(verts), len(verts) + seg)))
            verts += [(scale[0] * r * math.cos(2 * math.pi * k / seg), scale[1] * r * math.sin(2 * math.pi * k / seg), z)
                      for k in range(seg)]
    for i in range(len(rings) - 1):
        A, B = rings[i], rings[i + 1]; m = mat_of(i, prof[i][0])
        if len(A) == 1 and len(B) == 1: continue
        for k in range(seg):
            k1 = (k + 1) % seg
            if len(A) == 1:   faces.append((A[0], B[k1], B[k]))
            elif len(B) == 1: faces.append((A[k], A[k1], B[0]))
            else:             faces.append((A[k], A[k1], B[k1], B[k]))
            fm.append(m)
    ob = mesh_obj(name, verts, faces, mats=mats, face_mat=fm, smooth=True)
    es = ob.modifiers.new('EdgeSplit', 'EDGE_SPLIT'); es.split_angle = math.radians(40)
    return ob

def box(name, sx, sy, sz, z0, mat, bevel=0.0006):
    x, y = sx / 2, sy / 2; z1 = z0 + sz
    v = [(-x, -y, z0), (x, -y, z0), (x, y, z0), (-x, y, z0), (-x, -y, z1), (x, -y, z1), (x, y, z1), (-x, y, z1)]
    f = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    ob = mesh_obj(name, v, f, mats=[mat])
    if bevel:
        b = ob.modifiers.new('Bevel', 'BEVEL'); b.width = bevel; b.segments = 3
    return ob

def track(ob, target, axis='TRACK_NEGATIVE_Z'):
    c = ob.constraints.new('TRACK_TO'); c.target = target; c.track_axis = axis; c.up_axis = 'UP_Y'

def sheet(name, w, h, mat, nx=40, ny=56, flip=False):
    """flat sheet, origin at its centre, UV v=1 at local +y (flip=True: texture turned 180°)."""
    verts, uvs, faces = [], [], []
    for j in range(ny + 1):
        for i in range(nx + 1):
            verts.append(((i / nx - 0.5) * w, (j / ny - 0.5) * h, 0.0)); uvs.append((1 - i / nx, 1 - j / ny) if flip else (i / nx, j / ny))
    for j in range(ny):
        for i in range(nx):
            a = j * (nx + 1) + i; faces.append((a, a + 1, a + nx + 2, a + nx + 1))
    return mesh_obj(name, verts, faces, uvs=uvs, mats=[mat], smooth=True)

# ---------------------------------------------------------------- υλικά
def new_mat(name):
    m = bpy.data.materials.new(name)
    try: m.use_nodes = True
    except AttributeError: pass
    nt = m.node_tree
    for n in list(nt.nodes): nt.nodes.remove(n)
    bs = nt.nodes.new('ShaderNodeBsdfPrincipled')
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    nt.links.new(bs.outputs[0], out.inputs['Surface'])
    return m, nt, bs, out

def setin(node, name, v):
    if name in node.inputs: node.inputs[name].default_value = v

def plug(nt, src, dst):
    if isinstance(src, (int, float, tuple)): dst.default_value = src
    else: nt.links.new(src, dst)

def M(nt, op, a, b=None, clamp=False):
    n = nt.nodes.new('ShaderNodeMath'); n.operation = op; n.use_clamp = clamp
    plug(nt, a, n.inputs[0])
    if b is not None: plug(nt, b, n.inputs[1])
    return n.outputs[0]

def mix_rgb(nt, fac, a, b, blend='MIX'):
    n = nt.nodes.new('ShaderNodeMix'); n.data_type = 'RGBA'; n.blend_type = blend
    fin = [s for s in n.inputs if s.type == 'VALUE'][0]
    cin = [s for s in n.inputs if s.type == 'RGBA']
    plug(nt, fac, fin); plug(nt, a, cin[0]); plug(nt, b, cin[1])
    return [s for s in n.outputs if s.type == 'RGBA'][0]

def value(nt, v, label):
    n = nt.nodes.new('ShaderNodeValue'); n.label = label; n.outputs[0].default_value = v
    return n.outputs[0]

def image(nt, fname, vec=None):
    img = bpy.data.images.load(os.path.join(TEX, fname), check_existing=True)
    n = nt.nodes.new('ShaderNodeTexImage'); n.image = img; n.interpolation = 'Cubic'; n.extension = 'REPEAT'
    if vec is not None: nt.links.new(vec, n.inputs['Vector'])
    return n

def noise(nt, vec, scale, detail=4.0, w=None):
    n = nt.nodes.new('ShaderNodeTexNoise')
    if w is not None: try_set(n, 'noise_dimensions', '4D')
    nt.links.new(vec, n.inputs['Vector']); n.inputs['Scale'].default_value = scale; n.inputs['Detail'].default_value = detail
    if w is not None and 'W' in n.inputs: plug(nt, w, n.inputs['W'])
    return n.outputs['Fac']

def simple(name, color, rough, metal=0.0, coat=0.0):
    m, nt, bs, _ = new_mat(name)
    setin(bs, 'Base Color', (*color, 1)); setin(bs, 'Roughness', rough); setin(bs, 'Metallic', metal)
    setin(bs, 'Coat Weight', coat)
    return m

def transparent_ok(m, blended=False):
    if not try_set(m, 'surface_render_method', 'BLENDED' if blended else 'DITHERED'):
        try_set(m, 'blend_method', 'BLEND' if blended else 'HASHED')
    try_set(m, 'use_transparent_shadow', True)

def mat_paper(name, tex):
    m, nt, bs, _ = new_mat(name)
    t = image(nt, tex); nt.links.new(t.outputs['Color'], bs.inputs['Base Color'])
    setin(bs, 'Roughness', 0.85); setin(bs, 'Specular IOR Level', 0.3)
    tc = nt.nodes.new('ShaderNodeTexCoord')
    bp = nt.nodes.new('ShaderNodeBump'); bp.inputs['Strength'].default_value = 0.08; setin(bp, 'Distance', 0.0002)
    nt.links.new(noise(nt, tc.outputs['UV'], 380, 6), bp.inputs['Height']); nt.links.new(bp.outputs['Normal'], bs.inputs['Normal'])
    return m

def mat_wood():
    m, nt, bs, _ = new_mat('DeskWood')
    tc = nt.nodes.new('ShaderNodeTexCoord'); mp = nt.nodes.new('ShaderNodeMapping'); mp.inputs['Scale'].default_value = (1.6, 1.6, 1)
    nt.links.new(tc.outputs['UV'], mp.inputs['Vector'])
    t = image(nt, 'wood.png', mp.outputs['Vector']); nt.links.new(t.outputs['Color'], bs.inputs['Base Color'])
    setin(bs, 'Roughness', 0.45); setin(bs, 'Coat Weight', 0.3); setin(bs, 'Coat Roughness', 0.2)
    return m

# ---------------------------------------------------------------- ύφος μελανιού (STYLE = 'ink')
# Καμία ρεαλιστική απόδοση: το φως μετριέται (Diffuse → Shader to RGB) και γίνεται ΤΟΝΟΣ, ο τόνος γίνεται
# γραμμές χαρακτικού (μονή, σταυρωτή, συμπαγές μελάνι), σε χαρτί και μελάνι μετρημένα στις πλάκες. Χρώμα μόνο η φωτιά.
INK_PAPER = (1.0, 0.930, 0.807, 1)          # sRGB 255/247/232
INK_INK = (0.0030, 0.0024, 0.0021, 1)       # sRGB 10/8/7

def ink_light(nt):
    d = nt.nodes.new('ShaderNodeBsdfDiffuse'); d.inputs['Color'].default_value = (1, 1, 1, 1)
    s2r = nt.nodes.new('ShaderNodeShaderToRGB'); nt.links.new(d.outputs[0], s2r.inputs[0])
    bw = nt.nodes.new('ShaderNodeRGBToBW'); nt.links.new(s2r.outputs['Color'], bw.inputs[0])
    return bw.outputs[0]

def _lines(nt, coord, angle, freq, width, wob):
    sx = nt.nodes.new('ShaderNodeSeparateXYZ'); nt.links.new(coord, sx.inputs[0])
    a = math.radians(angle)
    u = M(nt, 'ADD', M(nt, 'ADD', M(nt, 'MULTIPLY', sx.outputs['X'], math.cos(a)), M(nt, 'MULTIPLY', sx.outputs['Y'], math.sin(a))),
          M(nt, 'MULTIPLY', sx.outputs['Z'], 0.6))
    u = M(nt, 'ADD', M(nt, 'MULTIPLY', u, freq), wob)
    tri = M(nt, 'MULTIPLY', M(nt, 'ABSOLUTE', M(nt, 'SUBTRACT', M(nt, 'FRACT', u), 0.5)), 2.0)       # 1 on the line, 0 between
    lo = M(nt, 'SUBTRACT', M(nt, 'SUBTRACT', 1.0, width), 0.07)
    mr = nt.nodes.new('ShaderNodeMapRange'); mr.clamp = True
    plug(nt, tri, mr.inputs['Value']); plug(nt, lo, mr.inputs['From Min']); plug(nt, M(nt, 'ADD', lo, 0.14), mr.inputs['From Max'])
    return M(nt, 'MULTIPLY', mr.outputs['Result'], M(nt, 'MULTIPLY', width, 10.0, clamp=True))

def ink_hatch(nt, T, coord, angle=40.0, freq=330.0):
    """tone T (1 = paper, 0 = ink) -> ink coverage, like an etching: line width grows with darkness.
    Lines are laid on the FRAME (window coordinates, ~7 px apart at 1080×1920), not on the objects: far objects
    would otherwise turn to grey mush."""
    tcw = nt.nodes.new('ShaderNodeTexCoord'); mw = nt.nodes.new('ShaderNodeMapping')
    mw.inputs['Scale'].default_value = (1080.0 / 7.0, 1920.0 / 7.0, 0.0); nt.links.new(tcw.outputs['Window'], mw.inputs['Vector'])
    coord = mw.outputs['Vector']; freq = 1.0
    wob = M(nt, 'ADD', M(nt, 'MULTIPLY', M(nt, 'SUBTRACT', noise(nt, coord, 0.012, 2), 0.5), 3.0),
          M(nt, 'MULTIPLY', M(nt, 'SUBTRACT', noise(nt, coord, 0.09, 3), 0.5), 0.7))                 # lines bend, hand-drawn
    w1 = M(nt, 'MULTIPLY', M(nt, 'MULTIPLY', M(nt, 'SUBTRACT', 0.93, T), 1.4, clamp=True), 0.9)
    w2 = M(nt, 'MULTIPLY', M(nt, 'MULTIPLY', M(nt, 'SUBTRACT', 0.40, T), 2.4, clamp=True), 0.9)
    l1 = _lines(nt, coord, angle, freq, w1, wob)
    l2 = _lines(nt, coord, angle + 75.0, freq * 0.95, w2, wob)
    solid = M(nt, 'SUBTRACT', 1.0, sh_range(nt, T, 0.13, 0.24))      # darks go solid black, as in the plates
    return M(nt, 'MAXIMUM', M(nt, 'MAXIMUM', l1, l2), solid)

def ink_emit(nt, inkcov, fire_col=None, fire_mask=None):
    tcw = nt.nodes.new('ShaderNodeTexCoord'); mw = nt.nodes.new('ShaderNodeMapping')
    mw.inputs['Scale'].default_value = (1080.0, 1920.0, 0.0); nt.links.new(tcw.outputs['Window'], mw.inputs['Vector'])
    px = mw.outputs['Vector']
    dry = sh_range(nt, noise(nt, px, 0.35, 3), 0.70, 0.76)                       # ink skipping on the paper grain
    inkcov = M(nt, 'MULTIPLY', inkcov, M(nt, 'SUBTRACT', 1.0, M(nt, 'MULTIPLY', dry, 0.30)))
    grain = M(nt, 'ADD', 0.955, M(nt, 'MULTIPLY', noise(nt, px, 0.02, 4), 0.09))  # uneven paper tone
    paper = mix_rgb(nt, 1.0, INK_PAPER, grain, 'MULTIPLY')
    col = mix_rgb(nt, inkcov, paper, INK_INK)
    if fire_mask is not None: col = mix_rgb(nt, fire_mask, col, fire_col)
    em = nt.nodes.new('ShaderNodeEmission'); nt.links.new(col, em.inputs['Color']); em.inputs['Strength'].default_value = 1.0
    return em.outputs[0]

def ink_material(name, albedo, tex=None, angle=40.0, freq=330.0, coord='Object', tex_vec=None):
    m, nt, bs, out = new_mat(name); nt.nodes.remove(bs)
    tc = nt.nodes.new('ShaderNodeTexCoord')
    alb = albedo
    if tex is not None:
        t = image(nt, tex, tex_vec(nt, tc) if tex_vec else None)
        bw = nt.nodes.new('ShaderNodeRGBToBW'); nt.links.new(t.outputs['Color'], bw.inputs[0])
        alb = M(nt, 'MULTIPLY', bw.outputs[0], albedo)
    T = M(nt, 'MULTIPLY', M(nt, 'MULTIPLY', ink_light(nt), alb), INK_GAIN)
    nt.links.new(ink_emit(nt, ink_hatch(nt, T, tc.outputs[coord], angle, freq)), out.inputs['Surface'])
    return m

_OUTLINE = []
def outline_mat():
    if not _OUTLINE:
        m, nt, bs, out = new_mat('InkOutline'); nt.nodes.remove(bs)
        em = nt.nodes.new('ShaderNodeEmission'); em.inputs['Color'].default_value = INK_INK
        nt.links.new(em.outputs[0], out.inputs['Surface']); m.use_backface_culling = True
        _OUTLINE.append(m)
    return _OUTLINE[0]

def add_outline(ob, t=0.0012):
    """inverted hull: the ink contour of an object (any Blender version, renders in EEVEE)."""
    if STYLE != 'ink': return ob
    n = len(ob.data.materials); ob.data.materials.append(outline_mat())
    so = ob.modifiers.new('InkOutline', 'SOLIDIFY'); so.thickness = t; so.offset = 1.0
    so.use_flip_normals = True; so.use_rim = False; so.material_offset = n
    return ob

if STYLE == 'ink':
    def simple(name, color, rough, metal=0.0, coat=0.0):
        lum = 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2]
        return ink_material(name, max(0.06, min(1.0, lum * 1.8 + (0.25 if metal else 0.0))))
    def mat_paper(name, tex):
        return ink_material(name, 1.0, tex=tex, coord='UV', freq=150.0)
    def mat_wood():
        def vec(nt, tc):
            mp = nt.nodes.new('ShaderNodeMapping'); mp.inputs['Scale'].default_value = (1.6, 1.6, 1)
            nt.links.new(tc.outputs['UV'], mp.inputs['Vector']); return mp.outputs['Vector']
        return ink_material('DeskWood', 3.5, tex='wood.png', angle=0.0, freq=300.0, tex_vec=vec)
    _lathe0, _box0 = lathe, box
    def lathe(*a, **k): return add_outline(_lathe0(*a, **k))
    def box(*a, **k): return add_outline(_box0(*a, **k))

# ---------------------------------------------------------------- φωτιά: ΕΝΑ πεδίο καύσης για όλα
# Κάθε σημείο της σελίδας έχει μια σταθερή «ώρα ανάφλεξης» burn_t (Python, παρακάτω). Το μέτωπο P(t) προχωρά και
# age = P − burn_t. Από το age διαβάζουν ΟΛΑ: το χρώμα (καψάλισμα → άκρη που φλέγεται → κάρβουνο → στάχτη),
# η γεωμετρία (κυρτώνει, σπάει), οι φλόγες (γεννιούνται πάνω στην άκρη και σβήνουν λίγο πίσω της) και οι καύτρες
# (σταθερές στη θέση τους: σβήνουν, δεν ταξιδεύουν). Έτσι φλόγα και χαρτί δεν μπορούν να ξεσυγχρονιστούν.
def _h(ix, iy, s):
    h = (ix * 374761393 + iy * 668265263 + s * 1442695041) & 0xffffffff
    h = ((h ^ (h >> 13)) * 1274126177) & 0xffffffff
    return (h ^ (h >> 16)) / 4294967296.0

def vnoise(x, y, s):
    ix, iy = math.floor(x), math.floor(y); fx, fy = x - ix, y - iy
    fx, fy = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)
    a, b = _h(ix, iy, s), _h(ix + 1, iy, s); c, d = _h(ix, iy + 1, s), _h(ix + 1, iy + 1, s)
    return a + (b - a) * fx + (c - a) * fy + (a - b - c + d) * fx * fy

def fbm(x, y, s, oct=4):
    v, amp, tot = 0.0, 1.0, 0.0
    for o in range(oct):
        v += amp * vnoise(x, y, s + o * 17); tot += amp; x, y, amp = x * 2.03, y * 2.03, amp * 0.5
    return v / tot

def smooth(e0, e1, x):
    t = min(1.0, max(0.0, (x - e0) / (e1 - e0))); return t * t * (3 - 2 * t)

def burn_page(name, W, H, corner, mat, nx=110, ny=156):
    """the list page with its static burn attributes (all in the page's local space, metres)."""
    ob = sheet(name, W, H, mat, nx, ny, flip=True)
    me = ob.data; diag = math.hypot(W, H)
    import random as _r
    rnd = _r.Random(11)
    T, SEED, SEED2, ANCH, BORD, CN, HN, WV = [], [], [], [], [], [], [], []
    for v in me.vertices:
        x, y = v.co.x, v.co.y
        d = math.hypot(x - corner[0], y - corner[1]) / diag
        f1 = fbm(x / 0.055, y / 0.055, 3); f2 = fbm(x / 0.018, y / 0.018, 7, 3)
        T.append(max(0.0, d * (1.0 + 0.55 * (f1 - 0.5)) + 0.05 * (f2 - 0.5)))      # fingers, uneven speed
        SEED.append(rnd.random()); SEED2.append(rnd.random()); ANCH.append(1.0 if rnd.random() < 0.09 else 0.0)
        b = max(abs(x) / (W / 2), abs(y) / (H / 2)); BORD.append(smooth(0.55, 1.0, b))
        CN.append(fbm(x / 0.012, y / 0.012, 23, 3)); HN.append(fbm(x / 0.010, y / 0.010, 31, 3)); WV.append(fbm(x / 0.04, y / 0.04, 41, 3))
    for nm, vals in (('burn_t', T), ('seed', SEED), ('seed2', SEED2), ('anchor', ANCH), ('border', BORD),
                     ('crinkle', CN), ('holes', HN), ('wave', WV)):
        a = me.attributes.new(nm, 'FLOAT', 'POINT'); a.data.foreach_set('value', vals)
    return ob, T

def gn_math(nt, op, a, b=None, clamp=False):
    return M(nt, op, a, b, clamp)

def gn_range(nt, x, a, b, smooth_=True):
    n = nt.nodes.new('ShaderNodeMapRange'); n.clamp = True
    if smooth_: n.interpolation_type = 'SMOOTHSTEP'
    plug(nt, x, n.inputs['Value']); n.inputs['From Min'].default_value = a; n.inputs['From Max'].default_value = b
    n.inputs['To Min'].default_value = 0.0; n.inputs['To Max'].default_value = 1.0
    return n.outputs['Result']

def gn_attr(nt, name):
    n = nt.nodes.new('GeometryNodeInputNamedAttribute'); n.data_type = 'FLOAT'; n.inputs['Name'].default_value = name
    return n.outputs['Attribute']

def gn_store(nt, geo, name, val):
    n = nt.nodes.new('GeometryNodeStoreNamedAttribute'); n.data_type = 'FLOAT'; n.domain = 'POINT'
    nt.links.new(geo, n.inputs['Geometry']); n.inputs['Name'].default_value = name; plug(nt, val, n.inputs['Value'])
    return n.outputs['Geometry']

def gn_xyz(nt, x, y, z):
    n = nt.nodes.new('ShaderNodeCombineXYZ')
    for s, v in zip(n.inputs, (x, y, z)): plug(nt, v, s)
    return n.outputs[0]

def build_burn_gn(page, cam, card, smoke):
    """geometry nodes on the page: age, curl, broken char, flames instanced exactly on the burning edge."""
    ng = bpy.data.node_groups.new('BurnGN', 'GeometryNodeTree')
    ng.interface.new_socket('Geometry', in_out='INPUT', socket_type='NodeSocketGeometry')
    ng.interface.new_socket('Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')
    nt = ng; gi = nt.nodes.new('NodeGroupInput'); go = nt.nodes.new('NodeGroupOutput')
    P = nt.nodes.new('ShaderNodeValue'); P.label = 'front'; P.outputs[0].default_value = 0.0
    age = gn_math(nt, 'SUBTRACT', P.outputs[0], gn_attr(nt, 'burn_t'))
    geo = gn_store(nt, gi.outputs[0], 'age', age)
    # shape: the burnt part lifts and crinkles, the sheet's borders curl up, the live edge rises a little
    lift = gn_range(nt, age, 0.0, 0.18)
    bord = gn_attr(nt, 'border')
    dz = gn_math(nt, 'MULTIPLY', lift, gn_math(nt, 'ADD', gn_math(nt, 'ADD', gn_math(nt, 'MULTIPLY', gn_attr(nt, 'crinkle'), 0.008),
                                                                     gn_math(nt, 'MULTIPLY', gn_attr(nt, 'wave'), 0.012)),
                                                gn_math(nt, 'MULTIPLY', gn_math(nt, 'MULTIPLY', bord, bord), 0.012)))
    dz = gn_math(nt, 'ADD', dz, gn_math(nt, 'MULTIPLY', gn_range(nt, age, 0.2, 0.7), gn_math(nt, 'MULTIPLY', gn_math(nt, 'POWER', bord, 1.5), 0.022)))
    edge = gn_math(nt, 'MULTIPLY', gn_range(nt, age, -0.01, 0.02), gn_math(nt, 'SUBTRACT', 1.0, gn_range(nt, age, 0.03, 0.15)))
    dz = gn_math(nt, 'ADD', dz, gn_math(nt, 'MULTIPLY', edge, 0.0025))
    sp = nt.nodes.new('GeometryNodeSetPosition'); nt.links.new(geo, sp.inputs['Geometry'])
    nt.links.new(gn_xyz(nt, 0.0, 0.0, dz), sp.inputs['Offset']); geo = sp.outputs['Geometry']
    page_geo = geo
    # flames: anchors (6 % of the vertices) light when the front reaches them and die a little behind it
    seed = gn_attr(nt, 'seed'); seed2 = gn_attr(nt, 'seed2')
    die = gn_math(nt, 'ADD', 0.014, gn_math(nt, 'MULTIPLY', seed, 0.030))
    life = gn_math(nt, 'MULTIPLY', gn_range(nt, age, 0.0, 0.006),
                   gn_math(nt, 'SUBTRACT', 1.0, gn_range(nt, gn_math(nt, 'SUBTRACT', age, die), 0.0, 0.022)))
    fgeo = gn_store(nt, geo, 'life', life)
    sel = gn_math(nt, 'MULTIPLY', gn_attr(nt, 'anchor'), gn_math(nt, 'GREATER_THAN', life, 0.02))
    mp = nt.nodes.new('GeometryNodeMeshToPoints'); mp.mode = 'VERTICES'
    nt.links.new(fgeo, mp.inputs['Mesh']); plug(nt, sel, mp.inputs['Selection'])
    pts = mp.outputs['Points']
    # size per flame, in metres (the card is 1×1, base at z=0)
    L = gn_attr(nt, 'life'); s1 = gn_attr(nt, 'seed'); s2 = gn_attr(nt, 'seed2')
    h = gn_math(nt, 'MULTIPLY', gn_math(nt, 'POWER', L, 0.7), gn_math(nt, 'ADD', 0.006, gn_math(nt, 'MULTIPLY', gn_math(nt, 'POWER', s2, 2.5), 0.048)))
    w = gn_math(nt, 'ADD', gn_math(nt, 'MULTIPLY', h, gn_math(nt, 'ADD', 0.55, gn_math(nt, 'MULTIPLY', s1, 0.30))), 0.003)
    # face the camera (turn about z only)
    ci = nt.nodes.new('GeometryNodeObjectInfo'); ci.transform_space = 'RELATIVE'; ci.inputs['Object'].default_value = cam
    pos = nt.nodes.new('GeometryNodeInputPosition')
    dv = nt.nodes.new('ShaderNodeVectorMath'); dv.operation = 'SUBTRACT'
    nt.links.new(ci.outputs['Location'], dv.inputs[0]); nt.links.new(pos.outputs[0], dv.inputs[1])
    sx = nt.nodes.new('ShaderNodeSeparateXYZ'); nt.links.new(dv.outputs[0], sx.inputs[0])
    yaw = gn_math(nt, 'SUBTRACT', gn_math(nt, 'ARCTAN2', sx.outputs['Y'], sx.outputs['X']), math.pi / 2)
    rot = gn_xyz(nt, 0.0, 0.0, yaw)
    try:
        e2r = nt.nodes.new('FunctionNodeEulerToRotation'); nt.links.new(rot, e2r.inputs[0]); rot = e2r.outputs[0]
    except RuntimeError:
        pass
    co = nt.nodes.new('GeometryNodeObjectInfo'); co.transform_space = 'ORIGINAL'; co.inputs['Object'].default_value = card
    ip = nt.nodes.new('GeometryNodeInstanceOnPoints')
    nt.links.new(pts, ip.inputs['Points']); nt.links.new(co.outputs['Geometry'], ip.inputs['Instance'])
    nt.links.new(rot, ip.inputs['Rotation']); nt.links.new(gn_xyz(nt, w, w, h), ip.inputs['Scale'])
    ri = nt.nodes.new('GeometryNodeRealizeInstances'); nt.links.new(ip.outputs['Instances'], ri.inputs[0])
    # smoke: a taller, wider, faint card above every flame (same points, so it rises from the edge too)
    so = nt.nodes.new('GeometryNodeObjectInfo'); so.transform_space = 'ORIGINAL'; so.inputs['Object'].default_value = smoke
    ip2 = nt.nodes.new('GeometryNodeInstanceOnPoints')
    nt.links.new(pts, ip2.inputs['Points']); nt.links.new(so.outputs['Geometry'], ip2.inputs['Instance'])
    nt.links.new(rot, ip2.inputs['Rotation'])
    nt.links.new(gn_xyz(nt, gn_math(nt, 'MULTIPLY', w, 1.8), gn_math(nt, 'MULTIPLY', w, 1.8), gn_math(nt, 'MULTIPLY', h, 3.2)), ip2.inputs['Scale'])
    ti = nt.nodes.new('GeometryNodeTranslateInstances'); nt.links.new(ip2.outputs['Instances'], ti.inputs['Instances'])
    nt.links.new(gn_xyz(nt, 0.0, 0.0, gn_math(nt, 'MULTIPLY', h, 0.9)), ti.inputs['Translation']); try_set(ti, 'use_local_space', False)
    ri2 = nt.nodes.new('GeometryNodeRealizeInstances'); nt.links.new(ti.outputs['Instances'], ri2.inputs[0])
    jg = nt.nodes.new('GeometryNodeJoinGeometry')
    nt.links.new(ri2.outputs[0], jg.inputs[0]); nt.links.new(ri.outputs[0], jg.inputs[0]); nt.links.new(page_geo, jg.inputs[0])
    nt.links.new(jg.outputs[0], go.inputs[0])
    mod = page.modifiers.new('Burn', 'NODES'); mod.node_group = ng
    return P.outputs[0]

def sh_attr(nt, name):
    n = nt.nodes.new('ShaderNodeAttribute'); n.attribute_type = 'GEOMETRY'; n.attribute_name = name
    return n.outputs['Fac']

def sh_range(nt, x, a, b):
    n = nt.nodes.new('ShaderNodeMapRange'); n.clamp = True; n.interpolation_type = 'SMOOTHSTEP'
    plug(nt, x, n.inputs['Value']); n.inputs['From Min'].default_value = a; n.inputs['From Max'].default_value = b
    return n.outputs['Result']

def mat_char():
    """the list page: paper → scorch → glowing edge → char with embers (fixed in place) → grey ash."""
    m, nt, bs, out = new_mat('ListBurn')
    tc = nt.nodes.new('ShaderNodeTexCoord'); uv = tc.outputs['UV']
    tex = image(nt, 'list.png')
    TW = value(nt, 0.0, 'time'); EG = value(nt, 1.0, 'ember_gain'); ED = value(nt, 0.0, 'ember_dying'); EK = value(nt, 0.0, 'ember_kill')
    A = sh_attr(nt, 'age')
    sc = nt.nodes.new('ShaderNodeMapping'); sc.inputs['Scale'].default_value = (1.0, 1.41, 1.0); nt.links.new(uv, sc.inputs['Vector'])
    U = sc.outputs['Vector']                                                         # square-ish uv for noise
    A2 = M(nt, 'ADD', A, M(nt, 'MULTIPLY', M(nt, 'SUBTRACT', noise(nt, U, 55, 4), 0.5), 0.010))
    scorch = sh_range(nt, A2, -0.030, 0.0)
    charf = sh_range(nt, A2, 0.0, 0.010)
    ash = M(nt, 'MULTIPLY', sh_range(nt, A2, 0.20, 0.8), sh_range(nt, noise(nt, U, 38, 6), 0.55, 0.70))
    col = mix_rgb(nt, sh_range(nt, scorch, 0.0, 0.6), tex.outputs['Color'], (0.42, 0.25, 0.10, 1))
    col = mix_rgb(nt, sh_range(nt, scorch, 0.5, 1.0), col, (0.09, 0.045, 0.02, 1))
    col = mix_rgb(nt, charf, col, (0.017, 0.015, 0.014, 1))
    col = mix_rgb(nt, M(nt, 'MULTIPLY', ash, 0.7), col, (0.12, 0.115, 0.11, 1))
    nt.links.new(col, bs.inputs['Base Color'])
    nt.links.new(M(nt, 'ADD', 0.88, M(nt, 'MULTIPLY', charf, -0.18)), bs.inputs['Roughness'])
    # the burning edge: a thin bright rim exactly at age≈0
    rim = M(nt, 'SUBTRACT', 1.0, sh_range(nt, M(nt, 'ABSOLUTE', M(nt, 'SUBTRACT', A2, 0.002)), 0.0, 0.0045))
    # embers: cracks + specks, STATIC (uv), dying with age at an uneven rate; flicker only changes brightness
    nv = nt.nodes.new('ShaderNodeTexNoise'); nt.links.new(U, nv.inputs['Vector']); nv.inputs['Scale'].default_value = 7.0
    dvec = nt.nodes.new('ShaderNodeVectorMath'); dvec.operation = 'MULTIPLY_ADD'
    nt.links.new(nv.outputs['Color'], dvec.inputs[0]); dvec.inputs[1].default_value = (0.09, 0.09, 0.09); nt.links.new(U, dvec.inputs[2])
    vor = nt.nodes.new('ShaderNodeTexVoronoi'); try_set(vor, 'feature', 'DISTANCE_TO_EDGE'); nt.links.new(dvec.outputs[0], vor.inputs['Vector'])
    vor.inputs['Scale'].default_value = 24.0
    cracks = M(nt, 'SUBTRACT', 1.0, sh_range(nt, vor.outputs['Distance'], 0.0, 0.03))
    cracks = M(nt, 'MULTIPLY', cracks, sh_range(nt, noise(nt, U, 11, 3), 0.55, 0.70))
    specks = sh_range(nt, noise(nt, U, 90, 2), 0.64, 0.72)
    mask = M(nt, 'MAXIMUM', cracks, specks)
    ns = noise(nt, U, 6, 2)
    decay = M(nt, 'SUBTRACT', 1.0, sh_range(nt, M(nt, 'MULTIPLY', A2, M(nt, 'ADD', 0.6, ns)), 0.02, 0.20))
    live = M(nt, 'MULTIPLY', M(nt, 'MULTIPLY', decay, EG), sh_range(nt, A2, 0.002, 0.02))
    keep = M(nt, 'GREATER_THAN', noise(nt, U, 30, 1), EK)                              # D: embers go out one by one
    dying = M(nt, 'MULTIPLY', M(nt, 'MULTIPLY', ED, keep), charf)
    dying = M(nt, 'MULTIPLY', dying, sh_range(nt, noise(nt, U, 3.5, 2), 0.48, 0.62))      # in a few clusters, not glitter
    flick = M(nt, 'ADD', 0.7, M(nt, 'MULTIPLY', noise(nt, U, 4, 1, w=TW), 0.6))
    emb = M(nt, 'MULTIPLY', M(nt, 'MULTIPLY', M(nt, 'MAXIMUM', live, dying), mask), flick)
    setin(bs, 'Emission Color', (1.0, 0.22, 0.03, 1))
    after = M(nt, 'MULTIPLY', M(nt, 'MULTIPLY', sh_range(nt, A2, 0.0, 0.004), M(nt, 'SUBTRACT', 1.0, sh_range(nt, A2, 0.004, 0.035))),
              M(nt, 'MULTIPLY', M(nt, 'ADD', 0.4, noise(nt, U, 25, 3)), EG))                # char just behind the edge glows red
    glow = M(nt, 'ADD', M(nt, 'ADD', M(nt, 'MULTIPLY', rim, 3.5), M(nt, 'MULTIPLY', emb, 6.0)), M(nt, 'MULTIPLY', after, 2.5))
    nt.links.new(glow, bs.inputs['Emission Strength'])
    bp = nt.nodes.new('ShaderNodeBump'); bp.inputs['Strength'].default_value = 0.2; setin(bp, 'Distance', 0.0004)
    nt.links.new(M(nt, 'MULTIPLY', noise(nt, U, 90, 5), charf), bp.inputs['Height']); nt.links.new(bp.outputs['Normal'], bs.inputs['Normal'])
    # old char breaks into holes with soft, irregular edges
    nf = nt.nodes.new('ShaderNodeTexNoise'); nt.links.new(U, nf.inputs['Vector']); nf.inputs['Scale'].default_value = 4.0
    dv2 = nt.nodes.new('ShaderNodeVectorMath'); dv2.operation = 'MULTIPLY_ADD'
    nt.links.new(nf.outputs['Color'], dv2.inputs[0]); dv2.inputs[1].default_value = (0.12, 0.12, 0.12); nt.links.new(U, dv2.inputs[2])
    vf = nt.nodes.new('ShaderNodeTexVoronoi'); try_set(vf, 'feature', 'DISTANCE_TO_EDGE'); nt.links.new(dv2.outputs[0], vf.inputs['Vector'])
    vf.inputs['Scale'].default_value = 8.0
    cw = M(nt, 'MULTIPLY', sh_range(nt, A2, 0.25, 1.1), M(nt, 'ADD', 0.012, M(nt, 'MULTIPLY', noise(nt, U, 20, 3), 0.03)))
    crack = M(nt, 'SUBTRACT', 1.0, sh_range(nt, M(nt, 'SUBTRACT', vf.outputs['Distance'], cw), -0.004, 0.004))
    crack = M(nt, 'MULTIPLY', crack, sh_range(nt, noise(nt, U, 5, 2), 0.52, 0.60))
    thr = M(nt, 'SUBTRACT', 0.74, M(nt, 'MULTIPLY', sh_range(nt, A2, 0.12, 0.9), 0.10))
    pit = M(nt, 'MULTIPLY', sh_range(nt, A2, 0.10, 0.16), sh_range(nt, M(nt, 'SUBTRACT', noise(nt, U, 16, 5), thr), 0.0, 0.015))
    hole = M(nt, 'MAXIMUM', crack, pit)
    surf = bs.outputs[0]
    if STYLE == 'ink':
        bw = nt.nodes.new('ShaderNodeRGBToBW'); nt.links.new(col, bw.inputs[0])
        T = M(nt, 'MULTIPLY', M(nt, 'MULTIPLY', ink_light(nt), M(nt, 'MULTIPLY', bw.outputs[0], 1.25)), INK_GAIN)
        hc = nt.nodes.new('ShaderNodeMapping'); hc.inputs['Scale'].default_value = (0.172, 0.2435, 0.0); nt.links.new(uv, hc.inputs['Vector'])
        inkcov = ink_hatch(nt, T, hc.outputs['Vector'], 40.0, 330.0)
        hl = M(nt, 'MULTIPLY', M(nt, 'MULTIPLY', charf, M(nt, 'SUBTRACT', 1.0, sh_range(nt, vor.outputs['Distance'], 0.0, 0.02))),
               sh_range(nt, noise(nt, U, 8, 2), 0.55, 0.66))                                  # woodcut: pale scratches in the black
        inkcov = M(nt, 'MULTIPLY', inkcov, M(nt, 'SUBTRACT', 1.0, M(nt, 'MULTIPLY', hl, 0.5)))
        fm = sh_range(nt, glow, 0.45, 0.85)
        fcol = mix_rgb(nt, sh_range(nt, glow, 1.8, 2.4), (0.95, 0.22, 0.02, 1), (1.0, 0.70, 0.22, 1))
        surf = ink_emit(nt, inkcov, fcol, fm)
    tr = nt.nodes.new('ShaderNodeBsdfTransparent'); ms = nt.nodes.new('ShaderNodeMixShader')
    nt.links.new(hole, ms.inputs[0]); nt.links.new(surf, ms.inputs[1]); nt.links.new(tr.outputs[0], ms.inputs[2])
    nt.links.new(ms.outputs[0], out.inputs['Surface'])
    transparent_ok(m)
    return m, TW, EG, ED, EK

def mat_fire(name, fixed_life=None):
    """flame card: rising turbulence (noise scrolls up), tongues that detach, red → orange → yellow."""
    m, nt, bs, out = new_mat(name)
    for n in list(nt.nodes):
        if n.type == 'BSDF_PRINCIPLED': nt.nodes.remove(n)
    tc = nt.nodes.new('ShaderNodeTexCoord'); sep = nt.nodes.new('ShaderNodeSeparateXYZ')
    nt.links.new(tc.outputs['UV'], sep.inputs[0]); u, v = sep.outputs['X'], sep.outputs['Y']
    TW = value(nt, 0.0, 'time')
    sd = M(nt, 'MULTIPLY', sh_attr(nt, 'seed'), 17.0) if fixed_life is None else 3.3
    life = sh_attr(nt, 'life') if fixed_life is None else fixed_life
    def scroll(sx, sy, speed, off):
        c = nt.nodes.new('ShaderNodeCombineXYZ')
        plug(nt, M(nt, 'ADD', M(nt, 'MULTIPLY', u, sx), sd), c.inputs[0])
        plug(nt, M(nt, 'SUBTRACT', M(nt, 'MULTIPLY', v, sy), M(nt, 'MULTIPLY', TW, speed)), c.inputs[1])
        plug(nt, M(nt, 'ADD', sd, off), c.inputs[2])
        return c.outputs[0]
    n1 = noise(nt, scroll(1.6, 1.3, 2.6, 0.0), 2.2, 3)
    n2 = noise(nt, scroll(2.5, 2.0, 4.0, 5.0), 3.5, 2)
    uw = M(nt, 'ADD', u, M(nt, 'MULTIPLY', M(nt, 'SUBTRACT', n1, 0.5), M(nt, 'MULTIPLY', v, 1.3 if STYLE == 'ink' else 0.9)))
    wid = M(nt, 'ADD', M(nt, 'MULTIPLY', M(nt, 'MULTIPLY', M(nt, 'POWER', v, 0.3), M(nt, 'POWER', M(nt, 'SUBTRACT', 1.0, v, clamp=True), 1.1 if STYLE == 'ink' else 0.75)), 0.62 if STYLE == 'ink' else 0.95), 0.002)
    shape = M(nt, 'SUBTRACT', 1.0, M(nt, 'DIVIDE', M(nt, 'ABSOLUTE', M(nt, 'SUBTRACT', uw, 0.5)), wid), clamp=True)
    top = M(nt, 'SUBTRACT', 1.0, sh_range(nt, M(nt, 'ADD', v, M(nt, 'MULTIPLY', M(nt, 'SUBTRACT', n2, 0.5), 0.9 if STYLE == 'ink' else 0.7)),
                                          0.30 if STYLE == 'ink' else 0.45, 0.88 if STYLE == 'ink' else 0.95))
    base = sh_range(nt, v, 0.0, 0.04)
    dens = M(nt, 'MULTIPLY', M(nt, 'MULTIPLY', M(nt, 'POWER', shape, 0.7), top), base)
    I = M(nt, 'MULTIPLY', dens, M(nt, 'POWER', life, 0.5))
    col = mix_rgb(nt, sh_range(nt, I, 0.05, 0.45), (0.85, 0.10, 0.01, 1), (1.0, 0.42, 0.06, 1))
    col = mix_rgb(nt, sh_range(nt, I, 0.45, 0.95), col, (1.0, 0.78, 0.38, 1))
    strength = M(nt, 'ADD', 0.5, M(nt, 'MULTIPLY', I, 1.5)); alpha = M(nt, 'MULTIPLY', I, 1.25, clamp=True)
    if STYLE == 'ink':
        def step(x, e): return sh_range(nt, x, e - 0.015, e + 0.015)
        col = mix_rgb(nt, step(I, 0.24), (0.95, 0.30, 0.03, 1), (1.0, 0.42, 0.05, 1))
        col = mix_rgb(nt, step(I, 0.46), col, (1.0, 0.66, 0.18, 1))
        col = mix_rgb(nt, step(I, 0.70), col, (1.0, 0.92, 0.70, 1))
        strength = 1.0; alpha = M(nt, 'MAXIMUM', step(I, 0.20), M(nt, 'MULTIPLY', sh_range(nt, I, 0.03, 0.20), 0.35))
    em = nt.nodes.new('ShaderNodeEmission'); nt.links.new(col, em.inputs['Color'])
    plug(nt, strength, em.inputs['Strength'])
    tr = nt.nodes.new('ShaderNodeBsdfTransparent'); ms = nt.nodes.new('ShaderNodeMixShader')
    plug(nt, alpha, ms.inputs[0])
    nt.links.new(tr.outputs[0], ms.inputs[1]); nt.links.new(em.outputs[0], ms.inputs[2])
    nt.links.new(ms.outputs[0], out.inputs['Surface'])
    transparent_ok(m, blended=True)
    try_set(m, 'use_backface_culling', False)
    return m, TW

def mat_smoke(name):
    """thin grey smoke: slow rising noise, very low opacity, fades out at the top."""
    m, nt, bs, out = new_mat(name)
    tc = nt.nodes.new('ShaderNodeTexCoord'); sep = nt.nodes.new('ShaderNodeSeparateXYZ')
    nt.links.new(tc.outputs['UV'], sep.inputs[0]); u, v = sep.outputs['X'], sep.outputs['Y']
    TW = value(nt, 0.0, 'time'); sd = M(nt, 'MULTIPLY', sh_attr(nt, 'seed'), 13.0)
    c = nt.nodes.new('ShaderNodeCombineXYZ')
    plug(nt, M(nt, 'ADD', M(nt, 'MULTIPLY', u, 1.2), sd), c.inputs[0])
    plug(nt, M(nt, 'SUBTRACT', v, M(nt, 'MULTIPLY', TW, 0.9)), c.inputs[1]); plug(nt, sd, c.inputs[2])
    n = noise(nt, c.outputs[0], 3.0, 4)
    uw = M(nt, 'ADD', u, M(nt, 'MULTIPLY', M(nt, 'SUBTRACT', n, 0.5), M(nt, 'MULTIPLY', v, 1.2)))
    side = M(nt, 'SUBTRACT', 1.0, sh_range(nt, M(nt, 'ABSOLUTE', M(nt, 'SUBTRACT', uw, 0.5)), 0.05, 0.45))
    vert = M(nt, 'MULTIPLY', sh_range(nt, v, 0.0, 0.25), M(nt, 'SUBTRACT', 1.0, sh_range(nt, v, 0.45, 1.0)))
    a = M(nt, 'MULTIPLY', M(nt, 'MULTIPLY', M(nt, 'MULTIPLY', side, vert), sh_range(nt, n, 0.35, 0.75)),
          M(nt, 'MULTIPLY', sh_attr(nt, 'life'), 0.16))
    setin(bs, 'Base Color', (0.02, 0.019, 0.018, 1)); setin(bs, 'Roughness', 1.0)
    sm = bs.outputs[0]
    if STYLE == 'ink':
        em = nt.nodes.new('ShaderNodeEmission'); em.inputs['Color'].default_value = (0.62, 0.58, 0.52, 1); sm = em.outputs[0]
        a = 0.0                                    # no smoke in the drawing
    tr = nt.nodes.new('ShaderNodeBsdfTransparent'); ms = nt.nodes.new('ShaderNodeMixShader')
    plug(nt, a, ms.inputs[0]); nt.links.new(tr.outputs[0], ms.inputs[1]); nt.links.new(sm, ms.inputs[2])
    nt.links.new(ms.outputs[0], out.inputs['Surface'])
    transparent_ok(m, blended=True); try_set(m, 'use_backface_culling', False)
    return m, TW

def flame_card(name, w, h, mat):
    """vertical card in the xz plane, base at z=0 (faces ±y)."""
    v = [(-w / 2, 0, 0), (w / 2, 0, 0), (w / 2, 0, h), (-w / 2, 0, h)]
    return mesh_obj(name, v, [(0, 1, 2, 3)], uvs=[(0, 0), (1, 0), (1, 1), (0, 1)], mats=[mat])

# ---------------------------------------------------------------- σκηνή
wipe_scene()
r = scene.render
r.resolution_x, r.resolution_y = 1080, 1920
r.resolution_percentage = _arg('--pct', int, 50 if PREVIEW else 100)
r.fps = FPS; r.fps_base = 1.0; scene.frame_start, scene.frame_end = 1, LENGTH

world = bpy.data.worlds.new('Room'); scene.world = world
try: world.use_nodes = True
except AttributeError: pass
bg = world.node_tree.nodes.get('Background') or world.node_tree.nodes.new('ShaderNodeBackground')
bg.inputs['Color'].default_value = (0.9, 0.92, 1.0, 1); bg.inputs['Strength'].default_value = 0.035
wo = world.node_tree.nodes.get('World Output') or world.node_tree.nodes.new('ShaderNodeOutputWorld')
world.node_tree.links.new(bg.outputs[0], wo.inputs['Surface'])
if STYLE == 'ink':                     # the camera sees ink where there is nothing; the world still lights the room
    wn = world.node_tree.nodes; lp = wn.new('ShaderNodeLightPath'); inkbg = wn.new('ShaderNodeBackground')
    inkbg.inputs['Color'].default_value = INK_INK; wmix = wn.new('ShaderNodeMixShader')
    world.node_tree.links.new(lp.outputs['Is Camera Ray'], wmix.inputs[0])
    world.node_tree.links.new(bg.outputs[0], wmix.inputs[1]); world.node_tree.links.new(inkbg.outputs[0], wmix.inputs[2])
    world.node_tree.links.new(wmix.outputs[0], wo.inputs['Surface'])

desk = sheet('Desk', 1.4, 1.0, mat_wood(), nx=1, ny=1)
cam_data = bpy.data.cameras.new('Cam'); cam_data.sensor_fit = 'AUTO'; cam_data.sensor_width = 36
cam_data.clip_start = 0.005; cam_data.clip_end = 20
cam = link_obj(bpy.data.objects.new('Cam', cam_data)); scene.camera = cam
AIM = empty('CamAim'); track(cam, AIM)
FOCUS = empty('Focus'); cam_data.dof.use_dof = True; cam_data.dof.focus_object = FOCUS
try_set(cam_data.dof, 'aperture_blades', 7)

def light(name, kind, loc, energy, color, **kw):
    ld = bpy.data.lights.new(name, kind); ld.energy = energy; ld.color = color
    for k, v in kw.items(): try_set(ld, k, v)
    ob = link_obj(bpy.data.objects.new(name, ld)); ob.location = loc
    return ob

brass = simple('Brass', (0.42, 0.32, 0.17), 0.35, metal=1.0)
black = simple('BlackLacquer', (0.012, 0.011, 0.012), 0.2, coat=0.8)
names_mat = mat_paper('NamesPaper', 'names.png')

def names_paper(loc, rot_deg):
    p = sheet('NamesPaper', 0.15, 0.21, names_mat, 30, 42)
    p.location = loc; p.rotation_euler = (0, 0, math.radians(rot_deg))
    s = p.modifiers.new('Thickness', 'SOLIDIFY'); s.thickness = 0.0003; s.offset = -1
    return add_outline(p, 0.0008)

def paper_local(p_loc, rot_deg, lx, ly, z=0.0004):
    """point on a paper (local x, y in metres) -> world."""
    a = math.radians(rot_deg)
    return (p_loc[0] + lx * math.cos(a) - ly * math.sin(a), p_loc[1] + lx * math.sin(a) + ly * math.cos(a), z)

# ================================================================= CLIP C · ΤΟ ΧΑΡΤΙ
if CLIP == 'C':
    cam_data.lens = 35; cam_data.dof.aperture_fstop = 1.8
    # props on the commander's desk (officer side = +y, camera behind it; visitors' side = −y)
    cloth = simple('CapCloth', (0.10, 0.11, 0.10), 0.85)
    cap = lathe('Cap', [(0.0, 0), (0.0, 0.085), (0.04, 0.088), (0.055, 0.095), (0.075, 0.13), (0.083, 0.13),
                        (0.088, 0.11), (0.09, 0)], [cloth])
    cap.location = (-0.20, 0.10, 0.0); cap.rotation_euler = (0, 0, math.radians(25))
    visor = lathe('Visor', [(0.0, 0), (0.0, 0.08), (0.004, 0.1), (0.006, 0)], [black], scale=(1, 0.55))
    visor.parent = cap; visor.location = (0, -0.05, 0.002); visor.rotation_euler = (math.radians(-12), 0, 0)
    if STYLE == 'ink':                 # drawn, the lathe cap reads as a bowl in every version: leave it out
        bpy.data.objects.remove(visor, do_unlink=True); bpy.data.objects.remove(cap, do_unlink=True)
    glass = simple('InkGlass', (0.02, 0.02, 0.025), 0.05, coat=1.0)
    ink = lathe('Inkwell', [(0, 0), (0, 0.032), (0.035, 0.032), (0.04, 0.012), (0.05, 0.012), (0.05, 0)], [glass])
    ink.location = (0.09, 0.22, 0)
    pen = lathe('Pen', [(0, 0), (0.0004, 0.0006), (0.012, 0.0034), (0.0125, 0.0042), (0.124, 0.0055), (0.128, 0.004), (0.13, 0)],
                [brass, black], mat_of=lambda i, z: 0 if z < 0.0122 else 1)
    pen.location = (0.10, 0.14, 0.0055); pen.rotation_euler = (math.radians(90), 0, math.radians(-70))
    stack = box('Orders', 0.21, 0.297, 0.012, 0.0, mat_paper('OrdersPaper', 'list.png'), 0.0004)
    stack.location = (0.10, -0.25, 0); stack.rotation_euler = (0, 0, math.radians(-8))
    # desk lamp (shade above the paper, its spot makes the pool of light)
    base = lathe('LampBase', [(0, 0), (0, 0.07), (0.015, 0.07), (0.02, 0.02), (0.085, 0.012), (0.09, 0)], [black])
    base.location = (0.26, 0.05, 0)
    shade = lathe('LampShade', [(0, 0.16), (0.10, 0.03), (0.11, 0)], [black]); shade.location = (0.15, 0.05, 0.47)
    shade.rotation_euler = (0, math.radians(25), 0)
    light('LampSpot', 'SPOT', (0.12, 0.04, 0.28), 2.0, (1.0, 0.86, 0.66), spot_size=math.radians(70),
          spot_blend=0.5, shadow_soft_size=0.03)
    light('Window', 'AREA', (-0.6, -0.4, 0.7), 6, (0.75, 0.82, 1.0), size=0.8)
    ob = scene.objects['Window']; track(ob, empty('WinAim', (0, 0, 0)))
    ob2 = scene.objects['LampSpot']; track(ob2, empty('LampAim', (0.0, 0.03, 0)))

    PF = (0.0, 0.05, 0.0006); ROT = 180 - 4                   # final place; text upright for the officer/camera
    paper = names_paper((0.02, -0.40, 0.0006), 180 + 9)
    key(paper, 'location', CF(1), (0.02, -0.40, 0.0006)); key(paper, 'rotation_euler', CF(1), (0, 0, math.radians(180 + 9)))
    key(paper, 'location', CF(60), (0.02, -0.40, 0.0006)); key(paper, 'rotation_euler', CF(60), (0, 0, math.radians(180 + 9)))
    key(paper, 'location', CF(138), PF); key(paper, 'rotation_euler', CF(138), (0, 0, math.radians(ROT)))
    # camera: wide over the desk, then a slow push to the names (the hook reuses the last 7.9″)
    for f, c, a in ((1, (0.02, 0.62, 0.40), (0.0, -0.06, 0)), (140, (0.0, 0.58, 0.37), (0.0, -0.02, 0)),
                    (170, (0.0, 0.56, 0.35), (0.0, 0.0, 0)), (290, (0.0, 0.25, 0.17), PF), (360, (0.0, 0.235, 0.16), PF)):
        key(cam, 'location', CF(f), c); key(AIM, 'location', CF(f), a)
    L1 = paper_local(PF, ROT, 0.0, 0.045); L2 = paper_local(PF, ROT, 0.0, 0.031)
    for f, fo in ((1, (-0.20, 0.10, 0.06)), (55, (-0.20, 0.10, 0.06)), (100, (0.02, -0.2, 0.0)),
                  (140, PF), (240, L1), (285, L1), (315, L2), (360, L2)):
        key(FOCUS, 'location', CF(f), fo)

# ================================================================= CLIP B / D · Η ΛΙΣΤΑ, Η ΣΤΑΧΤΗ
else:
    cam_data.lens = 50 if CLIP == 'B' else 35; cam_data.dof.aperture_fstop = 2.8 if CLIP == 'B' else 2.8
    tray = lathe('Tray', [(0, 0), (0, 0.20), (0.004, 0.205), (0.018, 0.215), (0.02, 0.212), (0.006, 0.19), (0.004, 0)],
                 [simple('TrayBrass', (0.20, 0.15, 0.08), 0.55, metal=1.0)])
    wax = simple('Wax', (0.36, 0.32, 0.25), 0.6)
    candle = lathe('Candle', [(0, 0), (0, 0.018), (0.16, 0.018), (0.162, 0.016), (0.162, 0)], [wax])
    holder = lathe('Holder', [(0, 0), (0, 0.045), (0.006, 0.045), (0.012, 0.022), (0.03, 0.022), (0.03, 0)], [brass])
    holder.location = (0.33, -0.30, 0); candle.location = (0.33, -0.30, 0.012)
    cmat, cTW = mat_fire('CandleFlame', fixed_life=0.8)
    cfl = flame_card('CandleFlame', 0.014, 0.040, cmat); cfl.location = (0.33, -0.30, 0.172)
    lt = cfl.constraints.new('LOCKED_TRACK'); lt.target = cam; lt.track_axis = 'TRACK_NEGATIVE_Y'; lt.lock_axis = 'LOCK_Z'
    cl = light('CandleLight', 'POINT', (0.33, -0.30, 0.215), 0.3, (1.0, 0.62, 0.28), shadow_soft_size=0.01)
    light('Room', 'AREA', (-0.5, 0.6, 0.8), 3.0, (0.75, 0.82, 1.0), size=0.8)
    track(scene.objects['Room'], empty('RoomAim', (0, 0, 0)))
    icon = box('Icon', 0.16, 0.012, 0.22, 0.0, simple('IconBoard', (0.25, 0.18, 0.08), 0.5), 0.002)
    icon.location = (0.0, -0.36, 0); icon.rotation_euler = (math.radians(-12), 0, math.radians(10))
    if STYLE == 'ink':
        wall = sheet('Wall', 1.6, 0.9, simple('WallPlaster', (0.30, 0.29, 0.27), 0.9), 1, 1)
        wall.location = (0.0, -0.62, 0.45); wall.rotation_euler = (math.radians(90), 0, 0)

    # the list page in the tray, burning from its far-right corner (as the camera sees it)
    W_, H_ = 0.21 * 0.82, 0.297 * 0.82
    CORNER = (-W_ / 2, -H_ / 2)
    PROT = -8.0; PAGE_Z = 0.0065
    bmat, TW, EG, ED, EK = mat_char()
    page, BT = burn_page('ListPage', W_, H_, CORNER, bmat)
    page.location = (0.0, 0.0, PAGE_Z); page.rotation_euler = (0, 0, math.radians(PROT))
    fmat, fTW = mat_fire('Flame')
    card = flame_card('FlameCard', 1.0, 1.0, fmat); card.location = (0, 0, -3.0)          # source only, kept off-stage
    card.hide_render = True
    smat, sTW = mat_smoke('Smoke')
    scard = flame_card('SmokeCard', 1.0, 1.0, smat); scard.location = (0, 0, -3.2); scard.hide_render = True
    Pv = build_burn_gn(page, cam, card, scard)
    PV = [v.co.copy() for v in page.data.vertices]

    def on_page(lx, ly, z):
        a = math.radians(PROT)
        return (lx * math.cos(a) - ly * math.sin(a), lx * math.sin(a) + ly * math.cos(a), PAGE_Z + z)

    if CLIP == 'B':
        P0, P1 = 0.0, 0.80
        def Pof(f): return P0 + (P1 - P0) * ((f - 1) / (LENGTH - 1)) ** 1.25
    else:
        def Pof(f): return 1.6
    for f in list(range(1, LENGTH + 1, 6)) + [LENGTH]:
        key(Pv, 'default_value', f, Pof(f), 'LINEAR')
    for s_ in (TW, fTW, cTW, sTW):
        key(s_, 'default_value', 1, 0.0, 'LINEAR'); key(s_, 'default_value', LENGTH, LENGTH / 30.0, 'LINEAR')

    if CLIP == 'B':
        # the fire's light follows the real burning edge (same field as the flames)
        fire = light('FireLight', 'POINT', on_page(CORNER[0] + 0.02, CORNER[1] + 0.02, 0.04), 0.0, (1.0, 0.45, 0.12),
                     shadow_soft_size=0.03)
        for f in range(1, LENGTH + 1, 3):
            P = Pof(f); xs = ys = 0.0; n = 0
            for co, t in zip(PV, BT):
                a = P - t
                if 0.0 < a < 0.035: xs += co.x; ys += co.y; n += 1
            if n:
                key(fire, 'location', f, on_page(xs / n, ys / n, 0.06), 'LINEAR')
            e = min(0.55, 0.10 + n / 2000.0) * (0.85 + 0.15 * math.sin(f * 1.9) + 0.08 * math.sin(f * 0.6))
            key(fire.data, 'energy', f, e, 'LINEAR')
        for f, ex in ((1, (-0.02, 0.40, 0.31)), (210, (0.0, 0.31, 0.25))):
            key(cam, 'location', f, ex)
        key(AIM, 'location', 1, on_page(CORNER[0] * 0.4, CORNER[1] * 0.4, 0)); key(AIM, 'location', 210, (0.0, 0.0, 0.0))
        key(FOCUS, 'location', 1, on_page(CORNER[0] + 0.03, CORNER[1] + 0.03, 0)); key(FOCUS, 'location', 210, on_page(-0.01, -0.03, 0))
    else:
        # D: everything has burnt; the last embers go out one by one; then up and back to the tray and the names
        key(EG, 'default_value', 1, 0.0, 'CONSTANT')
        for f, d_, k_ in ((1, 1.0, 0.63), (90, 0.8, 0.72), (150, 0.55, 0.80), (195, 0.3, 0.88), (225, 0.0, 1.0)):
            key(ED, 'default_value', f, d_); key(EK, 'default_value', f, k_)
        np_ = names_paper((0.30, -0.05, 0.0006), 180 - 14)
        for f, c, a in ((1, (0.03, 0.15, 0.11), (0.01, -0.02, 0.0)), (150, (0.12, 0.38, 0.36), (0.13, -0.04, 0.0)),
                        (240, (0.16, 0.50, 0.52), (0.16, -0.04, 0.0))):
            key(cam, 'location', f, c); key(AIM, 'location', f, a)
        for f, fo in ((1, (0.0, -0.02, 0.0)), (110, (0.05, -0.04, 0.0)), (170, (0.30, -0.05, 0.0)), (240, (0.30, -0.05, 0.0))):
            key(FOCUS, 'location', f, fo)
    cgain = 3.0 if (STYLE == 'ink' and CLIP == 'D') else 1.0
    for f in range(1, LENGTH + 1, 2):                                                   # the candle flickers
        key(cl.data, 'energy', f, cgain * (0.27 + 0.07 * abs(math.sin(f * 1.7)) + 0.05 * abs(math.sin(f * 0.53))), 'LINEAR')

# ---------------------------------------------------------------- render
if ENGINE == 'CYCLES':
    r.engine = 'CYCLES'; try_set(scene.cycles, 'samples', 64 if PREVIEW else 256)
else:
    for e in ('BLENDER_EEVEE_NEXT', 'BLENDER_EEVEE'):
        if try_set(r, 'engine', e): break
    ee = scene.eevee
    try_set(ee, 'taa_render_samples', _arg('--samples', int, 24 if PREVIEW else 80))
    for a, v in (('use_raytracing', True), ('use_shadows', True), ('shadow_ray_count', 2), ('shadow_step_count', 8),
                 ('use_gtao', True), ('use_bloom', True)):
        try_set(ee, a, v)
if not (try_set(r, 'use_motion_blur', True) and try_set(r, 'motion_blur_shutter', 0.5)):
    try_set(scene.eevee, 'use_motion_blur', True)
vs = scene.view_settings
try_set(vs, 'view_transform', 'Standard'); try_set(vs, 'look', 'None'); vs.exposure = EXPOSURE
if STYLE == 'ink':                     # drawings have no lens blur or motion blur; colours must come out exactly
    vs.exposure = 0.0; cam_data.dof.use_dof = False
    try_set(r, 'use_motion_blur', False); try_set(scene.eevee, 'use_motion_blur', False)
    try_set(scene.eevee, 'use_raytracing', False); try_set(scene.eevee, 'use_gtao', False)
    try_set(scene.eevee, 'taa_render_samples', _arg('--samples', int, 16))

def out_png():
    ims = r.image_settings
    if hasattr(ims, 'media_type'): ims.media_type = 'IMAGE'
    ims.file_format = 'PNG'

def out_video():
    os.makedirs(OUTDIR, exist_ok=True); ims = r.image_settings
    try:
        if hasattr(ims, 'media_type'): ims.media_type = 'VIDEO'
        ims.file_format = 'FFMPEG'
        r.ffmpeg.format = 'MPEG4'; r.ffmpeg.codec = 'H264'
        try_set(r.ffmpeg, 'constant_rate_factor', 'HIGH'); try_set(r.ffmpeg, 'ffmpeg_preset', 'GOOD')
        r.filepath = os.path.join(OUTDIR, NAME + '_')
    except (AttributeError, TypeError):
        out_png(); r.filepath = os.path.join(OUTDIR, NAME + '_frames', 'f_')

out_video()
apply_interp()
scene.frame_set(1)
if bpy.app.background:
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(os.path.dirname(TEX), NAME + '.blend'))
if '--stills' in argv:
    out_png(); os.makedirs(OUTDIR, exist_ok=True)
    picks = {'B': (1, 60, 120, 200), 'C': (1, 80, 150, 240, 300, 355), 'D': (1, 90, 170, 235)}[CLIP]
    if '--frames' in argv: picks = [int(x) for x in argv[argv.index('--frames') + 1].split(',')]
    for f in picks:
        # SUBF: llvmpipe (sandbox) hangs on a few frames (B 176); 0.02 of a frame later renders and looks the same
        scene.frame_set(f, subframe=float(os.environ.get('SUBF', '0'))); r.filepath = os.path.join(OUTDIR, 'stills_%s_%03d.png' % (CLIP, f))
        bpy.ops.render.render(write_still=True)
    out_video()
if '--render' in argv:
    bpy.ops.render.render(animation=True)
print('Mind Twist GR · επ. 05 · κλιπ', CLIP, '·', LENGTH, 'καρέ ·', TEX)
