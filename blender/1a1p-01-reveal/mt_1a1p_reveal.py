# =====================================================================================
#  Mind Twist GR · ΜΙΑ ΑΛΗΘΕΙΑ | ΕΝΑ ΨΕΜΑ · 01 · Η αποκάλυψη σε 3D · v2 (10″, 1080×1920, 30 fps)
#  v2: παλέτα και φως του 2D original, Standard αντί για AgX, σωστές καμπύλες κίνησης (η σφραγίδα δεν «προφαίνεται»).
#  Blender 4.2 LTS ή νεότερο (και 5.x). Όλα σε κώδικα: γεωμετρία, υλικά, φως, κίνηση.
#  Τα 4 textures του φακέλου textures/ τα ζωγράφισε η δική μας 2D μηχανή (καμία εικόνα AI).
#
#  ΧΡΗΣΗ (α) μέσα στο Blender: Scripting → Open → αυτό το αρχείο → Run Script.
#            Η σκηνή χτίζεται. F12 = ένα καρέ, Ctrl+F12 = όλο το βίντεο.
#        (β) από τερματικό, χωρίς να ανοίξει παράθυρο:
#            blender -b -P mt_1a1p_reveal.py -- --stills     (7 καρέ-δείγματα σε PNG, γρήγορο τεστ)
#            blender -b -P mt_1a1p_reveal.py -- --render     (όλο το mp4)
#        Τα αποτελέσματα πάνε στον φάκελο render/ δίπλα στο script.
# =====================================================================================
import bpy, math, os, sys

# ---------------------------------------------------------------- ρυθμίσεις
ENGINE   = 'EEVEE'     # 'EEVEE' (δευτερόλεπτα/καρέ) ή 'CYCLES' (πιο φυσικό φως, λεπτά/καρέ)
PREVIEW  = False       # True = μισή ανάλυση και λίγα samples, για γρήγορο έλεγχο
EXPOSURE = 0.0         # αν βγει σκοτεινό/φωτεινό: +0.5 / -0.5 κ.ο.κ.
TEX_DIR  = ''          # άδειο = ο φάκελος textures/ δίπλα στο script

FPS, F0, F1 = 30, 1, 300
IMPACT = 112           # το καρέ που η σφραγίδα ακουμπά το χαρτί

# ---------------------------------------------------------------- βοηθητικά
def find_tex_dir():
    c = []
    if TEX_DIR: c.append(TEX_DIR)
    try: c.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'textures'))
    except NameError: pass
    for t in bpy.data.texts:
        if t.filepath: c.append(os.path.join(os.path.dirname(bpy.path.abspath(t.filepath)), 'textures'))
    if bpy.data.filepath: c.append(os.path.join(os.path.dirname(bpy.data.filepath), 'textures'))
    for d in c:
        if os.path.isfile(os.path.join(d, 'card_A.png')): return d
    raise RuntimeError('Δεν βρέθηκε ο φάκελος textures/. Βάλε τη διαδρομή του στο TEX_DIR.')

TEX = find_tex_dir()
OUTDIR = os.path.join(os.path.dirname(TEX), 'render')

scene = bpy.context.scene
_KEYS = []

def key(target, attr, frame, value, interp='BEZIER'):
    """θέτει τιμή και κλειδί· το interp ορίζει την καμπύλη ΜΕΤΑ από αυτό το κλειδί (εφαρμόζεται στο τέλος)."""
    setattr(target, attr, value)
    target.keyframe_insert(attr, frame=frame)
    _KEYS.append((target.id_data, target.path_from_id(attr), frame, interp))

def fcurves_of(idb):
    ad = idb.animation_data
    if ad is None or ad.action is None: return []
    act = ad.action
    fcs = getattr(act, 'fcurves', None)                 # Blender 4.x
    if fcs is not None and len(fcs): return list(fcs)
    out = []                                            # Blender 4.4+/5.x: layers → strips → channelbags
    for layer in getattr(act, 'layers', []):
        for strip in layer.strips:
            for bag in getattr(strip, 'channelbags', []): out += list(bag.fcurves)
    return out

def apply_interp():
    groups = {}
    for idb, path, f, ip in _KEYS:
        groups.setdefault((type(idb).__name__, idb.name_full), (idb, {}))[1][(path, f)] = ip
    n = 0
    for idb, table in groups.values():
        for fc in fcurves_of(idb):
            for kp in fc.keyframe_points:
                ip = table.get((fc.data_path, int(round(kp.co[0]))))
                if ip and ip != 'BEZIER': kp.interpolation = ip; n += 1
            fc.update()
    print('interpolation set on', n, 'keys')

def wipe_scene():
    for ob in list(bpy.data.objects): bpy.data.objects.remove(ob, do_unlink=True)
    for coll in (bpy.data.meshes, bpy.data.materials, bpy.data.lights, bpy.data.cameras,
                 bpy.data.images, bpy.data.actions, bpy.data.worlds):
        for d in list(coll):
            if d.users == 0: coll.remove(d)

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

def lathe(name, prof, mats, mat_of=lambda i, z: 0, seg=56):
    """σώμα εκ περιστροφής γύρω από τον Z· prof = [(z, r)], r=0 → κορυφή/κέντρο καπακιού."""
    verts, faces, fm, rings = [], [], [], []
    for z, r in prof:
        if r < 1e-7: rings.append([len(verts)]); verts.append((0.0, 0.0, z))
        else:
            rings.append(list(range(len(verts), len(verts) + seg)))
            verts += [(r * math.cos(2 * math.pi * k / seg), r * math.sin(2 * math.pi * k / seg), z) for k in range(seg)]
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
    es = ob.modifiers.new('EdgeSplit', 'EDGE_SPLIT'); es.split_angle = math.radians(40)   # καθαρές ακμές στα σκαλοπάτια
    return ob

def box(name, sx, sy, sz, z0, mat, bevel=0.0006):
    x, y = sx / 2, sy / 2; z1 = z0 + sz
    v = [(-x, -y, z0), (x, -y, z0), (x, y, z0), (-x, y, z0), (-x, -y, z1), (x, -y, z1), (x, y, z1), (-x, y, z1)]
    f = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    ob = mesh_obj(name, v, f, mats=[mat])
    if bevel:
        b = ob.modifiers.new('Bevel', 'BEVEL'); b.width = bevel; b.segments = 3
        try: b.harden_normals = True
        except AttributeError: pass
    return ob

def track(ob, target):
    c = ob.constraints.new('TRACK_TO'); c.target = target
    c.track_axis = 'TRACK_NEGATIVE_Z'; c.up_axis = 'UP_Y'

# ---------------------------------------------------------------- υλικά (nodes)
def new_mat(name):
    m = bpy.data.materials.new(name)
    try: m.use_nodes = True
    except AttributeError: pass
    nt = m.node_tree
    for n in list(nt.nodes): nt.nodes.remove(n)
    bs = nt.nodes.new('ShaderNodeBsdfPrincipled'); bs.location = (300, 0)
    out = nt.nodes.new('ShaderNodeOutputMaterial'); out.location = (600, 0)
    nt.links.new(bs.outputs[0], out.inputs['Surface'])
    return m, nt, bs

def setin(node, name, v):
    if name in node.inputs: node.inputs[name].default_value = v

def plug(nt, src, dst):
    if isinstance(src, (int, float)): dst.default_value = src
    elif isinstance(src, tuple): dst.default_value = src
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

def image(nt, fname, noncolor=False):
    img = bpy.data.images.load(os.path.join(TEX, fname), check_existing=True)
    if noncolor:
        for cs in ('Non-Color', 'Raw', 'Linear Rec.709'):
            try: img.colorspace_settings.name = cs; break
            except TypeError: pass
    n = nt.nodes.new('ShaderNodeTexImage'); n.image = img; n.interpolation = 'Cubic'; n.extension = 'EXTEND'
    return n

def simple(name, color, rough, metal=0.0, coat=0.0):
    m, nt, bs = new_mat(name)
    setin(bs, 'Base Color', (*color, 1)); setin(bs, 'Roughness', rough); setin(bs, 'Metallic', metal)
    setin(bs, 'Coat Weight', coat); setin(bs, 'Coat Roughness', 0.08)
    return m

def paper_bump(nt, bs, strength=0.10):
    tc = nt.nodes.new('ShaderNodeTexCoord')
    nz = nt.nodes.new('ShaderNodeTexNoise'); nz.inputs['Scale'].default_value = 420; nz.inputs['Detail'].default_value = 6
    nt.links.new(tc.outputs['UV'], nz.inputs['Vector'])
    bp = nt.nodes.new('ShaderNodeBump'); bp.inputs['Strength'].default_value = strength
    setin(bp, 'Distance', 0.0002)
    nt.links.new(nz.outputs['Fac'], bp.inputs['Height']); nt.links.new(bp.outputs['Normal'], bs.inputs['Normal'])

STAMP_RED = (0.434, 0.025, 0.021)      # το κόκκινο της 2D σφραγίδας, σε γραμμικό χώρο
INK       = (0.023, 0.017, 0.016)      # το μελάνι της 2D μηχανής

def mat_card_A():
    m, nt, bs = new_mat('Card_A')
    tex = image(nt, 'card_A.png'); mask = image(nt, 'stamp_mask_A.png', True)
    imprint = value(nt, 0.0, 'imprint')                                          # 0 → 1 τη στιγμή της επαφής
    fac = M(nt, 'MULTIPLY', M(nt, 'MULTIPLY', mask.outputs['Color'], imprint), 0.93)
    col = mix_rgb(nt, fac, tex.outputs['Color'], (*STAMP_RED, 1), 'MULTIPLY')
    nt.links.new(col, bs.inputs['Base Color'])
    setin(bs, 'Roughness', 0.82); setin(bs, 'Specular IOR Level', 0.35)
    paper_bump(nt, bs)
    return m, imprint

def mat_card_B():
    m, nt, bs = new_mat('Card_B')
    tex = image(nt, 'card_B.png'); wipe = image(nt, 'x_wipe_B.png', True)
    sep = nt.nodes.new('ShaderNodeSeparateColor'); nt.links.new(wipe.outputs['Color'], sep.inputs[0])
    P = value(nt, 0.0, 'pen_progress'); dark = value(nt, 0.0, 'darken'); wet = value(nt, 0.0, 'wet_ink')
    front = M(nt, 'MULTIPLY', M(nt, 'SUBTRACT', P, sep.outputs[0]), 50.0, clamp=True)   # μαλακό μέτωπο 0.02
    ink = M(nt, 'MULTIPLY', front, sep.outputs[1])
    base = mix_rgb(nt, dark, tex.outputs['Color'], (0, 0, 0, 1))
    col = mix_rgb(nt, M(nt, 'MULTIPLY', ink, 0.92), base, (*INK, 1))
    nt.links.new(col, bs.inputs['Base Color'])
    rough = M(nt, 'ADD', M(nt, 'MULTIPLY', M(nt, 'MULTIPLY', ink, wet), -0.62), 0.82)  # φρέσκο μελάνι γυαλίζει
    nt.links.new(rough, bs.inputs['Roughness']); setin(bs, 'Specular IOR Level', 0.35)
    paper_bump(nt, bs)
    return m, P, dark, wet

def mat_wood(name, dark, light, rough, stretch=(3, 70, 70), scale=1.0):
    m, nt, bs = new_mat(name)
    tc = nt.nodes.new('ShaderNodeTexCoord'); mp = nt.nodes.new('ShaderNodeMapping')
    mp.inputs['Scale'].default_value = stretch
    nt.links.new(tc.outputs['Object'], mp.inputs['Vector'])
    g = nt.nodes.new('ShaderNodeTexNoise'); g.inputs['Scale'].default_value = scale
    g.inputs['Detail'].default_value = 9; setin(g, 'Distortion', 1.6)
    nt.links.new(mp.outputs['Vector'], g.inputs['Vector'])
    ramp = nt.nodes.new('ShaderNodeValToRGB')
    e = ramp.color_ramp.elements; e[0].position = 0.30; e[0].color = (*dark, 1); e[1].position = 0.72; e[1].color = (*light, 1)
    nt.links.new(g.outputs['Fac'], ramp.inputs['Fac']); nt.links.new(ramp.outputs['Color'], bs.inputs['Base Color'])
    setin(bs, 'Roughness', rough); setin(bs, 'Coat Weight', 0.25); setin(bs, 'Coat Roughness', 0.18)
    bp = nt.nodes.new('ShaderNodeBump'); bp.inputs['Strength'].default_value = 0.05; setin(bp, 'Distance', 0.0003)
    nt.links.new(g.outputs['Fac'], bp.inputs['Height']); nt.links.new(bp.outputs['Normal'], bs.inputs['Normal'])
    return m

def mat_desk():
    """το γραφείο του 2D: σκούρο «χαρτί στη σκιά» (PAPER × 0.10–0.22), χωρίς νερά ξύλου."""
    m, nt, bs = new_mat('DeskPaper')
    tc = nt.nodes.new('ShaderNodeTexCoord')
    n1 = nt.nodes.new('ShaderNodeTexNoise'); n1.inputs['Scale'].default_value = 7; n1.inputs['Detail'].default_value = 5
    n2 = nt.nodes.new('ShaderNodeTexNoise'); n2.inputs['Scale'].default_value = 60; n2.inputs['Detail'].default_value = 4
    for n in (n1, n2): nt.links.new(tc.outputs['Object'], n.inputs['Vector'])
    v = M(nt, 'ADD', M(nt, 'MULTIPLY', n1.outputs['Fac'], 0.7), M(nt, 'MULTIPLY', n2.outputs['Fac'], 0.3))
    ramp = nt.nodes.new('ShaderNodeValToRGB'); e = ramp.color_ramp.elements
    e[0].position = 0.30; e[0].color = (0.0095, 0.0085, 0.0070, 1); e[1].position = 0.70; e[1].color = (0.030, 0.027, 0.022, 1)
    nt.links.new(v, ramp.inputs['Fac']); nt.links.new(ramp.outputs['Color'], bs.inputs['Base Color'])
    setin(bs, 'Roughness', 0.78); setin(bs, 'Specular IOR Level', 0.3)
    bp = nt.nodes.new('ShaderNodeBump'); bp.inputs['Strength'].default_value = 0.04; setin(bp, 'Distance', 0.0003)
    nt.links.new(n2.outputs['Fac'], bp.inputs['Height']); nt.links.new(bp.outputs['Normal'], bs.inputs['Normal'])
    return m

# ---------------------------------------------------------------- χαρτιά
CW_M, CH_M = 0.18, 0.138                # η κάρτα είναι 900×690 px → 18 × 13,8 εκ.
PXM = CW_M / 900.0
REST = 0.00036

def curl_z(xl, yl, c):
    u, v = xl / (CW_M / 2), yl / (CH_M / 2)
    return c * (0.62 * u * u + 0.30 * v * v + 0.22 * u * v)

def card(name, mat, cx, cy, rot_deg, c):
    nx, ny = 60, 46; verts, uvs, faces = [], [], []
    for j in range(ny + 1):
        for i in range(nx + 1):
            xl = (i / nx - 0.5) * CW_M; yl = (j / ny - 0.5) * CH_M
            verts.append((xl, yl, curl_z(xl, yl, c))); uvs.append((i / nx, j / ny))
    for j in range(ny):
        for i in range(nx):
            a = j * (nx + 1) + i; faces.append((a, a + 1, a + nx + 2, a + nx + 1))
    ob = mesh_obj(name, verts, faces, uvs=uvs, mats=[mat], smooth=True)
    s = ob.modifiers.new('Thickness', 'SOLIDIFY'); s.thickness = 0.00035; s.offset = -1
    ob.location = (cx, cy, REST); ob.rotation_euler = (0, 0, math.radians(rot_deg))
    return ob

def on_card(cardspec, px, py, lift=0.0):
    """σημείο της κάρτας (px της 2D κάρτας) → κόσμος."""
    cx, cy, rot, c = cardspec
    xl, yl = (px - 450) * PXM, -(py - 345) * PXM
    r = math.radians(rot)
    return (cx + xl * math.cos(r) - yl * math.sin(r), cy + xl * math.sin(r) + yl * math.cos(r),
            REST + curl_z(xl, yl, c) + lift)

def qbez(p0, p1, p2, t):
    return tuple((1 - t) ** 2 * a + 2 * (1 - t) * t * b + t * t * cc for a, b, cc in zip(p0, p1, p2))

def point_at_proj(S, s):
    """σημείο της πινελιάς S που η προβολή του στο p0→p2 ισούται με s (όπως ο wipe της 2D μηχανής)."""
    p0, p2 = S[0], S[2]; dx, dy = p2[0] - p0[0], p2[1] - p0[1]; L2 = dx * dx + dy * dy
    pts = [qbez(*S, k / 400) for k in range(401)]
    pr = [((x - p0[0]) * dx + (y - p0[1]) * dy) / L2 for x, y in pts]
    s = min(max(s, pr[0]), pr[-1])
    for k in range(400):
        if pr[k] <= s <= pr[k + 1]:
            f = (s - pr[k]) / max(pr[k + 1] - pr[k], 1e-9)
            return (pts[k][0] + f * (pts[k + 1][0] - pts[k][0]), pts[k][1] + f * (pts[k + 1][1] - pts[k][1]))
    return pts[-1]

# ================================================================= ΧΤΙΣΙΜΟ
wipe_scene()

# ---- κόσμος & γραφείο
world = bpy.data.worlds.new('Room'); scene.world = world
try: world.use_nodes = True
except AttributeError: pass
bg = world.node_tree.nodes.get('Background') or world.node_tree.nodes.new('ShaderNodeBackground')
bg.inputs['Color'].default_value = (0.95, 0.97, 1.0, 1); bg.inputs['Strength'].default_value = 0.33   # απαλό διάχυτο φως δωματίου
wo = world.node_tree.nodes.get('World Output') or world.node_tree.nodes.new('ShaderNodeOutputWorld')
world.node_tree.links.new(bg.outputs[0], wo.inputs['Surface'])

desk = mesh_obj('Desk', [(-0.8, -0.8, 0), (0.8, -0.8, 0), (0.8, 0.8, 0), (-0.8, 0.8, 0)], [(0, 1, 2, 3)],
                mats=[mat_desk()])

# ---- οι δύο κάρτες (θέσεις/κλίσεις όπως στο βίντεο)
A_SPEC = (0.000, 0.078, 1.6, 0.0010)
B_SPEC = (0.004, -0.076, -1.2, 0.0011)
matA, IMPRINT = mat_card_A()
matB, PEN_P, DARKEN, WET = mat_card_B()
cardA = card('Card_A', matA, A_SPEC[0], A_SPEC[1], A_SPEC[2], A_SPEC[3])
cardB = card('Card_B', matB, B_SPEC[0], B_SPEC[1], B_SPEC[2], B_SPEC[3])

# ---- σκηνικά αντικείμενα: το χάνκο του Γιαμαγκούτσι και η κόκκινη μελανοταινία
ivory = simple('Bone', (0.56, 0.47, 0.36), 0.45, coat=0.1)        # PAPER_D
redink = simple('RedInk', (0.552, 0.037, 0.021), 0.6)             # VERM
hanko = lathe('Hanko', [(0, 0), (0, 0.0072), (0.0006, 0.008), (0.057, 0.008), (0.0577, 0.0074), (0.058, 0)],
              [redink, ivory], mat_of=lambda i, z: 0 if i == 0 else 1)
hanko.rotation_euler = (0, math.radians(90), math.radians(28)); hanko.location = (-0.095, 0.172, 0.008)
tin, felt = simple('Tin', (0.30, 0.29, 0.28), 0.5, metal=0.9), simple('Felt', (0.36, 0.024, 0.014), 1.0)
pad = lathe('InkPad', [(0, 0), (0, 0.0316), (0.0006, 0.032), (0.0114, 0.032), (0.012, 0.0314), (0.012, 0.0300),
                       (0.0095, 0.0295), (0.0095, 0)], [tin, felt], mat_of=lambda i, z: 1 if i == 6 else 0)
pad.location = (0.098, 0.196, 0)

# ---- η σφραγίδα «ΑΛΗΘΕΙΑ» (η αρχή του rig = κέντρο της λαστιχένιας βάσης)
STAMP_RIG = empty('Stamp')
rubber = box('Stamp_rubber', 0.137, 0.031, 0.0025, 0.0, simple('Rubber', (0.20, 0.015, 0.013), 0.75), 0.0003)
base = box('Stamp_base', 0.145, 0.040, 0.014, 0.0025,
           mat_wood('Beech', (0.12, 0.085, 0.055), (0.26, 0.19, 0.12), 0.45, (3, 90, 90), 1.0), 0.0012)
handle = lathe('Stamp_handle', [(0.0165, 0), (0.0165, 0.012), (0.019, 0.0105), (0.024, 0.0068), (0.040, 0.0062)] +
               [(0.052 + 0.014 * math.sin(a), 0.016 * math.cos(a)) for a in [math.radians(d) for d in range(-60, 91, 10)]],
               [simple('Lacquer', (0.023, 0.017, 0.016), 0.3, coat=0.5)])
for p in (rubber, base, handle): p.parent = STAMP_RIG

STAMP_PX = (535.0, 336.5)                              # κέντρο του αποτυπώματος στην κάρτα Α (px)
SX, SY, _ = on_card(A_SPEC, *STAMP_PX)
_c, _s = math.cos(math.radians(9)), math.sin(math.radians(9))
SZ = max(on_card(A_SPEC, STAMP_PX[0] + (u * _c - v * _s) / PXM, STAMP_PX[1] - (u * _s + v * _c) / PXM)[2]
         for u in (-0.0685, 0, 0.0685) for v in (-0.0155, 0, 0.0155)) + 0.00004   # το ψηλότερο σημείο κάτω από το λάστιχο
SROT = math.radians(A_SPEC[2] + 9.0)                   # ίδια κλίση με το 2D αποτύπωμα

def stamp_key(f, dx, dy, z, drot, interp='BEZIER'):
    key(STAMP_RIG, 'location', f, (SX + dx, SY + dy, z), interp)
    key(STAMP_RIG, 'rotation_euler', f, (0, 0, SROT + math.radians(drot)), interp)

stamp_key(F0, 0.20, 0.34, 0.24, 28)
stamp_key(55, 0.20, 0.34, 0.24, 28)
stamp_key(92, 0.0, 0.0, SZ + 0.068, 4)                 # αιωρείται πάνω από την κάρτα
stamp_key(104, 0.0, 0.0, SZ + 0.084, 2.5, 'LINEAR')    # μικρή προετοιμασία προς τα πάνω
stamp_key(109, 0.0, 0.0, SZ + 0.040, 0.6, 'LINEAR')    # επιτάχυνση
stamp_key(IMPACT, 0.0, 0.0, SZ, 0.0)                   # ΜΠΑΜ
stamp_key(121, 0.0, 0.0, SZ, 0.0)                      # πίεση
stamp_key(125, 0.0, 0.0, SZ + 0.004, -0.5)             # ξεκολλάει
stamp_key(142, 0.09, 0.13, 0.17, -14)                  # φεύγει
stamp_key(F1, 0.09, 0.13, 0.17, -14)

key(IMPRINT, 'default_value', F0, 0.0, 'CONSTANT')
key(IMPRINT, 'default_value', IMPACT, 1.0, 'CONSTANT')

# η κάρτα Β τινάζεται λίγο από το χτύπημα στο γραφείο
for f, dz in ((F0, 0), (IMPACT, 0), (IMPACT + 2, 0.0009), (IMPACT + 5, 0), (IMPACT + 7, 0.0002), (IMPACT + 9, 0)):
    key(cardB, 'location', f, (B_SPEC[0], B_SPEC[1], REST + dz))

# ---- η πένα που σβήνει την ιστορία Β με ένα Χ
gold = simple('Brass', (0.50, 0.38, 0.20), 0.35, metal=1.0)
black = simple('PenInk', (0.023, 0.017, 0.016), 0.2, coat=0.8)    # INK
pen_prof = [(0, 0), (0.0004, 0.0006), (0.012, 0.0034), (0.0125, 0.0042), (0.030, 0.0049), (0.0302, 0.0058),
            (0.034, 0.0058), (0.0342, 0.0062), (0.124, 0.0062), (0.128, 0.0052), (0.1305, 0.0028), (0.131, 0)]
pen = lathe('Pen', pen_prof, [gold, black], mat_of=lambda i, z: 0 if z < 0.0122 or 0.0301 <= z < 0.0341 else 1)
d = (0.50, -0.55, 0.62)                                # πιο χαμηλή κλίση: λιγότερη πένα μέσα στον φακό; n = math.sqrt(sum(c * c for c in d)); d = tuple(c / n for c in d)
ax = (-d[1], d[0], 0.0); na = math.hypot(ax[0], ax[1])
pen.rotation_mode = 'AXIS_ANGLE'; pen.rotation_axis_angle = (math.acos(d[2]), ax[0] / na, ax[1] / na, 0.0)

S1 = ((70, 120), (430, 330), (840, 620)); S2 = ((830, 90), (480, 360), (90, 640))   # οι πινελιές της 2D μηχανής
def tip(S, s, lift=0.0):
    px, py = point_at_proj(S, s); return on_card(B_SPEC, px, py, lift + 0.00005)

D1, D2 = (185, 196), (203, 214)                        # τα δύο χτυπήματα της πένας (καρέ)
s0, e1 = tip(S1, 0.0), tip(S1, 1.0); s2, e2 = tip(S2, 0.0), tip(S2, 1.0)
key(pen, 'location', F0, (s0[0] + 0.13, s0[1] - 0.20, 0.11))
key(pen, 'location', 158, (s0[0] + 0.13, s0[1] - 0.20, 0.11))
key(pen, 'location', 180, (s0[0], s0[1], s0[2] + 0.010))
key(pen, 'location', D1[0] - 1, (s0[0], s0[1], s0[2] + 0.0015), 'LINEAR')
for f in range(D1[0], D1[1] + 1):
    key(pen, 'location', f, tip(S1, (f - D1[0]) / (D1[1] - D1[0]) + 0.02), 'LINEAR')
key(pen, 'location', 199, ((e1[0] + s2[0]) / 2, (e1[1] + s2[1]) / 2, e1[2] + 0.011))
key(pen, 'location', D2[0] - 1, (s2[0], s2[1], s2[2] + 0.0015), 'LINEAR')
for f in range(D2[0], D2[1] + 1):
    key(pen, 'location', f, tip(S2, (f - D2[0]) / (D2[1] - D2[0]) + 0.02), 'LINEAR')
key(pen, 'location', 219, (e2[0] + 0.01, e2[1] - 0.006, e2[2] + 0.014))
key(pen, 'location', 242, (e2[0] + 0.16, e2[1] - 0.10, 0.11))
key(pen, 'location', F1, (e2[0] + 0.16, e2[1] - 0.10, 0.11))

key(PEN_P, 'default_value', F0, 0.0, 'CONSTANT')
key(PEN_P, 'default_value', D1[0], 0.0, 'LINEAR')
key(PEN_P, 'default_value', D1[1], 0.5, 'CONSTANT')
key(PEN_P, 'default_value', D2[0], 0.5, 'LINEAR')
key(PEN_P, 'default_value', D2[1], 1.03, 'CONSTANT')
key(DARKEN, 'default_value', D1[0], 0.0); key(DARKEN, 'default_value', D2[1] + 6, 0.09)
key(WET, 'default_value', D1[0] - 1, 1.0); key(WET, 'default_value', F1, 0.55)

# ---- κάμερα: τροχιά, εστίαση, τράνταγμα στο χτύπημα
cam_data = bpy.data.cameras.new('Cam'); cam_data.lens = 50; cam_data.sensor_fit = 'AUTO'; cam_data.sensor_width = 36
cam_data.clip_start = 0.005; cam_data.clip_end = 20
cam = link_obj(bpy.data.objects.new('Cam', cam_data)); scene.camera = cam
TARGET = empty('CamTarget'); AIM = empty('CamAim'); AIM.parent = TARGET
track(cam, AIM)
FOCUS = empty('Focus')
cam_data.dof.use_dof = True; cam_data.dof.focus_object = FOCUS; cam_data.dof.aperture_fstop = 3.2
try: cam_data.dof.aperture_blades = 7
except AttributeError: pass

TILT = math.radians(18)
CAM_KEYS = [(F0, (-0.015, 0.000), 0.66), (70, (0.000, 0.030), 0.52), (100, (0.018, 0.080), 0.40),
            (140, (0.020, 0.082), 0.385), (172, (0.003, -0.072), 0.40), (222, (0.003, -0.074), 0.39),
            (262, (0.000, 0.003), 0.60), (F1, (0.000, 0.003), 0.63)]
for f, (tx, ty), dist in CAM_KEYS:
    key(TARGET, 'location', f, (tx, ty, 0.0))
    key(cam, 'location', f, (tx, ty - dist * math.sin(TILT), dist * math.cos(TILT)))

for f, (fx, fy) in ((F0, (0.0, 0.02)), (95, (SX, SY)), (140, (SX, SY)), (172, (B_SPEC[0], B_SPEC[1])),
                    (222, (B_SPEC[0], B_SPEC[1])), (262, (0.0, 0.003)), (F1, (0.0, 0.003))):
    key(FOCUS, 'location', f, (fx, fy, 0.0006))

def aim_offset(f):
    br = (0.00045 * math.sin(f * 0.071) + 0.00025 * math.sin(f * 0.173 + 1.3),        # ανάσα χειριστή
          0.00040 * math.sin(f * 0.059 + 0.7) + 0.00022 * math.sin(f * 0.191))
    k = f - IMPACT
    if 0 <= k <= 14:
        a = 0.0024 * math.exp(-k / 2.6)
        br = (br[0] + a * math.sin(k * 2.1 + 0.4), br[1] - a * math.cos(k * 2.7))
    return br
for f in range(F0, F1 + 1):
    if f % 3 == 1 or IMPACT - 1 <= f <= IMPACT + 15:
        ox, oy = aim_offset(f); key(AIM, 'location', f, (ox, oy, 0.0))

# ---- φως: σαν το 2D, χαρτί ≈ τα χρώματα του texture (ήλιος 0.8 × cos + δωμάτιο 0.33 ≈ 1)
def light(name, kind, loc, aim, energy, color, **kw):
    ld = bpy.data.lights.new(name, kind); ld.energy = energy; ld.color = color
    for k, v in kw.items(): setattr(ld, k, v)
    ob = link_obj(bpy.data.objects.new(name, ld)); ob.location = loc
    track(ob, empty(name + '_aim', aim)); return ob
light('Sun', 'SUN', (-0.35, 0.42, 0.70), (0, 0, 0), 0.80, (1.0, 0.96, 0.90), angle=math.radians(5))

# ---------------------------------------------------------------- render
r = scene.render
r.resolution_x, r.resolution_y = 1080, 1920
r.resolution_percentage = 50 if PREVIEW else 100
r.fps = FPS; r.fps_base = 1.0; scene.frame_start, scene.frame_end = F0, F1

def try_set(obj, attr, val):
    try: setattr(obj, attr, val); return True
    except (AttributeError, TypeError, ValueError): return False

if ENGINE == 'CYCLES':
    r.engine = 'CYCLES'
    try_set(scene.cycles, 'samples', 64 if PREVIEW else 256); try_set(scene.cycles, 'use_denoising', True)
else:
    for e in ('BLENDER_EEVEE_NEXT', 'BLENDER_EEVEE'):
        if try_set(r, 'engine', e): break
    ee = scene.eevee
    try_set(ee, 'taa_render_samples', 24 if PREVIEW else 96)
    for a, v in (('use_raytracing', True), ('use_shadows', True), ('shadow_ray_count', 2), ('shadow_step_count', 10),
                 ('use_gtao', True), ('use_soft_shadows', True)):
        try_set(ee, a, v)
if not (try_set(r, 'use_motion_blur', True) and try_set(r, 'motion_blur_shutter', 0.5)):
    try_set(scene.eevee, 'use_motion_blur', True)

vs = scene.view_settings
try_set(vs, 'view_transform', 'Standard'); try_set(vs, 'look', 'None')   # τα textures βγαίνουν με τα χρώματα του 2D
vs.exposure = EXPOSURE

def out_video():
    os.makedirs(OUTDIR, exist_ok=True); ims = r.image_settings
    try:
        if hasattr(ims, 'media_type'): ims.media_type = 'VIDEO'
        ims.file_format = 'FFMPEG'
        r.ffmpeg.format = 'MPEG4'; r.ffmpeg.codec = 'H264'
        try_set(r.ffmpeg, 'constant_rate_factor', 'HIGH'); try_set(r.ffmpeg, 'ffmpeg_preset', 'GOOD')
        r.filepath = os.path.join(OUTDIR, '1A1P_reveal_3d_v2_')
    except (AttributeError, TypeError):
        out_png(); r.filepath = os.path.join(OUTDIR, 'frames', 'f_')

def out_png():
    ims = r.image_settings
    if hasattr(ims, 'media_type'): ims.media_type = 'IMAGE'
    ims.file_format = 'PNG'

out_video()
apply_interp()
scene.frame_set(F0)

argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
if bpy.app.background:
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(os.path.dirname(TEX), '1A1P_reveal_3d.blend'))
if '--stills' in argv:
    out_png(); os.makedirs(OUTDIR, exist_ok=True)
    for f in (1, 92, IMPACT, 128, 190, 210, 300):
        scene.frame_set(f); r.filepath = os.path.join(OUTDIR, 'still_%03d.png' % f)
        bpy.ops.render.render(write_still=True)
    out_video()
if '--render' in argv:
    bpy.ops.render.render(animation=True)
print('Mind Twist GR · σκηνή έτοιμη ·', TEX)
