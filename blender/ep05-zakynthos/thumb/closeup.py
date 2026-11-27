# Check renders of the pen up close (sandbox only, not part of the scene). Opens ep05_thumb.blend, adds a
# temporary camera aimed at a point, renders a small CPU preview, and does not save the .blend.
# usage: blender -b ep05_thumb.blend -P closeup.py -- <out.png> <tx> <ty> <tz> <dist> <elev°> <azim°> <lens> [samples]
import bpy, math, sys
from mathutils import Vector
a = sys.argv[sys.argv.index('--') + 1:]
out = a[0]; T = Vector(tuple(float(v) for v in a[1:4])); dist, elev, azim, lens = (float(v) for v in a[4:8])
samples = int(a[8]) if len(a) > 8 else 48
sc = bpy.context.scene
cd = bpy.data.cameras.new('Close'); cd.lens = lens; cd.sensor_fit = 'HORIZONTAL'; cd.clip_start = 0.002
c = bpy.data.objects.new('Close', cd); sc.collection.objects.link(c)
e, z = math.radians(elev), math.radians(azim)
c.location = T + Vector((math.cos(e) * math.sin(z), -math.cos(e) * math.cos(z), math.sin(e))) * dist
c.rotation_euler = (T - c.location).to_track_quat('-Z', 'Y').to_euler()
cd.dof.use_dof = True; cd.dof.aperture_fstop = 11; cd.dof.focus_distance = dist
sc.camera = c
r = sc.render; r.resolution_x, r.resolution_y, r.resolution_percentage = 960, 640, 100
sc.cycles.device = 'CPU'; sc.cycles.samples = samples
r.image_settings.file_format = 'PNG'; r.image_settings.color_depth = '8'
r.filepath = out
bpy.ops.render.render(write_still=True)
