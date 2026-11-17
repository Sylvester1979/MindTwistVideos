#!/bin/bash
# One-time setup of a fresh render sandbox (fonts, map data, depth model deps).
# Usage: MT_ROOT=/home/user/ep03 tools/setup.sh
set -e
ROOT=${MT_ROOT:-/home/user/ww1}
mkdir -p "$ROOT"/{fonts,map,img,plates,depth,audio,sfx,chunks,test}
G=https://github.com/google/fonts/raw/main
cd "$ROOT/fonts"
[ -f Caveat.ttf ]       || curl -sfSL -o Caveat.ttf       "$G/ofl/caveat/Caveat%5Bwght%5D.ttf"
[ -f EBGaramond.ttf ]   || curl -sfSL -o EBGaramond.ttf   "$G/ofl/ebgaramond/EBGaramond%5Bwght%5D.ttf"
[ -f SpecialElite.ttf ] || curl -sfSL -o SpecialElite.ttf "$G/apache/specialelite/SpecialElite-Regular.ttf"
cd "$ROOT/map"
B=https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson
for f in ne_10m_land ne_10m_rivers_europe ne_10m_rivers_lake_centerlines ne_10m_admin_0_boundary_lines_land; do
  [ -f $f.geojson ] || curl -sfSL -o $f.geojson $B/$f.geojson
done
# torch + transformers only for engine/depth.py (Depth-Anything-V2). ~3 min.
if ! PYTHONPATH=/home/user/pylib python3 -c "import torch, transformers" 2>/dev/null; then
  export PYTHONPATH= HOME=/home/user TMPDIR=/home/user/tmp; mkdir -p /home/user/tmp
  python3 -m pip install -q --no-cache-dir --target /home/user/pylib --index-url https://download.pytorch.org/whl/cpu torch
  python3 -m pip install -q --no-cache-dir --target /home/user/pylib transformers
fi
echo "setup ok -> $ROOT"
