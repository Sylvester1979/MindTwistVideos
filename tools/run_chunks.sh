#!/bin/bash
# Render an episode in parallel chunks, then join them losslessly.
# Usage: tools/run_chunks.sh <episode_dir> <K> [chunk indices...]
#   K = number of chunks (4 per XL sandbox run keeps each run under the 15-min limit).
#   With indices, renders only those chunks (e.g. after a change in one scene).
# Output: $MT_ROOT/chunks/c<i>.mp4 and $MT_ROOT/video_raw.mp4
set -e
EP=$(cd "$1" && pwd); K=$2; shift 2
ROOT=${MT_ROOT:-/home/user/ww1}
mkdir -p "$ROOT/chunks"
T=$(python3 -c "import json;print(json.load(open('$ROOT/timeline.json'))['T'])")
N=$(python3 -c "print(int(round($T*30)))")
B=($(python3 -c "print(' '.join(str(round($N*i/$K)) for i in range($K+1)))"))
IDX=${@:-$(seq 0 $((K-1)))}
export OMP_NUM_THREADS=1
for k in $IDX; do
  rm -f "$ROOT/chunks/c$k.mp4"
  python3 "$EP/render.py" video ${B[$k]} ${B[$((k+1))]} "$ROOT/chunks/c$k.mp4" > "$ROOT/chunks/log$k.txt" 2>&1 &
done
wait
: > "$ROOT/chunks/list.txt"
for k in $(seq 0 $((K-1))); do echo "file 'c$k.mp4'" >> "$ROOT/chunks/list.txt"; done
if [ $(ls "$ROOT"/chunks/c*.mp4 2>/dev/null | grep -v part | wc -l) -eq $K ]; then
  ffmpeg -v error -y -f concat -safe 0 -i "$ROOT/chunks/list.txt" -c copy "$ROOT/video_raw.mp4"
  echo "joined -> $ROOT/video_raw.mp4"
else
  echo "not all chunks present yet"; ls "$ROOT/chunks"
fi
