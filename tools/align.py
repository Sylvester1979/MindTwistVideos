"""Word-level transcription for checking narration and timing subtitles.
Usage: python3 tools/align.py <audio> <out.json> [model=medium]
  - 'medium' for full narration timing; 'large-v3' to verify doubtful words on short clips.
Prints every segment and flags words with probability < 0.6, plus silence boundaries
(use silences, not whisper end times, to cut pieces: whisper often cuts the last word short)."""
import json, sys, subprocess, re
from faster_whisper import WhisperModel

src, out = sys.argv[1], sys.argv[2]
model = sys.argv[3] if len(sys.argv) > 3 else 'medium'
m = WhisperModel(model, device='cpu', compute_type='int8', cpu_threads=4)
segs, _ = m.transcribe(src, language='el', word_timestamps=True, beam_size=5,
                       vad_filter=False, condition_on_previous_text=False)
res = []
for s in segs:
    res.append({'start': s.start, 'end': s.end, 'text': s.text,
                'words': [{'w': w.word, 's': w.start, 'e': w.end, 'p': w.probability} for w in s.words]})
json.dump(res, open(out, 'w'), ensure_ascii=False, indent=0)
for s in res:
    low = [f"{w['w'].strip()}({w['p']:.2f})" for w in s['words'] if w['p'] < 0.6]
    print(f"{s['start']:7.2f}-{s['end']:7.2f} |{s['text']}" + ('   LOW: ' + ' '.join(low) if low else ''))
log = subprocess.run(['ffmpeg', '-i', src, '-af', 'silencedetect=n=-42dB:d=0.18', '-f', 'null', '-'],
                     capture_output=True, text=True).stderr
print('silences:', ' '.join(re.findall(r'silence_(?:start|end): ([0-9.]+)', log)))
