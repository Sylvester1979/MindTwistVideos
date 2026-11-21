# 03 · Ο γιατρός που δεν έφυγε

Γιάνους Κόρτσακ, Βαρσοβία, Αύγουστος 1942. Διάρκεια περίπου 2:14.
- **Φωνή:** ElevenLabs v4 «François Louis»
- **Μουσική:** Suno «The Doctor Who Stayed (piano)», η εκδοχή Β (id 46e4309e)
- **Εφέ:** ElevenLabs text_to_sound_v3

## Αρχεία στο `$MT_ROOT`

**Εικόνες:** `img/` με 24 πλάνα, v3 μελάνι.
- Τα πλάνα με τον Κόρτσακ είναι από nano-banana-2.1, με αναφορά το B01 και το A01.
- `img/old` περιέχει τις απορριφθείσες εκδοχές.

**Ήχος:** στο `audio/`.
- `voice.mp3` είναι η λήψη του χρήστη. `voice_edit.wav` είναι η ίδια με μεγαλύτερες παύσεις.
- `align_voice.json` και `align_edit.json` είναι οι στοιχίσεις.
- `music.wav` και `mix.wav`.

**Εφέ:** στο `sfx/`: `water`, `pen`, `hall`, `whistle`, `steps`, `breeze` (mp3).

## Σειρά
```
python3 prep03.py && PYTHONPATH=/tmp/pylib python3 ../../engine/depth.py <keys>
python3 ../../tools/align.py $MT_ROOT/audio/voice.mp3 $MT_ROOT/audio/align_voice.json
python3 edit_voice.py                       # παύσεις που αγνόησε το TTS
python3 timeline_voice.py audio/align_edit.json
python3 render.py stills 2 16 25 63 112 130
tools/run_chunks.sh . 8                     # δύο εκτελέσεις XL των 4
python3 mix.py && ffmpeg -i video_raw.mp4 -i audio/mix.wav -map 0:v -map 1:a -c:v libx264 -crf 24 -c:a aac -b:a 256k -shortest out.mp4
```
