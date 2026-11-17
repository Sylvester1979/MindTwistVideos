# 02 · Ένα λεπτό πριν την ειρήνη

11 Νοεμβρίου 1918. Η ανακωχή υπογράφεται στις 5:12 και ισχύει από τις 11:00. Στο ενδιάμεσο σκοτώνονται περίπου 2.738 άνθρωποι.
Διάρκεια 2:12 · φωνή: ElevenLabs v4 «François Louis» · μουσική: Suno «Before the Silence».

## Γεγονότα και πηγές
| Ισχυρισμός | Πηγή |
|---|---|
| Υπογραφή 5:12–5:20, ισχύς 11:00, ~2.738 νεκροί την τελευταία μέρα | AP 2018, Veterans Breakfast Club |
| Τρεμπυσόν: 40, βοσκός, αγγελιοφόρος, σφαίρα ~10:45, σημείωμα «Rassemblement à 11h30 pour le ravitaillement», σταυρός «10 novembre» | Wikipedia, AP, Connexion France |
| Δύο εξηγήσεις για την ημερομηνία (ντροπή / πένθος οικογενειών) | AP (N. Czubak), Connexion France |
| Πράις: 25, Καναδός, ελεύθερος σκοπευτής, στο στήθος, 10:58, Ville-sur-Haine | CBC, Canadian Encyclopedia, Veterans Affairs Canada |
| Γκάνθερ: 23, υποβιβασμός λόγω γράμματος, οι Γερμανοί του φώναζαν να γυρίσει, 10:59 | Western Front Association, History.com, Wikipedia |
| Παρ (21/8/1914, έφηβος) και Έλισον (11/11/1918, 9:30), αντικριστοί τάφοι στο St Symphorien | CWGC, Royal British Legion |

## Αρχεία στο `$MT_ROOT`
- `img/P01_carriage … P14_letter` (Flux 1.1 pro ultra). Χρησιμοποιούνται τα `P08_sniper_b` και `P11_bugler_b`.
- `audio/voice_1803.mp3` (κύρια φωνή), `fix_treb3.mp3` («Μια σφαίρα…»), `fix_mons4.mp3` («Στον άλλον, ο τελευταίος Βρετανός.»), `music.mp3`
- `sfx/tick.mp3`, `artillery.mp3` (ElevenLabs) · `rumble.mp3`, `shot.mp3` (Suno) · `trumpet_user.mp3` (ElevenLabs, από τον χρήστη)
- Τα align JSON ξαναβγαίνουν με `tools/align.py`: `align_1803.json`, `fix_*_words.json`. Τα words-JSON είναι η λίστα `words` όλων των segments.

## Σειρά
```
python3 tools/align.py $MT_ROOT/audio/voice_1803.mp3 $MT_ROOT/audio/align_1803.json
python3 timeline.py      # TEMPO=1.04 · T≈131.6″
python3 render.py stills 3 13 21 33 51 74 80 118
tools/run_chunks.sh . 8  # δύο εκτελέσεις XL των 4
python3 mix.py && ffmpeg -i $MT_ROOT/video_raw.mp4 -i $MT_ROOT/audio/mix.wav -map 0:v -map 1:a -c:v libx264 -crf 25 -c:a aac -b:a 256k -shortest out.mp4
```

## Πακέτο ανεβάσματος
- **Τίτλος:** Σκοτώθηκε 1 λεπτό πριν τελειώσει ο Α΄ Παγκόσμιος
- **Thumbnail:** το ρολόι στις 10:59, «ΣΚΟΤΩΘΗΚΕ 10:59 / Ο πόλεμος τελείωσε στις 11:00»
- **Λεπτομέρεια εποχής:** στο καντράν δεν υπάρχει VI (εκεί είναι το καντράν δευτερολέπτων) και γράφεται IIII αντί για IV, όπως στα ρολόγια τσέπης του 1918.
