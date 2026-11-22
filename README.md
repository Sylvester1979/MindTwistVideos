# Mind Twist GR: μηχανή βίντεο

Κώδικας για τα ιστορικά Shorts του καναλιού **Mind Twist GR**: κάθετο βίντεο 9:16 με κινούμενες ζωγραφιές, όπου όλη η κίνηση φτιάχνεται από εμάς με κώδικα. Δεν χρησιμοποιούμε AI video.

| # | Επεισόδιο | Κατάσταση |
|---|---|---|
| 01 | Το ημερολόγιο της Τάνιας Σάβιτσεβα (Λένινγκραντ, 1941–42) | Δημοσιεύτηκε. Ο κώδικάς του χάθηκε πριν φτιαχτεί αυτό το repo, και τα κομμάτια του ξαναγράφτηκαν στο `engine/` |
| 02 | Ένα λεπτό πριν την ειρήνη (11 Νοεμβρίου 1918) | Ολοκληρώθηκε. Ο κώδικας είναι στο `episodes/02-…` |
| 03 | Ο γιατρός που δεν έφυγε (Γιάνους Κόρτσακ, Βαρσοβία 1942) | Ολοκληρώθηκε. `episodes/03-…` |
| 04 | Η λίστα του Σίντλερ | Ολοκληρώθηκε, δημοσίευση 23/11/2026. `episodes/04-…` |
| 05 | Καλάβρυτα (13 Δεκεμβρίου 1943) | Επόμενο, δημοσίευση 13/12 |

## Δομή

```
engine/      κοινά εργαλεία για όλα τα επεισόδια
  eng.py       κάμερα με βάθος, ομίχλη, βροχή, φιλμ, μεταβάσεις μελάνης, κείμενα
  gfx.py       ρολόι τσέπης, χάρτης, χειρόγραφο σημείωμα, μετρητής σταυρών, πλάκα τάφου
  prep.py      τόνος κάρβουνου, σβήσιμο υπογραφών, μάσκα χαρτιού
  depth.py     χάρτες βάθους (Depth-Anything-V2)
  detext.py    αντικατάσταση ψευδο-κειμένου AI με ουδέτερες γραμμές
episodes/NN-…/
  timeline.py  κόβει και ενώνει τη φωνή, βγάζει χρονισμούς και υπότιτλους
  render.py    η λίστα πλάνων του επεισοδίου
  mix.py       φωνή, μουσική και εφέ, ένταση −14 LUFS
  thumb.py     thumbnail
tools/       setup.sh · align.py (έλεγχος λέξεων) · run_chunks.sh (παράλληλο ρεντερ)
docs/        WORKFLOW.md (η διαδικασία) · LESSONS.md (τα λάθη που πληρώσαμε)
```

## Κώδικας και δεδομένα είναι χωριστά

Ο κώδικας βρίσκεται εδώ. Τα δεδομένα κάθε επεισοδίου (ζωγραφιές, φωνή, μουσική, εφέ) **δεν μπαίνουν στο repo** και ζουν σε φάκελο εργασίας, που ορίζεται με τη μεταβλητή `MT_ROOT`:

```
$MT_ROOT/img  plates  depth  audio  sfx  fonts  map  chunks  timeline.json
```

## Γρήγορη χρήση (στο sandbox του μοντάζ)

```bash
git clone --depth 1 https://github.com/Sylvester1979/MindTwistVideos /home/user/mt
export MT_ROOT=/home/user/ep03 && /home/user/mt/tools/setup.sh
# ζωγραφιές στο $MT_ROOT/img, μετά:
python3 /home/user/mt/engine/prep.py P01_x P02_y …
PYTHONPATH=/home/user/pylib HF_HOME=/home/user/.hf python3 /home/user/mt/engine/depth.py P01_x P02_y …
python3 /home/user/mt/tools/align.py $MT_ROOT/audio/voice.mp3 $MT_ROOT/audio/align.json
python3 EP/timeline.py && python3 EP/render.py stills 3 10 25 …   # έλεγχος καρέ
/home/user/mt/tools/run_chunks.sh EP 4      # ρεντερ
python3 EP/mix.py                            # ήχος
```
