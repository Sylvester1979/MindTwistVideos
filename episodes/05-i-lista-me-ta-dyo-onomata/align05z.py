"""Ep05 Zakynthos: align voice_full (script v7) to its beats via silences (no ASR here)."""
import re, sys, subprocess, json
import numpy as np
SR = 48000
A = '/home/claude/ep05z/audio'
def load(p):
    raw = subprocess.run(['ffmpeg','-v','error','-i',p,'-f','f32le','-ac','1','-ar',str(SR),'-'],capture_output=True,check=True).stdout
    return np.frombuffer(raw,np.float32).copy()
def silences(x, thr=-40, dmin=0.12):
    hop=SR//100; n=len(x)//hop; db=20*np.log10(np.sqrt(np.mean(x[:n*hop].reshape(n,hop)**2,1)+1e-12))
    q=db<thr; out=[]; i=0
    while i<n:
        if q[i]:
            j=i
            while j<n and q[j]: j+=1
            if (j-i)/100>=dmin or i==0 or j==n: out.append((i/100,j/100))
            i=j
        else: i+=1
    return out, db
def islands(sil,dur):
    sp=[];t=0.0
    for a,b in sil:
        if a>t+0.02: sp.append((t,a))
        t=b
    if t<dur-0.02: sp.append((t,dur))
    return sp
def letters(t): return len(re.sub(r'[^\wά-ώΆ-Ώ]','',t))
v8 = open("/home/claude/ep05z/script_v8.txt").read()
v7 = v8.replace("Οι αντάρτες, τους προειδοποιούν για τις εφόδους.","Οι αντάρτες τους προειδοποιούν για τις εφόδους.").replace(
     "Και τότε, ο δήμαρχος και ο Μητροπολίτης παραδίδουν στον διοικητή ένα χαρτί.","Και τότε, οι δυο τους του δίνουν ένα χαρτί.")
# beats = text between pause tags
parts = re.split(r'\[(?:long )?pause\]', v7)
beats = [re.sub(r'\[[^\]]*\]','',p).strip() for p in parts]
beats = [b for b in beats if b]
x = load(f'{A}/voice_full.wav'); dur=len(x)/SR
sil, db = silences(x); isl = islands(sil,dur)
n,m=len(isl),len(beats)
L=np.array([letters(b) for b in beats],float); speech=sum(b-a for a,b in isl); rate=speech/L.sum(); exp=L*rate
INF=1e18; D=np.full((m+1,n+1),INF); D[0,0]=0; back=np.zeros((m+1,n+1),int)
for i in range(1,m+1):
    for j in range(i,n-(m-i)+1):
        for k in range(max(i-1,j-8),j):
            if D[i-1,k]>=INF: continue
            s0,s1=isl[k][0],isl[j-1][1]; d=s1-s0
            c=D[i-1,k]+((d-exp[i-1])/max(exp[i-1],0.6))**2
            if c<D[i,j]: D[i,j]=c; back[i,j]=k
res=[]; j=n
for i in range(m,0,-1):
    k=back[i,j]; res.append((isl[k][0],isl[j-1][1],j-k)); j=k
res=res[::-1]
print(f'dur {dur:.2f}  islands {n}  beats {m}  rate {rate*1000:.0f} ms/letter')
out=[]
for b,(s0,s1,ni),e in zip(beats,res,exp):
    r=(s1-s0)/e
    flag='  <<<' if r<0.75 or r>1.3 else ''
    print(f'{s0:7.2f}-{s1:7.2f} {s1-s0:5.2f}s x{r:4.2f} [{ni}] {b[:70]}{flag}')
    out.append(dict(t0=s0,t1=s1,text=b))
json.dump(out,open('/home/claude/ep05z/work/beats_full.json','w'),ensure_ascii=False,indent=1)
