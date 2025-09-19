import sys,os,numpy as np,pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
if len(sys.argv)<2: raise SystemExit("usage: plot_fatigue.py <csv> [fail_deg]")
csv=sys.argv[1]; D=float(sys.argv[2]) if len(sys.argv)>2 else 22.0
W=2.0; OUT="logs/plots"
df=pd.read_csv(csv)
df=df[pd.to_numeric(df["t_rel"],errors="coerce").notna()]
df=df[pd.to_numeric(df["drift"],errors="coerce").notna()]
t=df["t_rel"].astype(float).to_numpy()
d=np.abs(df["drift"].astype(float).to_numpy())
d=np.maximum.accumulate(d)
M=np.clip(1.0-d/D,0.0,1.0)  # 1 at start → 0 at failure (no sign flip)
mask=(t>=W)&np.isfinite(M)
if mask.sum()>=10:
    t0=t[mask][0]; m,b=np.polyfit(t[mask]-t0,M[mask],1)
    tf=t[mask]; yf=m*(tf-t0)+b
else:
    m=0.0; tf=yf=np.array([])
fig,ax=plt.subplots(figsize=(10,5))
ax.plot(t,M,label="margin 1→0",linewidth=2)
if tf.size: ax.plot(tf,yf,"r--",label=f"slope {m:.4f}/s")
for _,r in df.iterrows():
    ch=r.get("channel"); tr=r.get("t_rel")
    if pd.notna(ch) and pd.notna(tr):
        if ch==1: ax.axvline(tr,color="green",alpha=0.35)
        elif ch==2: ax.axvline(tr,color="red",alpha=0.35)
ax.set_xlabel("Trial time (s)")
ax.set_ylabel("Performance margin (1=best,0=fail)")
ax.set_title("Fatigue (margin declines)")
ax.legend()
ax.set_ylim(0,1)           # normal axis: 0 bottom, 1 top
# no invert_yaxis, no negative values
os.makedirs(OUT,exist_ok=True)
out=os.path.join(OUT,os.path.basename(csv).replace(".csv","_fatigue.png"))
plt.savefig(out,dpi=140); print(out)
