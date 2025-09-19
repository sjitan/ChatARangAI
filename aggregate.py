import os,csv,pandas as pd,numpy as np,matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
os.makedirs("logs/plots",exist_ok=True); rep="logs/reports/trial_summaries.csv"
if not os.path.exists(rep) or os.path.getsize(rep)==0: raise SystemExit("no_summaries")
df=pd.read_csv(rep)
df["global_slope_deg_s"]=pd.to_numeric(df["global_slope_deg_s"],errors="coerce")
df["ttf_s"]=pd.to_numeric(df["ttf_s"],errors="coerce")
g=df.groupby("condition",as_index=False).agg(n=("trial_id","count"),slope_med=("global_slope_deg_s","median"),slope_mean=("global_slope_deg_s","mean"),ttf_med=("ttf_s","median"),ttf_mean=("ttf_s","mean"))
g.to_csv("logs/reports/aggregate_summary.csv",index=False)
plt.figure()
for c,sub in df.groupby("condition"):
    y=sub["global_slope_deg_s"].dropna().values
    x=np.random.normal({"control":0,"static":1,"interrupt":2}[c],0.04,size=len(y))
    plt.scatter(x,y,label=c)
plt.xticks([0,1,2],["control","static","interrupt"])
plt.ylabel("global_slope_deg_s")
plt.legend(); plt.tight_layout()
plt.savefig("logs/plots/aggregate_slope_scatter.png",dpi=120); plt.close()
plt.figure()
for c,sub in df.groupby("condition"):
    plt.hist(sub["ttf_s"].dropna().values,bins=20,alpha=0.5,label=c)
plt.xlabel("TTF (s)"); plt.ylabel("count"); plt.legend(); plt.tight_layout()
plt.savefig("logs/plots/aggregate_ttf_hist.png",dpi=120); plt.close()
print("OK")

# AUTO_OPEN_AGG
import os,glob,subprocess
try:
  ps=sorted(glob.glob('logs/plots/aggregate_*png'))
  [subprocess.run(['open',q],check=False) for q in ps[-2:]]
except Exception:
  pass
