import argparse,glob,time,os,pandas as pd,matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
ap=argparse.ArgumentParser()
ap.add_argument("--file",type=str)
ap.add_argument("--cli",action="store_true")
a=ap.parse_args()
def newest():
    xs=sorted(glob.glob("logs/session_*.csv"))
    return xs[-1] if xs else None
path=a.file or newest()
if not path: raise SystemExit("no_csv")
if a.cli:
    t0=None
    while True:
        try:
            df=pd.read_csv(path)
            tv=pd.to_numeric(df["t_vid"],errors="coerce")
            ang=pd.to_numeric(df["avg_angle"],errors="coerce")
            drift=pd.to_numeric(df["drift"],errors="coerce") if "drift" in df else None
            if t0 is None and (df["event"]=="start").any(): t0=float(tv[df["event"]=="start"].iloc[0])
            msg=f"n={len(df)} slope5s=? drift={drift.iloc[-1]:.2f}" if drift is not None and len(df)>0 else f"n={len(df)}"
            print(msg, flush=True)
        except: pass
        time.sleep(1)
else:
    plt.ion(); fig,ax=plt.subplots()
    while plt.fignum_exists(fig.number):
        try:
            df=pd.read_csv(path)
            tv=pd.to_numeric(df["t_vid"],errors="coerce")
            ang=pd.to_numeric(df["avg_angle"],errors="coerce")
            ax.clear(); ax.plot(tv,ang); ax.set_xlabel("time (s)"); ax.set_ylabel("avg_angle (deg)"); fig.canvas.draw(); fig.canvas.flush_events()
        except: pass
        plt.pause(1.0)
