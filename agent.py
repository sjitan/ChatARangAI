import argparse,csv,os,sys,time,math,subprocess,numpy as np,cv2,mediapipe as mp,pandas as pd,matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import deque

LM=mp.solutions.pose.PoseLandmark
def _xy(p,w,h): return np.array([p.x*w,p.y*h],dtype=np.float32)
def hip_angle_deg(lms,w,h):
    def s(sh,hp,an):
        v1=hp-sh; v2=an-hp
        den=(np.linalg.norm(v1)*np.linalg.norm(v2))+1e-6
        ang=math.degrees(math.acos(np.clip(np.dot(v1,v2)/den,-1,1)))
        return abs(180-ang)
    L=s(_xy(lms[LM.LEFT_SHOULDER],w,h),_xy(lms[LM.LEFT_HIP],w,h),_xy(lms[LM.LEFT_ANKLE],w,h))
    R=s(_xy(lms[LM.RIGHT_SHOULDER],w,h),_xy(lms[LM.RIGHT_HIP],w,h),_xy(lms[LM.RIGHT_ANKLE],w,h))
    return (L+R)/2.0

class EMA:
    def __init__(self,a): self.a=a; self.v=None
    def u(self,x): self.v=x if self.v is None else self.a*x+(1-self.a)*self.v; return self.v

def say(t):
    try: subprocess.run(["say",t],check=False)
    except: pass

def slope_last(buf,win=8.0,dead=0.15):
    if len(buf)<8: return 0.0
    t0=buf[-1][0]
    pts=[(t,y) for (t,y) in buf if t0-t<=win]
    if len(pts)<8: return 0.0
    x=np.array([p[0] for p in pts]); x=x-x.mean()
    y=np.array([p[1] for p in pts])
    m=float(np.polyfit(x,y,1)[0])
    return 0.0 if abs(m)<dead else m

ap=argparse.ArgumentParser()
g=ap.add_mutually_exclusive_group()
g.add_argument("--device",type=int)
g.add_argument("--seed",type=str)
ap.add_argument("--subject",required=True)
ap.add_argument("--condition",required=True,choices=["control","static","interrupt"])
ap.add_argument("--ema_alpha",type=float,default=0.10)
ap.add_argument("--calib_time",type=float,default=3.0)
ap.add_argument("--warmup",type=float,default=8.0)
ap.add_argument("--need_drift",type=float,default=6.0)
ap.add_argument("--need_hold",type=float,default=1.2)
ap.add_argument("--hyst_reset",type=float,default=2.0)
ap.add_argument("--period",type=float,default=3.0)
ap.add_argument("--static_period",type=float,default=12.0)
ap.add_argument("--safety_drift",type=float,default=12.0)
ap.add_argument("--safety_hold",type=float,default=2.0)
a=ap.parse_args()

os.makedirs("logs/plots",exist_ok=True); os.makedirs("logs/reports",exist_ok=True)

src=a.seed if a.seed else (0 if a.device is None else a.device)
cap=cv2.VideoCapture(src)
if not cap.isOpened(): sys.exit(1)
is_seed=bool(a.seed)
if isinstance(src,int):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1920); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)
    t0=time.time(); ok=False
    while time.time()-t0<10:
        ok,_=cap.read()
        if ok: break
        cv2.waitKey(1)
    if not ok: sys.exit(2)

trial_id=str(int(time.time()))
seed_id=""
if is_seed:
    import hashlib
    h=hashlib.sha256()
    with open(a.seed,'rb') as f:
        for ch in iter(lambda:f.read(1<<20),b''): h.update(ch)
    seed_id=h.hexdigest()[:12]

csv_path=f"logs/session_{a.subject}_{a.condition}_{trial_id}.csv"
f=open(csv_path,"w",newline=""); cw=csv.writer(f)
cw.writerow(["subject","condition","trial_id","seed_id","t_abs","t_rel","t_vid","event","hip_angle","avg_angle","slope_5s","drift","pose_valid","channel","cmd_id","t_cmd_vid","t_confirm_vid","latency_s","confirm_source","expected_dir","ttf_s"])

pose=mp.solutions.pose.Pose(model_complexity=1,enable_segmentation=False,min_detection_confidence=0.3,min_tracking_confidence=0.3)
ema=EMA(a.ema_alpha); med=deque(maxlen=9); samples=deque(maxlen=7200)

baseline=None; t_start=None; last_pose_t=None; first_pose=False
need_since=None; last_tick=-1e9; pending=None; next_id=1; rearm=True
fail_since=None; overlay_dir=None; overlay_until=0.0

fps=cap.get(cv2.CAP_PROP_FPS) or 30.0; frame_idx=0; t_ref=time.monotonic()
win="ChatARangAI"; cv2.namedWindow(win,cv2.WINDOW_NORMAL)

def log(ev,tvid,hip,avg,slope,drift,pose_ok,channel="",extra=None):
    e=extra or {}
    t_abs=time.time(); t_rel=0.0 if t_start is None else (tvid - t_start)
    cw.writerow([a.subject,a.condition,trial_id,seed_id,f"{t_abs:.3f}",f"{t_rel:.3f}",f"{tvid:.3f}",ev,
                f"{hip:.3f}" if hip is not None else "", f"{avg:.3f}" if avg is not None else "", f"{slope:.4f}" if slope is not None else "",
                f"{drift:.3f}" if drift is not None else "", 1 if pose_ok else 0, channel,
                e.get("cmd_id",""), e.get("t_cmd_vid",""), e.get("t_confirm_vid",""), e.get("latency",""), e.get("confirm_source",""),
                e.get("expected_dir",""), e.get("ttf_s","")])
    f.flush()

def finalize_and_plot():
    try:
        df=pd.read_csv(csv_path)
        tv=pd.to_numeric(df["t_vid"],errors="coerce")
        ang=pd.to_numeric(df["avg_angle"],errors="coerce")
        if (df["event"]=="start").any(): t0=float(tv[df["event"]=="start"].iloc[0])
        else: t0=float(tv.iloc[0])
        base=float(np.median(ang[(tv>=t0)&(tv<=t0+5.0)])) if np.isfinite(t0) else float(np.median(ang.head(30)))
        msk=tv>=t0+a.warmup
        xs=tv[msk].to_numpy()-(t0+a.warmup)
        ys=(ang-base)[msk].to_numpy()
        gs=float(np.polyfit(xs,ys,1)[0]) if len(xs)>=10 else 0.0
        if abs(gs)<0.05: gs=0.0
        sfile="logs/reports/trial_summaries.csv"; hdr=not os.path.exists(sfile) or os.path.getsize(sfile)==0
        with open(sfile,"a",newline="") as sf:
            sw=csv.writer(sf)
            if hdr: sw.writerow(["subject","condition","trial_id","seed_id","ttf_s","global_slope_deg_s","hits","n_cmd","median_latency_s"])
            ttfv=df.loc[df["event"]=="failure","t_vid"]; ttf="" if ttfv.empty else f"{float(ttfv.iloc[-1])-t0:.3f}"
            sw.writerow([df["subject"].iloc[0],df["condition"].iloc[0],trial_id,seed_id,ttf,gs,int((df['event']=='confirm').sum()),int((df['event']=='cmd').sum()),""])
        fig,ax=plt.subplots(); ax.plot(tv,ang); ax.set_xlabel("time (s)"); ax.set_ylabel("avg_angle (deg)"); fig.savefig(f"logs/plots/{os.path.basename(csv_path).replace('.csv','_angle.png')}",dpi=120,bbox_inches="tight"); plt.close(fig)
    except Exception:
        pass

while True:
    ok,frame=cap.read()
    if not ok:
        if is_seed: break
        cv2.waitKey(1); continue
    frame_idx+=1
    tvid=(cap.get(cv2.CAP_PROP_POS_MSEC)/1000.0 if is_seed else (time.monotonic()-t_ref))
    if is_seed and (not np.isfinite(tvid) or tvid<=0): tvid=frame_idx/(fps if fps>0 else 30.0)

    h,W=frame.shape[:2]
    res=pose.process(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    from mediapipe import solutions as _s
    if res.pose_landmarks:
        _s.drawing_utils.draw_landmarks(frame,res.pose_landmarks,_s.pose.POSE_CONNECTIONS,_s.drawing_styles.get_default_pose_landmarks_style())

    hip=avg=slope=drift=None; pose_ok=False
    if res.pose_landmarks:
        lms=res.pose_landmarks.landmark
        vis=[lms[i.value].visibility for i in [LM.LEFT_SHOULDER,LM.RIGHT_SHOULDER,LM.LEFT_HIP,LM.RIGHT_HIP,LM.LEFT_ANKLE,LM.RIGHT_ANKLE]]
        if min(vis)>=0.1:
            first_pose=True; last_pose_t=tvid; pose_ok=True
            hip=hip_angle_deg(lms,W,h)
            a=ema.u(hip); med.append(a); avg=float(np.median(list(med))); samples.append((tvid,avg))

    slope=slope_last(samples,8.0,0.15) if len(samples)>=8 else 0.0

    if baseline is None and avg is not None and tvid>=a.calib_time:
        base_vals=[v for (t,v) in samples if tvid-t<=5.0]
        if len(base_vals)>=4:
            baseline=float(np.median(base_vals)); t_start=tvid
            log("calibrated",tvid,hip,avg,0.0,0.0,pose_ok); log("start",tvid,hip,avg,0.0,0.0,pose_ok); say("start")

    if baseline is not None and avg is not None:
        drift=avg-baseline
        log("sample",tvid,hip,avg,slope,drift,pose_ok)

    fail=False
    if baseline is not None and avg is not None:
        if abs(drift)>=a.safety_drift:
            if fail_since is None: fail_since=tvid
            if (tvid - fail_since)>=a.safety_hold:
                fail=True
        else:
            fail_since=None
    if first_pose and last_pose_t is not None and (tvid - last_pose_t)>=12.0:
        fail=True

    need=False; expected=""
    if baseline is not None and avg is not None and not fail:
        if drift>=a.need_drift: need=True; expected="down"
        elif drift<=-a.need_drift: need=True; expected="up"
        if need:
            if need_since is None: need_since=tvid
        else:
            need_since=None
        need_ok = need and (tvid - need_since)>=a.need_hold
        if a.condition=="interrupt" and pending is None and rearm and need_ok and (tvid - last_tick)>=a.period:
            last_tick=tvid; rearm=False
            ch="lower_back" if expected=="down" else "quads"
            cid=next_id; next_id+=1
            pending={"cmd_id":cid,"channel":ch,"t_cmd_vid":tvid,"expected_dir":expected}
            log("cmd",tvid,hip,avg,slope,drift,pose_ok,ch,{"cmd_id":cid,"t_cmd_vid":f"{tvid:.3f}","expected_dir":expected})
            overlay_dir="up" if ch=="quads" else "down"; overlay_until=tvid+1.5
            say("channel one up" if ch=="quads" else "channel two down")

    if a.condition=="static" and baseline is not None and not fail and pending is None and (tvid - last_tick)>=a.static_period:
        last_tick=tvid
        ch="quads" if ((int((tvid - t_start)//a.static_period))%2==0) else "lower_back"
        cid=next_id; next_id+=1
        pending={"cmd_id":cid,"channel":ch,"t_cmd_vid":tvid,"expected_dir":"up" if ch=="quads" else "down"}
        log("cmd",tvid,hip,avg,slope,drift,pose_ok,ch,{"cmd_id":cid,"t_cmd_vid":f"{tvid:.3f}","expected_dir":pending["expected_dir"]})
        overlay_dir="up" if ch=="quads" else "down"; overlay_until=tvid+1.5
        say("channel one up" if ch=="quads" else "channel two down")

    if baseline is not None and avg is not None and ((not rearm and abs(drift)<=a.hyst_reset) or (tvid - last_tick)>(a.period+2.0)):
        rearm=True
    if overlay_dir and tvid>=overlay_until:
        overlay_dir=None; pending=None

    overlay=frame.copy()
    if overlay_dir and tvid<overlay_until:
        cx,cy=W//2,h//2
        if overlay_dir=="up":
            cv2.arrowedLine(overlay,(cx,cy+140),(cx,cy-140),(0,255,0),12,tipLength=0.25)
            cv2.putText(overlay,"UP",(cx-30,cy-170),cv2.FONT_HERSHEY_SIMPLEX,1.4,(0,255,0),3)
        else:
            cv2.arrowedLine(overlay,(cx,cy-140),(cx,cy+140),(0,0,255),12,tipLength=0.25)
            cv2.putText(overlay,"DOWN",(cx-60,cy+180),cv2.FONT_HERSHEY_SIMPLEX,1.4,(0,0,255),3)
    hud=f"{a.subject}/{a.condition}  t:{0.0 if t_start is None else (tvid-t_start):.1f}s  drift:{0.0 if drift is None else drift:.2f}  slope:{0.0 if slope is None else slope:.3f}"
    cv2.putText(overlay,hud,(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
    out=cv2.addWeighted(overlay,0.35,frame,0.65,0); cv2.imshow(win,out)

    k=cv2.waitKey(1)&0xFF
    if k==ord('q'): break
    elif k==ord('e'):
        ttf=(tvid - t_start) if t_start is not None else ""
        log("failure",tvid,hip,avg,slope,drift,pose_ok,"",{"ttf_s":f"{ttf:.3f}" if ttf!="" else ""})
        say("good job"); break
    elif k==ord('c'):
        if avg is not None:
            baseline=float(np.median(list(med))); t_start=tvid
            log("calibrated",tvid,hip,avg,0.0,0.0,pose_ok); log("start",tvid,hip,avg,0.0,0.0,pose_ok); say("start")

    if fail:
        ttf=(tvid - t_start) if t_start is not None else ""
        log("failure",tvid,hip,avg,slope,drift,pose_ok,"",{"ttf_s":f"{ttf:.3f}" if ttf!="" else ""})
        say("good job"); break

cap.release(); cv2.destroyAllWindows(); f.close()
finalize_and_plot()
