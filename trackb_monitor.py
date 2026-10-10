#!/usr/bin/env python3
"""
DDN Track-B live monitor.
Read-only monitoring dashboard for the collector databases/process/log/disk.

Run:
    python trackb_monitor.py
Optional:
    python trackb_monitor.py --interval 10
    python trackb_monitor.py --once

No external packages required.
"""
from __future__ import annotations
import argparse, json, os, shutil, sqlite3, subprocess, sys, time
from datetime import datetime
from pathlib import Path

DEFAULT_HOME = Path("/Users/12609/Documents/hazus")
HOME = Path(os.getenv("CLIP_HOME", str(DEFAULT_HOME))).expanduser()
LIVE_DB = Path(os.getenv("CLIP_NATIONWIDE_INTEGRATED_DB",
              str(HOME / "clip_nationwide_integrated_master.db"))).expanduser()
CORE_DB = Path(os.getenv("TRACKB_CORE_DB", str(HOME / "clip_trackb_core.db"))).expanduser()
WEATHER_DB = Path(os.getenv("TRACKB_WEATHER_DB", str(HOME / "clip_trackb_weather.db"))).expanduser()
BUILDING_TABLE = os.getenv("BUILDING_OUTPUT_TABLE", "nationwide_integrated_sheet")
MIN_FREE_GB = float(os.getenv("TRACKB_MIN_FREE_GB", "20"))
STALL_MINUTES = int(os.getenv("TRACKB_STALL_MINUTES", "20"))

def ro(path):
    if not path.exists():
        return None
    c = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA busy_timeout=5000")
    return c

def scalar(c, sql, args=(), default=None):
    try:
        r = c.execute(sql, args).fetchone()
        return r[0] if r else default
    except Exception:
        return default

def table_exists(c, name):
    return bool(scalar(c, "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,), 0))

def file_size(path):
    try: return path.stat().st_size
    except: return 0

def fmt_bytes(n):
    units=["B","KB","MB","GB","TB"]; x=float(n or 0)
    for u in units:
        if x < 1024 or u=="TB": return f"{x:.2f} {u}"
        x/=1024

def age_minutes(ts):
    if not ts: return None
    try:
        dt=datetime.fromisoformat(str(ts))
        if dt.tzinfo:
            return max(0,(datetime.now(dt.tzinfo)-dt).total_seconds()/60)
        return max(0,(datetime.now()-dt).total_seconds()/60)
    except: return None

def process_state():
    # macOS/Linux: observational only. Failure is reported as UNKNOWN, not interpreted as stopped.
    try:
        p=subprocess.run(["pgrep","-af","collect_trackb_data"],capture_output=True,text=True,timeout=2)
        lines=[x.strip() for x in p.stdout.splitlines() if x.strip() and "trackb_monitor.py" not in x]
        return lines
    except Exception:
        return []

def collect():
    out={"time":datetime.now().isoformat(timespec="seconds"),"alerts":[]}
    usage=shutil.disk_usage(HOME if HOME.exists() else Path.home())
    free_gb=usage.free/1024**3
    out["disk"]={"free_gb":free_gb,"total_gb":usage.total/1024**3}
    if free_gb < MIN_FREE_GB:
        out["alerts"].append(("CRITICAL",f"Disk free {free_gb:.1f} GiB < guard {MIN_FREE_GB:.1f} GiB"))

    out["files"]={
        "live":fmt_bytes(file_size(LIVE_DB)),
        "core":fmt_bytes(file_size(CORE_DB)),
        "core_wal":fmt_bytes(file_size(Path(str(CORE_DB)+"-wal"))),
        "weather":fmt_bytes(file_size(WEATHER_DB)),
        "weather_wal":fmt_bytes(file_size(Path(str(WEATHER_DB)+"-wal"))),
    }

    # Source count
    src=ro(LIVE_DB)
    if src:
        out["source_buildings"]=scalar(src,f'SELECT COUNT(*) FROM "{BUILDING_TABLE}"',default=None)
        src.close()
    else:
        out["source_buildings"]=None
        out["alerts"].append(("CRITICAL",f"Source DB missing: {LIVE_DB}"))

    core=ro(CORE_DB)
    if core:
        names=["trackb_building_features","trackb_fire_events","trackb_sprinkler_evidence",
               "trackb_flood_evidence","trackb_spatial_features","trackb_runs","trackb_checkpoints"]
        out["counts"]={n: scalar(core,f'SELECT COUNT(*) FROM "{n}"',default=None) if table_exists(core,n) else None for n in names}
        out["valid_coords"]=scalar(core,"SELECT COUNT(*) FROM trackb_building_features WHERE lat IS NOT NULL AND lon IS NOT NULL",default=0)
        out["unique_grids"]=scalar(core,"SELECT COUNT(*) FROM (SELECT DISTINCT nx,ny FROM trackb_weather_grid_map)",default=0) if table_exists(core,"trackb_weather_grid_map") else 0
        out["exact_fire"]=scalar(core,"SELECT COUNT(*) FROM trackb_fire_events WHERE match_status='EXACT_UNIQUE'",default=0)
        out["known_sprinkler"]=scalar(core,"SELECT COUNT(*) FROM trackb_sprinkler_evidence WHERE match_status='EXACT_UNIQUE' AND sprinkler_status IN ('YES','NO')",default=0)
        out["positive_flood"]=scalar(core,"SELECT COUNT(*) FROM trackb_flood_evidence WHERE match_status='EXACT_UNIQUE' AND observed_flood='YES'",default=0)

        out["runs"]=[]
        if table_exists(core,"trackb_runs"):
            for r in core.execute("""SELECT run_id,stage,status,started_at,finished_at,attempted,written,unresolved,last_error
                                     FROM trackb_runs ORDER BY started_at DESC LIMIT 8"""):
                out["runs"].append(dict(r))
                if r["status"]=="FAILED":
                    out["alerts"].append(("ERROR",f"{r['stage']} failed: {r['last_error'] or 'unknown'}"))

        out["checkpoints"]=[]
        if table_exists(core,"trackb_checkpoints"):
            for r in core.execute("""SELECT stage,partition_key,cursor_value,status,updated_at
                                     FROM trackb_checkpoints ORDER BY updated_at DESC LIMIT 12"""):
                d=dict(r); out["checkpoints"].append(d)
                age=age_minutes(r["updated_at"])
                if r["status"]=="OK" and age is not None and age>STALL_MINUTES:
                    # Only flag as possible stall if collector process is actually observed below.
                    d["_age_min"]=age
        core.close()
    else:
        out["counts"]={}
        out["runs"]=[]; out["checkpoints"]=[]
        out["alerts"].append(("WARN",f"Core DB not created yet: {CORE_DB}"))

    w=ro(WEATHER_DB)
    if w:
        out["weather_rows"]=scalar(w,"SELECT COUNT(*) FROM weather_observations",default=0) if table_exists(w,"weather_observations") else 0
        out["weather_failed_grids"]=scalar(w,"SELECT COUNT(*) FROM weather_checkpoints WHERE status='FAILED'",default=0) if table_exists(w,"weather_checkpoints") else 0
        if out["weather_failed_grids"]:
            out["alerts"].append(("WARN",f"KMA failed grid checkpoints: {out['weather_failed_grids']:,}"))
        w.close()
    else:
        out["weather_rows"]=0; out["weather_failed_grids"]=0

    procs=process_state()
    out["processes"]=procs
    if procs:
        stale=[x for x in out.get("checkpoints",[]) if x.get("_age_min",0)>STALL_MINUTES]
        if stale:
            out["alerts"].append(("WARN",f"Collector process exists but newest shown checkpoint is > {STALL_MINUTES} min old; inspect log/API/DB lock."))

    sb=out.get("source_buildings")
    tb=out.get("counts",{}).get("trackb_building_features")
    out["building_pct"]=(tb/sb*100) if sb and tb is not None else None
    return out

def bar(pct,width=32):
    if pct is None: return "["+"?"*width+"]"
    pct=max(0,min(100,pct)); n=round(width*pct/100)
    return "["+"#"*n+"-"*(width-n)+"]"

def render(x):
    os.system("clear" if os.name!="nt" else "cls")
    print("="*86)
    print(" DDN TRACK-B LIVE MONITOR   ",x["time"])
    print("="*86)
    pct=x.get("building_pct")
    print(f" BUILDINGS  {bar(pct)}  {pct:7.3f}%" if pct is not None else " BUILDINGS  [source/count unavailable]")
    print(f" source={x.get('source_buildings')!s:>12}  trackb={x.get('counts',{}).get('trackb_building_features')!s:>12}  valid_coord={x.get('valid_coords',0):,}")
    print(f" KMA unique grids={x.get('unique_grids',0):,}  weather rows={x.get('weather_rows',0):,}  failed grids={x.get('weather_failed_grids',0):,}")
    print(f" fire exact={x.get('exact_fire',0):,}  sprinkler known={x.get('known_sprinkler',0):,}  flood positive={x.get('positive_flood',0):,}")
    print("-"*86)
    d=x["disk"]; print(f" DISK free={d['free_gb']:.1f} GiB / total={d['total_gb']:.1f} GiB")
    f=x["files"]; print(f" DB live={f['live']} core={f['core']} (+WAL {f['core_wal']}) weather={f['weather']} (+WAL {f['weather_wal']})")
    print("-"*86)
    print(" PROCESS")
    if x["processes"]:
        for p in x["processes"][:4]: print("  RUNNING:",p[:76])
    else:
        print("  No collect_trackb_data process observed (this alone is not an error).")
    print("-"*86)
    print(" RECENT RUNS")
    if not x.get("runs"): print("  none")
    for r in x.get("runs",[])[:6]:
        print(f"  {r['status']:<8} {r['stage']:<20} attempted={r['attempted']:,} written={r['written']:,} unresolved={r['unresolved']:,}")
        if r.get("last_error"): print("    error:",str(r["last_error"])[:72])
    print("-"*86)
    print(" CHECKPOINTS")
    if not x.get("checkpoints"): print("  none")
    for c in x.get("checkpoints",[])[:8]:
        age=age_minutes(c.get("updated_at"))
        age_txt=f"{age:.1f}m ago" if age is not None else "?"
        print(f"  {c['stage']:<20} {c['partition_key']:<16} cursor={str(c['cursor_value'])[:16]:<16} {c['status']:<7} {age_txt}")
    print("-"*86)
    if x["alerts"]:
        print(" ALERTS")
        for level,msg in x["alerts"]: print(f"  [{level}] {msg}")
    else:
        print(" ALERTS: none detected by monitor checks")
    print("="*86)
    print("Monitor is read-only. 'No alert' means these checks passed; it does not prove external API completeness.")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--interval",type=float,default=10.0)
    ap.add_argument("--once",action="store_true")
    ap.add_argument("--json",action="store_true")
    a=ap.parse_args()
    while True:
        x=collect()
        if a.json: print(json.dumps(x,ensure_ascii=False,indent=2))
        else: render(x)
        if a.once: break
        try: time.sleep(max(2,a.interval))
        except KeyboardInterrupt: break

if __name__=="__main__":
    main()
