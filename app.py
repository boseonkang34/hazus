#!/usr/bin/env python3
"""
DDN Track-B Streamlit App
=========================
Purpose
- Read the DB structure produced by collect_trackb_data_v2.py.
- Show collection health before any risk result.
- Search an actual building from trackb_building_features.
- Show only evidence that really exists for that building.
- Show current/forecast KMA data by the building's stored KMA grid.
- Show spatial centroid proxy with an explicit non-wall-gap warning.
- Show model scores ONLY when a real score table exists. Never fabricate a score.
- Keep typhoon/WeatherNext disabled until a validated model/data pipeline is connected.

Run:
    streamlit run app.py

Environment:
    /Users/12609/Documents/hazus/.env
"""
from __future__ import annotations

import json
import math
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# [APP-CONFIG-01] Same .env contract as collector/monitor.
# -----------------------------------------------------------------------------
DEFAULT_HOME = Path("/Users/12609/Documents/hazus")
load_dotenv(DEFAULT_HOME / ".env", override=False)
HOME = Path(os.getenv("CLIP_HOME", str(DEFAULT_HOME))).expanduser()
load_dotenv(HOME / ".env", override=False)

LIVE_DB = Path(os.getenv(
    "CLIP_NATIONWIDE_INTEGRATED_DB",
    str(HOME / "clip_nationwide_integrated_master.db")
)).expanduser()
CORE_DB = Path(os.getenv(
    "TRACKB_CORE_DB",
    str(HOME / "clip_trackb_core.db")
)).expanduser()
WEATHER_DB = Path(os.getenv(
    "TRACKB_WEATHER_DB",
    str(HOME / "clip_trackb_weather.db")
)).expanduser()
RAW_DIR = Path(os.getenv("TRACKB_RAW_DIR", str(HOME / "trackb_raw"))).expanduser()

FIRE_SCORE_TABLE = os.getenv("TRACKB_FIRE_SCORE_TABLE", "trackb_score_fire")
FLOOD_SCORE_TABLE = os.getenv("TRACKB_FLOOD_SCORE_TABLE", "trackb_score_flood")
STALL_MINUTES = int(os.getenv("TRACKB_STALL_MINUTES", "20"))
MIN_FREE_GB = float(os.getenv("TRACKB_MIN_FREE_GB", "20"))
KST = timezone(timedelta(hours=9))

st.set_page_config(
    page_title="DDN Track-B",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# [DB-READ-01] App is read-only. It must not mutate collector databases.
# -----------------------------------------------------------------------------
def ro(path: Path) -> Optional[sqlite3.Connection]:
    if not path.exists():
        return None
    c = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=10)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA busy_timeout=10000")
    return c


def table_exists(c: sqlite3.Connection, table: str) -> bool:
    return bool(c.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone())


def table_columns(c: sqlite3.Connection, table: str) -> list[str]:
    if not table_exists(c, table):
        return []
    return [r["name"] for r in c.execute(f'PRAGMA table_info("{table}")')]


def scalar(c: sqlite3.Connection, sql: str, params: tuple = (), default=None):
    try:
        row = c.execute(sql, params).fetchone()
        return row[0] if row else default
    except sqlite3.Error:
        return default


def df_query(c: sqlite3.Connection, sql: str, params: tuple = ()) -> pd.DataFrame:
    return pd.read_sql_query(sql, c, params=params)


def fmt_int(v: Any) -> str:
    try:
        return f"{int(v):,}"
    except Exception:
        return "-"


def fmt_num(v: Any, digits: int = 1) -> str:
    try:
        x=float(v)
        return f"{x:,.{digits}f}" if math.isfinite(x) else "-"
    except Exception:
        return "-"


def age_minutes(ts: Any) -> Optional[float]:
    if not ts:
        return None
    try:
        d=datetime.fromisoformat(str(ts))
        now=datetime.now(d.tzinfo) if d.tzinfo else datetime.now()
        return max(0.0,(now-d).total_seconds()/60)
    except Exception:
        return None


# -----------------------------------------------------------------------------
# [HEALTH-01] Collection health is shown before risk interpretation.
# -----------------------------------------------------------------------------
@st.cache_data(ttl=10, show_spinner=False)
def health_snapshot() -> dict:
    import shutil
    out={"alerts":[]}
    try:
        usage=shutil.disk_usage(HOME if HOME.exists() else Path.home())
        out["disk_free_gb"]=usage.free/1024**3
        if out["disk_free_gb"] < MIN_FREE_GB:
            out["alerts"].append(("CRITICAL",f"남은 디스크 {out['disk_free_gb']:.1f}GB"))
    except Exception:
        out["disk_free_gb"]=None

    c=ro(CORE_DB)
    if not c:
        out["alerts"].append(("CRITICAL","Track-B CORE DB가 없습니다."))
        return out

    try:
        out["buildings"]=scalar(c,"SELECT COUNT(*) FROM trackb_building_features",default=0)
        out["valid_coords"]=scalar(c,"SELECT COUNT(*) FROM trackb_building_features WHERE lat IS NOT NULL AND lon IS NOT NULL",default=0)
        out["fire"]=scalar(c,"SELECT COUNT(*) FROM trackb_fire_events",default=0) if table_exists(c,"trackb_fire_events") else 0
        out["sprinkler"]=scalar(c,"SELECT COUNT(*) FROM trackb_sprinkler_evidence",default=0) if table_exists(c,"trackb_sprinkler_evidence") else 0
        out["flood"]=scalar(c,"SELECT COUNT(*) FROM trackb_flood_evidence",default=0) if table_exists(c,"trackb_flood_evidence") else 0
        out["spatial"]=scalar(c,"SELECT COUNT(*) FROM trackb_spatial_features",default=0) if table_exists(c,"trackb_spatial_features") else 0

        if table_exists(c,"trackb_runs"):
            out["last_runs"]=[dict(r) for r in c.execute("""
                SELECT stage,status,started_at,finished_at,attempted,written,unresolved,last_error
                FROM trackb_runs ORDER BY started_at DESC LIMIT 6
            """)]
            for r in out["last_runs"]:
                if r["status"]=="FAILED":
                    out["alerts"].append(("ERROR",f"{r['stage']} 실패: {r['last_error'] or '원인 미기록'}"))
        else:
            out["last_runs"]=[]

        if table_exists(c,"trackb_checkpoints"):
            out["checkpoints"]=[dict(r) for r in c.execute("""
                SELECT stage,partition_key,cursor_value,status,updated_at
                FROM trackb_checkpoints ORDER BY updated_at DESC LIMIT 8
            """)]
        else:
            out["checkpoints"]=[]
    finally:
        c.close()

    w=ro(WEATHER_DB)
    if w:
        try:
            out["weather_rows"]=scalar(w,"SELECT COUNT(*) FROM weather_observations",default=0) if table_exists(w,"weather_observations") else 0
            out["weather_failed"]=scalar(w,"SELECT COUNT(*) FROM weather_checkpoints WHERE status='FAILED'",default=0) if table_exists(w,"weather_checkpoints") else 0
            if out["weather_failed"]:
                out["alerts"].append(("WARN",f"KMA 실패 checkpoint {out['weather_failed']:,}개"))
        finally:
            w.close()
    else:
        out["weather_rows"]=0
        out["weather_failed"]=0

    return out


# -----------------------------------------------------------------------------
# [SEARCH-01] Indexed exact ID/PNU and bounded address/name LIKE search.
# For nationwide fuzzy search, a separate FTS index should be built offline.
# -----------------------------------------------------------------------------
@st.cache_data(ttl=30, show_spinner=False)
def search_buildings(q: str, limit: int = 100) -> pd.DataFrame:
    q=q.strip()
    if not q:
        return pd.DataFrame()
    c=ro(CORE_DB)
    if not c:
        return pd.DataFrame()
    try:
        if q.isdigit():
            rid=int(q)
            return df_query(c,"""
                SELECT * FROM trackb_building_features
                WHERE building_rowid=? OR pnu=?
                LIMIT ?
            """,(rid,q,limit))
        nq="".join(q.lower().split())
        # address_norm is indexed for exact normalized address.
        exact=df_query(c,"""
            SELECT * FROM trackb_building_features
            WHERE address_norm=? OR pnu=?
            LIMIT ?
        """,(nq,q,limit))
        if not exact.empty:
            return exact
        # Bounded fallback; correct but not claimed to be optimal for 7.8m fuzzy search.
        return df_query(c,"""
            SELECT * FROM trackb_building_features
            WHERE address LIKE ? OR name LIKE ?
            LIMIT ?
        """,(f"%{q}%",f"%{q}%",limit))
    finally:
        c.close()


# -----------------------------------------------------------------------------
# [OBJECT-01] Fetch one building and only its real evidence.
# -----------------------------------------------------------------------------
@st.cache_data(ttl=15, show_spinner=False)
def object_bundle(building_rowid: int) -> dict:
    out={}
    c=ro(CORE_DB)
    if not c:
        return out
    try:
        b=c.execute(
            "SELECT * FROM trackb_building_features WHERE building_rowid=?",
            (int(building_rowid),)
        ).fetchone()
        if not b:
            return out
        out["building"]=dict(b)

        if table_exists(c,"trackb_fire_events"):
            out["fire"]=df_query(c,"""
                SELECT event_date,event_type,match_status,source_id,source_reference,collected_at
                FROM trackb_fire_events WHERE building_rowid=?
                ORDER BY event_date DESC
            """,(building_rowid,))
        if table_exists(c,"trackb_sprinkler_evidence"):
            out["sprinkler"]=df_query(c,"""
                SELECT observed_at,sprinkler_status,match_status,source_id,source_reference,collected_at
                FROM trackb_sprinkler_evidence WHERE building_rowid=?
                ORDER BY collected_at DESC
            """,(building_rowid,))
        if table_exists(c,"trackb_flood_evidence"):
            out["flood"]=df_query(c,"""
                SELECT observed_at,observed_flood,match_status,source_id,source_reference,collected_at
                FROM trackb_flood_evidence WHERE building_rowid=?
                ORDER BY collected_at DESC
            """,(building_rowid,))
        if table_exists(c,"trackb_spatial_features"):
            r=c.execute(
                "SELECT * FROM trackb_spatial_features WHERE building_rowid=?",
                (building_rowid,)
            ).fetchone()
            out["spatial"]=dict(r) if r else None

        out["scores"]={}
        for hazard,table in [("fire",FIRE_SCORE_TABLE),("flood",FLOOD_SCORE_TABLE)]:
            if table_exists(c,table):
                cols=table_columns(c,table)
                score_col=next((x for x in ["score","risk_score","prediction_score"] if x in cols),None)
                if score_col and "building_rowid" in cols:
                    r=c.execute(
                        f'SELECT "{score_col}" AS score,* FROM "{table}" WHERE building_rowid=? LIMIT 1',
                        (building_rowid,)
                    ).fetchone()
                    if r:
                        out["scores"][hazard]=dict(r)
    finally:
        c.close()

    b=out.get("building",{})
    nx,ny=b.get("kma_nx"),b.get("kma_ny")
    w=ro(WEATHER_DB)
    if w and nx is not None and ny is not None:
        try:
            if table_exists(w,"weather_observations"):
                out["weather"]=df_query(w,"""
                    SELECT operation,base_date,base_time,forecast_date,forecast_time,
                           category,value,collected_at
                    FROM weather_observations
                    WHERE nx=? AND ny=?
                    ORDER BY base_date DESC,base_time DESC,forecast_date,forecast_time
                    LIMIT 200
                """,(int(nx),int(ny)))
        finally:
            w.close()
    return out


# -----------------------------------------------------------------------------
# [MAP-01] Only actual coordinates are mapped. No random/fabricated coordinates.
# -----------------------------------------------------------------------------
def render_map(b: dict):
    lat,lon=b.get("lat"),b.get("lon")
    if lat is None or lon is None:
        st.info("이 건물에는 검증 가능한 좌표가 없어 지도를 표시하지 않습니다.")
        return
    st.map(pd.DataFrame([{"lat":float(lat),"lon":float(lon)}]),zoom=16)


# -----------------------------------------------------------------------------
# [SCORE-01] Score semantics are conservative.
# Case-control CatBoost output is not automatically an annual loss probability.
# -----------------------------------------------------------------------------
def render_score(label: str, score_record: Optional[dict]):
    if not score_record:
        st.metric(label,"미산출")
        st.caption("실제 score table에 이 건물 결과가 없으므로 점수를 만들지 않았습니다.")
        return
    v=score_record.get("score")
    st.metric(label,fmt_num(v,2))
    st.caption("모델 상대위험 점수. 별도 확률 보정이 없다면 사고확률·보험료율로 해석하지 않습니다.")


# -----------------------------------------------------------------------------
# [PAGE] Dashboard
# -----------------------------------------------------------------------------
def page_dashboard():
    st.title("🏢 DDN Track-B")
    st.caption("전국 건축물 기반 화재·침수·기상·공간 위험정보 — evidence first")

    h=health_snapshot()
    a,b,c,d,e=st.columns(5)
    a.metric("건축물",fmt_int(h.get("buildings")))
    b.metric("좌표 확보",fmt_int(h.get("valid_coords")))
    c.metric("화재 evidence",fmt_int(h.get("fire")))
    d.metric("소방시설 evidence",fmt_int(h.get("sprinkler")))
    e.metric("침수 evidence",fmt_int(h.get("flood")))

    a,b,c,d=st.columns(4)
    a.metric("공간 feature",fmt_int(h.get("spatial")))
    b.metric("기상 rows",fmt_int(h.get("weather_rows")))
    c.metric("KMA 실패",fmt_int(h.get("weather_failed")))
    free=h.get("disk_free_gb")
    d.metric("남은 디스크",f"{free:.1f} GB" if free is not None else "-")

    if h.get("alerts"):
        st.subheader("🚨 수집 상태 경고")
        for level,msg in h["alerts"]:
            (st.error if level in {"ERROR","CRITICAL"} else st.warning)(f"[{level}] {msg}")
    else:
        st.success("현재 앱이 확인한 DB 상태에서 즉시 표시할 경고는 없습니다.")
        st.caption("이 문구는 외부 API 전체 완전성이나 모델 정확도를 보증하지 않습니다.")

    st.subheader("최근 수집 작업")
    if h.get("last_runs"):
        st.dataframe(pd.DataFrame(h["last_runs"]),use_container_width=True,hide_index=True)
    else:
        st.info("수집 run 기록이 없습니다.")

    st.subheader("Checkpoint")
    if h.get("checkpoints"):
        cp=pd.DataFrame(h["checkpoints"])
        st.dataframe(cp,use_container_width=True,hide_index=True)
    else:
        st.info("Checkpoint가 없습니다.")


# -----------------------------------------------------------------------------
# [PAGE] Building risk/evidence lookup
# -----------------------------------------------------------------------------
def page_building():
    st.title("🔎 건축물 위험 조회")
    q=st.text_input("주소 / 건물명 / building_rowid / PNU",placeholder="예: 서울특별시 ...")
    if not q:
        st.info("검색어를 입력하세요.")
        return

    result=search_buildings(q)
    if result.empty:
        st.warning("일치하는 건물을 찾지 못했습니다.")
        return

    label_cols=[x for x in ["building_rowid","address","name","purpose"] if x in result.columns]
    choices=[]
    for _,r in result.iterrows():
        label=" | ".join(str(r.get(x) or "") for x in label_cols)
        choices.append((int(r["building_rowid"]),label))
    selected=st.selectbox("건물 선택",choices,format_func=lambda x:x[1])
    rid=selected[0]
    data=object_bundle(rid)
    b=data.get("building")
    if not b:
        st.error("건물 상세정보를 읽지 못했습니다.")
        return

    st.subheader(b.get("name") or b.get("address") or f"building_rowid={rid}")
    st.caption(f"building_rowid={rid} · source={b.get('source_id','-')}")

    c1,c2,c3,c4=st.columns(4)
    c1.metric("주용도",str(b.get("purpose") or "-"))
    c2.metric("연면적",f"{fmt_num(b.get('area'))} ㎡")
    c3.metric("높이",f"{fmt_num(b.get('height'))} m")
    c4.metric("지상층",fmt_num(b.get("floors"),0))

    c1,c2,c3,c4=st.columns(4)
    c1.metric("구조",str(b.get("structure") or "-"))
    c2.metric("지하층",fmt_num(b.get("basement"),0))
    c3.metric("사용승인",str(b.get("approval") or "-"))
    c4.metric("KMA Grid",f"{b.get('kma_nx','-')}, {b.get('kma_ny','-')}")

    render_map(b)

    st.divider()
    st.subheader("모델 점수")
    c1,c2=st.columns(2)
    with c1: render_score("🔥 화재 상대위험",data.get("scores",{}).get("fire"))
    with c2: render_score("🌊 침수 상대위험",data.get("scores",{}).get("flood"))

    st.divider()
    tabs=st.tabs(["🔥 화재이력","🧯 소방시설","🌊 침수흔적","📐 공간","🌦 기상"])
    with tabs[0]:
        x=data.get("fire",pd.DataFrame())
        if x.empty: st.info("매칭된 화재 evidence가 없습니다. 이것을 '화재 없음'으로 해석하지 않습니다.")
        else: st.dataframe(x,use_container_width=True,hide_index=True)
    with tabs[1]:
        x=data.get("sprinkler",pd.DataFrame())
        if x.empty: st.info("소방시설 evidence가 없습니다. 설치되지 않았다는 뜻이 아닙니다.")
        else: st.dataframe(x,use_container_width=True,hide_index=True)
    with tabs[2]:
        x=data.get("flood",pd.DataFrame())
        if x.empty: st.info("침수 evidence가 없습니다. 침수 이력이 없다는 뜻이 아닙니다.")
        else: st.dataframe(x,use_container_width=True,hide_index=True)
    with tabs[3]:
        x=data.get("spatial")
        if not x:
            st.info("공간 feature가 아직 계산되지 않았습니다.")
        else:
            a,bx=st.columns(2)
            a.metric("100m 이내 건물 수",fmt_int(x.get("neighbor_100m_count")))
            bx.metric("최근접 중심점 거리",f"{fmt_num(x.get('nearest_centroid_m'),1)} m")
            st.warning("현재 값은 건물 중심점 기준 proxy입니다. 실제 외벽-외벽 이격거리가 아닙니다.")
            st.caption(f"method={x.get('spatial_method')}")
    with tabs[4]:
        x=data.get("weather",pd.DataFrame())
        if x.empty:
            st.info("해당 KMA 격자의 수집된 실황/예보가 없습니다.")
        else:
            st.dataframe(x,use_container_width=True,hide_index=True)
            st.warning("현재/예보 기상자료입니다. 과거 사고시점의 학습용 역사기상으로 사용하면 안 됩니다.")


# -----------------------------------------------------------------------------
# [PAGE] Portfolio upload: score lookup only, never synthetic expected loss.
# -----------------------------------------------------------------------------
def page_portfolio():
    st.title("📁 다중물건 조회")
    st.caption("CSV의 building_rowid를 실제 DB와 결합합니다. 존재하지 않는 모델점수는 생성하지 않습니다.")
    up=st.file_uploader("CSV 업로드",type=["csv"])
    if up is None:
        st.info("필수 열: building_rowid")
        return
    df=pd.read_csv(up)
    if "building_rowid" not in df.columns:
        st.error("building_rowid 열이 필요합니다.")
        return
    ids=pd.to_numeric(df["building_rowid"],errors="coerce")
    valid=df.loc[ids.notna()].copy()
    valid["building_rowid"]=ids.loc[ids.notna()].astype("int64")
    if valid.empty:
        st.error("유효한 building_rowid가 없습니다.")
        return

    c=ro(CORE_DB)
    if not c:
        st.error("CORE DB가 없습니다.")
        return
    try:
        parts=[]
        unique_ids=valid["building_rowid"].drop_duplicates().tolist()
        for i in range(0,len(unique_ids),900):
            part=unique_ids[i:i+900]
            marks=",".join("?" for _ in part)
            q=f"""SELECT building_rowid,address,name,purpose,lat,lon
                  FROM trackb_building_features WHERE building_rowid IN ({marks})"""
            parts.append(df_query(c,q,tuple(part)))
        master=pd.concat(parts,ignore_index=True) if parts else pd.DataFrame()

        for hz,table in [("fire",FIRE_SCORE_TABLE),("flood",FLOOD_SCORE_TABLE)]:
            if table_exists(c,table):
                cols=table_columns(c,table)
                score_col=next((x for x in ["score","risk_score","prediction_score"] if x in cols),None)
                if score_col and "building_rowid" in cols:
                    scores=[]
                    for i in range(0,len(unique_ids),900):
                        part=unique_ids[i:i+900]
                        marks=",".join("?" for _ in part)
                        scores.append(df_query(
                            c,f'SELECT building_rowid,"{score_col}" AS {hz}_score FROM "{table}" WHERE building_rowid IN ({marks})',
                            tuple(part)
                        ))
                    if scores:
                        master=master.merge(pd.concat(scores,ignore_index=True),on="building_rowid",how="left")
    finally:
        c.close()

    result=valid.merge(master,on="building_rowid",how="left")
    st.metric("입력 물건",fmt_int(len(valid)))
    st.metric("건물 DB 매칭",fmt_int(result["address"].notna().sum() if "address" in result else 0))
    st.dataframe(result,use_container_width=True,hide_index=True)
    st.download_button(
        "결과 CSV 다운로드",
        result.to_csv(index=False).encode("utf-8-sig"),
        file_name="ddn_trackb_portfolio_result.csv",
        mime="text/csv",
    )


# -----------------------------------------------------------------------------
# [PAGE] Typhoon is intentionally gated.
# -----------------------------------------------------------------------------
def page_typhoon():
    st.title("🌀 태풍 위험")
    st.warning("현재 운영 점수 산출 비활성화")
    st.write(
        "실제 WeatherNext 경로/강도 데이터, 건물별 취약도 함수, 검증된 태풍 피해 Y와 "
        "손해율 보정이 연결되기 전에는 예상손해액이나 피해확률을 계산하지 않습니다."
    )
    st.code(
        "WeatherNext/공식 태풍외력 → 건물별 풍속·강우 외력 → 검증된 취약도 → "
        "실제 피해자료 검증 → calibration → 운영점수",
        language=None,
    )


# -----------------------------------------------------------------------------
# [APP] Navigation
# -----------------------------------------------------------------------------
def main():
    with st.sidebar:
        st.title("DDN")
        page=st.radio(
            "Track-B",
            ["운영 현황","건축물 위험 조회","다중물건","태풍"],
            index=0,
        )
        st.divider()
        st.caption(f"CORE: {CORE_DB.name}")
        st.caption(f"WEATHER: {WEATHER_DB.name}")
        if st.button("🔄 캐시 새로고침"):
            st.cache_data.clear()
            st.rerun()

    if page=="운영 현황":
        page_dashboard()
    elif page=="건축물 위험 조회":
        page_building()
    elif page=="다중물건":
        page_portfolio()
    else:
        page_typhoon()


if __name__=="__main__":
    main()
