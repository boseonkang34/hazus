#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 HAZUS Track-B — 기존 UI 보존형 운영 앱
================================================
기준:
1) 기존 Fire / Flood / Typhoon 메뉴와 주요 화면 구성은 유지한다.
2) 가짜 target, random 좌표, hash/mod 평균, SHAP 합산 점수, 임의 손해율은 제거한다.
3) 수집 V2의 CORE/WEATHER DB와 .env를 공통 사용한다.
4) 모델은 검증된 training table에 실제 y/label/target이 있을 때만 학습한다.
5) 모델 score = CatBoost predict_proba * 100. SHAP은 설명 전용이다.
6) 인근/동일업종 평균은 실제 score table 집계만 사용한다.
7) 태풍 UI는 유지하되 실제 태풍 forcing + 검증 취약도/손해모형이 없으면 손실 산출을 잠근다.
"""
from __future__ import annotations

import hashlib, math, os, pickle, re, sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
from dotenv import load_dotenv
import streamlit as st
import platform
import matplotlib.pyplot as plt
import pydeck as pdk

from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold, GroupKFold
from catboost import CatBoostClassifier, Pool

# =========================================================================
# [구획 1/5] ENV / DB / 대용량 안전 설정
# =========================================================================
KST = timezone(timedelta(hours=9))
DEFAULT_BASE = Path("/Users/12609/Documents/hazus")
load_dotenv(DEFAULT_BASE / ".env", override=False)
BASE_DIR = Path(os.getenv("CLIP_HOME", str(DEFAULT_BASE))).expanduser()
load_dotenv(BASE_DIR / ".env", override=False)

LIVE_DB = Path(os.getenv("CLIP_NATIONWIDE_INTEGRATED_DB",
                          str(BASE_DIR / "clip_nationwide_integrated_master.db"))).expanduser()
CORE_DB = Path(os.getenv("TRACKB_CORE_DB",
                          str(BASE_DIR / "clip_trackb_core.db"))).expanduser()
WEATHER_DB = Path(os.getenv("TRACKB_WEATHER_DB",
                             str(BASE_DIR / "clip_trackb_weather.db"))).expanduser()
BUILDING_TABLE = os.getenv("BUILDING_OUTPUT_TABLE", "nationwide_integrated_sheet")

TRAIN_TABLE = {
    "fire": os.getenv("TRACKB_FIRE_TRAIN_TABLE", "trackb_training_fire"),
    "flood": os.getenv("TRACKB_FLOOD_TRAIN_TABLE", "trackb_training_flood"),
    "typhoon": os.getenv("TRACKB_TYPHOON_TRAIN_TABLE", "trackb_training_typhoon"),
}
SCORE_TABLE = {
    "fire": os.getenv("TRACKB_FIRE_SCORE_TABLE", "trackb_score_fire"),
    "flood": os.getenv("TRACKB_FLOOD_SCORE_TABLE", "trackb_score_flood"),
    "typhoon": os.getenv("TRACKB_TYPHOON_SCORE_TABLE", "trackb_score_typhoon"),
}
MODEL_DIR = Path(os.getenv("CLIP_TRACKB_MODEL_DIR", str(BASE_DIR / "models_trackb"))).expanduser()
MODEL_DIR.mkdir(parents=True, exist_ok=True)

N_SPLITS = int(os.getenv("CLIP_N_SPLITS", "4"))
RANDOM_STATE = int(os.getenv("CLIP_RANDOM_STATE", "42"))
ITERATIONS = int(os.getenv("TRACKB_CATBOOST_ITERATIONS", "700"))
DEPTH = int(os.getenv("TRACKB_CATBOOST_DEPTH", "7"))
LR = float(os.getenv("TRACKB_CATBOOST_LR", "0.035"))
SCORE_CHUNK = int(os.getenv("TRACKB_SCORE_CHUNK", "25000"))
MIN_POS = int(os.getenv("TRACKB_MIN_POSITIVE", "20"))
MIN_NEG = int(os.getenv("TRACKB_MIN_NEGATIVE", "40"))

if platform.system() == "Darwin":
    plt.rc("font", family="AppleGothic")
elif platform.system() == "Windows":
    plt.rc("font", family="Malgun Gothic")
plt.rc("axes", unicode_minus=False)

st.set_page_config(page_title="HAZUS Track-B 관제센터", page_icon="🏢", layout="wide",
                   initial_sidebar_state="expanded")

def now():
    return datetime.now(KST).isoformat(timespec="seconds")

def qi(s):
    return '"' + str(s).replace('"', '""') + '"'

def connect(path: Path, readonly=False):
    if readonly:
        c = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30)
    else:
        c = sqlite3.connect(str(path), timeout=120)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA busy_timeout=120000")
    c.execute("PRAGMA temp_store=FILE")
    return c

def exists(c, t):
    return c.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (t,)).fetchone() is not None

def cols(c, t):
    return [r["name"] for r in c.execute(f"PRAGMA table_info({qi(t)})")] if exists(c, t) else []

def scalar(c, sql, params=(), default=None):
    try:
        r = c.execute(sql, params).fetchone()
        return r[0] if r else default
    except Exception:
        return default

# =========================================================================
# [구획 2/5] 스키마 / 검색 / 실제 모델 엔진
# =========================================================================
ALIASES = {
    "key": ["_row_key", "mgmBldrgstPk", "bldrgstPk"],
    "address": ["newPlatPlc", "platPlc", "roadAddr", "address", "도로명주소", "대지위치"],
    "name": ["bldNm", "buildingName", "건물명"],
    "pnu": ["pnu", "PNU", "platPlcPnu"],
    "sigungu": ["_query_sigungu_cd", "sigunguCd", "sigungu_cd", "sgg_cd"],
    "bjdong": ["_query_bjdong_cd", "bjdongCd", "bjdong_cd", "bjd_cd"],
    "purpose": ["mainPurpsCdNm", "mainPurpsNm", "purpsCdNm", "purpose"],
    "structure": ["strctCdNm", "strctNm", "structure"],
    "roof": ["roofCdNm", "roofNm", "roof"],
    "area": ["totArea", "totalArea", "archArea"],
    "arch_area": ["archArea", "buildingArea"],
    "height": ["heit", "height", "bldHeight"],
    "ground_floors": ["grndFlrCnt", "groundFloorCount"],
    "underground_floors": ["ugrndFlrCnt", "undergroundFloorCount", "basementFloorCount"],
    "approval": ["useAprDay", "useAprDate", "approvalDate", "useAprYear"],
    "lat": ["lat", "latitude", "y", "위도"],
    "lon": ["lon", "lng", "longitude", "x", "경도"],
}

def pick(cs, aliases):
    low={x.lower():x for x in cs}
    for a in aliases:
        if a.lower() in low: return low[a.lower()]
    for x in cs:
        if any(a.lower() in x.lower() for a in aliases): return x
    return None

@st.cache_data(ttl=120, show_spinner=False)
def schema():
    c=connect(LIVE_DB, readonly=True)
    try:
        cs=cols(c, BUILDING_TABLE)
    finally:
        c.close()
    if not cs:
        raise RuntimeError(f"마스터 DB '{BUILDING_TABLE}' 테이블을 찾을 수 없습니다.")
    return {k:pick(cs,a) for k,a in ALIASES.items()}, cs

def age(v):
    s=re.sub(r"\D","",str(v or ""))
    if len(s)>=4:
        y=int(s[:4]); cy=datetime.now(KST).year
        if 1800<=y<=cy: return cy-y
    return np.nan

def mpath(hz):
    return MODEL_DIR / f"trackb_{hz}.pkl"

def clean(d, num, cat):
    X=pd.DataFrame(index=d.index)
    for c in num:
        X[c]=pd.to_numeric(d[c],errors="coerce") if c in d else np.nan
    for c in cat:
        X[c]=d[c].fillna("UNKNOWN").astype(str).replace({"":"UNKNOWN","nan":"UNKNOWN"}) if c in d else "UNKNOWN"
    return X.replace([np.inf,-np.inf],np.nan)

def _training_source(hz):
    """검증된 training table만 허용. 마스터 물리특성으로 y를 만들지 않는다."""
    table=TRAIN_TABLE[hz]
    # training table은 CORE 우선, LIVE 보조. 실제 target 컬럼 필수.
    for db in (CORE_DB, LIVE_DB):
        if not db.exists(): continue
        c=connect(db, readonly=True)
        try:
            if not exists(c,table): continue
            cs=cols(c,table)
            target=next((x for x in ["y","target","label","event_label"] if x in cs),None)
            if target is None:
                continue
            return db,table,target,cs
        finally:
            c.close()
    raise RuntimeError(
        f"{hz} 학습 중단: '{table}'에 검증된 실제 target(y/target/label/event_label)이 없습니다. "
        "미관측 건물을 음성(0)으로 간주하거나 물리특성으로 target을 만들지 않습니다."
    )

def load_train(hz):
    db,table,target,cs=_training_source(hz)
    c=connect(db, readonly=True)
    try:
        # 대규모 전체 마스터를 RAM에 올리지 않고, 이미 생성된 training table만 읽는다.
        d=pd.read_sql_query(f"SELECT * FROM {qi(table)}",c)
    finally:
        c.close()
    if d.empty: raise RuntimeError("학습 테이블이 비어 있습니다.")
    y=pd.to_numeric(d[target],errors="coerce")
    keep=y.isin([0,1])
    d=d.loc[keep].copy()
    d["y"]=y.loc[keep].astype(int)
    pos=int((d["y"]==1).sum()); neg=int((d["y"]==0).sum())
    if pos<MIN_POS or neg<MIN_NEG:
        raise RuntimeError(f"학습 중단: 실제 양성 {pos:,} / 음성 {neg:,}; 최소 {MIN_POS}/{MIN_NEG} 미달")

    forbidden={"y","target","label","event_label","building_rowid","rowid","pnu","address","name",
               "risk_score","score","loss","damage_ratio","predicted_loss_amount","sample_weight"}
    candidates=[x for x in d.columns if x.lower() not in forbidden]
    cat=[x for x in candidates if d[x].dtype=="object" or str(d[x].dtype).startswith("string")]
    num=[x for x in candidates if x not in cat]
    # 고카디널리티 식별자성 텍스트 제외
    cat=[x for x in cat if d[x].nunique(dropna=True) <= min(5000,max(50,len(d)//5))]
    num=[x for x in num if pd.to_numeric(d[x],errors="coerce").notna().mean()>=0.2]
    if not num and not cat: raise RuntimeError("사용 가능한 학습 feature가 없습니다.")
    d["sample_weight"]=pd.to_numeric(d.get("sample_weight",1.0),errors="coerce").fillna(1.0)
    return d,num,cat

def groups(d):
    for c in ["sigungu","sigungu_cd","bjdong","bjdong_cd"]:
        if c in d:
            g=d[c].fillna("UNKNOWN").astype(str)
            if g.nunique()>=N_SPLITS: return g
    return None

def train_pipeline(hz):
    d,num,cat=load_train(hz)
    y=d["y"].astype(int); w=d["sample_weight"].astype(float)
    X=clean(d,num,cat)
    cat_idx=[X.columns.get_loc(c) for c in cat if c in X.columns]
    g=groups(d)
    class_min=int(y.value_counts().min())
    splits_n=min(N_SPLITS,class_min)
    if splits_n<2: raise RuntimeError("교차검증 가능한 실제 양/음성 표본이 부족합니다.")
    splitter=(GroupKFold(splits_n).split(X,y,g) if g is not None and g.nunique()>=splits_n
              else StratifiedKFold(splits_n,shuffle=True,random_state=RANDOM_STATE).split(X,y))
    oof=np.full(len(X),np.nan); folds=[]
    for f,(tr,va) in enumerate(splitter,1):
        if y.iloc[tr].nunique()<2 or y.iloc[va].nunique()<2: continue
        m=CatBoostClassifier(iterations=ITERATIONS,depth=DEPTH,learning_rate=LR,loss_function="Logloss",
            eval_metric="AUC",random_seed=RANDOM_STATE+f,verbose=False,thread_count=-1,allow_writing_files=False)
        m.fit(X.iloc[tr],y.iloc[tr],sample_weight=w.iloc[tr],cat_features=cat_idx,
              eval_set=(X.iloc[va],y.iloc[va]),early_stopping_rounds=50,verbose=False)
        p=m.predict_proba(X.iloc[va])[:,1]; oof[va]=p
        folds.append({"fold":f,"auc":float(roc_auc_score(y.iloc[va],p)),
                      "pr_auc":float(average_precision_score(y.iloc[va],p)),"samples":len(va)})
    valid=np.isfinite(oof)
    auc=float(roc_auc_score(y[valid],oof[valid])) if valid.sum() and y[valid].nunique()>1 else np.nan
    ap=float(average_precision_score(y[valid],oof[valid])) if valid.sum() and y[valid].nunique()>1 else np.nan
    final=CatBoostClassifier(iterations=ITERATIONS,depth=DEPTH,learning_rate=LR,loss_function="Logloss",
        eval_metric="AUC",random_seed=RANDOM_STATE,verbose=False,thread_count=-1,allow_writing_files=False)
    final.fit(X,y,sample_weight=w,cat_features=cat_idx,verbose=False)
    b={"hazard":hz,"numeric":num,"categorical":cat,"features":list(X.columns),"model":final,
       "metrics":{"auc":auc,"pr_auc":ap,"folds":folds,"rows":len(d),
                  "positive":int((y==1).sum()),"negative":int((y==0).sum())},
       "trained_at":now(),"score_semantics":"relative_model_score_not_annual_probability"}
    with open(mpath(hz),"wb") as f: pickle.dump(b,f)
    return b

@st.cache_data(ttl=30, show_spinner=False)
def search_building(q,limit=20):
    s,_=schema(); clean_q=str(q).strip()
    if not clean_q:return []
    c=connect(LIVE_DB,readonly=True)
    try:
        # numeric rowid fast path
        if clean_q.isdigit():
            sel=["rowid building_rowid"]+[f"{qi(v)} {qi(k)}" for k,v in s.items() if v]
            d=pd.read_sql_query(f"SELECT {','.join(sel)} FROM {qi(BUILDING_TABLE)} WHERE rowid=? LIMIT ?",
                                c,params=(int(clean_q),limit))
            if not d.empty:return d.replace({np.nan:None}).to_dict("records")
        targets=[x for x in [s.get("address"),s.get("name"),s.get("pnu"),s.get("key")] if x]
        words=[w for w in clean_q.split() if w]
        clauses=[]; params=[]
        for x in targets:
            sub=[]
            for w in words:
                sub.append(f"CAST({qi(x)} AS TEXT) LIKE ?"); params.append(f"%{w}%")
            clauses.append("("+" AND ".join(sub)+")")
        sel=["rowid building_rowid"]+[f"{qi(v)} {qi(k)}" for k,v in s.items() if v]
        d=pd.read_sql_query(f"SELECT {','.join(sel)} FROM {qi(BUILDING_TABLE)} WHERE {' OR '.join(clauses)} LIMIT ?",
                            c,params=(*params,limit))
        return d.replace({np.nan:None}).to_dict("records")
    finally:c.close()

def _feature_row(rowid,b):
    # Prefer CORE feature table because collector V2 materializes validated features there.
    c=connect(CORE_DB,readonly=True)
    try:
        if not exists(c,"trackb_building_features"): return pd.DataFrame()
        d=pd.read_sql_query("SELECT * FROM trackb_building_features WHERE building_rowid=?",c,params=(rowid,))
    finally:c.close()
    if d.empty:return d
    # Model may use training feature names not present in core; missing remains unknown/NaN.
    return d

def predict_one(rowid,b):
    raw=_feature_row(rowid,b)
    if raw.empty:return None
    X=clean(raw,b["numeric"],b["categorical"])
    for f in b["features"]:
        if f not in X:X[f]=np.nan
    X=X[b["features"]]
    return float(b["model"].predict_proba(X)[:,1][0]*100.0)

def compute_object_shap_impact(rowid,hz,b):
    raw=_feature_row(rowid,b)
    if raw.empty:return []
    X=clean(raw,b["numeric"],b["categorical"])
    for f in b["features"]:
        if f not in X:X[f]=np.nan
    X=X[b["features"]]
    cat_idx=[X.columns.get_loc(c) for c in b["categorical"] if c in X]
    p=Pool(X,cat_features=cat_idx)
    sv=b["model"].get_feature_importance(p,type="ShapValues")[0]
    vals=sv[:-1]
    # PCA에 의미를 임의 부여하지 않는다. 원 feature 이름 그대로 설명.
    rec=[{"물리적 분석 변수 팩터":f,"리스크 기여 수치(SHAP)":float(v),
          "종합 진단 방향성":"위험 점수 상승 방향" if v>0 else "위험 점수 하락 방향"}
         for f,v in zip(X.columns,vals)]
    return sorted(rec,key=lambda x:abs(x["리스크 기여 수치(SHAP)"]),reverse=True)

def score_table_stats(hz,rowid,purpose=None,lat=None,lon=None):
    """실제 score table이 있을 때만 선택/인근/동일업종 평균을 계산."""
    c=connect(CORE_DB,readonly=True)
    try:
        t=SCORE_TABLE[hz]
        if not exists(c,t): return None,None,None
        cs=cols(c,t); sc=next((x for x in ["score","risk_score","prediction_score"] if x in cs),None)
        if not sc:return None,None,None
        mine=scalar(c,f"SELECT {qi(sc)} FROM {qi(t)} WHERE building_rowid=?",(rowid,))
        # 인근 평균: score table에 lat/lon이 있을 때 실제 반경 근사 bounding box 후 평균
        neigh=None
        if lat is not None and lon is not None and "lat" in cs and "lon" in cs:
            dlat=0.009; dlon=0.011
            neigh=scalar(c,f"""SELECT AVG({qi(sc)}) FROM {qi(t)}
                              WHERE lat BETWEEN ? AND ? AND lon BETWEEN ? AND ?""",
                         (float(lat)-dlat,float(lat)+dlat,float(lon)-dlon,float(lon)+dlon))
        catavg=None
        if purpose is not None and "purpose" in cs:
            catavg=scalar(c,f"SELECT AVG({qi(sc)}) FROM {qi(t)} WHERE purpose=?",(purpose,))
        return mine,neigh,catavg
    finally:c.close()

def render_3d_spatial_map(bld_lat,bld_lon,hz_code,target_rowid,bundle_model):
    try:
        lat=float(bld_lat); lon=float(bld_lon)
        if not (33<=lat<=39 and 124<=lon<=132):raise ValueError
    except Exception:
        st.warning("검증 가능한 좌표가 없어 3D 지도를 표시하지 않습니다.")
        return
    purpose=None
    core=connect(CORE_DB,readonly=True)
    try:
        if exists(core,"trackb_building_features"):
            r=core.execute("SELECT purpose FROM trackb_building_features WHERE building_rowid=?",(target_rowid,)).fetchone()
            purpose=r[0] if r else None
    finally:core.close()
    model_score=predict_one(target_rowid,bundle_model)
    db_score,neighbor_avg,category_avg=score_table_stats(hz_code,target_rowid,purpose,lat,lon)
    my_score=db_score if db_score is not None else model_score
    if my_score is None:
        st.warning("실제 모델 점수를 산출할 수 없어 비교 지도를 표시하지 않습니다.");return
    def color(x):
        if x is None:return [140,140,140,180]
        if x>=50:return [235,52,52,200]
        if x>=15:return [235,180,52,200]
        return [52,235,82,200]
    rows=[{"name":"현재 선택한 분석 물건","lat":lat,"lon":lon,"score":my_score,"height":my_score*8,
           "color":[30,144,255,255]}]
    if neighbor_avg is not None:
        rows.append({"name":"인근 지역 실제 score 평균","lat":lat+0.0005,"lon":lon+0.0005,
                     "score":neighbor_avg,"height":neighbor_avg*8,"color":color(neighbor_avg)})
    if category_avg is not None:
        rows.append({"name":"동일 업종 실제 score 평균","lat":lat-0.0005,"lon":lon+0.0005,
                     "score":category_avg,"height":category_avg*8,"color":color(category_avg)})
    st.pydeck_chart(pdk.Deck(
        layers=[pdk.Layer("ColumnLayer",pd.DataFrame(rows),get_position="[lon, lat]",
                          get_elevation="height",radius=45,get_fill_color="color",pickable=True,auto_highlight=True)],
        initial_view_state=pdk.ViewState(latitude=lat,longitude=lon,zoom=16.5,pitch=60,bearing=30),
        tooltip={"text":"분류: {name}\n모델 상대위험 점수: {score}"}
    ))
    if neighbor_avg is None or category_avg is None:
        st.caption("회색/누락 대조군은 실제 score 집계 자료가 아직 없어 생성하지 않았습니다.")

def run_prediction_for_uploaded(df_uploaded,hz,b):
    d=df_uploaded.copy()
    if "building_rowid" not in d:
        for x in ["id","고유번호"]:
            if x in d:d=d.rename(columns={x:"building_rowid"});break
    if "building_rowid" not in d:return pd.DataFrame()
    d["building_rowid"]=pd.to_numeric(d["building_rowid"],errors="coerce")
    d=d.dropna(subset=["building_rowid"]).copy(); d["building_rowid"]=d["building_rowid"].astype(int)
    rows=[]
    for rid in d["building_rowid"].drop_duplicates():
        sc=predict_one(int(rid),b)
        if sc is not None:rows.append({"building_rowid":int(rid),"risk_score":sc})
    return d.merge(pd.DataFrame(rows),on="building_rowid",how="inner") if rows else pd.DataFrame()

# =========================================================================
# [운영 상태] 기존 UI에 추가만 함 — 삭제 없음
# =========================================================================
@st.cache_data(ttl=10,show_spinner=False)
def health():
    out={}
    if CORE_DB.exists():
        c=connect(CORE_DB,readonly=True)
        try:
            out["buildings"]=scalar(c,"SELECT COUNT(*) FROM trackb_building_features",default=0) if exists(c,"trackb_building_features") else 0
            out["fire"]=scalar(c,"SELECT COUNT(*) FROM trackb_fire_events",default=0) if exists(c,"trackb_fire_events") else 0
            out["flood"]=scalar(c,"SELECT COUNT(*) FROM trackb_flood_evidence",default=0) if exists(c,"trackb_flood_evidence") else 0
            out["sprinkler"]=scalar(c,"SELECT COUNT(*) FROM trackb_sprinkler_evidence",default=0) if exists(c,"trackb_sprinkler_evidence") else 0
            out["last"]=c.execute("SELECT stage,status,updated_at FROM trackb_checkpoints ORDER BY updated_at DESC LIMIT 1").fetchone() if exists(c,"trackb_checkpoints") else None
        finally:c.close()
    return out

def render_health_strip():
    h=health()
    with st.expander("🖥️ Track-B 수집/DB 상태",expanded=False):
        a,b,c,d=st.columns(4)
        a.metric("건축물",f"{h.get('buildings',0):,}")
        b.metric("화재 evidence",f"{h.get('fire',0):,}")
        c.metric("침수 evidence",f"{h.get('flood',0):,}")
        d.metric("소방시설 evidence",f"{h.get('sprinkler',0):,}")
        if h.get("last"):
            st.caption(f"최근 checkpoint: {h['last']['stage']} / {h['last']['status']} / {h['last']['updated_at']}")

# =========================================================================
# [구획 4/5] FIRE — 기존 화면 구조 유지
# =========================================================================
def _profile(target):
    p1,p2,p3,p4=st.columns(4)
    p1.metric("주용도 명칭",str(target.get("purpose") or "미등기/UNKNOWN"))
    p2.metric("건물 구조",str(target.get("structure") or "UNKNOWN"))
    app=str(target.get("approval") or ""); av=age(app)
    p3.metric("건축년도 (사용승인일)",f"{app[:4]}년" if len(app)>=4 else "UNKNOWN")
    p4.metric("건물 노후도 (경과년수)",f"{av:.0f}년" if not pd.isna(av) else "UNKNOWN")

def _legend():
    st.markdown("### 관제 레이더 리스크 범례 표")
    st.info("기존 3대 대조군 UI를 유지합니다. 실제 집계가 없는 대조군은 값을 만들지 않고 미산출 처리합니다.")
    st.dataframe(pd.DataFrame({
        "기둥 종류 (식별)":["파란색 기둥","위험등급 색상 기둥","위험등급 색상 기둥"],
        "대상 분석 집단 명칭":["현재 선택한 분석 물건","인근 지역 실제 score 평균","동일 업종 실제 score 평균"],
        "시각적 의미":["선택 목적물","실제 score 집계","실제 score 집계"]}),use_container_width=True,hide_index=True)
    st.dataframe(pd.DataFrame({
        "위험 등급":["고위험군 (High)","중위험군 (Medium)","안전군 (Safe)"],
        "AI 리스크 점수 스펙":["50.00~100.00","15.00~49.99","0.00~14.99"],
        "투영 색상":["Ruby Red","Amber","Emerald"]}),use_container_width=True,hide_index=True)

def _analysis_block(target,hz,b,label):
    _profile(target)
    lat=target.get("lat");lon=target.get("lon")
    st.write("---")
    st.markdown(f"##### `[{target.get('name') or '선택 목적물'}]` 중심 인근 지역 3대 핵심 원기둥 지도 및 리스크 범례")
    cm,cl=st.columns([1.3,0.7])
    with cm:render_3d_spatial_map(lat,lon,hz,target["building_rowid"],b)
    with cl:_legend()

    model_score=predict_one(target["building_rowid"],b)
    db_score,navg,cavg=score_table_stats(hz,target["building_rowid"],target.get("purpose"),lat,lon)
    my=db_score if db_score is not None else model_score
    st.write("---");st.markdown("##### 3대 다차원 리스크 비교 분석 매트릭스 (선택 목적물 vs 인근 평균 vs 동일 업종 평균)")
    vals=[my,navg,cavg]; names=["현재 선택한 물건","인근 전체 물건 평균","동일 업종 집단 평균"]
    real=[(n,v) for n,v in zip(names,vals) if v is not None]
    if real:
        fig,ax=plt.subplots(figsize=(12,3.5))
        bars=ax.barh([x[0] for x in real],[x[1] for x in real],height=.55)
        ax.set_xlim(0,100);ax.set_xlabel(f"AI 종합 {label} 상대위험 점수 (0~100)")
        ax.grid(axis="x",linestyle=":",alpha=.6)
        for bar in bars:
            ax.text(bar.get_width()+1,bar.get_y()+bar.get_height()/2,f"{bar.get_width():.2f}점",va="center")
        st.pyplot(fig);plt.close(fig)
    else:st.info("실제 점수 데이터가 아직 없습니다.")

    shap_res=compute_object_shap_impact(target["building_rowid"],hz,b) if my is not None else []
    st.write("---");st.markdown(f"### 🔍 `[{target.get('name') or '선택 목적물'}]` 리스크 팩터별 정밀 분석 보고서")
    if my is not None:st.metric(f"{label} 모델 상대위험 점수",f"{my:.2f}")
    st.caption("점수는 predict_proba 기반 상대위험 점수이며 별도 calibration 전에는 연간 사고확률/보험요율이 아닙니다.")
    if shap_res:
        st.dataframe(pd.DataFrame(shap_res[:10]),use_container_width=True,hide_index=True)
        st.caption("SHAP은 점수 산출식이 아니라 이미 산출된 모델 예측의 설명값입니다.")
    else:st.info("SHAP 설명을 생성할 실제 모델/feature가 없습니다.")

def render_fire_page():
    st.header("🔥 화재 내재위험 분석 및 업종별 자산 4분면 모니터링")
    st.subheader("STEP 1: 4-Fold 교차 검증 및 대용량 피처 모델 파이프라인")
    if st.button("🔥 화재 내재위험 분석 및 4-Fold CV 실행",use_container_width=True):
        with st.spinner("검증된 화재 training table로 CatBoost CV 연산 중..."):
            try:train_pipeline("fire");st.success("화재 모델 학습 및 검증 완료")
            except Exception as e:st.error(str(e))
    if not mpath("fire").exists():
        st.info("검증된 화재 모델이 아직 없습니다. 기존 분석 UI는 유지되며 학습 데이터 준비 후 활성화됩니다.");return
    b=pickle.load(open(mpath("fire"),"rb"))
    with st.expander("모형 검증 성능 리포트 보기"):
        st.dataframe(pd.DataFrame(b["metrics"].get("folds",[])),use_container_width=True)
        st.caption(f"OOF AUC {b['metrics'].get('auc',np.nan):.4f} | rows {b['metrics'].get('rows',0):,}")
    st.write("---");st.subheader("개별 목적물 주소 조회 및 다차원 리스크 비교 분석")
    st.session_state.setdefault("fire_active_target",None)
    q=st.text_input("건물 마스터 검색기 (주소, 건물명, PNU 입력)",key="fire_search")
    if q.strip():
        rs=search_building(q)
        if not rs:st.warning("현재 DB에서 일치 목적물을 찾지 못했습니다.")
        for item in rs:
            if st.button(f"🏢 {item.get('name') or '명칭 미상'} | 주소: {item.get('address') or '-'} (ID: {item['building_rowid']})",
                         key=f"f_{item['building_rowid']}",use_container_width=True):
                st.session_state["fire_active_target"]=item
    if st.session_state["fire_active_target"]:
        t=st.session_state["fire_active_target"]
        st.write("---");st.markdown(f"### `[{t.get('name') or '선택 목적물'}]` 종합 위험도 관제 포트폴리오")
        _analysis_block(t,"fire",b,"화재")

    st.write("---");st.subheader("STEP 2: 보험금액 및 피해액 자산 데이터 결합")
    up=st.file_uploader("화재 분석 대상 물건의 자산 명세 파일을 드롭하세요.",type=["xlsx","csv"],key="fire_upload")
    if up is not None:
        d=pd.read_csv(up) if up.name.lower().endswith(".csv") else pd.read_excel(up)
        d=d.rename(columns={"가입금액":"보험금액","보험가입금액":"보험금액","보장금액":"보험금액","손실액":"피해액","손해액":"피해액"})
        if "보험금액" not in d:st.error("'보험금액' 또는 '가입금액' 컬럼이 필요합니다.");return
        m=run_prediction_for_uploaded(d,"fire",b)
        if m.empty:st.error("building_rowid가 DB와 일치하지 않습니다.");return
        st.write("---");st.subheader("STEP 3: 업종별 위험-자산 4분면 매트릭스")
        m["보험금액"]=pd.to_numeric(m["보험금액"],errors="coerce").fillna(0)
        m["피해액"]=pd.to_numeric(m.get("피해액",0),errors="coerce").fillna(0)
        m["업종"]=m["업종"] if "업종" in m else "일반 목적물"
        ms=float(m["risk_score"].median());ma=float(m["보험금액"].median())
        m["4분면 영역"]=m.apply(lambda r:
            "I (High Risk / High Value)" if r.risk_score>=ms and r["보험금액"]>=ma else
            "II (Low Risk / High Value)" if r.risk_score<ms and r["보험금액"]>=ma else
            "III (Low Risk / Low Value)" if r.risk_score<ms and r["보험금액"]<ma else
            "IV (High Risk / Low Value)",axis=1)
        fig,ax=plt.subplots(figsize=(12,5))
        for name,g in m.groupby("업종"):
            ax.scatter(g["risk_score"],g["보험금액"],label=str(name),s=90)
        ax.axvline(ms,linestyle="--");ax.axhline(ma,linestyle="--")
        if (m["보험금액"]>0).all():ax.set_yscale("log")
        ax.legend();st.pyplot(fig);plt.close(fig)
        st.dataframe(m,use_container_width=True)

# =========================================================================
# [구획 4/5] FLOOD — 기존 검색/지도/비교/보고서 화면 유지
# =========================================================================
def render_flood_page():
    st.header("🌊 침수 수문지형 위험 분석 및 업종별 자산 4분면 모니터링")
    st.subheader("STEP 1: 4-Fold 교차 검증 및 피처 모델 파이프라인")
    if st.button("🌊 침수 내재위험 분석 및 4-Fold CV 실행",use_container_width=True):
        with st.spinner("검증된 침수 training table로 CatBoost CV 연산 중..."):
            try:train_pipeline("flood");st.success("침수 모델 학습 및 검증 완료")
            except Exception as e:st.error(str(e))
    if not mpath("flood").exists():
        st.info("검증된 침수 모델이 아직 없습니다. 기존 분석 UI는 유지되며 학습 데이터 준비 후 활성화됩니다.");return
    b=pickle.load(open(mpath("flood"),"rb"))
    with st.expander("모형 검증 성능 리포트 보기"):
        st.dataframe(pd.DataFrame(b["metrics"].get("folds",[])),use_container_width=True)
        st.caption(f"OOF AUC {b['metrics'].get('auc',np.nan):.4f} | rows {b['metrics'].get('rows',0):,}")
    st.write("---");st.subheader("개별 목적물 주소 조회 및 다차원 리스크 비교 분석")
    st.session_state.setdefault("flood_active_target",None)
    q=st.text_input("건물 마스터 검색기 (주소, 건물명, PNU 입력)",key="flood_search")
    if q.strip():
        rs=search_building(q)
        if not rs:st.warning("현재 DB에서 일치 목적물을 찾지 못했습니다.")
        for item in rs:
            if st.button(f"🏢 {item.get('name') or '명칭 미상'} | 주소: {item.get('address') or '-'} (ID: {item['building_rowid']})",
                         key=f"fl_{item['building_rowid']}",use_container_width=True):
                st.session_state["flood_active_target"]=item
    if st.session_state["flood_active_target"]:
        t=st.session_state["flood_active_target"]
        st.write("---");st.markdown(f"### `[{t.get('name') or '선택 목적물'}]` 종합 침수위험 관제 포트폴리오")
        _analysis_block(t,"flood",b,"침수")

# =========================================================================
# [구획 5/5] TYPHOON — UI 유지, 가짜 WeatherNext/손해율 제거
# =========================================================================
# 화면 선택지는 유지하되 이것은 "참고 경로 UI 프리셋"이며 WeatherNext 출력이라고 주장하지 않는다.
TYPHOON_PRESETS = {
    "제주 진입 후 남해안 상륙 참고 경로": [
        {"step_name":"1단계","lat":33.2,"lon":126.9},
        {"step_name":"2단계","lat":34.8,"lon":128.4},
        {"step_name":"3단계","lat":36.5,"lon":129.2},
    ],
    "서해안 북상 참고 경로": [
        {"step_name":"1단계","lat":33.4,"lon":125.1},
        {"step_name":"2단계","lat":35.8,"lon":125.8},
        {"step_name":"3단계","lat":37.8,"lon":126.4},
    ],
    "대한해협 통과 참고 경로": [
        {"step_name":"1단계","lat":32.8,"lon":126.5},
        {"step_name":"2단계","lat":34.9,"lon":129.1},
        {"step_name":"3단계","lat":37.2,"lon":131.2},
    ],
}

def render_typhoon_page():
    st.header("🌀 WeatherNext/공식 태풍외력 연계 태풍 시뮬레이터")
    st.subheader("STEP 1: 태풍 취약성 모델 및 검증 상태")
    if st.button("🌀 태풍 고유 취약성 분석 및 4-Fold CV 실행",use_container_width=True):
        with st.spinner("검증된 태풍 training table 확인 중..."):
            try:train_pipeline("typhoon");st.success("태풍 취약성 모델 학습/검증 완료")
            except Exception as e:st.error(str(e))
    if mpath("typhoon").exists():
        b=pickle.load(open(mpath("typhoon"),"rb"))
        with st.expander("4-Fold Cross Validation 태풍 모형 검증 성능 보기"):
            st.dataframe(pd.DataFrame(b["metrics"].get("folds",[])),use_container_width=True)
    else:
        st.warning("실제 태풍 피해 target이 포함된 training table이 없어 모델 산출을 잠갔습니다.")

    st.write("---");st.subheader("STEP 2: 한국 태풍 경로 및 이동 타임라인 제어")
    selected=st.sidebar.radio("태풍 경로 보기",list(TYPHOON_PRESETS.keys()),key="typhoon_path")
    nodes=TYPHOON_PRESETS[selected]; idx=st.slider("태풍 추적 타임라인 단계",0,len(nodes)-1,0)
    active=nodes[idx]
    st.markdown(f"### 현재 진행 단계: `{active['step_name']}`")
    st.info(f"참고 경로 좌표: 위도 {active['lat']}, 경도 {active['lon']}")
    st.caption("이 좌표열은 UI 확인용 참고 경로이며 Google WeatherNext 예측값이 아닙니다.")

    df=pd.DataFrame(nodes)
    path_data=[{"path":df[["lon","lat"]].values.tolist()}]
    layers=[
        pdk.Layer("PathLayer",path_data,get_path="path",get_color=[255,140,0,200],width_min_pixels=5,pickable=True),
        pdk.Layer("ScatterplotLayer",[active],get_position="[lon, lat]",get_radius=30000,
                  get_fill_color=[255,0,0,60],pickable=True)
    ]
    st.pydeck_chart(pdk.Deck(layers=layers,initial_view_state=pdk.ViewState(
        latitude=35.5,longitude=127.5,zoom=6.5,pitch=35,bearing=0)))

    st.write("---");st.subheader("💡 STEP 3: 자산 포트폴리오 업로드 및 손실 계산")
    up=st.file_uploader("태풍 영향도 분석 대상 자산 파일을 업로드하세요.",type=["xlsx","csv"],key="typhoon_upload")
    if up is None:
        st.info("기존 업로드 인터페이스는 유지됩니다.")
    else:
        d=pd.read_csv(up) if up.name.lower().endswith(".csv") else pd.read_excel(up)
        st.dataframe(d.head(100),use_container_width=True)
    st.error(
        "운영 손실 계산 잠금: 실제 WeatherNext/공식 태풍 시계열(경로·기압·풍속·강풍반경), "
        "건물별 도달 외력, 검증된 취약도/손해율 모델이 연결되기 전에는 damage_ratio와 예상손해액을 만들지 않습니다."
    )

# =========================================================================
# [MAIN] 기존 3메뉴 + 상태창 추가
# =========================================================================
def main():
    st.sidebar.title("🏢 HAZUS 관제 센터")
    st.sidebar.caption("CatBoost 위험도 및 자산 4분면 통합 대시보드")
    menu=st.sidebar.radio("분석 재해 선택",[
        "1. 화재 내재위험 분석 (Fire)",
        "2. 침수 내재위험 분석 (Flood)",
        "3. 태풍 WeatherNext 예측 (Typhoon)"
    ])
    st.sidebar.write("---")
    st.sidebar.info(
        "**통합 운영 가이드**\n"
        "- 기존 Fire/Flood/Typhoon 화면 구조를 유지합니다.\n"
        "- 실제 evidence/score가 없으면 UNKNOWN/미산출로 표시합니다.\n"
        "- SHAP은 설명 전용입니다."
    )
    render_health_strip()

    if "1." in menu:render_fire_page()
    elif "2." in menu:render_flood_page()
    else:render_typhoon_page()

if __name__=="__main__":
    main()
