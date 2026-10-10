#!/usr/bin/env python3
"""
DDN Track-B production-oriented evidence collector v2.

Design goals
------------
1) Millions of buildings are streamed in bounded chunks. No nationwide DataFrame/dict/KDTree.
2) Every long-running stage has a persistent checkpoint and can resume.
3) External evidence is append-only. Missing evidence is UNKNOWN, never fabricated NO/Y=0.
4) Weather is stored once per KMA grid/time, not duplicated per building.
5) Weather is isolated in a separate SQLite DB to prevent the core evidence DB from ballooning.
6) Raw external responses/files can be preserved separately for audit/reproducibility.
7) Spatial centroid proxies are processed cell-by-cell; actual wall gaps require building footprints.
8) Integrity checks, disk-space guards, WAL checkpoints, run/audit tables, source/version metadata.
9) Current KMA observations/forecasts are operational overlays, NOT historical training weather.
10) No source is called unless its endpoint/schema has been explicitly implemented and configured.

This file deliberately does NOT invent:
- fire events
- sprinkler installation status
- flood history
- wall-to-wall gaps
- historical weather
- loss ratios
- model labels

Python: 3.10+
Required: requests, python-dotenv
Optional for spatial: numpy, scipy
"""
from __future__ import annotations

# [IMPORT-01] Standard library only for DB, checkpoints, raw archives and resilient HTTP.
import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import shutil
import sqlite3
import sys
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

# [IMPORT-02] External libraries kept minimal. scipy/numpy are imported only inside spatial stage.
import requests
from dotenv import load_dotenv

# ============================================================================
# [CONFIG] Paths, source DB and API configuration
# ============================================================================

# [CONFIG-01] Existing nationwide building registry remains the authoritative building master.
DEFAULT_HOME = Path("/Users/12609/Documents/hazus")
load_dotenv(DEFAULT_HOME / ".env", override=False)
HOME = Path(os.getenv("CLIP_HOME", str(DEFAULT_HOME))).expanduser()
load_dotenv(HOME / ".env", override=False)

LIVE_DB = Path(os.getenv(
    "CLIP_NATIONWIDE_INTEGRATED_DB",
    str(HOME / "clip_nationwide_integrated_master.db"),
)).expanduser()

# [CONFIG-02] Core evidence/features and high-volume weather are physically separated.
CORE_DB = Path(os.getenv(
    "TRACKB_CORE_DB",
    str(HOME / "clip_trackb_core.db"),
)).expanduser()
WEATHER_DB = Path(os.getenv(
    "TRACKB_WEATHER_DB",
    str(HOME / "clip_trackb_weather.db"),
)).expanduser()

# [CONFIG-03] Raw external source material is preserved outside SQLite.
RAW_DIR = Path(os.getenv("TRACKB_RAW_DIR", str(HOME / "trackb_raw"))).expanduser()
BUILDING_TABLE = os.getenv("BUILDING_OUTPUT_TABLE", "nationwide_integrated_sheet")

# [CONFIG-04] Disk guard is intentionally configurable; default keeps 20 GiB free.
MIN_FREE_GB = float(os.getenv("TRACKB_MIN_FREE_GB", "20"))
BUILDING_CHUNK = int(os.getenv("TRACKB_BUILDING_CHUNK", "25000"))
EVIDENCE_CHUNK = int(os.getenv("TRACKB_EVIDENCE_CHUNK", "5000"))

# [CONFIG-05] KMA current/forecast API. This is not historical observation storage.
KMA_KEY = os.getenv("KMA_SERVICE_KEY", "")
KMA_BASE = "https://apis.data.go.kr/1360000/VilageFcstInfoService_2.0"

KST = timezone(timedelta(hours=9))
UNKNOWN = "UNKNOWN"
YES = "YES"
NO = "NO"

# ============================================================================
# [UTIL] General helpers
# ============================================================================

# [UTIL-01] Consistent audit timestamp.
def now() -> str:
    return datetime.now(KST).isoformat(timespec="seconds")


# [UTIL-02] Quote SQL identifiers. Values are always passed as parameters separately.
def qi(value: str) -> str:
    return '"' + str(value).replace('"', '""') + '"'


# [UTIL-03] Conservative numeric parser; NaN/Inf become NULL.
def fnum(value: Any) -> Optional[float]:
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


# [UTIL-04] Korean address matching normalizer. It removes whitespace only; it does not guess addresses.
def norm_addr(value: Any) -> str:
    return "".join(str(value or "").strip().lower().split())


# [UTIL-05] Stable SHA-256 ID for append-only evidence/raw objects.
def sha256_text(*parts: Any) -> str:
    payload = "|".join("" if p is None else str(p) for p in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# [UTIL-06] Prevent a long ingestion from filling the disk and corrupting an active DB transaction.
def require_disk_space(path: Path, minimum_gb: float = MIN_FREE_GB) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(path.parent)
    free_gb = usage.free / (1024 ** 3)
    if free_gb < minimum_gb:
        raise RuntimeError(
            f"Insufficient free disk: {free_gb:.1f} GiB < required {minimum_gb:.1f} GiB "
            f"at {path.parent}"
        )


# [UTIL-07] Source DB must exist; destination DB may be created by db().
def ensure_source_exists() -> None:
    if not LIVE_DB.exists():
        raise FileNotFoundError(f"Nationwide building DB not found: {LIVE_DB}")


# ============================================================================
# [DB] SQLite connection, schema, WAL and audit/checkpoint controls
# ============================================================================

# [DB-01] One connection policy for all Track-B SQLite databases.
# NORMAL+WAL gives durable commits with better throughput; foreign_keys guards relational mistakes.
def db(path: Path, *, readonly: bool = False) -> sqlite3.Connection:
    if readonly and not path.exists():
        raise FileNotFoundError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if readonly:
        uri = f"file:{path}?mode=ro"
        c = sqlite3.connect(uri, uri=True, timeout=120)
    else:
        c = sqlite3.connect(str(path), timeout=120)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA busy_timeout=120000")
    c.execute("PRAGMA foreign_keys=ON")
    if not readonly:
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA synchronous=NORMAL")
        c.execute("PRAGMA temp_store=FILE")
        c.execute("PRAGMA wal_autocheckpoint=10000")
    return c


# [DB-02] Generic transaction wrapper: rollback on failure, commit only on success.
@contextmanager
def transaction(c: sqlite3.Connection):
    try:
        c.execute("BEGIN")
        yield
        c.commit()
    except Exception:
        c.rollback()
        raise


# [DB-03] Core DB schema.
# Evidence tables are append-only by evidence_id; no REPLACE that erases history.
def init_core() -> None:
    require_disk_space(CORE_DB)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with db(CORE_DB) as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS trackb_schema_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS trackb_source_registry (
            source_id TEXT PRIMARY KEY,
            agency TEXT NOT NULL,
            source_name TEXT NOT NULL,
            scope TEXT,
            source_url TEXT,
            license_note TEXT,
            verification TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS trackb_runs (
            run_id TEXT PRIMARY KEY,
            stage TEXT NOT NULL,
            source_id TEXT,
            status TEXT NOT NULL,
            started_at TEXT NOT NULL,
            finished_at TEXT,
            attempted INTEGER NOT NULL DEFAULT 0,
            written INTEGER NOT NULL DEFAULT 0,
            unresolved INTEGER NOT NULL DEFAULT 0,
            last_error TEXT,
            note TEXT
        );

        CREATE TABLE IF NOT EXISTS trackb_checkpoints (
            stage TEXT NOT NULL,
            partition_key TEXT NOT NULL,
            cursor_value TEXT,
            status TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY(stage, partition_key)
        );

        CREATE TABLE IF NOT EXISTS trackb_building_features (
            building_rowid INTEGER PRIMARY KEY,
            address TEXT,
            address_norm TEXT,
            name TEXT,
            sigungu TEXT,
            bjdong TEXT,
            pnu TEXT,
            purpose TEXT,
            approval TEXT,
            area REAL,
            arch_area REAL,
            height REAL,
            floors REAL,
            basement REAL,
            structure TEXT,
            roof TEXT,
            lat REAL,
            lon REAL,
            spatial_cell TEXT,
            kma_nx INTEGER,
            kma_ny INTEGER,
            source_id TEXT NOT NULL,
            source_row_hash TEXT NOT NULL,
            collected_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS trackb_address_match (
            address_norm TEXT NOT NULL,
            building_rowid INTEGER NOT NULL,
            PRIMARY KEY(address_norm, building_rowid),
            FOREIGN KEY(building_rowid) REFERENCES trackb_building_features(building_rowid)
        ) WITHOUT ROWID;

        CREATE TABLE IF NOT EXISTS trackb_weather_grid_map (
            building_rowid INTEGER PRIMARY KEY,
            nx INTEGER NOT NULL,
            ny INTEGER NOT NULL,
            assigned_at TEXT NOT NULL,
            FOREIGN KEY(building_rowid) REFERENCES trackb_building_features(building_rowid)
        );

        CREATE TABLE IF NOT EXISTS trackb_fire_events (
            evidence_id TEXT PRIMARY KEY,
            event_date TEXT,
            address TEXT,
            address_norm TEXT,
            event_type TEXT,
            building_rowid INTEGER,
            match_status TEXT NOT NULL,
            source_id TEXT NOT NULL,
            source_reference TEXT,
            raw_sha256 TEXT NOT NULL,
            collected_at TEXT NOT NULL,
            FOREIGN KEY(building_rowid) REFERENCES trackb_building_features(building_rowid)
        );

        CREATE TABLE IF NOT EXISTS trackb_sprinkler_evidence (
            evidence_id TEXT PRIMARY KEY,
            observed_at TEXT,
            address TEXT,
            address_norm TEXT,
            building_name TEXT,
            sprinkler_status TEXT NOT NULL,
            building_rowid INTEGER,
            match_status TEXT NOT NULL,
            source_id TEXT NOT NULL,
            source_reference TEXT,
            raw_sha256 TEXT NOT NULL,
            collected_at TEXT NOT NULL,
            CHECK(sprinkler_status IN ('YES','NO','UNKNOWN')),
            FOREIGN KEY(building_rowid) REFERENCES trackb_building_features(building_rowid)
        );

        CREATE TABLE IF NOT EXISTS trackb_flood_evidence (
            evidence_id TEXT PRIMARY KEY,
            observed_at TEXT,
            address TEXT,
            address_norm TEXT,
            observed_flood TEXT NOT NULL,
            building_rowid INTEGER,
            match_status TEXT NOT NULL,
            source_id TEXT NOT NULL,
            source_reference TEXT,
            raw_sha256 TEXT NOT NULL,
            collected_at TEXT NOT NULL,
            CHECK(observed_flood IN ('YES','NO','UNKNOWN')),
            FOREIGN KEY(building_rowid) REFERENCES trackb_building_features(building_rowid)
        );

        CREATE TABLE IF NOT EXISTS trackb_spatial_features (
            building_rowid INTEGER PRIMARY KEY,
            neighbor_100m_count INTEGER,
            nearest_centroid_m REAL,
            spatial_method TEXT NOT NULL,
            measured_at TEXT NOT NULL,
            FOREIGN KEY(building_rowid) REFERENCES trackb_building_features(building_rowid)
        );

        CREATE TABLE IF NOT EXISTS trackb_quality_events (
            quality_id INTEGER PRIMARY KEY AUTOINCREMENT,
            stage TEXT NOT NULL,
            severity TEXT NOT NULL,
            code TEXT NOT NULL,
            message TEXT NOT NULL,
            affected_count INTEGER,
            recorded_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_bld_addr_norm
            ON trackb_building_features(address_norm);
        CREATE INDEX IF NOT EXISTS idx_bld_sigungu
            ON trackb_building_features(sigungu);
        CREATE INDEX IF NOT EXISTS idx_bld_pnu
            ON trackb_building_features(pnu);
        CREATE INDEX IF NOT EXISTS idx_bld_cell
            ON trackb_building_features(spatial_cell);
        CREATE INDEX IF NOT EXISTS idx_bld_kma
            ON trackb_building_features(kma_nx, kma_ny);
        CREATE INDEX IF NOT EXISTS idx_bld_latlon
            ON trackb_building_features(lat, lon);
        CREATE INDEX IF NOT EXISTS idx_fire_building
            ON trackb_fire_events(building_rowid);
        CREATE INDEX IF NOT EXISTS idx_spr_building
            ON trackb_sprinkler_evidence(building_rowid);
        CREATE INDEX IF NOT EXISTS idx_flood_building
            ON trackb_flood_evidence(building_rowid);
        """)

        stamp = now()
        c.execute("""
            INSERT INTO trackb_schema_meta(key,value,updated_at)
            VALUES('schema_version','2.0',?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value,updated_at=excluded.updated_at
        """, (stamp,))

        # [DB-04] Registry distinguishes confirmed source existence from implemented automatic collectors.
        sources = [
            ("building_hub", "국토교통부", "건축HUB 건축물대장",
             "전국 건축물 원장", "https://www.data.go.kr/data/15134735/openapi.do",
             "공공데이터포털 이용조건 확인", "SOURCE_CONFIRMED; MASTER_DB_ALREADY_COLLECTED"),
            ("kma_short", "기상청", "단기예보 조회서비스",
             "현재 실황/예보 격자", "https://www.data.go.kr/data/15084084/openapi.do",
             "공공데이터포털 이용조건 확인", "SOURCE_CONFIRMED; AUTOMATED_CURRENT_ONLY"),
            ("safemap_flood", "행정안전부", "생활안전지도 침수흔적도",
             "침수흔적 공간자료", "https://www.safemap.go.kr",
             "상업적 이용 등 개별 데이터 이용조건 재확인 필요", "SOURCE_CONFIRMED; AUTOMATION_NOT_ASSERTED"),
            ("nfa_sprinkler", "소방청", "특정소방대상물 소방시설정보",
             "제공 범위 내 소방시설", "https://www.data.go.kr",
             "API별 제공범위/이용조건 확인", "SOURCE_CONFIRMED; NATIONWIDE_COVERAGE_NOT_ASSUMED"),
            ("fire_event_source", "소방 관련 공공 원천", "건물 단위 화재사건 원천",
             "실제 사건 CASE", "", "원천별 검증 필요", "NO_UNVERIFIED_ENDPOINT_HARDCODED"),
            ("building_footprint", "공간정보 원천", "건물 폴리곤",
             "실제 외벽간격 계산", "", "원천/라이선스 확인 필요", "NOT_CONFIGURED"),
        ]
        c.executemany("""
            INSERT INTO trackb_source_registry
            (source_id,agency,source_name,scope,source_url,license_note,verification,updated_at)
            VALUES(?,?,?,?,?,?,?,?)
            ON CONFLICT(source_id) DO UPDATE SET
                agency=excluded.agency,
                source_name=excluded.source_name,
                scope=excluded.scope,
                source_url=excluded.source_url,
                license_note=excluded.license_note,
                verification=excluded.verification,
                updated_at=excluded.updated_at
        """, [(*r, stamp) for r in sources])
        c.commit()


# [DB-05] Weather DB is independent because time-series volume can dwarf the building master.
def init_weather() -> None:
    require_disk_space(WEATHER_DB)
    with db(WEATHER_DB) as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS weather_observations (
            nx INTEGER NOT NULL,
            ny INTEGER NOT NULL,
            operation TEXT NOT NULL,
            base_date TEXT NOT NULL,
            base_time TEXT NOT NULL,
            forecast_date TEXT NOT NULL,
            forecast_time TEXT NOT NULL,
            category TEXT NOT NULL,
            value TEXT,
            source_id TEXT NOT NULL,
            collected_at TEXT NOT NULL,
            PRIMARY KEY(
                nx,ny,operation,base_date,base_time,
                forecast_date,forecast_time,category
            )
        ) WITHOUT ROWID;

        CREATE TABLE IF NOT EXISTS weather_runs (
            run_id TEXT PRIMARY KEY,
            operation TEXT NOT NULL,
            status TEXT NOT NULL,
            started_at TEXT NOT NULL,
            finished_at TEXT,
            grids_attempted INTEGER NOT NULL DEFAULT 0,
            grids_success INTEGER NOT NULL DEFAULT 0,
            last_error TEXT
        );

        CREATE TABLE IF NOT EXISTS weather_checkpoints (
            operation TEXT NOT NULL,
            nx INTEGER NOT NULL,
            ny INTEGER NOT NULL,
            last_base_date TEXT,
            last_base_time TEXT,
            status TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY(operation,nx,ny)
        ) WITHOUT ROWID;

        CREATE INDEX IF NOT EXISTS idx_weather_time
            ON weather_observations(base_date,base_time,operation);
        """)
        c.commit()


# [DB-06] Persistent run records make interrupted/failed stages auditable.
def start_run(c: sqlite3.Connection, stage: str, source_id: Optional[str], note: str = "") -> str:
    run_id = uuid.uuid4().hex
    c.execute("""
        INSERT INTO trackb_runs(run_id,stage,source_id,status,started_at,note)
        VALUES(?,?,?,?,?,?)
    """, (run_id, stage, source_id, "RUNNING", now(), note))
    c.commit()
    return run_id


def finish_run(
    c: sqlite3.Connection, run_id: str, status: str,
    attempted: int, written: int, unresolved: int = 0, error: Optional[str] = None
) -> None:
    c.execute("""
        UPDATE trackb_runs
        SET status=?,finished_at=?,attempted=?,written=?,unresolved=?,last_error=?
        WHERE run_id=?
    """, (status, now(), attempted, written, unresolved, error, run_id))
    c.commit()


# [DB-07] Checkpoints are updated only after a successful chunk commit.
def get_checkpoint(c: sqlite3.Connection, stage: str, partition: str = "ALL") -> Optional[str]:
    row = c.execute("""
        SELECT cursor_value FROM trackb_checkpoints
        WHERE stage=? AND partition_key=? AND status='OK'
    """, (stage, partition)).fetchone()
    return row["cursor_value"] if row else None


def set_checkpoint(
    c: sqlite3.Connection, stage: str, cursor: Any,
    partition: str = "ALL", status: str = "OK"
) -> None:
    c.execute("""
        INSERT INTO trackb_checkpoints(stage,partition_key,cursor_value,status,updated_at)
        VALUES(?,?,?,?,?)
        ON CONFLICT(stage,partition_key) DO UPDATE SET
            cursor_value=excluded.cursor_value,
            status=excluded.status,
            updated_at=excluded.updated_at
    """, (stage, partition, str(cursor), status, now()))


# [DB-08] WAL checkpoint prevents an indefinitely growing -wal file during long operations.
def checkpoint_wal(path: Path, mode: str = "PASSIVE") -> tuple:
    with db(path) as c:
        return tuple(c.execute(f"PRAGMA wal_checkpoint({mode})").fetchone())


# [DB-09] SQLite integrity checks are explicit operational commands, not assumed.
def integrity(path: Path, quick: bool = True) -> str:
    if not path.exists():
        return f"MISSING: {path}"
    with db(path, readonly=True) as c:
        pragma = "quick_check" if quick else "integrity_check"
        rows = [r[0] for r in c.execute(f"PRAGMA {pragma}")]
    return "\n".join(rows)


# ============================================================================
# [BUILDING] Nationwide building master -> bounded Track-B materialization
# ============================================================================

# [BUILD-01] Inspect source schema dynamically instead of assuming one vendor spelling.
def source_columns() -> list[str]:
    ensure_source_exists()
    with db(LIVE_DB, readonly=True) as c:
        return [r["name"] for r in c.execute(f"PRAGMA table_info({qi(BUILDING_TABLE)})")]


def choose(columns: list[str], aliases: list[str]) -> Optional[str]:
    low = {x.lower(): x for x in columns}
    for a in aliases:
        if a.lower() in low:
            return low[a.lower()]
    return None


def building_schema() -> dict[str, Optional[str]]:
    cs = source_columns()
    if not cs:
        raise RuntimeError(f"Missing source table: {BUILDING_TABLE}")
    aliases = {
        "address": ["newPlatPlc", "platPlc", "roadAddr", "address", "도로명주소", "대지위치"],
        "name": ["bldNm", "buildingName", "건물명"],
        "sigungu": ["_query_sigungu_cd", "sigunguCd", "sigungu_cd", "sgg_cd"],
        "bjdong": ["_query_bjdong_cd", "bjdongCd", "bjdong_cd", "bjd_cd"],
        "pnu": ["pnu", "PNU", "platPlcPnu"],
        "purpose": ["mainPurpsCdNm", "mainPurpsNm", "purpose"],
        "approval": ["useAprDay", "useAprDate", "approvalDate"],
        "area": ["totArea", "totalArea"],
        "arch_area": ["archArea", "buildingArea"],
        "height": ["heit", "height", "bldHeight"],
        "floors": ["grndFlrCnt", "groundFloorCount"],
        "basement": ["ugrndFlrCnt", "undergroundFloorCount", "basementFloorCount"],
        "structure": ["strctCdNm", "strctNm", "structure"],
        "roof": ["roofCdNm", "roofNm", "roof"],
        "lat": ["lat", "latitude", "위도"],
        "lon": ["lon", "lng", "longitude", "경도"],
    }
    return {k: choose(cs, v) for k, v in aliases.items()}


# [BUILD-02] 0.05-degree cells keep local spatial jobs bounded; this is only a processing partition.
def spatial_cell(lat: Optional[float], lon: Optional[float], step: float = 0.05) -> Optional[str]:
    if lat is None or lon is None:
        return None
    iy = math.floor(lat / step)
    ix = math.floor(lon / step)
    return f"{iy}:{ix}"


# [KMA-GRID-01] Official village-forecast Lambert grid transform.
def latlon_to_grid(lat: float, lon: float) -> tuple[int, int]:
    re_km, grid = 6371.00877, 5.0
    slat1, slat2, olon, olat, xo, yo = 30.0, 60.0, 126.0, 38.0, 43.0, 136.0
    degrad = math.pi / 180.0
    re = re_km / grid
    sl1, sl2, olo, ola = [v * degrad for v in (slat1, slat2, olon, olat)]
    sn = math.log(math.cos(sl1) / math.cos(sl2)) / math.log(
        math.tan(math.pi * 0.25 + sl2 * 0.5) / math.tan(math.pi * 0.25 + sl1 * 0.5)
    )
    sf = math.tan(math.pi * 0.25 + sl1 * 0.5) ** sn * math.cos(sl1) / sn
    ro = re * sf / math.tan(math.pi * 0.25 + ola * 0.5) ** sn
    ra = re * sf / math.tan(math.pi * 0.25 + lat * degrad * 0.5) ** sn
    theta = lon * degrad - olo
    if theta > math.pi:
        theta -= 2 * math.pi
    if theta < -math.pi:
        theta += 2 * math.pi
    theta *= sn
    nx = int(math.floor(ra * math.sin(theta) + xo + 0.5))
    ny = int(math.floor(ro - ra * math.cos(theta) + yo + 0.5))
    return nx, ny


# [BUILD-03] Stream source rows by rowid. Resume begins after the last committed rowid.
def ingest_buildings(chunk: int = BUILDING_CHUNK, limit: Optional[int] = None, reset_checkpoint: bool = False) -> None:
    init_core()
    ensure_source_exists()
    require_disk_space(CORE_DB)
    s = building_schema()
    fields = [
        "address","name","sigungu","bjdong","pnu","purpose","approval","area","arch_area",
        "height","floors","basement","structure","roof","lat","lon"
    ]
    select = ["rowid AS building_rowid"]
    for k in fields:
        select.append(f"{qi(s[k])} AS {qi(k)}" if s.get(k) else f"NULL AS {qi(k)}")

    with db(CORE_DB) as dst:
        if reset_checkpoint:
            dst.execute("DELETE FROM trackb_checkpoints WHERE stage='buildings' AND partition_key='ALL'")
            dst.commit()
        cp = get_checkpoint(dst, "buildings")
        last = int(cp or 0)
        run_id = start_run(dst, "buildings", "building_hub", f"resume_after_rowid={last}")

    attempted = written = 0
    try:
        with db(LIVE_DB, readonly=True) as src, db(CORE_DB) as dst:
            while True:
                require_disk_space(CORE_DB)
                n = min(chunk, limit - attempted) if limit is not None else chunk
                if n <= 0:
                    break
                rows = src.execute(
                    f"SELECT {','.join(select)} FROM {qi(BUILDING_TABLE)} "
                    "WHERE rowid>? ORDER BY rowid LIMIT ?",
                    (last, n),
                ).fetchall()
                if not rows:
                    break

                payload = []
                addr_payload = []
                grid_payload = []
                stamp = now()

                for r in rows:
                    rid = int(r["building_rowid"])
                    lat, lon = fnum(r["lat"]), fnum(r["lon"])
                    if lat is None or lon is None or not (33.0 <= lat <= 39.5 and 124.0 <= lon <= 132.0):
                        lat = lon = None

                    addr = r["address"]
                    addr_norm = norm_addr(addr) or None
                    nx = ny = None
                    if lat is not None and lon is not None:
                        nx, ny = latlon_to_grid(lat, lon)

                    canonical = {
                        "address": addr, "name": r["name"], "sigungu": r["sigungu"],
                        "bjdong": r["bjdong"], "pnu": r["pnu"], "purpose": r["purpose"],
                        "approval": r["approval"], "area": fnum(r["area"]),
                        "arch_area": fnum(r["arch_area"]), "height": fnum(r["height"]),
                        "floors": fnum(r["floors"]), "basement": fnum(r["basement"]),
                        "structure": r["structure"], "roof": r["roof"], "lat": lat, "lon": lon,
                    }
                    row_hash = sha256_text(json.dumps(canonical, ensure_ascii=False, sort_keys=True, default=str))
                    payload.append((
                        rid, addr, addr_norm, r["name"], r["sigungu"], r["bjdong"], r["pnu"],
                        r["purpose"], r["approval"], canonical["area"], canonical["arch_area"],
                        canonical["height"], canonical["floors"], canonical["basement"],
                        r["structure"], r["roof"], lat, lon, spatial_cell(lat, lon), nx, ny,
                        "building_hub", row_hash, stamp,
                    ))
                    if addr_norm:
                        addr_payload.append((addr_norm, rid))
                    if nx is not None and ny is not None:
                        grid_payload.append((rid, nx, ny, stamp))

                # [BUILD-04] Building upsert is idempotent; address/grid mapping is updated in the same transaction.
                with transaction(dst):
                    dst.executemany("""
                        INSERT INTO trackb_building_features(
                            building_rowid,address,address_norm,name,sigungu,bjdong,pnu,purpose,
                            approval,area,arch_area,height,floors,basement,structure,roof,lat,lon,
                            spatial_cell,kma_nx,kma_ny,source_id,source_row_hash,collected_at
                        ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                        ON CONFLICT(building_rowid) DO UPDATE SET
                            address=excluded.address,address_norm=excluded.address_norm,
                            name=excluded.name,sigungu=excluded.sigungu,bjdong=excluded.bjdong,
                            pnu=excluded.pnu,purpose=excluded.purpose,approval=excluded.approval,
                            area=excluded.area,arch_area=excluded.arch_area,height=excluded.height,
                            floors=excluded.floors,basement=excluded.basement,
                            structure=excluded.structure,roof=excluded.roof,lat=excluded.lat,
                            lon=excluded.lon,spatial_cell=excluded.spatial_cell,
                            kma_nx=excluded.kma_nx,kma_ny=excluded.kma_ny,
                            source_row_hash=excluded.source_row_hash,collected_at=excluded.collected_at
                    """, payload)

                    # [BUILD-05] Rebuild mappings only for this chunk; never build a 7.8m-row Python address dict.
                    ids = [p[0] for p in payload]
                    for start in range(0, len(ids), 900):
                        part = ids[start:start+900]
                        marks = ",".join("?" for _ in part)
                        dst.execute(f"DELETE FROM trackb_address_match WHERE building_rowid IN ({marks})", part)
                    dst.executemany(
                        "INSERT OR IGNORE INTO trackb_address_match(address_norm,building_rowid) VALUES(?,?)",
                        addr_payload,
                    )
                    dst.executemany("""
                        INSERT INTO trackb_weather_grid_map(building_rowid,nx,ny,assigned_at)
                        VALUES(?,?,?,?)
                        ON CONFLICT(building_rowid) DO UPDATE SET
                            nx=excluded.nx,ny=excluded.ny,assigned_at=excluded.assigned_at
                    """, grid_payload)

                    last = int(rows[-1]["building_rowid"])
                    set_checkpoint(dst, "buildings", last)

                attempted += len(rows)
                written += len(payload)
                print(f"[BUILD] run={run_id[:8]} rows={written:,} checkpoint={last:,}")

                # [BUILD-06] Periodically fold WAL pages back into the main DB.
                if written and written % (chunk * 20) == 0:
                    dst.execute("PRAGMA wal_checkpoint(PASSIVE)")

            finish_run(dst, run_id, "SUCCESS", attempted, written)
            dst.execute("PRAGMA wal_checkpoint(PASSIVE)")
        print(f"[BUILD] SUCCESS written={written:,} last_rowid={last:,}")
    except Exception as exc:
        with db(CORE_DB) as dst:
            finish_run(dst, run_id, "FAILED", attempted, written, error=str(exc))
        raise


# ============================================================================
# [RAW] Immutable raw archive
# ============================================================================

# [RAW-01] Every imported evidence row is archived as gzip JSONL before/with normalized ingestion.
def raw_archive_path(source_id: str, run_id: str) -> Path:
    d = RAW_DIR / source_id / datetime.now(KST).strftime("%Y/%m/%d")
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{run_id}.jsonl.gz"


def archive_records(source_id: str, run_id: str, records: Iterable[dict]) -> tuple[Path, int]:
    path = raw_archive_path(source_id, run_id)
    count = 0
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True, default=str) + "\n")
            count += 1
    return path, count


# ============================================================================
# [MATCH] SQL-indexed exact address matching, bounded memory
# ============================================================================

# [MATCH-01] Exact normalized-address match is resolved in SQLite, not a nationwide Python dict.
def match_address_sql(c: sqlite3.Connection, address: Any) -> tuple[Optional[int], str]:
    a = norm_addr(address)
    if not a:
        return None, "NO_ADDRESS"
    rows = c.execute(
        "SELECT building_rowid FROM trackb_address_match WHERE address_norm=? LIMIT 2", (a,)
    ).fetchall()
    if len(rows) == 1:
        return int(rows[0]["building_rowid"]), "EXACT_UNIQUE"
    if len(rows) > 1:
        return None, "AMBIGUOUS"
    return None, "UNMATCHED"


# [MATCH-02] No fuzzy match is silently promoted to an exact building ID.
def normalized_yes_no(value: Any) -> str:
    x = str(value or "").strip().upper()
    return {
        "Y": YES, "YES": YES, "1": YES, "TRUE": YES, "설치": YES, "있음": YES,
        "N": NO, "NO": NO, "0": NO, "FALSE": NO, "미설치": NO, "없음": NO,
    }.get(x, UNKNOWN)


# ============================================================================
# [EVIDENCE] Append-only CSV/JSONL import with raw archive + resume
# ============================================================================

# [EVIDENCE-01] Streaming reader avoids records=list(...) on large evidence files.
def iter_records(path: Path) -> Iterator[dict]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            yield from csv.DictReader(f)
    elif suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
    else:
        raise ValueError("Supported evidence files: .csv, .jsonl")


# [EVIDENCE-02] Import is idempotent by evidence_id. Existing evidence is never overwritten.
def import_evidence(kind: str, path: Path, source_id: str, chunk: int = EVIDENCE_CHUNK) -> None:
    init_core()
    require_disk_space(CORE_DB)
    if kind not in {"fire", "sprinkler", "flood"}:
        raise ValueError("kind must be fire|sprinkler|flood")
    if not path.exists():
        raise FileNotFoundError(path)

    required = {
        "fire": {"address", "event_date"},
        "sprinkler": {"address", "sprinkler_status"},
        "flood": {"address", "observed_flood"},
    }[kind]

    with db(CORE_DB) as c:
        run_id = start_run(c, f"import_{kind}", source_id, f"file={path.name}")

    # [EVIDENCE-03] Raw source is preserved first. A second streaming pass performs normalization.
    raw_path, raw_count = archive_records(source_id, run_id, iter_records(path))
    attempted = written = unresolved = 0

    try:
        batch: list[dict] = []
        with db(CORE_DB) as c:
            for record in iter_records(path):
                if not required.issubset(record.keys()):
                    raise ValueError(f"Missing required columns {sorted(required)}")
                batch.append(record)
                if len(batch) >= chunk:
                    a, w, u = _write_evidence_batch(c, kind, source_id, batch)
                    attempted += a; written += w; unresolved += u
                    batch.clear()
                    require_disk_space(CORE_DB)
            if batch:
                a, w, u = _write_evidence_batch(c, kind, source_id, batch)
                attempted += a; written += w; unresolved += u
            finish_run(
                c, run_id, "SUCCESS", attempted, written, unresolved,
                error=None
            )
        print(
            f"[IMPORT] SUCCESS kind={kind} source={source_id} attempted={attempted:,} "
            f"new={written:,} unresolved={unresolved:,} raw={raw_path} raw_rows={raw_count:,}"
        )
    except Exception as exc:
        with db(CORE_DB) as c:
            finish_run(c, run_id, "FAILED", attempted, written, unresolved, str(exc))
        raise


# [EVIDENCE-04] One bounded transaction per batch.
def _write_evidence_batch(
    c: sqlite3.Connection, kind: str, source_id: str, batch: list[dict]
) -> tuple[int, int, int]:
    stamp = now()
    attempted = len(batch)
    unresolved = 0
    before = c.total_changes

    with transaction(c):
        for r in batch:
            rid, match_status = match_address_sql(c, r.get("address"))
            if match_status != "EXACT_UNIQUE":
                unresolved += 1

            raw_json = json.dumps(r, ensure_ascii=False, sort_keys=True, default=str)
            raw_hash = hashlib.sha256(raw_json.encode("utf-8")).hexdigest()
            ref = str(r.get("source_reference") or "")
            addr = r.get("address")
            addr_norm = norm_addr(addr) or None

            # [EVIDENCE-05] IDs include source+content, so reruns do not duplicate the same observation.
            evidence_id = sha256_text(kind, source_id, raw_hash)

            if kind == "fire":
                c.execute("""
                    INSERT OR IGNORE INTO trackb_fire_events(
                        evidence_id,event_date,address,address_norm,event_type,building_rowid,
                        match_status,source_id,source_reference,raw_sha256,collected_at
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    evidence_id, r.get("event_date"), addr, addr_norm, r.get("event_type"),
                    rid, match_status, source_id, ref, raw_hash, stamp
                ))

            elif kind == "sprinkler":
                status = normalized_yes_no(r.get("sprinkler_status"))
                c.execute("""
                    INSERT OR IGNORE INTO trackb_sprinkler_evidence(
                        evidence_id,observed_at,address,address_norm,building_name,
                        sprinkler_status,building_rowid,match_status,source_id,
                        source_reference,raw_sha256,collected_at
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    evidence_id, r.get("observed_at"), addr, addr_norm, r.get("building_name"),
                    status, rid, match_status, source_id, ref, raw_hash, stamp
                ))

            else:
                status = normalized_yes_no(r.get("observed_flood"))
                c.execute("""
                    INSERT OR IGNORE INTO trackb_flood_evidence(
                        evidence_id,observed_at,address,address_norm,observed_flood,
                        building_rowid,match_status,source_id,source_reference,raw_sha256,collected_at
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    evidence_id, r.get("observed_at"), addr, addr_norm, status,
                    rid, match_status, source_id, ref, raw_hash, stamp
                ))

    written = c.total_changes - before
    return attempted, written, unresolved


# ============================================================================
# [SPATIAL] Local-cell KDTree, never nationwide KDTree
# ============================================================================

# [SPATIAL-01] Parse the processing-cell key generated by spatial_cell().
def parse_cell(cell: str, step: float = 0.05) -> tuple[float, float, float, float]:
    iy, ix = map(int, cell.split(":"))
    lat0, lon0 = iy * step, ix * step
    return lat0, lat0 + step, lon0, lon0 + step


# [SPATIAL-02] Process one 0.05° cell at a time with a small halo.
# This computes centroid distance only; it must never be described as facade/wall gap.
def spatial_centroids(
    radius_m: float = 100.0,
    max_cells: Optional[int] = None,
    reset_checkpoint: bool = False,
) -> None:
    try:
        import numpy as np
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise RuntimeError("spatial stage requires numpy and scipy") from exc

    init_core()
    with db(CORE_DB) as c:
        if reset_checkpoint:
            c.execute("DELETE FROM trackb_checkpoints WHERE stage='spatial_cells'")
            c.commit()
        cells = [
            r["spatial_cell"] for r in c.execute("""
                SELECT DISTINCT spatial_cell
                FROM trackb_building_features
                WHERE spatial_cell IS NOT NULL
                ORDER BY spatial_cell
            """)
        ]
        done = {
            r["partition_key"] for r in c.execute("""
                SELECT partition_key FROM trackb_checkpoints
                WHERE stage='spatial_cells' AND status='OK'
            """)
        }
        cells = [x for x in cells if x not in done]
        if max_cells is not None:
            cells = cells[:max_cells]
        run_id = start_run(c, "spatial_cells", "geometry", f"cells={len(cells)}")

    attempted = written = 0
    try:
        with db(CORE_DB) as c:
            for idx, cell in enumerate(cells, 1):
                require_disk_space(CORE_DB)
                lat0, lat1, lon0, lon1 = parse_cell(cell)

                # [SPATIAL-03] Halo is approximate degrees, deliberately larger than 100 m.
                halo_lat = 0.002
                center_lat = (lat0 + lat1) / 2
                halo_lon = 0.002 / max(math.cos(math.radians(center_lat)), 0.3)

                core = c.execute("""
                    SELECT building_rowid,lat,lon FROM trackb_building_features
                    WHERE spatial_cell=? AND lat IS NOT NULL AND lon IS NOT NULL
                """, (cell,)).fetchall()
                if not core:
                    set_checkpoint(c, "spatial_cells", "EMPTY", cell)
                    c.commit()
                    continue

                halo = c.execute("""
                    SELECT building_rowid,lat,lon FROM trackb_building_features
                    WHERE lat BETWEEN ? AND ? AND lon BETWEEN ? AND ?
                      AND lat IS NOT NULL AND lon IS NOT NULL
                """, (lat0-halo_lat, lat1+halo_lat, lon0-halo_lon, lon1+halo_lon)).fetchall()

                h_ids = np.array([int(r["building_rowid"]) for r in halo], dtype=np.int64)
                h_lat = np.radians(np.array([float(r["lat"]) for r in halo], dtype=float))
                h_lon = np.radians(np.array([float(r["lon"]) for r in halo], dtype=float))
                xyz = np.column_stack((
                    np.cos(h_lat)*np.cos(h_lon),
                    np.cos(h_lat)*np.sin(h_lon),
                    np.sin(h_lat),
                ))
                tree = cKDTree(xyz)
                radius_chord = 2 * np.sin(radius_m / (2 * 6371000.0))
                id_to_idx = {int(v): i for i, v in enumerate(h_ids)}

                payload = []
                stamp = now()
                for r in core:
                    rid = int(r["building_rowid"])
                    i = id_to_idx[rid]
                    point = xyz[i]
                    neighbors = tree.query_ball_point(point, radius_chord)
                    neighbor_count = max(0, len(neighbors) - 1)

                    # [SPATIAL-04] k=2 returns self + nearest other centroid.
                    if len(halo) >= 2:
                        d, _ = tree.query(point, k=2)
                        chord = float(d[1])
                        nearest = 2 * 6371000.0 * math.asin(min(max(chord / 2, 0.0), 1.0))
                    else:
                        nearest = None
                    payload.append((
                        rid, neighbor_count, nearest,
                        "CENTROID_DISTANCE_PROXY_NOT_WALL_GAP", stamp
                    ))

                with transaction(c):
                    c.executemany("""
                        INSERT INTO trackb_spatial_features(
                            building_rowid,neighbor_100m_count,nearest_centroid_m,
                            spatial_method,measured_at
                        ) VALUES(?,?,?,?,?)
                        ON CONFLICT(building_rowid) DO UPDATE SET
                            neighbor_100m_count=excluded.neighbor_100m_count,
                            nearest_centroid_m=excluded.nearest_centroid_m,
                            spatial_method=excluded.spatial_method,
                            measured_at=excluded.measured_at
                    """, payload)
                    set_checkpoint(c, "spatial_cells", len(payload), cell)

                attempted += len(core)
                written += len(payload)
                if idx % 25 == 0 or idx == len(cells):
                    print(f"[SPATIAL] cells={idx}/{len(cells)} buildings={written:,}")

            finish_run(c, run_id, "SUCCESS", attempted, written)
            c.execute("PRAGMA wal_checkpoint(PASSIVE)")
        print(f"[SPATIAL] SUCCESS buildings={written:,}; metric=centroid proxy")
    except Exception as exc:
        with db(CORE_DB) as c:
            finish_run(c, run_id, "FAILED", attempted, written, error=str(exc))
        raise


# ============================================================================
# [KMA] Unique-grid current/forecast collection into separate weather DB
# ============================================================================

# [KMA-01] Base times are operational scheduling logic; resultCode still determines validity.
def kma_base(operation: str) -> tuple[str, str]:
    t = datetime.now(KST)
    if operation == "getUltraSrtNcst":
        t -= timedelta(minutes=45)
        return t.strftime("%Y%m%d"), t.strftime("%H00")
    if operation == "getUltraSrtFcst":
        t -= timedelta(minutes=55)
        minute = (t.minute // 30) * 30
        return t.strftime("%Y%m%d"), f"{t.hour:02d}{minute:02d}"
    if operation == "getVilageFcst":
        t -= timedelta(minutes=15)
        bases = [2,5,8,11,14,17,20,23]
        valid = [h for h in bases if h <= t.hour]
        if not valid:
            t -= timedelta(days=1)
            h = 23
        else:
            h = valid[-1]
        return t.strftime("%Y%m%d"), f"{h:02d}00"
    raise ValueError(operation)


# [KMA-02] Retry transient failures; never convert an API error/no-data response into weather.
def kma_request(operation: str, nx: int, ny: int, retries: int = 4) -> tuple[list[dict], str, str, dict]:
    if not KMA_KEY:
        raise RuntimeError("KMA_SERVICE_KEY is missing from .env")
    base_date, base_time = kma_base(operation)
    params = {
        "serviceKey": KMA_KEY,
        "pageNo": 1,
        "numOfRows": 1000,
        "dataType": "JSON",
        "base_date": base_date,
        "base_time": base_time,
        "nx": int(nx),
        "ny": int(ny),
    }
    last_error: Optional[Exception] = None
    for attempt in range(retries):
        try:
            res = requests.get(f"{KMA_BASE}/{operation}", params=params, timeout=(10, 30))
            res.raise_for_status()
            payload = res.json()
            response = payload["response"]
            header = response["header"]
            if str(header.get("resultCode")) != "00":
                raise RuntimeError(f"KMA resultCode={header.get('resultCode')} {header.get('resultMsg')}")
            items = response.get("body", {}).get("items", {}).get("item", [])
            if isinstance(items, dict):
                items = [items]
            if not isinstance(items, list):
                items = []
            return items, base_date, base_time, payload
        except (requests.RequestException, ValueError, KeyError, RuntimeError) as exc:
            last_error = exc
            time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"KMA request failed grid=({nx},{ny}): {last_error}")


# [KMA-03] Read DISTINCT grids from the mapping table. No building-level weather duplication.
def collect_kma(
    operation: str,
    max_grids: Optional[int] = None,
    sleep_seconds: float = 0.15,
    force: bool = False,
) -> None:
    if operation not in {"getUltraSrtNcst", "getUltraSrtFcst", "getVilageFcst"}:
        raise ValueError(operation)
    init_core()
    init_weather()
    require_disk_space(WEATHER_DB)

    with db(CORE_DB, readonly=True) as core:
        sql = "SELECT DISTINCT nx,ny FROM trackb_weather_grid_map ORDER BY nx,ny"
        params: tuple = ()
        if max_grids is not None:
            sql += " LIMIT ?"
            params = (int(max_grids),)
        grids = [(int(r["nx"]), int(r["ny"])) for r in core.execute(sql, params)]

    if not grids:
        raise RuntimeError("No KMA grid mapping. Run buildings first.")

    run_id = uuid.uuid4().hex
    with db(WEATHER_DB) as w:
        w.execute("""
            INSERT INTO weather_runs(run_id,operation,status,started_at)
            VALUES(?,?,?,?)
        """, (run_id, operation, "RUNNING", now()))
        w.commit()

    attempted = success = 0
    last_error = None
    try:
        with db(WEATHER_DB) as w:
            for nx, ny in grids:
                require_disk_space(WEATHER_DB)
                base_date, base_time = kma_base(operation)

                # [KMA-04] Same grid/base cycle is skipped unless --force is requested.
                if not force:
                    cp = w.execute("""
                        SELECT last_base_date,last_base_time,status
                        FROM weather_checkpoints
                        WHERE operation=? AND nx=? AND ny=?
                    """, (operation, nx, ny)).fetchone()
                    if cp and cp["status"] == "OK" and cp["last_base_date"] == base_date and cp["last_base_time"] == base_time:
                        continue

                attempted += 1
                try:
                    items, base_date, base_time, raw = kma_request(operation, nx, ny)

                    # [KMA-05] Preserve raw API response once per grid/cycle for audit.
                    raw_id = sha256_text(operation, nx, ny, base_date, base_time)
                    raw_path = RAW_DIR / "kma_short" / base_date
                    raw_path.mkdir(parents=True, exist_ok=True)
                    with gzip.open(raw_path / f"{raw_id}.json.gz", "wt", encoding="utf-8") as f:
                        json.dump(raw, f, ensure_ascii=False)

                    rows = []
                    stamp = now()
                    for item in items:
                        category = item.get("category")
                        if not category:
                            continue
                        value = item.get("obsrValue", item.get("fcstValue"))
                        rows.append((
                            nx, ny, operation, base_date, base_time,
                            str(item.get("fcstDate", base_date)),
                            str(item.get("fcstTime", base_time)),
                            str(category), None if value is None else str(value),
                            "kma_short", stamp,
                        ))

                    with transaction(w):
                        w.executemany("""
                            INSERT INTO weather_observations(
                                nx,ny,operation,base_date,base_time,forecast_date,
                                forecast_time,category,value,source_id,collected_at
                            ) VALUES(?,?,?,?,?,?,?,?,?,?,?)
                            ON CONFLICT(
                                nx,ny,operation,base_date,base_time,
                                forecast_date,forecast_time,category
                            ) DO UPDATE SET
                                value=excluded.value,
                                collected_at=excluded.collected_at
                        """, rows)
                        w.execute("""
                            INSERT INTO weather_checkpoints(
                                operation,nx,ny,last_base_date,last_base_time,status,updated_at
                            ) VALUES(?,?,?,?,?,?,?)
                            ON CONFLICT(operation,nx,ny) DO UPDATE SET
                                last_base_date=excluded.last_base_date,
                                last_base_time=excluded.last_base_time,
                                status=excluded.status,
                                updated_at=excluded.updated_at
                        """, (operation, nx, ny, base_date, base_time, "OK", stamp))
                    success += 1
                except Exception as exc:
                    last_error = str(exc)
                    w.execute("""
                        INSERT INTO weather_checkpoints(
                            operation,nx,ny,last_base_date,last_base_time,status,updated_at
                        ) VALUES(?,?,?,?,?,?,?)
                        ON CONFLICT(operation,nx,ny) DO UPDATE SET
                            last_base_date=excluded.last_base_date,
                            last_base_time=excluded.last_base_time,
                            status=excluded.status,
                            updated_at=excluded.updated_at
                    """, (operation, nx, ny, base_date, base_time, "FAILED", now()))
                    w.commit()
                    print(f"[KMA] FAILED grid=({nx},{ny}) {exc}", file=sys.stderr)

                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)

            w.execute("""
                UPDATE weather_runs
                SET status=?,finished_at=?,grids_attempted=?,grids_success=?,last_error=?
                WHERE run_id=?
            """, ("SUCCESS" if success == attempted else "PARTIAL",
                  now(), attempted, success, last_error, run_id))
            w.commit()
            w.execute("PRAGMA wal_checkpoint(PASSIVE)")
        print(f"[KMA] operation={operation} success={success:,}/{attempted:,}")
    except Exception as exc:
        with db(WEATHER_DB) as w:
            w.execute("""
                UPDATE weather_runs SET status='FAILED',finished_at=?,grids_attempted=?,
                    grids_success=?,last_error=? WHERE run_id=?
            """, (now(), attempted, success, str(exc), run_id))
            w.commit()
        raise


# ============================================================================
# [QUALITY] Coverage, integrity and operational diagnostics
# ============================================================================

# [QUALITY-01] Counts are reported; completeness is never inferred from table existence.
def status() -> None:
    init_core()
    init_weather()
    ensure_source_exists()
    with db(LIVE_DB, readonly=True) as src:
        live_count = int(src.execute(f"SELECT COUNT(*) FROM {qi(BUILDING_TABLE)}").fetchone()[0])

    with db(CORE_DB, readonly=True) as c:
        def count(table: str) -> int:
            return int(c.execute(f"SELECT COUNT(*) FROM {qi(table)}").fetchone()[0])

        buildings = count("trackb_building_features")
        valid_coords = int(c.execute("""
            SELECT COUNT(*) FROM trackb_building_features
            WHERE lat IS NOT NULL AND lon IS NOT NULL
        """).fetchone()[0])
        grids = int(c.execute("SELECT COUNT(DISTINCT nx||':'||ny) FROM trackb_weather_grid_map").fetchone()[0])
        fire_exact = int(c.execute("""
            SELECT COUNT(*) FROM trackb_fire_events WHERE match_status='EXACT_UNIQUE'
        """).fetchone()[0])
        sprinkler_known = int(c.execute("""
            SELECT COUNT(*) FROM trackb_sprinkler_evidence
            WHERE match_status='EXACT_UNIQUE' AND sprinkler_status IN ('YES','NO')
        """).fetchone()[0])
        flood_positive = int(c.execute("""
            SELECT COUNT(*) FROM trackb_flood_evidence
            WHERE match_status='EXACT_UNIQUE' AND observed_flood='YES'
        """).fetchone()[0])
        spatial = count("trackb_spatial_features")
        cp = {
            f"{r['stage']}:{r['partition_key']}": r["cursor_value"]
            for r in c.execute("""
                SELECT stage,partition_key,cursor_value
                FROM trackb_checkpoints
                WHERE partition_key='ALL'
            """)
        }

    with db(WEATHER_DB, readonly=True) as w:
        weather_rows = int(w.execute("SELECT COUNT(*) FROM weather_observations").fetchone()[0])

    result = {
        "source_buildings": live_count,
        "trackb_buildings": buildings,
        "building_coverage_pct": round(buildings / live_count * 100, 4) if live_count else None,
        "valid_coordinates": valid_coords,
        "unique_kma_grids": grids,
        "spatial_feature_rows": spatial,
        "exact_fire_events": fire_exact,
        "known_sprinkler_evidence_rows": sprinkler_known,
        "positive_flood_evidence_rows": flood_positive,
        "weather_rows": weather_rows,
        "core_db_bytes": CORE_DB.stat().st_size if CORE_DB.exists() else 0,
        "weather_db_bytes": WEATHER_DB.stat().st_size if WEATHER_DB.exists() else 0,
        "checkpoints": cp,
        "core_quick_check": integrity(CORE_DB, quick=True),
        "weather_quick_check": integrity(WEATHER_DB, quick=True),
        "non_claims": [
            "Missing fire evidence is not Y=0.",
            "Missing sprinkler evidence is UNKNOWN, not NO.",
            "Centroid distance is not wall-to-wall gap.",
            "Current KMA data is not historical event-time training weather.",
            "Flood WMS/image pixels are not automatically treated as validated flood polygons.",
            "No nationwide sprinkler coverage is claimed.",
        ],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


# [QUALITY-02] Explicit WAL maintenance command for clean shutdown/backup preparation.
def maintenance(truncate_wal: bool = False) -> None:
    init_core()
    init_weather()
    mode = "TRUNCATE" if truncate_wal else "PASSIVE"
    for path in (CORE_DB, WEATHER_DB):
        print(f"[MAINT] {path.name} wal_checkpoint={checkpoint_wal(path, mode)}")
        print(f"[MAINT] {path.name} quick_check={integrity(path, quick=True)}")


# ============================================================================
# [CLI] Operational entry point
# ============================================================================

# [CLI-01] Separate commands make every expensive stage explicit and resumable.
def main() -> None:
    p = argparse.ArgumentParser(
        description="DDN Track-B v2: large-scale evidence-first collector"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init", help="Create/upgrade core and weather schemas")
    sub.add_parser("status", help="Coverage + integrity report")

    b = sub.add_parser("buildings", help="Resume nationwide building materialization")
    b.add_argument("--chunk", type=int, default=BUILDING_CHUNK)
    b.add_argument("--limit", type=int)
    b.add_argument("--reset-checkpoint", action="store_true")

    e = sub.add_parser("import", help="Append verified external evidence")
    e.add_argument("--kind", choices=["fire", "sprinkler", "flood"], required=True)
    e.add_argument("--file", type=Path, required=True)
    e.add_argument("--source-id", required=True)
    e.add_argument("--chunk", type=int, default=EVIDENCE_CHUNK)

    s = sub.add_parser("spatial", help="Cell-by-cell centroid proximity proxy")
    s.add_argument("--radius-m", type=float, default=100.0)
    s.add_argument("--max-cells", type=int)
    s.add_argument("--reset-checkpoint", action="store_true")

    k = sub.add_parser("kma", help="Collect current/forecast KMA data by UNIQUE grid")
    k.add_argument(
        "--operation",
        choices=["getUltraSrtNcst", "getUltraSrtFcst", "getVilageFcst"],
        default="getUltraSrtNcst",
    )
    k.add_argument("--max-grids", type=int)
    k.add_argument("--sleep", type=float, default=0.15)
    k.add_argument("--force", action="store_true")

    m = sub.add_parser("maintenance", help="WAL checkpoint + quick integrity check")
    m.add_argument("--truncate-wal", action="store_true")

    args = p.parse_args()

    if args.cmd == "init":
        init_core(); init_weather()
        print(f"[INIT] core={CORE_DB}")
        print(f"[INIT] weather={WEATHER_DB}")
        print(f"[INIT] raw={RAW_DIR}")
    elif args.cmd == "status":
        status()
    elif args.cmd == "buildings":
        ingest_buildings(args.chunk, args.limit, args.reset_checkpoint)
    elif args.cmd == "import":
        import_evidence(args.kind, args.file, args.source_id, args.chunk)
    elif args.cmd == "spatial":
        spatial_centroids(args.radius_m, args.max_cells, args.reset_checkpoint)
    elif args.cmd == "kma":
        collect_kma(args.operation, args.max_grids, args.sleep, args.force)
    elif args.cmd == "maintenance":
        maintenance(args.truncate_wal)


if __name__ == "__main__":
    main()
