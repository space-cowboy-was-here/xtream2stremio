#!/usr/bin/env python3

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json
import logging
import os
import re
import time
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse, urlunparse

import aiohttp
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn




# ----------------------------
# Config helpers
# ----------------------------
def load_env_file(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v


def env(name: str, default: Optional[str] = None, required: bool = False) -> str:
    v = os.getenv(name, default)
    if required and (v is None or v.strip() == ""):
        raise SystemExit(f"Missing required env var: {name}")
    return (v or "").strip()

def env_str(k: str, d: str="") -> str:
    v = os.environ.get(k)
    return d if v is None or v=="" else v

def env_int(name: str, default: int = 0, required: bool = False) -> int:
    v = env(name, None, required)
    if v is None or v == "":
        return default
    try:
        return int(v)
    except Exception:
        raise SystemExit(f"Invalid int env var: {name}={v!r}")

def env_float(k: str, d: float) -> float:
    v = os.environ.get(k)
    if not v: return d
    try: return float(v)
    except: return d
    
def env_bool(k: str, d: bool) -> bool:
    v = os.environ.get(k)
    if not v: return d
    v = v.strip().lower()
    if v in ("1","true","yes","y","on"): return True
    if v in ("0","false","no","n","off"): return False
    return d


load_env_file(env_str("ENV_FILE",".env"))

# ----------------------------
# Logging
# ----------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s (xtream2stemio) [%(levelname)s] %(message)s",
)
log = logging.getLogger("xtream2stemio")


# ----------------------------
# ENVIRONMENT VARIABLES
# ----------------------------

XTREAM_BASE = env("XTREAM_BASE", required=True)  # e.g. https://host:port
XTREAM_API_BASE = XTREAM_BASE + "/player_api.php"
XTREAM_USERNAME = env("XTREAM_USERNAME", required=True)
XTREAM_PASSWORD = env("XTREAM_PASSWORD", required=True)

XTREAM_STREAM_BASE = env("XTREAM_STREAM_BASE", default="")  # e.g. https://host:port (optional)
CINEMETA_BASE = env("CINEMETA_BASE", default="https://v3-cinemeta.strem.io").rstrip("/")

# Hide endpoints behind a base path (recommended when exposing publicly)
BASE_PATH = env_str("BASE_PATH", "")
if BASE_PATH and not BASE_PATH.startswith("/"):
    BASE_PATH = "/" + BASE_PATH
    

RESET_CACHE = env_bool("RESET_CACHE", False)

ADDON_NAME = env_str("ADDON_NAME", "Xtream2Stremio")
ADDON_DESCRIPTION = env_str("ADDON_DESCRIPTION", "Get streams from your Xtream VOD library. Made with <3 by Space Cowboy.")

REFRESH_MINUTES = int(env("REFRESH_MINUTES", default="1440"))  # refresh Xtream lists every N minutes

WAIT_FOR_INDEX_ON_START = env("WAIT_FOR_INDEX_ON_START", default="1").lower() not in ("0", "false", "no")



MAX_STREAMS_PER_MATCH = int(env("MAX_STREAMS_PER_MATCH", default="30"))

HTTP_TIMEOUT_S = float(env("HTTP_TIMEOUT_S", default="30"))






# Xtream common stream URL shapes
VOD_PATH_PREFIX = env("VOD_PATH_PREFIX", default="movie").strip("/").strip()
SERIES_PATH_PREFIX = env("SERIES_PATH_PREFIX", default="series").strip("/").strip()



TITLE_CLEANING = env_str("TITLE_CLEANING", "[]")
try:
    parsed = json.loads(TITLE_CLEANING)
    if not isinstance(parsed, list):
        raise ValueError("TITLE_CLEANING must be a JSON list")
except Exception:
    log.warning("Invalid TITLE_CLEANING env var; expected JSON list of dicts, using empty list")
    parsed = []

# Normalize to list of (substring_lower, replacement) tuples
TITLE_CLEANING = []
for entry in parsed:
    if isinstance(entry, dict):
        for k, v in entry.items():
            TITLE_CLEANING.append((str(k).lower(), "" if v is None else str(v)))
            


CATEGORY_CLEANING = env_str("CATEGORY_CLEANING", "[]")
try:
    parsed = json.loads(CATEGORY_CLEANING)
    if not isinstance(parsed, list):
        raise ValueError("CATEGORY_CLEANING must be a JSON list")
except Exception:
    log.warning("Invalid CATEGORY_CLEANING env var; expected JSON list of dicts, using empty list")
    parsed = []

# Normalize to list of (substring_lower, replacement) tuples
CATEGORY_CLEANING = []
for entry in parsed:
    if isinstance(entry, dict):
        for k, v in entry.items():
            CATEGORY_CLEANING.append((str(k).lower(), "" if v is None else str(v)))



CATEGORY_EXCLUDE_FILTER = env_str("CATEGORY_EXCLUDE_FILTER", "[]")
try:
    parsed = json.loads(CATEGORY_EXCLUDE_FILTER)
    if not isinstance(parsed, list):
        raise ValueError("CATEGORY_EXCLUDE_FILTER must be a JSON list")
except Exception:
    log.warning("Invalid CATEGORY_EXCLUDE_FILTER env var; expected JSON list of strings, using empty list")
    parsed = []

CATEGORY_EXCLUDE_FILTER = [str(x).lower() for x in parsed]

TITLE_EXCLUDE_FILTER = env_str("TITLE_EXCLUDE_FILTER", "[]")
try:
    parsed = json.loads(TITLE_EXCLUDE_FILTER)
    if not isinstance(parsed, list):
        raise ValueError("TITLE_EXCLUDE_FILTER must be a JSON list")
except Exception:
    log.warning("Invalid TITLE_EXCLUDE_FILTER env var; expected JSON list of strings, using empty list")
    parsed = []

TITLE_EXCLUDE_FILTER = [str(x).lower() for x in parsed]

STREAM_NAME_TEMPLATE_MOVIE = env_str("STREAM_NAME_TEMPLATE_MOVIE", "VOD {quality}")
STREAM_DESC_TEMPLATE_MOVIE = env_str("STREAM_DESC_TEMPLATE_MOVIE", "{title} \nCategory: {category_name} \nXtream2Stremio")
STREAM_NAME_TEMPLATE_SERIES = env_str("STREAM_NAME_TEMPLATE_SERIES", "VOD {quality}")
STREAM_DESC_TEMPLATE_SERIES = env_str("STREAM_DESC_TEMPLATE_SERIES", "{title} \nCategory: {category_name} \nXtream2Stremio")

WRITE_SERIES_INFO = env_bool("WRITE_SERIES_INFO", True)

DATA_PATH = env_str("DATA_PATH","data")


HOST = env("HOST", "0.0.0.0")
PORT = int(env("PORT", "7348"))


def derive_stream_base(api_base: str) -> str:
    u = urlparse(api_base)
    return urlunparse((u.scheme, u.netloc, "", "", "", "")).rstrip("/")


if not XTREAM_STREAM_BASE:
    XTREAM_STREAM_BASE = derive_stream_base(XTREAM_API_BASE)





















def load_json(path: str, default: Any) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default



def detect_quality(name: str) -> str:
    s = name.lower()
    if "2160" in s or "4k" in s or "uhd" in s:
        return "4K"
    if "1080" in s:
        return "1080p"
    if "720" in s:
        return "720p"
    if "480" in s:
        return "480p"
    return ""


# ----------------------------
# Xtream models + HTTP
# ----------------------------
@dataclass
class VodStream:
    stream_id: str
    name: str
    ext: str
    norm: str
    year: Optional[int]
    raw: Dict[str, Any]


@dataclass
class SeriesItem:
    series_id: str
    name: str
    norm: str
    year: Optional[int]
    raw: Dict[str, Any]


@dataclass
class SeriesInfoCacheEntry:
    ts: float
    data: Dict[str, Any]


class HttpClient:
    def __init__(self) -> None:
        timeout = aiohttp.ClientTimeout(total=HTTP_TIMEOUT_S)
        self.session = aiohttp.ClientSession(timeout=timeout, raise_for_status=False)

    async def close(self) -> None:
        await self.session.close()

    async def get_json(self, url: str, params: Optional[Dict[str, Any]] = None, tries: int = 3, label: str = "") -> Any:
        last_err = None
        for i in range(tries):
            try:
                async with self.session.get(url, params=params) as r:
                    txt = await r.text()
                    if r.status >= 400:
                        # retry common transient failures
                        if r.status in (429, 500, 502, 503, 504) and i < tries - 1:
                            await asyncio.sleep(0.6 * (2 ** i))
                            continue
                        raise RuntimeError(f"HTTP {r.status} {label} url={url} body={txt[:200]}")
                    if not txt:
                        return None
                    return json.loads(txt)
            except Exception as e:
                last_err = e
                log.debug(f"Attempt {i+1}/{tries} failed for {label}: {e}")
                if i < tries - 1:
                    await asyncio.sleep(0.6 * (2 ** i))
        raise RuntimeError(f"Failed GET {label} {url}: {last_err}")


class XtreamIndex:
    # Keeps in-memory index of VOD/series items from Xtream API for matching.
    # Supports refreshing the index at custom frequency (REFRESH_MINUTES).
    # Saves cache to disk in case of restarts (set RESET_CACHE=true to force full reindex).
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self.ready = False

        self.vod_cache: Dict[str, List[Dict[str,str]]] = {}
        self.series_cache: Dict[str, List[Dict[str,str]]] = {}

        self.series_info_cache: Dict[str, SeriesInfoCacheEntry] = {}
        
        self.eps: Dict[str, List[Dict[str,str]]] = {}
        
    
    
    ### MAIN INDEX LOOP ###

    async def refresh(self, http: HttpClient) -> None:
        async with self._lock:
            t0 = time.time()
            log.info("Refreshing Xtream lists...")
            
            
            
            ## MOVIES
            
            log.info("Fetching Movies...")
            try:
                vod = await self._fetch_vod(http)
            except Exception as e:
                log.error("Error fetching VOD: %s", e)
                vod = []
            log.info("%d Movie Streams Found", len(vod))
                    
                
            log.info("Fetching Movie Categories...")
            try:
                vod_cats = await self._fetch_vod_categories(http)
            except Exception as e:
                log.error("Error fetching Movie Categories: %s", e)
                vod_cats = []
            if vod_cats:
                vod_cats_dict = {str(cat.get("category_id") or cat.get("id") or ""): cat for cat in vod_cats}
            log.info("%d Movie Categories Found", len(vod_cats))
            
            log.info("Building Movie Index...")
            try:
                self._build_vod(vod, vod_cats_dict)
            except Exception as e:
                log.error("Error building Movie Index: %s", e)
            log.info("Movie Index Built: total_streams=%d titles=%d", len(vod), len(self.vod_cache))


            ## SERIES

            log.info("Fetching Series...")
            try:
                series = await self._fetch_series(http)
            except Exception as e:
                log.error("Error fetching Series: %s", e)
                series = []
            log.info("%d Series Streams Found", len(series))

            log.info("Fetching Series Categories...")
            try:
                series_cats = await self._fetch_series_categories(http)
            except Exception as e:
                log.error("Error fetching Series Categories: %s", e)
                series_cats = []
            if series_cats:
                series_cats_dict = {str(cat.get("category_id") or cat.get("id") or ""): cat for cat in series_cats}
            log.info("%d Series Categories Found", len(series_cats))

            log.info("Building Series Index...")
            try:
                self._build_series(series, series_cats_dict)
            except Exception as e:
                log.error("Error building Series Index: %s", e)
            log.info("Series Index Built: total_streams=%d titles=%d", len(series), len(self.series_cache))


            self.ready = True
            log.info(
                "\n=========================================================================================\n" 
                + "=========================================================================================\n"
                + "Refresh done!\nSummary:\nMovie Streams: %d\nMovie Titles (after filters): %d\nSeries: %d\nSeries Titles (after filters): %d\nElapsed: %.1f sec",
                len(vod), len(self.vod_cache), len(series), len(self.series_cache), time.time() - t0
            )
            
            display_host = HOST
            if display_host == "0.0.0.0":
                display_host = "127.0.0.1"
            log.info(
                "\n=========================================================================================\n" 
                + "=========================================================================================\n"
                + "ADDON IS READY FOR USE\n"
                + f"LOCAL MANIFEST URL: http://{display_host}:{PORT}{BASE_PATH}/manifest.json\n"
                + "=========================================================================================\n"
                + "=========================================================================================\n"
            )
            
            
            # write cache to file
            with open(f"{DATA_PATH}/vod_cache.json", "w", encoding="utf-8") as f:
                json.dump(self.vod_cache, f, indent=2, ensure_ascii=True)
                
            with open(f"{DATA_PATH}/series_cache.json", "w", encoding="utf-8") as f:
                json.dump(self.series_cache, f, indent=2, ensure_ascii=True)



    
    ### FETCH WORKERS ###
    
    async def _fetch_vod(self, http: HttpClient) -> List[Dict[str, Any]]:
        params = {"username": XTREAM_USERNAME, "password": XTREAM_PASSWORD, "action": "get_vod_streams"}
        log.debug("Fetching VOD with params: %s", params)
        no_vod = True
        attempts = 0
        while no_vod and attempts < 5:
            try:
                data = await http.get_json(XTREAM_API_BASE, params=params, label="xtream.get_vod_streams")
            except aiohttp.ClientResponseError as e:
                if e.status == 403:
                    log.error("Access denied fetching VOD streams (403). Check your Xtream credentials and permissions.")
                    return []
            attempts += 1
            if isinstance(data, list) and data:
                no_vod = False
            else:
                log.warning("No VOD streams returned, attempt %d/5. Retrying...", attempts)
                log.debug("Fetched VOD data: %s", data)
                log.debug("Fetched VOD data count: %d", len(data) if isinstance(data, list) else 0)
                await asyncio.sleep(5 * attempts)
        return data if isinstance(data, list) else []

    async def _fetch_series(self, http: HttpClient) -> List[Dict[str, Any]]:
        params = {"username": XTREAM_USERNAME, "password": XTREAM_PASSWORD, "action": "get_series"}
        log.debug("Fetching Series with params: %s", params)
        no_series = True
        attempts = 0
        while no_series and attempts < 5:
            try:
                data = await http.get_json(XTREAM_API_BASE, params=params, label="xtream.get_series")
            except aiohttp.ClientResponseError as e:
                if e.status == 403:
                    log.error("Access denied fetching Series (403). Check your Xtream credentials and permissions.")
                    return []
            attempts += 1
            if isinstance(data, list) and data:
                no_series = False
            else:
                log.warning("No Series data returned, attempt %d/5. Retrying...", attempts)
                log.debug("Fetched Series data: %s", data)
                log.debug("Fetched Series data count: %d", len(data) if isinstance(data, list) else 0)
                await asyncio.sleep(5 * attempts)
        return data if isinstance(data, list) else []

    async def _fetch_series_info(self, http: HttpClient, series_id: str, info_sem: asyncio.Semaphore, sleep = 0) -> Dict[str, Any]:
        if not series_id:
            return None, {}
        
        async with info_sem:
        
            # send both series_id and series params for compatibility, okay to send both (only one is needed)
            params = {"username": XTREAM_USERNAME, "password": XTREAM_PASSWORD, "action": "get_series_info", "series_id": series_id, "series": series_id}
            try:
                data = await http.get_json(XTREAM_API_BASE, params=params, label=f"xtream.get_series_info.{series_id}")
            except Exception:
                return series_id, {}
            if isinstance(data, dict):
                return series_id, data
            elif isinstance(data, list):
                # some providers return a list with a single dict
                if data and isinstance(data[0], dict):
                    return series_id, data[0]
            if sleep > 0:
                await asyncio.sleep(sleep)
            return series_id, {}
    
    async def _fetch_vod_categories(self, http: HttpClient) -> List[Dict[str, Any]]:
        params = {"username": XTREAM_USERNAME, "password": XTREAM_PASSWORD, "action": "get_vod_categories"}
        data = await http.get_json(XTREAM_API_BASE, params=params, label="xtream.get_vod_categories")
        return data if isinstance(data, list) else []
    
    async def _fetch_series_categories(self, http: HttpClient) -> List[Dict[str, Any]]:
        params = {"username": XTREAM_USERNAME, "password": XTREAM_PASSWORD, "action": "get_series_categories"}
        data = await http.get_json(XTREAM_API_BASE, params=params, label="xtream.get_series_categories")
        return data if isinstance(data, list) else []
    
    
    
    
    
    ### INDEX BUILDERS ###

    def _build_vod(self, vod_items: List[Dict[str, Any]], categories: Dict[str, Any]) -> None:
        
        if (os.path.isfile(f"{DATA_PATH}/vod_cache.json")) and not self.vod_cache:
            with open(f"{DATA_PATH}/vod_cache.json", "r", encoding="utf-8") as f:
                self.vod_cache = json.load(f)
            log.info("Loaded existing VOD cache from file with %d titles", len(self.vod_cache))

        no_tmdb_count = 0
        skipped = 0
        for it in vod_items:
            title = str(it.get("name") or it.get("title") or "").strip()
            if not title:
                continue
            tmdb_id = str(it.get("tmdb") or it.get("tmdb_id") or it.get("moviedb") or it.get("moviedb_id") or 0)
            if not tmdb_id:
                no_tmdb_count += 1
                continue
            stream_id = str(it.get("stream_id") or it.get("id") or "").strip()
            if not stream_id:
                continue
            
            if self.vod_cache:
                if any(stream_id == s.get("stream_id") for s in self.vod_cache.get(tmdb_id, [])):
                    skipped += 1
                    # log.debug("Skipping existing VOD stream_id=%s tmdb_id=%s", stream_id, tmdb_id)
                    continue
            
            category_ids = list(it.get("category_ids") or it.get("categories") or it.get("category_id") or it.get("category") or [])
            category_name = categories.get(str(category_ids[0]))["category_name"] if category_ids else ""
            
            
            if CATEGORY_EXCLUDE_FILTER:
                skip = False
                cat_lower = category_name.lower()
                for sub in CATEGORY_EXCLUDE_FILTER:
                    if sub in cat_lower:
                        skip = True
                        break
                if skip:
                    # log.debug("Excluding VOD by category filter: stream_id=%s tmdb_id=%s category=%s", stream_id, tmdb_id, category_name)
                    continue
                
            if TITLE_EXCLUDE_FILTER:
                skip = False
                title_lower = title.lower()
                for sub in TITLE_EXCLUDE_FILTER:
                    if sub in title_lower:
                        skip = True
                        break
                if skip:
                    # log.debug("Excluding VOD by title filter: stream_id=%s tmdb_id=%s title=%s", stream_id, tmdb_id, title)
                    continue
            
            if CATEGORY_CLEANING:
                for sub, repl in CATEGORY_CLEANING:
                    category_name = re.sub(re.escape(sub), repl, category_name, flags=re.IGNORECASE).strip()

            q = detect_quality(title)
            ext = str(it.get("container_extension") or it.get("ext") or "mp4").strip().lstrip(".")
            source_url = vod_url(stream_id, ext)
            
            quality = "4K" if q == "4K" else "HD"
            
            
            
            
            # use title cleaning from env
            if TITLE_CLEANING:
                for sub, repl in TITLE_CLEANING:
                    title = re.sub(re.escape(sub), repl, title, flags=re.IGNORECASE).strip()
            
            try: 
                description = STREAM_DESC_TEMPLATE_MOVIE.format(
                    title=title,
                    category=category_name or "Unknown",
                    quality=quality
                )
            except Exception:
                description = f"{title} \nCategory: {category_name} \nXtream2Stremio" if category_name else title

            try:
                name = STREAM_NAME_TEMPLATE_MOVIE.format(
                    title=title,
                    category=category_name or "Unknown",
                    quality=quality
                )
            except Exception:
                name = f"VOD {quality}"

            self.vod_cache.setdefault(tmdb_id, []).append({"title": title, "name": name, "url": source_url, "description": description, "stream_id": stream_id})

        if no_tmdb_count > 0:
            log.warning("Skipped %d Movie items with no TMDB ID", no_tmdb_count)
            if (no_tmdb_count/len(vod_items))>0.9:
                log.warning("WARNING: MORE THAN %.2f%% PCT OF STREAMS HAVE NO TMDB ISSUE",(no_tmdb_count/len(vod_items))*100)
            
        if skipped > 0:
            log.info("Skipped %d existing VOD streams during reindex", skipped)




    def _build_series(self, series_items: List[Dict[str, Any]], categories: Dict[str, Any]) -> None:
        
        

        if (os.path.isfile(f"{DATA_PATH}/series_cache.json")) and not self.series_cache:
            with open(f"{DATA_PATH}/series_cache.json", "r", encoding="utf-8") as f:
                self.series_cache = json.load(f)
            log.info("Loaded existing Series cache from file with %d titles", len(self.series_cache))

        no_tmdb_count = 0
        skipped = 0
        for it in series_items:
            title = str(it.get("name") or it.get("title") or "").strip()
            if not title:
                continue
            tmdb_id = str(it.get("tmdb") or it.get("tmdb_id") or it.get("moviedb") or it.get("moviedb_id") or "")
            if not tmdb_id:
                no_tmdb_count += 1
                continue
            series_id = str(it.get("series_id") or it.get("id") or "").strip()
            if not series_id:
                continue

            if self.series_cache:
                if any(series_id == s.get("series_id") for s in self.series_cache.get(tmdb_id, [])):
                    skipped += 1
                    # log.debug("Skipping existing Series series_id=%s tmdb_id=%s", series_id, tmdb_id)
                    continue

            category_ids = list(it.get("category_ids") or it.get("categories") or it.get("category_id") or it.get("category") or [])
            if categories and category_ids:
                category_info = categories.get(str(category_ids[0]))
                category_name = category_info["category_name"] if category_info else ""
            else:
                category_name = ""
                
            q = detect_quality(title)
            
            if CATEGORY_EXCLUDE_FILTER:
                skip = False
                cat_lower = category_name.lower()
                for sub in CATEGORY_EXCLUDE_FILTER:
                    if sub in cat_lower:
                        skip = True
                        break
                if skip:
                    # log.debug("Excluding Series by category filter: series_id=%s tmdb_id=%s category=%s", series_id, tmdb_id, category_name)
                    continue
            
            if TITLE_EXCLUDE_FILTER:
                skip = False
                title_lower = title.lower()
                for sub in TITLE_EXCLUDE_FILTER:
                    if sub in title_lower:
                        skip = True
                        break
                if skip:
                    # log.debug("Excluding Series by title filter: series_id=%s tmdb_id=%s title=%s", series_id, tmdb_id, title)
                    continue
                
            
            if CATEGORY_CLEANING:
                for sub, repl in CATEGORY_CLEANING:
                    category_name = re.sub(re.escape(sub), repl, category_name, flags=re.IGNORECASE).strip()
            
                
            # use title cleaning from env
            if TITLE_CLEANING:
                for sub, repl in TITLE_CLEANING:
                    title = re.sub(re.escape(sub), repl, title, flags=re.IGNORECASE).strip()

            quality = "4K" if q == "4K" else "HD"

            self.series_cache.setdefault(tmdb_id, []).append({"series_id": series_id, "title": title, "category_name": category_name, "quality": quality})

        if no_tmdb_count > 0:
            log.warning("Skipped %d Series items with no TMDB ID", no_tmdb_count)
            if (no_tmdb_count/len(series_items))>0.5:
                log.warning("WARNING: MORE THAN %.2f%% PCT OF STREAMS HAVE NO TMDB ID. THESE STREAMS WON'T BE FOUND. PLEASE TRY A DIFFERENT XTREAM ACCOUNT.",(no_tmdb_count/len(series_items))*100)

        if skipped > 0:
            log.info("Skipped %d existing Series streams during reindex", skipped)

    ### MATCHING FUNCTIONS ###
    
    def tmdb_id_match_vod(self, tmdb_id: str) -> List[Dict[str, Any]]:
        if not hasattr(self, 'vod_cache'):
            return []
        items = self.vod_cache.get(tmdb_id)
        items = [{k: it.get(k) for k in ("title","name","url","description")} for it in items] if items else []
        if items:
            return items
        return []

    async def tmdb_id_match_series(self, http: HttpClient, tmdb_id: str, season: str, episode: str) -> List[Dict[str, Any]]:
        if not hasattr(self, 'series_cache'):
            return []
        series_items = self.series_cache.get(tmdb_id)
        if not series_items:
            return []
        items = []
        url = ""


        if self.eps is None and os.path.isfile(f"{DATA_PATH}/series_eps_cache.json"):
            with open(f"{DATA_PATH}/series_eps_cache.json", "r", encoding="utf-8") as f:
                self.eps = json.load(f)

        info_sem = asyncio.Semaphore(5)  # limit concurrent series_info fetches
        for it in series_items:
            # check if episodes are already cached
            # file exists
            series_id = str(it.get('series_id'))
            if series_id:
                if self.eps:
                    urls = self.eps.get(series_id)
                    if urls:
                        for u in urls:
                            if u.get(f"s{season}_e{episode}"):
                                url = u.get(f"s{season}_e{episode}")
                                break
                # fetch and cache season & episode from info cache if not found in self.eps
                if self.eps is None or not url:
                    if os.path.isfile(f"{DATA_PATH}/series_info_cache/{series_id}.json"):
                        with open(f"{DATA_PATH}/series_info_cache/{series_id}.json", "r", encoding="utf-8") as f:
                            data = json.load(f)
                    else:
                        _,data = await self._fetch_series_info(http, series_id, info_sem)
                        
                        if WRITE_SERIES_INFO:
                            # cache all info to file
                            os.makedirs(f"{DATA_PATH}/series_info_cache", exist_ok=True)
                            with open(f"{DATA_PATH}/series_info_cache/{series_id}.json", "w", encoding="utf-8") as f:
                                json.dump(data, f, indent=2, ensure_ascii=True)
                    
                    # parse episodes
                    if data.get('episodes'):
                        episodes_data = data['episodes']
                        # Normalize to a dict-like structure or iterate values if it's a dict
                        # If it's a list, we just iterate the items directly
                        
                        iter_items = []
                        if isinstance(episodes_data, dict):
                             iter_items = episodes_data.values()
                        elif isinstance(episodes_data, list):
                             iter_items = episodes_data
                        
                        for ep_list in iter_items:
                            # Sometimes the value itself is not a list but a single dict or empty
                            if not isinstance(ep_list, list): 
                                continue
                                
                            for ep in ep_list:
                                ep_season = str(ep.get('season') or ep.get('season_num'))
                                ep_episode = str(ep.get('episode') or ep.get('episode_num'))
                                ep_id = str(ep.get('episode_id') or ep.get('id'))
                                ext = str(ep.get('container_extension') or ep.get('ext') or 'mp4').strip().lstrip('.')
                                if ep_season and ep_episode and ep_id:
                                    ep_url = series_url(ep_id, ext)
                                    if ep_url:
                                        self.eps.setdefault(series_id, []).append({f"s{ep_season}_e{ep_episode}": ep_url})

                    # cache to file
                    with open(f"{DATA_PATH}/series_eps_cache.json", "w", encoding="utf-8") as f:
                        json.dump(self.eps, f, indent=2, ensure_ascii=True)
                    for u in self.eps.get(series_id, []):
                        if u.get(f"s{season}_e{episode}"):
                            url = u.get(f"s{season}_e{episode}")
            if url:
                
                title = it.get("title")
                category_name = it.get("category_name")
                quality = it.get("quality")
                season_format = season.zfill(2)
                episode_format = episode.zfill(2)
                try:
                    description = STREAM_DESC_TEMPLATE_SERIES.format(
                        title=title,
                        category=category_name or "Unknown",
                        quality=quality,
                        season=season_format,
                        episode=episode_format
                    )
                except Exception:
                    description = f"S{season_format}E{episode_format} \n{title} \nCategory: {category_name} \nXtream2Stremio" if category_name else title

                try:
                    name = STREAM_NAME_TEMPLATE_SERIES.format(
                        title=title,
                        category=category_name or "Unknown",
                        quality=quality,
                        season=season_format,
                        episode=episode_format
                    )
                except Exception:
                    name = f"VOD {quality}"
                items.append({"title": name, "name": name, "url": url, "description": description})
        if items:
            return items
        return []
    
    
    
    
    ### SERIES INFO CACHING LOOP ###
    
    async def cache_series_info_loop(self, http: HttpClient) -> None:
        if not self.series_cache:
            raise RuntimeError("Series cache not built yet")


        if not self.eps and os.path.isfile(f"{DATA_PATH}/series_eps_cache.json"):
            with open(f"{DATA_PATH}/series_eps_cache.json", "r", encoding="utf-8") as f:
                self.eps = json.load(f)
        
        num_series_items = len([it for items in self.series_cache.values() for it in items])
        progress_update = max(1, num_series_items // 100)
        progress_count = 0
        log.info("Starting series_info caching loop for %d series items...", num_series_items)

        if not num_series_items:
            raise RuntimeError("No series in cache to process for series_info")
        
        info_sem = asyncio.Semaphore(1)  # limit concurrent series_info fetches
        for items in self.series_cache.values():
            for it in items:
                series_id = str(it.get("series_id") or it.get("id") or "")
                if series_id:
                    # note: can't just check self.eps because there may be new episodes to cache in a refresh
                    if os.path.isfile(f"{DATA_PATH}/series_info_cache/{series_id}.json"):
                        with open(f"{DATA_PATH}/series_info_cache/{series_id}.json", "r", encoding="utf-8") as f:
                            data = json.load(f)
                    else:
                        _,data = await self._fetch_series_info(http, series_id, info_sem, sleep=2.0)
                        
                        if WRITE_SERIES_INFO:
                            # cache all info to file
                            os.makedirs(f"{DATA_PATH}/series_info_cache", exist_ok=True)
                            with open(f"{DATA_PATH}/series_info_cache/{series_id}.json", "w", encoding="utf-8") as f:
                                json.dump(data, f, indent=2, ensure_ascii=True)
                    
                    # parse episodes
                    if data.get('episodes'):
                        episodes_data = data['episodes']
                        # Normalize to a dict-like structure or iterate values if it's a dict
                        # If it's a list, we just iterate the items directly
                        
                        iter_items = []
                        if isinstance(episodes_data, dict):
                             iter_items = episodes_data.values()
                        elif isinstance(episodes_data, list):
                             iter_items = episodes_data
                        
                        for ep_list in iter_items:
                            # Sometimes the value itself is not a list but a single dict or empty
                            if not isinstance(ep_list, list): 
                                continue
                                
                            for ep in ep_list:
                                ep_season = str(ep.get('season') or ep.get('season_num'))
                                ep_episode = str(ep.get('episode') or ep.get('episode_num'))
                                # here we can check if already cached, continue if so to find any new episodes
                                if self.eps.get(series_id):
                                    if any(u.get(f"s{ep_season}_e{ep_episode}") for u in self.eps.get(series_id)):
                                        continue
                                ep_id = str(ep.get('episode_id') or ep.get('id'))
                                ext = str(ep.get('container_extension') or ep.get('ext') or 'mp4').strip().lstrip('.')
                                if ep_season and ep_episode and ep_id:
                                    ep_url = series_url(ep_id, ext)
                                    if ep_url:
                                        self.eps.setdefault(series_id, []).append({f"s{ep_season}_e{ep_episode}": ep_url})

                    # write cache to file
                    with open(f"{DATA_PATH}/series_eps_cache.json", "w", encoding="utf-8") as f:
                        json.dump(self.eps, f, indent=2, ensure_ascii=True)
                progress_count += 1
                if progress_count % progress_update == 0 or progress_count == num_series_items:
                    log.info("Series info caching progress: %d/%d (%.1f%%)", progress_count, num_series_items, (progress_count / num_series_items) * 100.0)
        log.info("Series info caching loop completed.")
        return



# ----------------------------
# Cinemeta helpers
# ----------------------------
async def cinemeta_meta(http: HttpClient, type_: str, id_: str) -> Dict[str, Any]:
    url = f"{CINEMETA_BASE}/meta/{type_}/{id_}.json"
    data = await http.get_json(url, label=f"cinemeta.meta.{type_}")
    if isinstance(data, dict) and isinstance(data.get("meta"), dict):
        return data["meta"]
    return {}


def parse_stremio_id(type_: str, id_: str) -> Tuple[str, Optional[int], Optional[int]]:
    # movie: id is meta id (imdb)
    if type_ != "series":
        return id_, None, None

    # series episode: imdb:season:episode
    parts = id_.split(":")
    if len(parts) >= 3 and parts[1].isdigit() and parts[2].isdigit():
        return parts[0], int(parts[1]), int(parts[2])
    return id_, None, None


# ----------------------------
# Stream URL builders
# ----------------------------
def vod_url(stream_id: str, ext: str) -> str:
    ext = (ext or "mp4").strip().lstrip(".") or "mp4"
    return f"{XTREAM_STREAM_BASE}/{VOD_PATH_PREFIX}/{XTREAM_USERNAME}/{XTREAM_PASSWORD}/{stream_id}.{ext}"


def series_url(episode_id: str, ext: str) -> str:
    ext = (ext or "mp4").strip().lstrip(".") or "mp4"
    return f"{XTREAM_STREAM_BASE}/{SERIES_PATH_PREFIX}/{XTREAM_USERNAME}/{XTREAM_PASSWORD}/{episode_id}.{ext}"


# ----------------------------
# FastAPI
# ----------------------------
manifest = {
    'id': 'org.xtream2stremio',
    'version': '1.0.0',

    'name': ADDON_NAME,
    'description': ADDON_DESCRIPTION,

    'types': ['movie', 'series'],

    'catalogs': [],

    'resources': [
        {'name': 'stream', 'types': [
            'movie', 'series'], 'idPrefixes': ['tt']}
    ]
}



def respond_with(data):
    resp = JSONResponse(data)
    resp.headers['Access-Control-Allow-Origin'] = '*'
    resp.headers['Access-Control-Allow-Headers'] = '*'
    return resp


@asynccontextmanager
async def lifespan(app: FastAPI):
    global http, BASE_PATH, RESET_CACHE, DATA_PATH, idx
    http = HttpClient()


    if BASE_PATH=="/8501c45d8045":
        log.warning("BASE_PATH is set to default value '8501c45d8045'. This will generate a random manifest URL on each start. Please set BASE_PATH to a fixed value to have a stable manifest URL.")
        import uuid
        BASE_PATH = "/" + str(uuid.uuid4().hex[:12])
        log.warning("Generated new BASE_PATH: %s", BASE_PATH)

    # record/log environemnt variables on startup
    os.makedirs(f"{DATA_PATH}", exist_ok=True)
    os.makedirs(f"{DATA_PATH}/env_cache", exist_ok=True)

    # if Xtream account has changed, force reindex
    if os.listdir(f"{DATA_PATH}/env_cache"):
        latest = sorted(os.listdir(f'{DATA_PATH}/env_cache'), reverse=True)[0]
        if os.path.isfile(f"{DATA_PATH}/env_cache/{latest}"):
            with open(f"{DATA_PATH}/env_cache/{latest}", "r", encoding="utf-8") as f:
                last_env = json.load(f)
                if last_env.get("XTREAM_API_BASE") != XTREAM_API_BASE or last_env.get("XTREAM_USERNAME") != XTREAM_USERNAME:
                    log.debug("Last env cache: XTREAM_API_BASE=%s, XTREAM_USERNAME=%s", last_env.get("XTREAM_API_BASE"), last_env.get("XTREAM_USERNAME"))
                    log.debug("Current Xtream env: XTREAM_API_BASE=%s, XTREAM_USERNAME=%s", XTREAM_API_BASE, XTREAM_USERNAME)
                    log.info("Xtream account appears to have changed since last run, forcing RESET_CACHE...")
                    global RESET_CACHE
                    RESET_CACHE = True
                else:
                    log.info("Xtream account appears unchanged since last run.")
        else:
            log.info("Latest env cache file not found, assuming first run.")
    else:
        log.info("No previous env cache files found in env_cache folder, assuming first run.")
    
    if RESET_CACHE:
        if os.listdir(f"{DATA_PATH}"):
            try:
                import shutil
                shutil.rmtree(f"{DATA_PATH}", )
                os.makedirs(f"{DATA_PATH}", exist_ok=True)
                os.makedirs(f"{DATA_PATH}/env_cache", exist_ok=True)
            except Exception as e:
                raise ValueError(f"Problem resetting cache: \n{e}")
    
    env_dict = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "XTREAM_BASE": XTREAM_BASE,
        "XTREAM_API_BASE": XTREAM_API_BASE,
        "XTREAM_STREAM_BASE": XTREAM_STREAM_BASE,
        "XTREAM_USERNAME": XTREAM_USERNAME,
        "CINEMETA_BASE": CINEMETA_BASE,
        "BASE_PATH": BASE_PATH,
        "RESET_CACHE": str(RESET_CACHE),
        "ADDON_NAME": ADDON_NAME,
        "ADDON_DESCRIPTION": ADDON_DESCRIPTION,
        "REFRESH_MINUTES": str(REFRESH_MINUTES),
        "WAIT_FOR_INDEX_ON_START": str(WAIT_FOR_INDEX_ON_START),
        "MAX_STREAMS_PER_MATCH": str(MAX_STREAMS_PER_MATCH),
        "HTTP_TIMEOUT_S": str(HTTP_TIMEOUT_S),
    }
    
    # remove all but last 5 env cache files
    os.makedirs(f"{DATA_PATH}", exist_ok=True)
    os.makedirs(f"{DATA_PATH}/env_cache", exist_ok=True)
    for f in sorted(os.listdir(f"{DATA_PATH}/env_cache")):
        if f not in sorted(os.listdir(f"{DATA_PATH}/env_cache"), reverse=True)[:5]:
            try:
                os.remove(os.path.join(f"{DATA_PATH}/env_cache", f))
            except Exception:
                pass

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    with open(f"{DATA_PATH}/env_cache/env_{timestamp}.json", "w", encoding="utf-8") as f:
        json.dump(env_dict, f, indent=2)

    async def background_worker() -> None:
        # Initial cycle
        try:
            await idx.refresh(http)
        except Exception as e:
            log.exception("Initial refresh failed: %s", e)
        
        # Start caching series info immediately after refresh
        # This await is non-blocking to the API, but sequential for this worker
        try:
            log.info("Starting background series info caching...")
            await idx.cache_series_info_loop(http)
        except Exception as e:
            log.exception("Initial series cache loop failed: %s", e)

        # Periodic cycle
        while True:
            await asyncio.sleep(REFRESH_MINUTES * 60)
            try:
                await idx.refresh(http)
                # Run cache loop again after periodic refresh
                await idx.cache_series_info_loop(http)
            except Exception as e:
                log.exception("Periodic refresh/cache failed: %s", e)

    # Start the worker task
    task = asyncio.create_task(background_worker())
    
    try:
        yield
    finally:
        # Cleanup on shutdown
        if not task.done():
            task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        if http:
            await http.close()
    
# Initialize apps with lifespan
# If BASE_PATH exists, root_app is the entry point.
# If BASE_PATH is empty, api becomes root_app and is the entry point.
# We assign lifespan to whichever will be the main application.
root_app = FastAPI(title=manifest["name"], lifespan=lifespan if BASE_PATH else None)
api = FastAPI(title=manifest["name"] + " (api)", lifespan=lifespan if not BASE_PATH else None)

http: Optional[HttpClient] = None
idx = XtreamIndex()


@api.get("/manifest.json")
async def get_manifest() -> JSONResponse:
    return respond_with(manifest)


@api.get("/stream/{type_}/{id_}.json")
async def get_stream(type_: str, id_: str) -> JSONResponse:
    if type_ not in ("movie", "series"):
        return respond_with({"streams": []})

    if WAIT_FOR_INDEX_ON_START and not idx.ready:
        for _ in range(40):
            if idx.ready:
                break
            await asyncio.sleep(0.25)

    if not idx.ready or http is None:
        return respond_with({"streams": []})

    imdb_id, season, episode = parse_stremio_id(type_, id_)
    meta = await cinemeta_meta(http, type_, imdb_id)
    tmdb_id = str(meta.get("tmdb") or meta.get("tmdb_id") or meta.get("moviedb") or meta.get("moviedb_id") or "")

    if not tmdb_id:
        return respond_with({"streams": []})
    
    
    if type_ == "movie":
        items = idx.tmdb_id_match_vod(tmdb_id)
        return respond_with({"streams": items})

    if type_ == "series":
        items = await idx.tmdb_id_match_series(http, tmdb_id, str(season), str(episode))
        return respond_with({"streams": items})


# Mount API under BASE_PATH (recommended) or at root
if BASE_PATH:
    root_app.mount(BASE_PATH, api)
else:
    root_app = api


def main() -> None:
    uvicorn.run(root_app, host=HOST, port=PORT, log_level=LOG_LEVEL.lower())
    
    


if __name__ == "__main__":
    main()
