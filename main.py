#!/usr/bin/env python3
"""
DHMZ radar + pljusak station tracker with satellite basemap, storm frame archiver,
walk-forward speed/heading scoring, adaptive velocity windowing, a continuous
parallel Shadow Challenger + background auto-tuner on the main dashboard (/),
and a dedicated Offline Replay & Shadow Auto-Tuner GUI (/replay).

Endpoints:
  /                    Live HTML dashboard (with Champion vs Shadow Challenger card)
  /replay              Dedicated Offline Replay & Shadow Auto-Tuner GUI
  /threat.png          Current frame: satellite/radar composite + overlays (with optional shadow vectors)
  /threat.gif          Animated composite
  /status              JSON: summary + local_weather + sensors + threats + archive + shadow
  /settings            GET/POST hot-swappable runtime settings
  /config              GET full JSON config (POST replace, DELETE reset)
  /archive             GET archived storm episodes & frame counts
  /replay/run          POST run shadow vs baseline evaluation across episodes
  /replay/frame        GET rendered PNG for an archived episode frame + shadow overlay
  /replay/autotune     POST run 2-stage shadow auto-tuner and save new profile iteration
  /shadow/select       POST set active shadow challenger profile
  /shadow/tune_now     POST trigger background 2-stage auto-tune immediately
  /profiles            GET/POST/DELETE saved parameter profiles (baseline + iterations)
  /profiles/promote    POST promote a saved profile to active runtime config
  /refresh             POST force refresh
  /accuracy            GET stats, /accuracy/predictions, /accuracy/events
  /accuracy/reset      POST clear scores
  /accuracy/replay     POST re-score
"""

import argparse
import io
import json
import math
import os
import random
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

import numpy as np
import requests
import yaml
from PIL import Image, ImageDraw, ImageEnhance, ImageSequence
from scipy import ndimage
from scipy.spatial import cKDTree
import hashlib

try:
    import pytesseract
    HAVE_TESSERACT = True
except ImportError:
    HAVE_TESSERACT = False
try:
    import paho.mqtt.client as mqtt
    HAVE_MQTT = True
except ImportError:
    HAVE_MQTT = False
try:
    from zoneinfo import ZoneInfo
    HAVE_ZONEINFO = True
except ImportError:
    HAVE_ZONEINFO = False

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
GIF_URL = "https://vrijeme.hr/anim_kompozit.gif"
RADAR_HEADERS = {"User-Agent": "Mozilla/5.0 (dhmz-radar-tracker/1.0)"}
PLJUSAK_URL = "https://pljusak.com/"
PLJUSAK_HEADERS = {
    "User-Agent": "Mozilla/5.0 (pljusak-station-puller/1.0)",
    "Referer": "https://pljusak.com/",
    "Accept": "text/html,application/xhtml+xml",
}

MIN_SAT = 0.45
MIN_VAL = 0.25
MIN_CLUSTER_PIXELS = 15
MAX_STORM_SPEED_KMH = 140.0
VELOCITY_WINDOW_MIN = 25.0
MIN_SPEED_KMH = 2.0

STALE_MIN = 20
VERY_STALE_MIN = 40

CLUSTER_DIGITS = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"

DEFAULT_SRI_THRESHOLDS = (2.5, 10.0, 30.0)

DHMZ_SRI_SCALE = [
    (0.2,   0.5,   (20,  50,  210), "0.2"),
    (0.5,   1.0,   (0,   110, 255), "0.5"),
    (1.0,   2.0,   (0,   185, 255), "1"),
    (2.0,   3.0,   (0,   230, 200), "2"),
    (3.0,   5.0,   (0,   225, 110), "3"),
    (5.0,   7.0,   (0,   195, 0),   "5"),
    (7.0,   10.0,  (140, 230, 0),   "7"),
    (10.0,  15.0,  (255, 255, 0),   "10"),
    (15.0,  20.0,  (255, 210, 0),   "15"),
    (20.0,  30.0,  (255, 155, 0),   "20"),
    (30.0,  40.0,  (255, 95,  0),   "30"),
    (40.0,  50.0,  (240, 40,  0),   "40"),
    (50.0,  70.0,  (215, 0,   0),   "50"),
    (70.0,  100.0, (170, 0,   35),  "70"),
    (100.0, 150.0, (200, 0,   180), "100"),
    (150.0, 999.0, (240, 130, 255), "150+"),
]

_DHMZ_SRI_TREE = cKDTree(np.array([c[2] for c in DHMZ_SRI_SCALE], dtype=np.float64))
_DHMZ_SRI_MIDPOINTS = np.array(
    [math.sqrt(c[0] * min(c[1], 200.0)) for c in DHMZ_SRI_SCALE], dtype=np.float32
)

COMPASS_16_DEG = {
    "N": 0.0, "NNE": 22.5, "NE": 45.0, "ENE": 67.5,
    "E": 90.0, "ESE": 112.5, "SE": 135.0, "SSE": 157.5,
    "S": 180.0, "SSW": 202.5, "SW": 225.0, "WSW": 247.5,
    "W": 270.0, "WNW": 292.5, "NW": 315.0, "NNW": 337.5,
}

MATCH_TOLERANCE_FACTOR = 1.5
MATCH_TOLERANCE_MIN = 60.0

DEFAULT_DISPLAY_TZ = "Europe/Zagreb"
DEFAULT_TIMESTAMP_REGION_STR = "0,0,260,22"

CITIES = [
    ("zg", "Zagreb",         45.8146, 15.9722),
    ("ri", "Rijeka",         45.3271, 14.4422),
    ("st", "Split",          43.5081, 16.4402),
    ("os", "Osijek",         45.5550, 18.6955),
    ("zd", "Zadar",          44.1194, 15.2314),
    ("db", "Dubrovnik",      42.6507, 18.0944),
    ("pu", "Pula",           44.8666, 13.8496),
    ("ka", "Karlovac",       45.4929, 15.5551),
    ("va", "Varaždin",       46.3057, 16.3366),
    ("lj", "Ljubljana",      46.0569, 14.5058),
    ("tb", "Sarajevo",       43.8633, 18.4131),
    ("sk", "Sisak",          45.4667, 16.3789),
    ("mk", "Makarska",       43.2969, 17.0178),
    ("šb", "Šibenik",        43.7350, 15.8956),
    ("vk", "Vukovar",        45.2551, 19.1122),
    ("sb", "Slavonski Brod", 45.1603, 18.0156),
    ("po", "Požega",         45.3403, 17.6853),
    ("bj", "Bjelovar",       45.8986, 16.8422),
    ("kt", "Kutina",         45.4772, 16.7783),
    ("bk", "Bjelolasica",    45.2540, 15.0160),
    ("og", "Ogulin",         45.2631, 15.2281),
    ("gs", "Gospić",         44.5461, 15.3747),
    ("kn", "Knin",           44.0409, 16.1969),
]

ADRIATIC_COAST = [
    (45.52, 13.57), (45.47, 13.58), (45.40, 13.55), (45.30, 13.58),
    (45.20, 13.60), (45.10, 13.62), (44.97, 13.65), (44.87, 13.83),
    (44.80, 13.92), (44.70, 14.00), (44.60, 14.10), (44.75, 14.20),
    (44.75, 14.30), (44.75, 14.40), (44.63, 14.50), (44.55, 14.45),
    (44.60, 14.60), (44.70, 14.70), (44.80, 14.75), (45.00, 14.75),
    (45.10, 14.70), (45.15, 14.65), (45.30, 14.45), (45.30, 14.35),
    (45.20, 14.30), (45.10, 14.20), (45.05, 14.15), (44.95, 14.15),
    (44.88, 14.20), (44.85, 14.30), (44.75, 14.55), (44.65, 14.75),
    (44.55, 14.80), (44.78, 14.78), (44.75, 14.85), (44.60, 14.95),
    (44.55, 14.90), (44.50, 14.85), (44.45, 15.05), (44.30, 15.10),
    (44.20, 15.10), (44.15, 15.20), (44.10, 15.23), (44.00, 15.20),
    (43.95, 15.30), (43.85, 15.50), (43.75, 15.65), (43.73, 15.89),
    (43.65, 15.95), (43.50, 16.10), (43.50, 16.44), (43.45, 16.55),
    (43.40, 16.65), (43.30, 16.90), (43.20, 17.10), (43.05, 17.45),
    (42.90, 17.55), (42.85, 17.70), (42.75, 18.00), (42.65, 18.10),
    (42.55, 18.25), (42.48, 18.35), (42.40, 18.52),
]

DEFAULT_CALIBRATION_JSON = (
    '{"map":{"x0":0,"y0":0,"x1":720,"y1":720},'
    '"points":['
    '{"pixel":[343,226],"city":"Zagreb"},'
    '{"pixel":[385,490],"city":"Split"},'
    '{"pixel":[566,258],"city":"Osijek"}'
    ']}'
)

# ----------------------------------------------------------------------------
# Runtime (hot-swappable)
# ----------------------------------------------------------------------------
_RUNTIME = {
    "approaching_radius_km": 20.0,
    "cluster_radius_weight": 1.0,
    "only_approach_deg": 180.0,
    "threat_radius_km": 15.0,
    "threat_angle_deg": 45.0,
    "threat_min_duration": 3,
    "threat_min_sri_mmh": 0.0,
    "min_duration_frames": 2,
    "min_size_px": 30,
    "velocity_window_min": 25.0,
    "velocity_adaptive": True,
    "radius_km": 150.0,
    "weather_radius_km": 5.0,
    "archive_enabled": True,
    "archive_radius_km": 200.0,
    "archive_min_pixels": 40,
    "shadow_enabled": True,
    "shadow_profile": "",
    "shadow_overlay_live": True,
    "shadow_autotune_enabled": True,
    "shadow_autotune_hours": 6.0,
    "poll_s": 300,
    "view_rotation_auto": True,
    "view_rotation": 0.0,
    "basemap": "auto",
    "basemap_saturation": 1.0,
    "basemap_brightness": 1.0,
    "basemap_contrast": 1.0,
    "show_cities": True,
    "web_scale": 1.0,
    "settings_tag": "baseline",
}
_RUNTIME_TYPES = {
    "approaching_radius_km": (float, 1.0, 500.0),
    "cluster_radius_weight": (float, 0.0, 5.0),
    "only_approach_deg": (float, 0.0, 180.0),
    "threat_radius_km": (float, 1.0, 500.0),
    "threat_angle_deg": (float, 0.0, 180.0),
    "threat_min_duration": (int, 1, 50),
    "threat_min_sri_mmh": (float, 0.0, 500.0),
    "min_duration_frames": (int, 1, 50),
    "min_size_px": (int, 5, 5000),
    "velocity_window_min": (float, 10.0, 90.0),
    "velocity_adaptive": (bool, None, None),
    "radius_km": (float, 20.0, 1000.0),
    "weather_radius_km": (float, 0.5, 100.0),
    "archive_enabled": (bool, None, None),
    "archive_radius_km": (float, 20.0, 500.0),
    "archive_min_pixels": (int, 10, 5000),
    "shadow_enabled": (bool, None, None),
    "shadow_profile": (str, None, None),
    "shadow_overlay_live": (bool, None, None),
    "shadow_autotune_enabled": (bool, None, None),
    "shadow_autotune_hours": (float, 0.5, 168.0),
    "poll_s": (int, 30, 7200),
    "view_rotation_auto": (bool, None, None),
    "view_rotation": (float, 0.0, 360.0),
    "basemap": (str, None, None),
    "basemap_saturation": (float, 0.0, 2.0),
    "basemap_brightness": (float, 0.2, 2.0),
    "basemap_contrast": (float, 0.2, 2.0),
    "show_cities": (bool, None, None),
    "web_scale": (float, 0.1, 4.0),
    "settings_tag": (str, None, None),
}
_TUNABLE_KEYS = (
    "velocity_window_min",
    "velocity_adaptive",
    "min_size_px",
    "min_duration_frames",
    "approaching_radius_km",
    "cluster_radius_weight",
    "only_approach_deg",
    "threat_radius_km",
    "threat_angle_deg",
    "threat_min_duration",
    "threat_min_sri_mmh",
)

_RUNTIME_LOCK = threading.Lock()
_FORCE_REFRESH = threading.Event()
_FORCE_TUNE = threading.Event()

# ----------------------------------------------------------------------------
# Static (restart-required)
# ----------------------------------------------------------------------------
_STATIC_DEFAULTS = {
    "location": None,
    "lat": None,
    "lon": None,
    "calibration_inline": DEFAULT_CALIBRATION_JSON,
    "calibration": None,
    "radar_intensity": "simple",
    "sri_legend_region": None,
    "sri_legend_orientation": "horizontal",
    "sri_legend_padding": 1,
    "sri_breaks": None,
    "sri_thresholds": "2.5,10,30",
    "frame_interval_min": 5.0,
    "auto_frame_interval": False,
    "timestamp_region": DEFAULT_TIMESTAMP_REGION_STR,
    "timestamp_enabled": True,
    "timestamp_tz": None,
    "timestamp_source": "last",
    "display_tz": DEFAULT_DISPLAY_TZ,
    "pljusak_enabled": True,
    "pljusak_url": PLJUSAK_URL,
    "pljusak_interval": 60,
    "station_confirm_radius": 15.0,
    "station_confirm_min_mm": 0.2,
    "station_confirm_max_age_min": 15.0,
    "station_confirm_fraction": 0.5,
    "station_confirm_min_count": 1,
    "web_enabled": True,
    "web_port": 8080,
    "mqtt_enabled": True,
    "mqtt_host": None,
    "mqtt_port": 1883,
    "mqtt_tls": False,
    "mqtt_discovery_prefix": "homeassistant",
    "mqtt_state_prefix": "dhmz_radar",
    "mqtt_device_id": None,
    "mqtt_device_name": "DHMZ Radar",
    "mqtt_sensors": "all",
    "mqtt_mark_offline": None,
    "mqtt_no_retain": False,
    "mqtt_threat_topic": None,
    "jsonl_dump": None,
    "accuracy_enabled": True,
    "mapbox_token": None,
    "mapbox_style": "mapbox/satellite-v9",
    "basemap_size": 640,
}
_STATIC = dict(_STATIC_DEFAULTS)
_STATIC_LOCK = threading.Lock()

_CLI_TO_RUNTIME = {
    "approaching_radius": "approaching_radius_km",
    "cluster_radius_weight": "cluster_radius_weight",
    "only_approach": "only_approach_deg",
    "threat_radius": "threat_radius_km",
    "threat_angle": "threat_angle_deg",
    "threat_min_duration": "threat_min_duration",
    "threat_min_sri": "threat_min_sri_mmh",
    "min_duration": "min_duration_frames",
    "min_size": "min_size_px",
    "velocity_window": "velocity_window_min",
    "radius": "radius_km",
    "weather_radius": "weather_radius_km",
    "archive_radius": "archive_radius_km",
    "archive_min_pixels": "archive_min_pixels",
    "shadow_profile": "shadow_profile",
    "shadow_autotune_hours": "shadow_autotune_hours",
    "poll": "poll_s",
    "view_rotation": "view_rotation",
    "basemap": "basemap",
    "basemap_saturation": "basemap_saturation",
    "basemap_brightness": "basemap_brightness",
    "basemap_contrast": "basemap_contrast",
    "web_scale": "web_scale",
    "settings_tag": "settings_tag",
}
_CLI_TO_STATIC = {
    "location": "location",
    "lat": "lat",
    "lon": "lon",
    "calibration": "calibration",
    "calibration_inline": "calibration_inline",
    "radar_intensity": "radar_intensity",
    "sri_legend_region": "sri_legend_region",
    "sri_legend_orientation": "sri_legend_orientation",
    "sri_legend_padding": "sri_legend_padding",
    "sri_breaks": "sri_breaks",
    "sri_thresholds": "sri_thresholds",
    "frame_interval_min": "frame_interval_min",
    "timestamp_region": "timestamp_region",
    "timestamp_tz": "timestamp_tz",
    "timestamp_source": "timestamp_source",
    "display_tz": "display_tz",
    "pljusak_url": "pljusak_url",
    "pljusak_interval": "pljusak_interval",
    "station_confirm_radius": "station_confirm_radius",
    "station_confirm_min_mm": "station_confirm_min_mm",
    "station_confirm_max_age_min": "station_confirm_max_age_min",
    "station_confirm_fraction": "station_confirm_fraction",
    "station_confirm_min_count": "station_confirm_min_count",
    "web_port": "web_port",
    "mqtt_host": "mqtt_host",
    "mqtt_port": "mqtt_port",
    "mqtt_tls": "mqtt_tls",
    "ha_discovery_prefix": "mqtt_discovery_prefix",
    "ha_state_prefix": "mqtt_state_prefix",
    "ha_device_id": "mqtt_device_id",
    "ha_device_name": "mqtt_device_name",
    "ha_sensors": "mqtt_sensors",
    "mqtt_no_retain": "mqtt_no_retain",
    "ha_threat_topic": "mqtt_threat_topic",
    "jsonl_dump": "jsonl_dump",
    "mapbox_token": "mapbox_token",
    "mapbox_style": "mapbox_style",
    "basemap_size": "basemap_size",
}
_CLI_TO_STATIC_BOOL = {
    "web_enabled": "web_enabled",
    "pljusak_enabled": "pljusak_enabled",
    "timestamp_enabled": "timestamp_enabled",
    "mqtt_enabled": "mqtt_enabled",
    "accuracy_enabled": "accuracy_enabled",
}
_CLI_TO_RUNTIME_BOOL = {
    "cities": "show_cities",
    "view_rotation_auto": "view_rotation_auto",
    "archive_enabled": "archive_enabled",
    "velocity_adaptive": "velocity_adaptive",
    "shadow_enabled": "shadow_enabled",
    "shadow_overlay_live": "shadow_overlay_live",
    "shadow_autotune_enabled": "shadow_autotune_enabled",
}

_VIZ_STATE = {
    "png": None, "gif": None, "updated": None, "status": {},
    "lock": threading.Lock(),
}


# ----------------------------------------------------------------------------
# Config helpers
# ----------------------------------------------------------------------------
def _load_json_file(path):
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"config: load error {e}", file=sys.stderr)
        return None


def _save_json_file(path, data):
    if not path:
        return False
    try:
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        try:
            os.chmod(tmp, 0o600)
        except OSError:
            pass
        if os.path.exists(path):
            try:
                os.replace(path, path + ".bak")
            except OSError:
                pass
        os.replace(tmp, path)
        return True
    except Exception as e:
        print(f"config: save error {e}", file=sys.stderr)
        return False


def _persist_runtime_to_disk():
    cfg_path = _VIZ_STATE.get("config_path")
    if not cfg_path:
        return
    current = _snapshot()
    existing = _load_json_file(cfg_path) or {}
    now = datetime.now(timezone.utc).isoformat()
    _save_json_file(cfg_path, {
        "version": 1,
        "created_at": existing.get("created_at", now),
        "updated_at": now,
        "runtime": current,
        "static": existing.get("static") or _static_snapshot(),
    })


def _coerce(key, value):
    if key == "basemap":
        v = str(value).lower().strip()
        return v if v in ("auto", "mapbox", "greyscale", "none") else "auto"
    t, lo, hi = _RUNTIME_TYPES[key]
    if t is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        return str(value).lower() in ("1", "true", "yes", "on", "checked")
    if t is str:
        return str(value)[:64]
    if t is int:
        v = int(round(float(value)))
    elif t is float:
        v = float(value)
    else:
        return value
    if lo is not None:
        v = max(lo, min(hi, v))
    return v


def _snapshot():
    with _RUNTIME_LOCK:
        return dict(_RUNTIME)


def _static_snapshot():
    with _STATIC_LOCK:
        return dict(_STATIC)


def _apply_static(d):
    with _STATIC_LOCK:
        _STATIC.update(d)


def _merge_config(cli_args, existing_cfg):
    existing_cfg = existing_cfg or {}
    runtime = dict(_RUNTIME)
    runtime.update(existing_cfg.get("runtime") or {})
    static = dict(_STATIC_DEFAULTS)
    static.update(existing_cfg.get("static") or {})

    for cli_key, cfg_key in _CLI_TO_RUNTIME.items():
        v = getattr(cli_args, cli_key, None)
        if v is not None:
            runtime[cfg_key] = _coerce(cfg_key, v)
    for cli_key, cfg_key in _CLI_TO_RUNTIME_BOOL.items():
        v = getattr(cli_args, cli_key, None)
        if v is not None:
            runtime[cfg_key] = bool(v)
    for cli_key, cfg_key in _CLI_TO_STATIC.items():
        v = getattr(cli_args, cli_key, None)
        if v is not None:
            static[cfg_key] = v
    for cli_key, cfg_key in _CLI_TO_STATIC_BOOL.items():
        v = getattr(cli_args, cli_key, None)
        if v is not None:
            static[cfg_key] = bool(v)

    if cli_args.location is not None:
        static["location"] = cli_args.location
        static["lat"] = None
        static["lon"] = None
    elif cli_args.lat is not None or cli_args.lon is not None:
        static["location"] = None
        if cli_args.lat is not None:
            static["lat"] = cli_args.lat
        if cli_args.lon is not None:
            static["lon"] = cli_args.lon

    if static.get("mqtt_host") is None:
        env_val = os.environ.get("MQTT_HOST")
        if env_val:
            static["mqtt_host"] = env_val
    if static.get("mqtt_port") in (None, 1883):
        env_val = os.environ.get("MQTT_PORT")
        if env_val:
            try:
                static["mqtt_port"] = int(env_val)
            except ValueError:
                pass
    if static.get("mapbox_token") is None:
        env_val = os.environ.get("MAPBOX_TOKEN")
        if env_val:
            static["mapbox_token"] = env_val

    return runtime, static


# ----------------------------------------------------------------------------
# Cities + timezone
# ----------------------------------------------------------------------------
def _city_lookup(query):
    import unicodedata
    import difflib

    def norm(s):
        s = unicodedata.normalize("NFKD", s)
        s = "".join(c for c in s if not unicodedata.combining(c))
        return s.lower().strip()

    q = norm(query)
    for label, name, lat, lon in CITIES:
        if q == norm(label) or q == norm(name):
            return name, lat, lon
    d = {norm(name): (name, lat, lon) for _, name, lat, lon in CITIES}
    d.update({norm(label): (name, lat, lon) for label, name, lat, lon in CITIES})
    m = difflib.get_close_matches(q, list(d.keys()), n=1, cutoff=0.6)
    return d[m[0]] if m else None


def resolve_city(query):
    return _city_lookup(query)


def city_label(name):
    for label, nm, _, _ in CITIES:
        if nm == name:
            return label
    return name[:2].lower()


def get_zoneinfo(name):
    if not HAVE_ZONEINFO:
        return timezone.utc, "UTC"
    cand = name or os.environ.get("TZ") or DEFAULT_DISPLAY_TZ
    try:
        return ZoneInfo(cand), cand
    except Exception:
        return timezone.utc, "UTC"


def tz_abbrev(dt):
    return dt.strftime("%Z") or dt.strftime("%z")


def cluster_label(idx):
    return CLUSTER_DIGITS[idx] if 0 <= idx < len(CLUSTER_DIGITS) else f"#{idx + 1}"


# ----------------------------------------------------------------------------
# Coast rotation (satellite edge detection + polyline fallback)
# ----------------------------------------------------------------------------
def detect_coast_rotation_from_satellite(basemap_north_up):
    if basemap_north_up is None:
        return None
    arr = np.asarray(basemap_north_up.convert("RGB"), dtype=np.float32)
    H, W = arr.shape[:2]
    if H < 32 or W < 32:
        return None

    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b

    water = (lum < 95.0) & (b >= r - 2.0) & (b >= g - 6.0)
    water |= (lum < 55.0) & (b >= r)

    struct = np.ones((5, 5), dtype=bool)
    water = ndimage.binary_closing(water, structure=struct)
    water = ndimage.binary_opening(water, structure=struct)
    water = ndimage.binary_fill_holes(water)

    margin = max(12, int(min(H, W) * 0.04))
    inner = water[margin:H - margin, margin:W - margin]
    water_frac = float(inner.mean())
    if water_frac < 0.04 or water_frac > 0.96:
        return None

    dil = ndimage.binary_dilation(water, structure=np.ones((7, 7), dtype=bool))
    ero = ndimage.binary_erosion(water, structure=np.ones((7, 7), dtype=bool))
    coast_band = dil & (~ero)
    coast_band[:margin, :] = False
    coast_band[H - margin:, :] = False
    coast_band[:, :margin] = False
    coast_band[:, W - margin:] = False

    if int(coast_band.sum()) < 25:
        return None

    smooth_water = ndimage.gaussian_filter(water.astype(np.float32), sigma=3.0)
    gx = ndimage.sobel(smooth_water, axis=1)
    gy = ndimage.sobel(smooth_water, axis=0)

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cy, cx = H / 2.0, W / 2.0
    sigma_pos = min(H, W) * 0.35
    spatial_w = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma_pos ** 2))

    w = spatial_w * coast_band.astype(np.float32)
    sx = float(np.sum(gx * w))
    sy = float(np.sum(gy * w))
    mag = math.hypot(sx, sy)
    total_grad = float(np.sum(np.hypot(gx, gy) * w))
    if total_grad <= 1e-6 or (mag / total_grad) < 0.15:
        return None

    seaward_bearing = math.degrees(math.atan2(sx, -sy)) % 360.0
    return seaward_bearing


def compute_coast_rotation(target_lat, target_lon, max_dist_km=100.0):
    best_dist = float("inf")
    best_bearing = None
    for (lat1, lon1), (lat2, lon2) in zip(ADRIATIC_COAST, ADRIATIC_COAST[1:]):
        coslat = math.cos(math.radians((lat1 + target_lat) / 2))
        sx = (lon2 - lon1) * 111.32 * coslat
        sy = (lat2 - lat1) * 111.32
        tx = (target_lon - lon1) * 111.32 * coslat
        ty = (target_lat - lat1) * 111.32
        seg_sq = sx * sx + sy * sy
        if seg_sq < 1e-9:
            continue
        t = (tx * sx + ty * sy) / seg_sq
        t = max(0.0, min(1.0, t))
        px, py = sx * t, sy * t
        dist = math.hypot(tx - px, ty - py)
        if dist < best_dist:
            best_dist = dist
            best_bearing = math.degrees(math.atan2(sx, sy)) % 360
    if best_dist > max_dist_km or best_bearing is None:
        return None, best_dist
    return (best_bearing + 90.0) % 360.0, best_dist


# ----------------------------------------------------------------------------
# Timestamp OCR
# ----------------------------------------------------------------------------
TIMESTAMP_PATTERNS = [
    (re.compile(r"(\d{4})-(\d{1,2})-(\d{1,2})\s+(\d{1,2}):(\d{2})(?::(\d{2}))?"),
     ("y", "mo", "d", "h", "mi", "s")),
    (re.compile(r"(\d{1,2})\.(\d{1,2})\.(\d{2,4})\.?\s+(\d{1,2}):(\d{2})(?::(\d{2}))?"),
     ("d", "mo", "y", "h", "mi", "s")),
    (re.compile(r"(\d{1,2})/(\d{1,2})/(\d{2,4})\s+(\d{1,2}):(\d{2})"),
     ("d", "mo", "y", "h", "mi", None)),
]


def parse_timestamp_text(text, tz):
    if not text:
        return None
    t = re.sub(r"[^\d.:/\- ]", " ", text.strip())
    for pat, keys in TIMESTAMP_PATTERNS:
        m = pat.search(t)
        if not m:
            continue
        parts = {}
        for k, v in zip(keys, m.groups()):
            if k is None:
                continue
            parts[k] = int(v) if v is not None else 0
        if "y" not in parts:
            continue
        y = parts["y"] + (2000 if parts["y"] < 100 else 0)
        try:
            return datetime(y, parts["mo"], parts["d"], parts["h"],
                            parts["mi"], parts.get("s", 0),
                            tzinfo=tz).astimezone(timezone.utc)
        except ValueError:
            continue
    return None


def _otsu(arr):
    hist, _ = np.histogram(arr, bins=256, range=(0, 256))
    total = arr.size
    sum_total = float(np.dot(np.arange(256), hist))
    sum_b, w_b = 0.0, 0
    best_t, best_var = 128, -1.0
    for i in range(256):
        w_b += hist[i]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += i * hist[i]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        var = w_b * w_f * (m_b - m_f) ** 2
        if var > best_var:
            best_var, best_t = var, i
    return best_t


def read_frame_timestamp(img, region, tz, source="last"):
    if not HAVE_TESSERACT or region is None:
        return None, None, 0
    try:
        nf = getattr(img, "n_frames", 1)
        idx = 0 if source == "first" else max(0, nf - 1)
        try:
            img.seek(idx)
        except (EOFError, OSError, ValueError):
            idx = 0
            img.seek(0)
        frame = img.convert("RGB")
        crop = frame.crop(region)
        w, h = crop.size
        if w < 10 or h < 5:
            return None, None, idx
        crop = crop.resize((w * 3, h * 3), Image.LANCZOS)
        gray = np.array(crop.convert("L"))
        t = _otsu(gray)
        binary = ((gray > t) * 255).astype(np.uint8)
        if binary.mean() < 128:
            binary = 255 - binary
        text = pytesseract.image_to_string(
            Image.fromarray(binary), config="--psm 7").strip()
        return parse_timestamp_text(text, tz), text, idx
    except Exception:
        return None, None, 0
    finally:
        try:
            img.seek(0)
        except Exception:
            pass


# ----------------------------------------------------------------------------
# Geometry + calibration
# ----------------------------------------------------------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def bearing_label(deg):
    dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    return dirs[int((deg + 11.25) % 360 // 22.5)]


def angular_diff_deg(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


@dataclass
class MapRegion:
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self): return self.x1 - self.x0
    @property
    def height(self): return self.y1 - self.y0

    def crop_mask(self, full_h, full_w):
        m = np.zeros((full_h, full_w), dtype=bool)
        y0, y1 = max(0, self.y0), min(full_h, self.y1)
        x0, x1 = max(0, self.x0), min(full_w, self.x1)
        m[y0:y1, x0:x1] = True
        return m


def validate_map_against_image(mr, size):
    w, h = size
    probs = []
    if mr.x0 < 0 or mr.y0 < 0:
        probs.append(f"origin ({mr.x0},{mr.y0}) negative")
    if mr.x1 > w: probs.append(f"x1={mr.x1} > width {w}")
    if mr.y1 > h: probs.append(f"y1={mr.y1} > height {h}")
    if mr.x1 <= mr.x0 or mr.y1 <= mr.y0: probs.append("empty region")
    if probs:
        raise ValueError(f"map region invalid for {w}x{h}: " + "; ".join(probs))


def parse_map_region(raw, image_size=None):
    if raw and "map" in raw:
        m = raw["map"]
        return MapRegion(int(m["x0"]), int(m["y0"]), int(m["x1"]), int(m["y1"]))
    if image_size:
        return MapRegion(0, 0, image_size[0], image_size[1])
    return None


VALID_EDGES = {"nw", "ne", "sw", "se"}


def _resolve_edge(name, mr):
    name = name.lower()
    if name not in VALID_EDGES:
        raise ValueError(f"Unknown edge '{name}'")
    return (0 if name in ("nw", "sw") else mr.width - 1,
            0 if name in ("nw", "ne") else mr.height - 1)


class Calibration:
    def __init__(self, points, map_region, anchor_cities=None):
        if len(points) < 3:
            raise ValueError(f"Need >=3 anchors, got {len(points)}")
        A, bl, bo = [], [], []
        for x, y, lat, lon in points:
            A.append([x, y, 1.0]); bl.append(lat); bo.append(lon)
        A = np.asarray(A, dtype=float)
        self.coef_lat, *_ = np.linalg.lstsq(A, np.asarray(bl), rcond=None)
        self.coef_lon, *_ = np.linalg.lstsq(A, np.asarray(bo), rcond=None)
        self.map_region = map_region
        self.anchor_cities = anchor_cities or []
        self._inv = np.linalg.inv(np.array([
            [self.coef_lat[0], self.coef_lat[1]],
            [self.coef_lon[0], self.coef_lon[1]],
        ]))

    def image_pixel_to_latlon(self, ix, iy):
        lx, ly = ix - self.map_region.x0, iy - self.map_region.y0
        return (float(self.coef_lat[0] * lx + self.coef_lat[1] * ly + self.coef_lat[2]),
                float(self.coef_lon[0] * lx + self.coef_lon[1] * ly + self.coef_lon[2]))

    def latlon_to_image_pixel(self, lat, lon):
        rhs = np.array([lat - self.coef_lat[2], lon - self.coef_lon[2]])
        xy = self._inv @ rhs
        return float(xy[0]) + self.map_region.x0, float(xy[1]) + self.map_region.y0

    def km_per_pixel(self, ix, iy):
        lat0, lon0 = self.image_pixel_to_latlon(ix, iy)
        lat1, lon1 = self.image_pixel_to_latlon(ix + 1, iy)
        lat2, lon2 = self.image_pixel_to_latlon(ix, iy + 1)
        return (haversine_km(lat0, lon0, lat1, lon1)
                + haversine_km(lat0, lon0, lat2, lon2)) / 2.0

    def latlon_offset_km(self, px, py, t_lat, t_lon):
        lat = self.coef_lat[0] * px + self.coef_lat[1] * py + self.coef_lat[2]
        lon = self.coef_lon[0] * px + self.coef_lon[1] * py + self.coef_lon[2]
        coslat = math.cos(math.radians(t_lat))
        return (lon - t_lon) * 111.32 * coslat, (lat - t_lat) * 111.32

    def radius_mask(self, h, w, t_lat, t_lon, radius_km):
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
        lx = xx - self.map_region.x0
        ly = yy - self.map_region.y0
        lat = self.coef_lat[0] * lx + self.coef_lat[1] * ly + self.coef_lat[2]
        lon = self.coef_lon[0] * lx + self.coef_lon[1] * ly + self.coef_lon[2]
        coslat = math.cos(math.radians(t_lat))
        dx = (lon - t_lon) * 111.32 * coslat
        dy = (lat - t_lat) * 111.32
        return (dx * dx + dy * dy) <= (radius_km * radius_km)


def load_calibration(path, inline, image_size=None):
    if inline:
        try:
            raw = json.loads(inline)
        except json.JSONDecodeError:
            raw = yaml.safe_load(inline)
    elif path:
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
    else:
        raise ValueError("No calibration given")
    if not raw or "points" not in raw:
        raise ValueError("Calibration missing 'points'")
    mr = parse_map_region(raw, image_size)
    if mr is None:
        raise ValueError("Calibration has no 'map' block and no image dims.")
    if image_size:
        validate_map_against_image(mr, image_size)
    used, anchors = [], []
    for i, entry in enumerate(raw["points"], start=1):
        edge_key = "edge" if "edge" in entry else "corner" if "corner" in entry else None
        has_pixel, has_edge = "pixel" in entry, edge_key is not None
        has_city, has_ll = "city" in entry, "latlon" in entry
        if has_city and has_ll:
            raise ValueError(f"Anchor #{i}: city or latlon, not both")
        if not (has_city or has_ll):
            raise ValueError(f"Anchor #{i}: need city or latlon")
        if has_pixel == has_edge:
            raise ValueError(f"Anchor #{i}: exactly one of pixel/edge")
        if has_pixel:
            px = entry["pixel"]; x, y = float(px[0]), float(px[1])
        else:
            x, y = _resolve_edge(entry[edge_key], mr)
        if has_city:
            r = _city_lookup(entry["city"])
            if not r:
                raise ValueError(f"Anchor #{i}: unknown city '{entry['city']}'")
            name, lat, lon = r
            anchors.append((city_label(name), lat, lon))
        else:
            lat, lon = float(entry["latlon"][0]), float(entry["latlon"][1])
        used.append((x, y, lat, lon))
    if len(used) < 3:
        raise ValueError(f"Only {len(used)} complete anchor(s); need >=3")
    return Calibration(used, mr, anchor_cities=anchors)


# ----------------------------------------------------------------------------
# SRI legend
# ----------------------------------------------------------------------------
class SRILegend:
    def __init__(self, region, orientation="horizontal",
                 breaks=None, break_positions=None, padding_px=1):
        if not breaks:
            raise ValueError("SRILegend needs a non-empty `breaks` list")
        if break_positions is not None and len(break_positions) != len(breaks):
            raise ValueError("break_positions length must match breaks")
        self.region = tuple(region)
        self.orientation = orientation
        self.breaks = list(breaks)
        if break_positions is None:
            n = len(breaks)
            self.break_positions = [(i / (n - 1)) if n > 1 else 0.0 for i in range(n)]
        else:
            self.break_positions = list(break_positions)
        self.padding_px = int(padding_px)
        self.samples_rgb = None
        self.samples_mmh = None
        self.tree = None
        self._ready = False

    def calibrate(self, frame):
        x0, y0, x1, y1 = self.region
        crop = frame.crop((x0, y0, x1, y1))
        arr = np.asarray(crop.convert("RGB"), dtype=np.float32)
        h, w = arr.shape[:2]
        pad = self.padding_px
        if self.orientation == "horizontal":
            usable = arr[:, pad:w - pad, :] if w > 2 * pad + 2 else arr
            bar_length = usable.shape[1]; axis = 1
        else:
            usable = arr[pad:h - pad, :, :] if h > 2 * pad + 2 else arr
            bar_length = usable.shape[0]; axis = 0
        if bar_length < 2:
            raise ValueError(f"SRI legend bar too short ({bar_length} px)")
        sample_rgb, sample_mmh = [], []
        for mmh, pos in zip(self.breaks, self.break_positions):
            idx = int(round(pos * (bar_length - 1)))
            idx = max(0, min(bar_length - 1, idx))
            lo, hi = max(0, idx - 1), min(bar_length, idx + 2)
            if axis == 1:
                color = usable[:, lo:hi, :].mean(axis=(0, 1))
            else:
                color = usable[lo:hi, :, :].mean(axis=(0, 1))
            sample_rgb.append(color); sample_mmh.append(mmh)
        self.samples_rgb = np.array(sample_rgb, dtype=np.float64)
        self.samples_mmh = np.array(sample_mmh, dtype=np.float64)
        self.tree = cKDTree(self.samples_rgb)
        self._ready = True

    def is_ready(self):
        return self._ready

    def decode(self, rgb_pixels):
        if not self._ready or len(rgb_pixels) == 0:
            return np.zeros(len(rgb_pixels), dtype=np.float32)
        _, indices = self.tree.query(rgb_pixels)
        return self.samples_mmh[indices].astype(np.float32)

    def print_samples(self):
        if not self._ready:
            print("SRI legend: not calibrated"); return
        print(f"SRI legend ({self.orientation}, {len(self.breaks)} breaks):")
        for rgb, mmh, pos in zip(self.samples_rgb, self.samples_mmh, self.break_positions):
            r, g, b = rgb.astype(int)
            print(f"  pos={pos:.3f}  #{r:02x}{g:02x}{b:02x} → {mmh:g} mm/h")


# ----------------------------------------------------------------------------
# Radar image analysis
# ----------------------------------------------------------------------------
def fetch_image(url, timeout=30):
    r = requests.get(url, headers=RADAR_HEADERS, timeout=timeout)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content))


def _dedupe_trailing_static_frames(frames):
    """Drop trailing frames pixel-identical to the last frame, keeping the
    FIRST frame of the trailing static run so that indices stay contiguous
    from zero and every downstream img.seek(idx) remains valid.

    Returns (deduped_frames, original_length).
    """
    if len(frames) < 2:
        return frames, len(frames)
    last_hash = hashlib.md5(frames[-1].tobytes()).hexdigest()
    keep_from = len(frames) - 1
    while keep_from > 0:
        if hashlib.md5(frames[keep_from - 1].tobytes()).hexdigest() != last_hash:
            break
        keep_from -= 1
    return frames[:keep_from + 1], len(frames)


def compute_rgb_and_mask(frame):
    rgb = np.asarray(frame.convert("RGB"), dtype=np.float32)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    diff = mx - mn
    sat = np.where(mx > 0, diff / mx, 0)
    val = mx / 255.0
    precip = (sat >= MIN_SAT) & (val >= MIN_VAL)
    return rgb, precip


def analyze_frame(frame, map_region, intensity_mode, sri_legend, legend_mask):
    rgb, precip = compute_rgb_and_mask(frame)
    precip &= map_region.crop_mask(precip.shape[0], precip.shape[1])
    if legend_mask is not None and legend_mask.shape == precip.shape:
        precip &= ~legend_mask
    mmh = np.zeros(precip.shape, dtype=np.float32)
    ys, xs = np.where(precip)
    if len(ys) > 0:
        rgbs = rgb[ys, xs]
        uniq, inv = np.unique(rgbs, axis=0, return_inverse=True)
        inv = inv.ravel()
        if intensity_mode != "simple" and sri_legend and sri_legend.is_ready():
            mmh[ys, xs] = sri_legend.decode(uniq)[inv]
        else:
            _, indices = _DHMZ_SRI_TREE.query(uniq)
            mmh[ys, xs] = _DHMZ_SRI_MIDPOINTS[indices][inv]
    return precip, mmh


# ----------------------------------------------------------------------------
# Storm Frame Archiver (deduplicated PNGs grouped by storm episode)
# ----------------------------------------------------------------------------
class StormArchiver:
    EPISODE_GAP_MIN = 45.0

    def __init__(self, data_dir, calib, target_lat, target_lon, target_name):
        self.data_dir = data_dir
        self.root_dir = os.path.join(data_dir, "archive") if data_dir else None
        self.calib = calib
        self.target_lat = target_lat
        self.target_lon = target_lon
        self.target_name = target_name
        self.lock = threading.Lock()
        self._mask_cache = {}
        self.total_frames_saved = 0
        self.total_episodes = 0
        self.last_episode_id = None
        self.last_saved_utc = None
        if self.root_dir:
            try:
                os.makedirs(self.root_dir, exist_ok=True)
                self._scan_existing()
            except OSError as e:
                print(f"archive: cannot create {self.root_dir}: {e}", file=sys.stderr)
                self.root_dir = None

    def _scan_existing(self):
        if not self.root_dir or not os.path.isdir(self.root_dir):
            return
        episodes = sorted(
            d for d in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, d))
        )
        self.total_episodes = len(episodes)
        if episodes:
            self.last_episode_id = episodes[-1]
        count = 0
        for ep in episodes:
            ep_dir = os.path.join(self.root_dir, ep)
            try:
                count += sum(1 for f in os.listdir(ep_dir) if f.endswith(".png"))
            except OSError:
                pass
        self.total_frames_saved = count

    def _get_radius_mask(self, h, w, radius_km):
        key = (h, w, round(float(radius_km), 1))
        m = self._mask_cache.get(key)
        if m is None:
            m = self.calib.radius_mask(h, w, self.target_lat, self.target_lon, radius_km)
            if len(self._mask_cache) > 8:
                self._mask_cache.clear()
            self._mask_cache[key] = m
        return m

    def _parse_episode_start(self, ep_name):
        try:
            return datetime.strptime(ep_name, "%Y-%m-%d_%H%MZ").replace(tzinfo=timezone.utc)
        except ValueError:
            return None

    def _resolve_episode_dir(self, frame_ts_utc):
        best_ep = None
        best_gap = float("inf")
        for d in sorted(os.listdir(self.root_dir)):
            ep_dir = os.path.join(self.root_dir, d)
            if not os.path.isdir(ep_dir):
                continue
            meta_path = os.path.join(ep_dir, "event.json")
            meta = _load_json_file(meta_path)
            start_dt = _safe_fromiso(meta.get("start_utc")) if meta else self._parse_episode_start(d)
            end_dt = _safe_fromiso(meta.get("end_utc")) if meta else start_dt
            if start_dt == datetime.min.replace(tzinfo=timezone.utc):
                continue
            if (start_dt - timedelta(minutes=self.EPISODE_GAP_MIN)) <= frame_ts_utc <= (
                    end_dt + timedelta(minutes=self.EPISODE_GAP_MIN)):
                gap = abs((frame_ts_utc - end_dt).total_seconds()) / 60.0
                if gap < best_gap:
                    best_gap = gap
                    best_ep = d

        if best_ep is not None:
            return best_ep, os.path.join(self.root_dir, best_ep), False

        ep_id = frame_ts_utc.strftime("%Y-%m-%d_%H%MZ")
        ep_dir = os.path.join(self.root_dir, ep_id)
        os.makedirs(ep_dir, exist_ok=True)
        return ep_id, ep_dir, True

    def process_animation(self, img, frame_masks_mmh, last_frame_ts_utc,
                          eff_interval_min, fallback_now_utc, rt,
                          local_weather=None, target_raining=False):
        if not self.root_dir or not rt.get("archive_enabled", True):
            return 0

        radius_km = float(rt.get("archive_radius_km", 200.0))
        min_px = int(rt.get("archive_min_pixels", 40))
        n_frames = len(frame_masks_mmh)
        if n_frames == 0:
            return 0

        anchor_utc = (last_frame_ts_utc or fallback_now_utc).replace(second=0, microsecond=0)

        saved_this_cycle = 0
        with self.lock:
            for idx, (mask, mmh) in enumerate(frame_masks_mmh):
                h, w = mask.shape
                rmask = self._get_radius_mask(h, w, radius_km)
                local_precip = mask & rmask
                px_count = int(local_precip.sum())
                if px_count < min_px:
                    continue

                minutes_before_last = (n_frames - 1 - idx) * eff_interval_min
                frame_ts = anchor_utc - timedelta(minutes=round(minutes_before_last))
                fname = frame_ts.strftime("%Y%m%d_%H%M%SZ.png")

                ep_id, ep_dir, is_new_ep = self._resolve_episode_dir(frame_ts)
                if is_new_ep:
                    self.total_episodes += 1

                fpath = os.path.join(ep_dir, fname)
                if os.path.exists(fpath):
                    continue

                try:
                    img.seek(idx)
                    frame_pil = img.convert("P", palette=Image.ADAPTIVE, colors=64)
                    tmp_path = fpath + ".tmp"
                    frame_pil.save(tmp_path, format="PNG", optimize=True)
                    os.replace(tmp_path, fpath)
                except Exception as e:
                    print(f"archive: frame save error {e}", file=sys.stderr)
                    continue

                peak_mmh = float(mmh[local_precip].max()) if px_count > 0 else 0.0
                mean_mmh = float(mmh[local_precip].mean()) if px_count > 0 else 0.0
                self._update_episode_meta(
                    ep_dir, ep_id, frame_ts, radius_km, px_count,
                    peak_mmh, mean_mmh,
                    local_weather if idx == n_frames - 1 else None,
                    target_raining if idx == n_frames - 1 else False,
                )
                saved_this_cycle += 1
                self.total_frames_saved += 1
                self.last_episode_id = ep_id
                self.last_saved_utc = frame_ts.isoformat()

        try:
            img.seek(0)
        except Exception:
            pass
        return saved_this_cycle

    def _update_episode_meta(self, ep_dir, ep_id, frame_ts, radius_km,
                             px_count, peak_mmh, mean_mmh,
                             local_weather, target_raining):
        meta_path = os.path.join(ep_dir, "event.json")
        meta = _load_json_file(meta_path) or {
            "episode_id": ep_id,
            "target": self.target_name,
            "target_lat": self.target_lat,
            "target_lon": self.target_lon,
            "archive_radius_km": radius_km,
            "start_utc": frame_ts.isoformat(),
            "end_utc": frame_ts.isoformat(),
            "frame_count": 0,
            "max_precip_pixels": 0,
            "peak_sri_mmh": 0.0,
            "target_hit": False,
        }
        ts_iso = frame_ts.isoformat()
        if ts_iso < meta["start_utc"]:
            meta["start_utc"] = ts_iso
        if ts_iso > meta["end_utc"]:
            meta["end_utc"] = ts_iso
        try:
            meta["frame_count"] = sum(
                1 for f in os.listdir(ep_dir) if f.endswith(".png")
            )
        except OSError:
            meta["frame_count"] = meta.get("frame_count", 0) + 1
        meta["max_precip_pixels"] = max(int(meta.get("max_precip_pixels", 0)), px_count)
        meta["peak_sri_mmh"] = round(max(float(meta.get("peak_sri_mmh", 0.0)), peak_mmh), 1)
        if target_raining:
            meta["target_hit"] = True
        if local_weather and local_weather.station_count > 0:
            meta["latest_local_weather"] = {
                "temp_c": local_weather.temp_c,
                "pressure_hpa": local_weather.pressure_hpa,
                "wind_speed_kmh": local_weather.wind_speed_kmh,
                "wind_gust_kmh": local_weather.wind_gust_kmh,
                "wind_dir_deg": local_weather.wind_dir_deg,
                "precip_today_mm": local_weather.precip_today_mm,
            }
        _save_json_file(meta_path, meta)

    def load_episode_frames(self, episode_id):
        if not self.root_dir:
            return []
        safe_id = os.path.basename(episode_id)
        ep_dir = os.path.join(self.root_dir, safe_id)
        if not os.path.isdir(ep_dir):
            return []
        files = sorted(f for f in os.listdir(ep_dir) if f.endswith(".png"))
        out = []
        for fn in files:
            try:
                ts = datetime.strptime(fn, "%Y%m%d_%H%M%SZ.png").replace(tzinfo=timezone.utc)
            except ValueError:
                ts = None
            out.append((ts, os.path.join(ep_dir, fn), fn))
        return out

    def summary(self):
        with self.lock:
            return {
                "enabled": bool(self.root_dir and _snapshot().get("archive_enabled", True)),
                "archive_dir": self.root_dir,
                "radius_km": _snapshot().get("archive_radius_km", 200.0),
                "total_episodes": self.total_episodes,
                "total_frames": self.total_frames_saved,
                "last_episode": self.last_episode_id,
                "last_saved_utc": self.last_saved_utc,
            }

    def list_episodes(self, limit=50):
        if not self.root_dir or not os.path.isdir(self.root_dir):
            return []
        out = []
        with self.lock:
            dirs = sorted(
                (d for d in os.listdir(self.root_dir)
                 if os.path.isdir(os.path.join(self.root_dir, d))),
                reverse=True,
            )[:limit]
            for d in dirs:
                ep_dir = os.path.join(self.root_dir, d)
                meta = _load_json_file(os.path.join(ep_dir, "event.json"))
                files = sorted(f for f in os.listdir(ep_dir) if f.endswith(".png"))
                if meta:
                    meta["frame_count"] = len(files)
                    meta["frames"] = files
                    out.append(meta)
                else:
                    out.append({"episode_id": d, "frame_count": len(files), "frames": files})
        return out


# ----------------------------------------------------------------------------
# Clustering + tracking with Adaptive Velocity Windowing
# ----------------------------------------------------------------------------
@dataclass
class PixelCluster:
    cx: float
    cy: float
    size: int
    label: int = 0
    max_mmh: float = 0.0
    mean_mmh: float = 0.0


def clusters_from_mask(mask, min_pixels, mmh=None):
    label_array, n = ndimage.label(mask, structure=np.ones((3, 3)))
    if n == 0:
        return [], label_array
    idx = np.arange(1, n + 1)
    sizes = ndimage.sum(mask, label_array, idx)
    geom_cents = ndimage.center_of_mass(mask, label_array, idx)
    if mmh is not None:
        weights = np.where(mask, np.maximum(mmh, 0.1), 0.0)
        cents = ndimage.center_of_mass(weights, label_array, idx)
        max_vals = ndimage.maximum(mmh, label_array, idx)
        mean_vals = ndimage.mean(mmh, label_array, idx)
    else:
        cents = geom_cents
        max_vals = np.zeros(n, dtype=np.float32)
        mean_vals = np.zeros(n, dtype=np.float32)
    out = []
    for lbl, sz, (cy, cx), (gcy, gcx), mx_v, mn_v in zip(
            idx, sizes, cents, geom_cents, max_vals, mean_vals):
        if sz < min_pixels:
            continue
        if not (math.isfinite(cx) and math.isfinite(cy)):
            cx, cy = gcx, gcy
        out.append(PixelCluster(float(cx), float(cy), int(sz), int(lbl),
                                float(mx_v), float(mn_v)))
    return out, label_array


@dataclass
class Track:
    cluster: PixelCluster
    cid: int = 0
    vx_kmh: float = 0.0
    vy_kmh: float = 0.0
    n_observations: int = 1
    span_frames: int = 0
    prev_size: Optional[int] = None
    event: Optional[str] = None
    eff_window_min: float = VELOCITY_WINDOW_MIN

    @property
    def speed_kmh(self):
        return math.hypot(self.vx_kmh, self.vy_kmh)


def _compute_adaptive_window_min(obs, base_window_min, frame_interval_min, adaptive=True):
    if not adaptive or len(obs) < 3:
        return float(base_window_min)

    tail = obs[-min(len(obs), 6):]
    step_bearings = []
    for (f0, c0), (f1, c1) in zip(tail, tail[1:]):
        dx = c1.cx - c0.cx
        dy = -(c1.cy - c0.cy)
        if math.hypot(dx, dy) >= 0.75:
            step_bearings.append(math.degrees(math.atan2(dx, dy)) % 360.0)

    turn_rate = 0.0
    if len(step_bearings) >= 2:
        diffs = [angular_diff_deg(b1, b0)
                 for b0, b1 in zip(step_bearings, step_bearings[1:])]
        turn_rate = float(np.mean(diffs))

    s0 = max(1.0, float(tail[0][1].size))
    s1 = max(1.0, float(tail[-1][1].size))
    n_steps = max(1, tail[-1][0] - tail[0][0])
    growth_per_step = abs(math.log(s1 / s0)) / n_steps

    turn_factor = max(0.0, min(1.0, 1.0 - (turn_rate / 25.0)))
    growth_factor = max(0.0, min(1.0, 1.0 - (growth_per_step / 0.35)))
    stability = 0.65 * turn_factor + 0.35 * growth_factor

    min_win = max(10.0, 2.0 * frame_interval_min)
    max_win = max(base_window_min, min(60.0, base_window_min * 1.35))
    return min_win + stability * (max_win - min_win)


def _select_recent_window(obs, frame_interval_min, split_frames=None,
                          base_window_min=VELOCITY_WINDOW_MIN, adaptive=True):
    if len(obs) <= 1:
        return obs, float(base_window_min)
    if split_frames:
        last_fidx = obs[-1][0]
        latest_split = max((sf for sf in split_frames if sf <= last_fidx), default=None)
        if latest_split is not None:
            post_split = [o for o in obs if o[0] >= latest_split]
            if len(post_split) >= 2:
                obs = post_split

    eff_win = _compute_adaptive_window_min(
        obs, base_window_min, frame_interval_min, adaptive=adaptive)
    last_fidx = obs[-1][0]
    max_frame_span = max(1, int(round(eff_win / max(1.0, frame_interval_min))))
    cutoff_fidx = last_fidx - max_frame_span
    window = [o for o in obs if o[0] >= cutoff_fidx]
    if len(window) < 2:
        window = obs[-2:]
    return window, eff_win


def _estimate_pixel_velocity(obs, frame_interval_min, split_frames=None,
                             base_window_min=VELOCITY_WINDOW_MIN, adaptive=True):
    window, _ = _select_recent_window(
        obs, frame_interval_min, split_frames, base_window_min, adaptive)
    if len(window) < 2:
        return 0.0, 0.0
    ts = np.array([o[0] for o in window], dtype=np.float64)
    xs = np.array([o[1].cx for o in window], dtype=np.float64)
    ys = np.array([o[1].cy for o in window], dtype=np.float64)
    age_steps = ts[-1] - ts
    recency_w = np.exp(-0.12 * age_steps)
    ws = recency_w * np.sqrt(np.maximum(1.0, np.array([o[1].size for o in window], dtype=np.float64)))
    w_sum = float(np.sum(ws))
    t_mean = float(np.sum(ws * ts) / w_sum)
    dt = ts - t_mean
    denom = float(np.sum(ws * dt * dt))
    if denom <= 1e-9:
        return 0.0, 0.0
    x_mean = float(np.sum(ws * xs) / w_sum)
    y_mean = float(np.sum(ws * ys) / w_sum)
    vx_px = float(np.sum(ws * dt * (xs - x_mean)) / denom)
    vy_px = float(np.sum(ws * dt * (ys - y_mean)) / denom)
    return vx_px, vy_px


def _estimate_velocity_kmh(obs, calib, frame_interval_min, split_frames=None,
                           base_window_min=VELOCITY_WINDOW_MIN, adaptive=True,
                           return_window=False):
    window, eff_win = _select_recent_window(
        obs, frame_interval_min, split_frames, base_window_min, adaptive)
    if len(window) < 2:
        return (0.0, 0.0, eff_win) if return_window else (0.0, 0.0)

    lat0, lon0 = calib.image_pixel_to_latlon(window[0][1].cx, window[0][1].cy)
    coslat = max(0.01, math.cos(math.radians(lat0)))

    ts_h = np.empty(len(window), dtype=np.float64)
    xs_km = np.empty(len(window), dtype=np.float64)
    ys_km = np.empty(len(window), dtype=np.float64)
    ws = np.empty(len(window), dtype=np.float64)

    last_fidx = window[-1][0]
    for idx, (fidx, c) in enumerate(window):
        lat, lon = calib.image_pixel_to_latlon(c.cx, c.cy)
        ts_h[idx] = fidx * frame_interval_min / 60.0
        xs_km[idx] = (lon - lon0) * 111.32 * coslat
        ys_km[idx] = (lat - lat0) * 111.32
        recency = math.exp(-0.12 * (last_fidx - fidx))
        ws[idx] = recency * math.sqrt(max(1.0, float(c.size)))

    w_sum = float(np.sum(ws))
    t_mean = float(np.sum(ws * ts_h) / w_sum)
    dt = ts_h - t_mean
    denom = float(np.sum(ws * dt * dt))
    if denom <= 1e-9:
        return (0.0, 0.0, eff_win) if return_window else (0.0, 0.0)

    x_mean = float(np.sum(ws * xs_km) / w_sum)
    y_mean = float(np.sum(ws * ys_km) / w_sum)
    vx = float(np.sum(ws * dt * (xs_km - x_mean)) / denom)
    vy = float(np.sum(ws * dt * (ys_km - y_mean)) / denom)

    speed = math.hypot(vx, vy)
    if speed > MAX_STORM_SPEED_KMH:
        scale = MAX_STORM_SPEED_KMH / speed
        vx *= scale
        vy *= scale
    return (vx, vy, eff_win) if return_window else (vx, vy)


def track_frame_sequence(frame_masks_mmh, calib, frame_interval_min,
                         target_lat, target_lon, rt=None):
    rt = rt or _snapshot()
    base_win = float(rt.get("velocity_window_min", VELOCITY_WINDOW_MIN))
    adaptive_win = bool(rt.get("velocity_adaptive", True))

    tracks_by_id = {}
    split_frames_by_id = {}
    next_id = 0
    prev_clusters, prev_ids = [], []
    n_frames = 0
    rain_history = []
    last_label_array = None
    last_mask = last_mmh = None
    last_frame_events = {}
    cluster_first_target_hit = {}

    mid_x = (calib.map_region.x0 + calib.map_region.x1) / 2.0
    mid_y = (calib.map_region.y0 + calib.map_region.y1) / 2.0
    kmpp = max(0.05, calib.km_per_pixel(mid_x, mid_y))
    max_step_km = MAX_STORM_SPEED_KMH * (max(1.0, frame_interval_min) / 60.0)
    base_max_jump_px = max(18.0, (max_step_km / kmpp) + 12.0)

    px, py = calib.latlon_to_image_pixel(target_lat, target_lon)
    xi, yi = int(round(px)), int(round(py))

    for i, (mask, mmh) in enumerate(frame_masks_mmh):
        n_frames += 1
        last_mask, last_mmh = mask, mmh
        h, w = mask.shape
        if 0 <= yi < h and 0 <= xi < w:
            rain_history.append(bool(mask[max(0, yi-1):yi+2, max(0, xi-1):xi+2].any()))
        else:
            rain_history.append(False)
        clusters, label_array = clusters_from_mask(mask, MIN_CLUSTER_PIXELS, mmh=mmh)
        last_label_array = label_array
        if not clusters:
            prev_clusters, prev_ids = [], []
            last_frame_events = {}
            continue

        predictions = []
        for j, p in enumerate(prev_clusters):
            pid = prev_ids[j]
            obs = tracks_by_id[pid]
            dt_f = max(1, i - obs[-1][0])
            if len(obs) >= 2:
                vx_px, vy_px = _estimate_pixel_velocity(
                    obs, frame_interval_min, split_frames_by_id.get(pid),
                    base_win, adaptive_win)
                pred_x = p.cx + vx_px * dt_f
                pred_y = p.cy + vy_px * dt_f
                allowed_px = base_max_jump_px * dt_f
            else:
                pred_x, pred_y = p.cx, p.cy
                allowed_px = base_max_jump_px * 1.25 * dt_f
            predictions.append((pred_x, pred_y, allowed_px))

        candidates = []
        for k, c in enumerate(clusters):
            for j, p in enumerate(prev_clusters):
                pred_x, pred_y, allowed_px = predictions[j]
                d = math.hypot(c.cx - pred_x, c.cy - pred_y)
                if d > allowed_px:
                    continue
                size_ratio = max(c.size, p.size) / max(1, min(c.size, p.size))
                cost = d * (1.0 + 0.25 * max(0.0, size_ratio - 1.5))
                candidates.append((cost, j, k))
        candidates.sort(key=lambda item: item[0])

        matched_prev = {}
        matched_curr = {}
        for _, j, k in candidates:
            if j in matched_prev or k in matched_curr:
                continue
            matched_prev[j] = k
            matched_curr[k] = prev_ids[j]

        events = {}
        for k, c in enumerate(clusters):
            if k in matched_curr:
                continue
            best_parent_cid = None
            best_d = base_max_jump_px * 0.85
            for j, p in enumerate(prev_clusters):
                if j not in matched_prev:
                    continue
                pred_x, pred_y, _ = predictions[j]
                d = math.hypot(c.cx - pred_x, c.cy - pred_y)
                if d < best_d:
                    best_d = d
                    best_parent_cid = prev_ids[j]
            if best_parent_cid is not None:
                split_frames_by_id.setdefault(best_parent_cid, set()).add(i)
                for m_k, m_cid in matched_curr.items():
                    if m_cid == best_parent_cid:
                        events[m_cid] = "split"

        curr_ids = []
        for k, c in enumerate(clusters):
            if k in matched_curr:
                cid = matched_curr[k]
                prev_obs_size = tracks_by_id[cid][-1][1].size
                ratio = c.size / max(1, prev_obs_size)
                if ratio > 2.5 or ratio < 0.4:
                    split_frames_by_id.setdefault(cid, set()).add(i)
            else:
                cid = next_id
                next_id += 1
                if any(math.hypot(c.cx - predictions[j][0], c.cy - predictions[j][1])
                       < base_max_jump_px * 0.85
                       for j in matched_prev):
                    events[cid] = "split"
            curr_ids.append(cid)
            tracks_by_id.setdefault(cid, []).append((i, c))

        last_frame_events = events
        if 0 <= yi < h and 0 <= xi < w and label_array[yi, xi] > 0:
            target_lbl = int(label_array[yi, xi])
            for k, c in enumerate(clusters):
                if c.label == target_lbl:
                    cid_here = curr_ids[k]
                    if cid_here not in cluster_first_target_hit:
                        cluster_first_target_hit[cid_here] = i
                    break
        prev_clusters, prev_ids = clusters, curr_ids

    tracks = []
    for j, c in enumerate(prev_clusters):
        tid = prev_ids[j]
        obs = tracks_by_id[tid]
        dt_frames = obs[-1][0] - obs[0][0]
        tr = Track(cluster=c, cid=tid, n_observations=len(obs), span_frames=dt_frames,
                   eff_window_min=base_win)
        if dt_frames > 0:
            tr.vx_kmh, tr.vy_kmh, tr.eff_window_min = _estimate_velocity_kmh(
                obs, calib, frame_interval_min, split_frames_by_id.get(tid),
                base_win, adaptive_win, return_window=True)
        if len(obs) >= 2:
            tr.prev_size = obs[-2][1].size
        tr.event = last_frame_events.get(tid)
        tracks.append(tr)
    return (last_mask, last_mmh, tracks, n_frames, rain_history,
            last_label_array, tracks_by_id, split_frames_by_id,
            cluster_first_target_hit)


def track_animation(img, calib, frame_interval_min, target_lat, target_lon,
                    intensity_mode, sri_legend, legend_mask, rt=None):
    raw = [f.convert("RGB") for f in ImageSequence.Iterator(img)]
    raw, _original_count = _dedupe_trailing_static_frames(raw)
    frame_masks_mmh = []
    for frame in raw:
        mask, mmh = analyze_frame(frame, calib.map_region, intensity_mode,
                                  sri_legend, legend_mask)
        frame_masks_mmh.append((mask, mmh))
    res = track_frame_sequence(frame_masks_mmh, calib, frame_interval_min,
                               target_lat, target_lon, rt=rt)
    return (*res, frame_masks_mmh)


# ----------------------------------------------------------------------------
# Walk-Forward Speed & Heading Evaluator + Two-Stage Shadow Auto-Tuner
# ----------------------------------------------------------------------------
def evaluate_walk_forward_episode(frame_masks_mmh, calib, frame_interval_min,
                                  target_lat, target_lon, rt, horizon_min=30.0):
    (last_mask, last_mmh, _, n_frames, rain_history,
     _, tracks_by_id, split_frames_by_id, _) = track_frame_sequence(
        frame_masks_mmh, calib, frame_interval_min, target_lat, target_lon, rt=rt)

    if n_frames < 3:
        return None

    base_win = float(rt.get("velocity_window_min", VELOCITY_WINDOW_MIN))
    adaptive_win = bool(rt.get("velocity_adaptive", True))
    min_size = int(rt.get("min_size_px", 30))
    min_dur = int(rt.get("min_duration_frames", 2))
    horizon_steps = max(1, int(round(horizon_min / max(1.0, frame_interval_min))))

    speed_errs = []
    heading_errs = []
    pos_miss_km = []
    eff_windows = []

    for cid, obs in tracks_by_id.items():
        if len(obs) <= horizon_steps + 1:
            continue
        obs_by_f = {fidx: c for fidx, c in obs}
        for idx in range(1, len(obs) - horizon_steps):
            f_now, c_now = obs[idx]
            if c_now.size < min_size:
                continue
            obs_prefix = obs[:idx + 1]
            if len(obs_prefix) < min_dur:
                continue
            f_fut = f_now + horizon_steps
            c_fut = obs_by_f.get(f_fut)
            if c_fut is None:
                continue

            vx_p, vy_p, eff_w = _estimate_velocity_kmh(
                obs_prefix, calib, frame_interval_min,
                split_frames_by_id.get(cid), base_win, adaptive_win,
                return_window=True)
            pred_speed = math.hypot(vx_p, vy_p)
            if pred_speed < MIN_SPEED_KMH:
                continue

            lat0, lon0 = calib.image_pixel_to_latlon(c_now.cx, c_now.cy)
            lat1, lon1 = calib.image_pixel_to_latlon(c_fut.cx, c_fut.cy)
            dt_h = (f_fut - f_now) * frame_interval_min / 60.0
            if dt_h <= 0:
                continue

            coslat = max(0.01, math.cos(math.radians((lat0 + lat1) / 2.0)))
            act_dx = (lon1 - lon0) * 111.32 * coslat
            act_dy = (lat1 - lat0) * 111.32
            act_dist = math.hypot(act_dx, act_dy)
            act_speed = act_dist / dt_h
            if act_speed < MIN_SPEED_KMH:
                continue

            pred_dx = vx_p * dt_h
            pred_dy = vy_p * dt_h
            miss = math.hypot(pred_dx - act_dx, pred_dy - act_dy)

            pred_brg = math.degrees(math.atan2(vx_p, vy_p)) % 360.0
            act_brg = math.degrees(math.atan2(act_dx, act_dy)) % 360.0
            hdg_err = angular_diff_deg(pred_brg, act_brg)

            speed_errs.append(abs(pred_speed - act_speed))
            heading_errs.append(hdg_err)
            pos_miss_km.append(miss)
            eff_windows.append(eff_w)

    tp = fp = fn = 0
    eta_errs = []
    lookahead_steps = max(2, int(round(60.0 / max(1.0, frame_interval_min))))
    for t in range(1, n_frames - 1):
        future_rain = rain_history[t + 1:min(n_frames, t + 1 + lookahead_steps)]
        actual_hit = any(future_rain)
        actual_eta = None
        if actual_hit:
            first_idx = next(k for k, r in enumerate(future_rain) if r)
            actual_eta = (first_idx + 1) * frame_interval_min

        active_tracks = []
        for cid, obs in tracks_by_id.items():
            pref = [o for o in obs if o[0] <= t]
            if not pref or pref[-1][0] != t:
                continue
            c_here = pref[-1][1]
            if c_here.size < min_size or len(pref) < min_dur:
                continue
            vx, vy = _estimate_velocity_kmh(
                pref, calib, frame_interval_min,
                split_frames_by_id.get(cid), base_win, adaptive_win)
            tr = Track(cluster=c_here, cid=cid, vx_kmh=vx, vy_kmh=vy,
                       n_observations=len(pref),
                       span_frames=pref[-1][0] - pref[0][0])
            active_tracks.append(tr)

        approaches = [assess(tr, target_lat, target_lon, calib) for tr in active_tracks]
        threats = [refine_threat(tr, a, rt) for tr, a in zip(active_tracks, approaches)]
        pred_hit = any(threats)

        if pred_hit and actual_hit:
            tp += 1
            etas = [a.eta_min for a, tf in zip(approaches, threats)
                    if tf and a.eta_min is not None]
            if etas and actual_eta is not None:
                eta_errs.append(abs(min(etas) - actual_eta))
        elif pred_hit and not actual_hit:
            fp += 1
        elif (not pred_hit) and actual_hit and not rain_history[t]:
            fn += 1

    return {
        "n_frames": n_frames,
        "kinematic_samples": len(pos_miss_km),
        "speed_mae_kmh": round(float(np.mean(speed_errs)), 2) if speed_errs else None,
        "heading_mae_deg": round(float(np.mean(heading_errs)), 1) if heading_errs else None,
        "pos_miss_30m_km": round(float(np.mean(pos_miss_km)), 2) if pos_miss_km else None,
        "avg_eff_window_min": round(float(np.mean(eff_windows)), 1) if eff_windows else None,
        "tp": tp, "fp": fp, "fn": fn,
        "eta_mae_min": round(float(np.mean(eta_errs)), 1) if eta_errs else None,
    }


def aggregate_walk_forward(ep_results):
    valid = [r for r in ep_results if r is not None]
    if not valid:
        return {
            "episodes_evaluated": 0,
            "kinematic_samples": 0,
            "speed_mae_kmh": None,
            "heading_mae_deg": None,
            "pos_miss_30m_km": None,
            "avg_eff_window_min": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "eta_mae_min": None,
            "composite_loss": None,
        }

    def _wmean(key):
        vals = [(r[key], max(1, r["kinematic_samples"]))
                for r in valid if r.get(key) is not None]
        if not vals:
            return None
        num = sum(v * w for v, w in vals)
        den = sum(w for _, w in vals)
        return round(num / den, 2)

    tp = sum(r["tp"] for r in valid)
    fp = sum(r["fp"] for r in valid)
    fn = sum(r["fn"] for r in valid)
    prec = tp / (tp + fp) if (tp + fp) > 0 else None
    rec = tp / (tp + fn) if (tp + fn) > 0 else None
    f1 = (2 * prec * rec / (prec + rec)) if (prec and rec and (prec + rec) > 0) else 0.0

    spd_mae = _wmean("speed_mae_kmh")
    hdg_mae = _wmean("heading_mae_deg")
    miss_30 = _wmean("pos_miss_30m_km")
    eta_mae = _wmean("eta_mae_min")

    loss = 0.0
    if miss_30 is not None:
        loss += miss_30 * 2.0
    if hdg_mae is not None:
        loss += hdg_mae * 0.25
    if spd_mae is not None:
        loss += spd_mae * 0.4
    if eta_mae is not None:
        loss += eta_mae * 0.3
    loss += (1.0 - f1) * 15.0

    return {
        "episodes_evaluated": len(valid),
        "kinematic_samples": sum(r["kinematic_samples"] for r in valid),
        "speed_mae_kmh": spd_mae,
        "heading_mae_deg": hdg_mae,
        "pos_miss_30m_km": miss_30,
        "avg_eff_window_min": _wmean("avg_eff_window_min"),
        "precision": round(prec, 3) if prec is not None else None,
        "recall": round(rec, 3) if rec is not None else None,
        "f1": round(f1, 3),
        "eta_mae_min": eta_mae,
        "composite_loss": round(loss, 2),
    }


class ProfileStore:
    def __init__(self, data_dir):
        self.path = os.path.join(data_dir, "profiles.json") if data_dir else None
        self.lock = threading.Lock()
        self.profiles = {}
        if self.path:
            self._load()

    def _load(self):
        raw = _load_json_file(self.path)
        if isinstance(raw, dict) and "profiles" in raw:
            self.profiles = raw["profiles"]

    def _save(self):
        if self.path:
            _save_json_file(self.path, {"version": 1, "profiles": self.profiles})

    def ensure_baseline(self, current_rt):
        with self.lock:
            if "baseline" not in self.profiles:
                self.profiles["baseline"] = {
                    "name": "baseline",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "parent": None,
                    "stage": "initial",
                    "params": {k: current_rt[k] for k in _TUNABLE_KEYS if k in current_rt},
                    "metrics": None,
                }
                self._save()

    def save_profile(self, name, params, metrics=None, parent="baseline", stage="manual"):
        with self.lock:
            clean_params = {k: params[k] for k in _TUNABLE_KEYS if k in params}
            self.profiles[name] = {
                "name": name,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "parent": parent,
                "stage": stage,
                "params": clean_params,
                "metrics": metrics,
            }
            self._save()
            return self.profiles[name]

    def next_iter_name(self):
        with self.lock:
            n = 1
            while f"iter-{n:02d}" in self.profiles:
                n += 1
            return f"iter-{n:02d}"

    def get(self, name):
        with self.lock:
            return dict(self.profiles[name]) if name in self.profiles else None

    def best_profile(self):
        with self.lock:
            scored = [
                p for p in self.profiles.values()
                if p.get("metrics") and p["metrics"].get("composite_loss") is not None
            ]
            if not scored:
                return dict(self.profiles["baseline"]) if "baseline" in self.profiles else None
            scored.sort(key=lambda p: p["metrics"]["composite_loss"])
            return dict(scored[0])

    def list_all(self):
        with self.lock:
            return list(self.profiles.values())

    def delete(self, name):
        if name == "baseline":
            return False
        with self.lock:
            if name in self.profiles:
                del self.profiles[name]
                self._save()
                return True
            return False


class OfflineReplayEngine:
    def __init__(self, archiver, calib, target_lat, target_lon,
                 frame_interval_min, intensity_mode, sri_legend):
        self.archiver = archiver
        self.calib = calib
        self.target_lat = target_lat
        self.target_lon = target_lon
        self.frame_interval_min = frame_interval_min
        self.intensity_mode = intensity_mode
        self.sri_legend = sri_legend
        self._cache = {}
        self.lock = threading.Lock()
        self.tune_lock = threading.Lock()
        self.last_autotune_utc = None
        self.last_autotune_frames = 0
        self.last_autotune_note = "idle"

    def get_episode_data(self, episode_id):
        if not self.archiver:
            return []
        entries = self.archiver.load_episode_frames(episode_id)
        if not entries:
            return []
        with self.lock:
            cached = self._cache.get(episode_id)
            if cached and cached[0] == len(entries):
                return cached[1]

        decoded = []
        for ts, fpath, fname in entries:
            try:
                with Image.open(fpath) as im:
                    mask, mmh = analyze_frame(
                        im, self.calib.map_region, self.intensity_mode,
                        self.sri_legend, None)
                decoded.append((ts, mask, mmh, fpath, fname))
            except Exception as e:
                print(f"replay: error decoding {fpath}: {e}", file=sys.stderr)

        with self.lock:
            if len(self._cache) > 12:
                self._cache.clear()
            self._cache[episode_id] = (len(entries), decoded)
        return decoded

    def evaluate_params(self, params, episode_ids=None):
        if not self.archiver:
            return {"error": "archiver disabled"}
        if not episode_ids:
            eps = self.archiver.list_episodes(limit=15)
            episode_ids = [e["episode_id"] for e in eps if e.get("frame_count", 0) >= 4]

        full_rt = _snapshot()
        full_rt.update({k: params[k] for k in _TUNABLE_KEYS if k in params})

        per_ep = {}
        results = []
        for ep_id in episode_ids:
            data = self.get_episode_data(ep_id)
            if len(data) < 4:
                continue
            masks_mmh = [(d[1], d[2]) for d in data]
            res = evaluate_walk_forward_episode(
                masks_mmh, self.calib, self.frame_interval_min,
                self.target_lat, self.target_lon, full_rt)
            if res:
                per_ep[ep_id] = res
                results.append(res)

        agg = aggregate_walk_forward(results)
        agg["per_episode"] = per_ep
        return agg

    def run_two_stage_autotune(self, base_params, episode_ids=None,
                               stage1_trials=14, stage2_trials=14):
        if not self.tune_lock.acquire(blocking=False):
            return {"ok": False, "error": "Auto-tuner is already running."}
        try:
            self.last_autotune_note = "running"
            if not episode_ids and self.archiver:
                eps = self.archiver.list_episodes(limit=12)
                episode_ids = [e["episode_id"] for e in eps if e.get("frame_count", 0) >= 4]
            if not episode_ids:
                self.last_autotune_note = "waiting for >=4 archived storm frames"
                return {"ok": False, "error": "No archived episodes with >=4 frames available yet."}

            decoded_eps = []
            for ep_id in episode_ids:
                d = self.get_episode_data(ep_id)
                if len(d) >= 4:
                    decoded_eps.append((ep_id, [(x[1], x[2]) for x in d]))
            if not decoded_eps:
                self.last_autotune_note = "no valid decoded frames"
                return {"ok": False, "error": "No valid archived frames found."}

            def _eval_fast(cand_rt):
                res_list = [
                    evaluate_walk_forward_episode(
                        m_list, self.calib, self.frame_interval_min,
                        self.target_lat, self.target_lon, cand_rt)
                    for _, m_list in decoded_eps
                ]
                return aggregate_walk_forward(res_list)

            base_rt = _snapshot()
            base_rt.update({k: base_params[k] for k in _TUNABLE_KEYS if k in base_params})
            baseline_metrics = _eval_fast(base_rt)

            best_s1 = dict(base_rt)
            best_s1_metrics = baseline_metrics

            def _kinematic_loss(m):
                if m["kinematic_samples"] == 0:
                    return 1e9
                return ((m["pos_miss_30m_km"] or 25.0) * 2.0
                        + (m["heading_mae_deg"] or 45.0) * 0.25
                        + (m["speed_mae_kmh"] or 15.0) * 0.4)

            best_s1_loss = _kinematic_loss(best_s1_metrics)
            rng = random.Random(int(time.time()))

            for _ in range(stage1_trials):
                cand = dict(base_rt)
                cand["velocity_window_min"] = round(rng.choice([15.0, 20.0, 25.0, 30.0, 35.0, 45.0]), 1)
                cand["velocity_adaptive"] = rng.choice([True, True, False])
                cand["min_size_px"] = rng.choice([20, 25, 30, 40, 50, 65])
                cand["min_duration_frames"] = rng.choice([2, 3, 4])
                m = _eval_fast(cand)
                loss = _kinematic_loss(m)
                if loss < best_s1_loss:
                    best_s1_loss = loss
                    best_s1 = cand
                    best_s1_metrics = m

            best_s2 = dict(best_s1)
            best_s2_metrics = best_s1_metrics
            best_s2_loss = best_s2_metrics.get("composite_loss") or 1e9

            for _ in range(stage2_trials):
                cand = dict(best_s1)
                cand["approaching_radius_km"] = round(
                    max(8.0, min(60.0, best_s1["approaching_radius_km"] + rng.choice([-8.0, -4.0, 0.0, 4.0, 8.0, 12.0]))), 1)
                cand["cluster_radius_weight"] = round(
                    max(0.2, min(2.0, best_s1["cluster_radius_weight"] + rng.choice([-0.3, -0.15, 0.0, 0.15, 0.3]))), 2)
                cand["threat_radius_km"] = round(
                    max(5.0, min(45.0, best_s1["threat_radius_km"] + rng.choice([-5.0, -2.5, 0.0, 2.5, 5.0, 8.0]))), 1)
                cand["threat_angle_deg"] = round(
                    max(15.0, min(90.0, best_s1["threat_angle_deg"] + rng.choice([-15.0, -7.5, 0.0, 7.5, 15.0]))), 1)
                cand["threat_min_sri_mmh"] = round(
                    rng.choice([0.0, 0.5, 1.0, 2.0, 3.0]), 1)
                m = _eval_fast(cand)
                loss = m.get("composite_loss") or 1e9
                if loss < best_s2_loss:
                    best_s2_loss = loss
                    best_s2 = cand
                    best_s2_metrics = m

            full_final = self.evaluate_params(best_s2, episode_ids=episode_ids)
            self.last_autotune_utc = datetime.now(timezone.utc).isoformat()
            if self.archiver:
                self.last_autotune_frames = self.archiver.total_frames_saved
            self.last_autotune_note = f"completed ({len(episode_ids)} episodes)"
            return {
                "ok": True,
                "episodes_used": episode_ids,
                "baseline_metrics": baseline_metrics,
                "stage1_metrics": best_s1_metrics,
                "tuned_metrics": full_final,
                "tuned_params": {k: best_s2[k] for k in _TUNABLE_KEYS},
            }
        finally:
            self.tune_lock.release()


def _execute_shadow_autotune(replay_engine, profile_store, set_active_challenger=True):
    """Runs 2-stage auto-tune and saves the new iteration without modifying active live params."""
    if not replay_engine or not profile_store:
        return {"ok": False, "error": "replay engine or profile store unavailable"}
    cur_rt = _snapshot()
    profile_store.ensure_baseline(cur_rt)

    # Start search from current active shadow profile if available, else current champion
    sh_name = cur_rt.get("shadow_profile") or ""
    sh_prof = profile_store.get(sh_name) if sh_name else None
    base_params = sh_prof["params"] if sh_prof else cur_rt

    res = replay_engine.run_two_stage_autotune(base_params)
    if not res.get("ok"):
        return res

    base_entry = profile_store.get("baseline")
    if base_entry and base_entry.get("metrics") is None:
        profile_store.save_profile(
            "baseline", base_entry["params"], metrics=res["baseline_metrics"],
            parent=None, stage="initial")

    iter_name = profile_store.next_iter_name()
    saved = profile_store.save_profile(
        iter_name, res["tuned_params"], metrics=res["tuned_metrics"],
        parent=sh_name or "baseline", stage="2-stage-autotune")
    res["saved_profile"] = iter_name

    # Adopt into SHADOW slot (never live champion) if it improves over current shadow
    if set_active_challenger:
        cur_sh_loss = (sh_prof.get("metrics") or {}).get("composite_loss") if sh_prof else None
        new_loss = (res["tuned_metrics"] or {}).get("composite_loss")
        if cur_sh_loss is None or (new_loss is not None and new_loss <= cur_sh_loss):
            with _RUNTIME_LOCK:
                _RUNTIME["shadow_profile"] = iter_name
            _persist_runtime_to_disk()
            res["adopted_as_shadow"] = True
        else:
            res["adopted_as_shadow"] = False
    return res


def _shadow_autotune_worker(replay_engine, profile_store, archiver):
    """Background daemon that periodically auto-tunes the shadow challenger as storm archive grows."""
    time.sleep(20)
    while True:
        try:
            rt = _snapshot()
            if rt.get("shadow_autotune_enabled", True) and archiver and replay_engine and profile_store:
                interval_s = max(1800.0, float(rt.get("shadow_autotune_hours", 6.0)) * 3600.0)
                now = datetime.now(timezone.utc)
                last_dt = _safe_fromiso(replay_engine.last_autotune_utc)
                elapsed_s = (now - last_dt).total_seconds()
                new_frames = archiver.total_frames_saved - replay_engine.last_autotune_frames
                should_run = _FORCE_TUNE.is_set() or (
                    elapsed_s >= interval_s and new_frames >= 4 and archiver.total_frames_saved >= 4
                )
                if should_run:
                    _FORCE_TUNE.clear()
                    print("shadow-tuner: starting background 2-stage auto-tune...")
                    res = _execute_shadow_autotune(replay_engine, profile_store, set_active_challenger=True)
                    if res.get("ok"):
                        print(f"shadow-tuner: saved {res.get('saved_profile')} "
                              f"(loss={res['tuned_metrics'].get('composite_loss')})")
                    else:
                        print(f"shadow-tuner: skipped ({res.get('error')})")
        except Exception as e:
            print(f"shadow-tuner worker error: {e}", file=sys.stderr)
        _FORCE_TUNE.wait(timeout=120.0)


# ----------------------------------------------------------------------------
# Approach + nowcast
# ----------------------------------------------------------------------------
@dataclass
class Approach:
    dist_km: float
    bearing_to_target: float
    bearing_of_motion: Optional[float]
    approach_angle: Optional[float]
    closest_km: Optional[float]
    eta_min: Optional[float]
    approaching: bool
    cluster_radius_km: float = 0.0


@dataclass
class Nowcast:
    raining_now: bool = False
    rain_duration_min: Optional[float] = None
    rain_end_eta_min: Optional[float] = None
    next_rain_eta_min: Optional[float] = None
    gap_min: Optional[float] = None
    summary: str = ""
    lines: list = field(default_factory=list)


def assess(track, target_lat, target_lon, calib):
    c = track.cluster
    clat, clon = calib.image_pixel_to_latlon(c.cx, c.cy)
    dist = haversine_km(clat, clon, target_lat, target_lon)
    kmpp = calib.km_per_pixel(c.cx, c.cy)
    c_rad = math.sqrt(max(0.0, c.size * kmpp * kmpp) / math.pi)
    coslat = math.cos(math.radians((clat + target_lat) / 2))
    dx_km = (target_lon - clon) * 111.32 * coslat
    dy_km = (target_lat - clat) * 111.32
    bearing = math.degrees(math.atan2(dx_km, dy_km)) % 360
    speed = track.speed_kmh
    if speed < MIN_SPEED_KMH:
        return Approach(dist, bearing, None, None, None, None, False, c_rad)
    bm = math.degrees(math.atan2(track.vx_kmh, track.vy_kmh)) % 360
    ux, uy = dx_km / dist, dy_km / dist
    angle = math.degrees(math.acos(max(-1.0, min(1.0,
                (track.vx_kmh * ux + track.vy_kmh * uy) / speed))))
    Rx, Ry = -dx_km, -dy_km
    t_star = -(Rx * track.vx_kmh + Ry * track.vy_kmh) / (speed * speed)
    if t_star > 0:
        cax = Rx + track.vx_kmh * t_star
        cay = Ry + track.vy_kmh * t_star
        return Approach(dist, bearing, bm, angle,
                        math.hypot(cax, cay), t_star * 60.0, True, c_rad)
    return Approach(dist, bearing, bm, angle, dist, None, False, c_rad)


def refine_approaching(track, a, rt):
    if not a.approaching:
        return False
    if a.approach_angle is None or a.closest_km is None:
        return False
    if a.approach_angle > rt["only_approach_deg"]:
        return False
    r_eff = rt["approaching_radius_km"] + a.cluster_radius_km * rt["cluster_radius_weight"]
    return a.closest_km <= r_eff


def refine_threat(track, a, rt):
    if not refine_approaching(track, a, rt):
        return False
    if track.n_observations < rt["threat_min_duration"]:
        return False
    min_sri = float(rt.get("threat_min_sri_mmh", 0.0))
    if min_sri > 0.0 and track.cluster.max_mmh < min_sri:
        return False
    if a.approach_angle is None or a.approach_angle > rt["threat_angle_deg"]:
        return False
    r_eff = rt["threat_radius_km"] + a.cluster_radius_km * rt["cluster_radius_weight"]
    return a.closest_km is not None and a.closest_km <= r_eff


def _project_along(calib, pixels_yx, t_lat, t_lon, ux, uy):
    dx, dy = calib.latlon_offset_km(pixels_yx[:, 1].astype(float),
                                    pixels_yx[:, 0].astype(float), t_lat, t_lon)
    return dx * ux + dy * uy


def predict_end_min(calib, pixels, t_lat, t_lon, vx, vy):
    speed = math.hypot(vx, vy)
    if speed < MIN_SPEED_KMH or len(pixels) == 0:
        return None
    proj = _project_along(calib, pixels, t_lat, t_lon, vx / speed, vy / speed)
    mn = float(proj.min())
    return 0.0 if mn >= 0 else -mn / speed * 60.0


def predict_arrival_min(calib, pixels, t_lat, t_lon, vx, vy):
    speed = math.hypot(vx, vy)
    if speed < MIN_SPEED_KMH or len(pixels) == 0:
        return None
    proj = _project_along(calib, pixels, t_lat, t_lon, vx / speed, vy / speed)
    mx = float(proj.max())
    return 0.0 if mx >= 0 else -mx / speed * 60.0


def is_upwind(calib, cluster, t_lat, t_lon, vx, vy):
    speed = math.hypot(vx, vy)
    if speed < MIN_SPEED_KMH:
        return False
    clat, clon = calib.image_pixel_to_latlon(cluster.cx, cluster.cy)
    coslat = math.cos(math.radians(t_lat))
    dx = (clon - t_lon) * 111.32 * coslat
    dy = (clat - t_lat) * 111.32
    return (dx * vx / speed + dy * vy / speed) < 0


def _find_upwind_candidates(calib, tracks, approaches, label_array,
                            t_lat, t_lon, rt, exclude_label=None):
    candidates = []
    for i, tr in enumerate(tracks):
        if exclude_label is not None and tr.cluster.label == exclude_label:
            continue
        if tr.speed_kmh < MIN_SPEED_KMH:
            continue
        if not is_upwind(calib, tr.cluster, t_lat, t_lon, tr.vx_kmh, tr.vy_kmh):
            continue
        a = approaches[i]
        if not refine_approaching(tr, a, rt):
            continue
        px_arr = np.argwhere(label_array == tr.cluster.label)
        arr = predict_arrival_min(calib, px_arr, t_lat, t_lon, tr.vx_kmh, tr.vy_kmh)
        if arr is not None and arr > 0:
            candidates.append((arr, i, tr))
    candidates.sort(key=lambda x: x[0])
    return candidates


def compute_nowcast(calib, t_lat, t_lon, tracks, approaches, rain_history,
                    label_array, interval_min, rt):
    nc = Nowcast()
    nc.raining_now = rain_history[-1] if rain_history else False
    if nc.raining_now:
        streak = 0
        for r in reversed(rain_history):
            if r: streak += 1
            else: break
        nc.rain_duration_min = streak * interval_min
        px, py = calib.latlon_to_image_pixel(t_lat, t_lon)
        xi, yi = int(round(px)), int(round(py))
        cov_label = 0
        if 0 <= yi < label_array.shape[0] and 0 <= xi < label_array.shape[1]:
            cov_label = int(label_array[yi, xi])
        cov_tr = next((t for t in tracks if t.cluster.label == cov_label), None)
        if cov_tr and cov_tr.speed_kmh >= MIN_SPEED_KMH:
            px_arr = np.argwhere(label_array == cov_label)
            nc.rain_end_eta_min = predict_end_min(calib, px_arr, t_lat, t_lon,
                                                  cov_tr.vx_kmh, cov_tr.vy_kmh)
        candidates = _find_upwind_candidates(calib, tracks, approaches, label_array,
                                             t_lat, t_lon, rt, exclude_label=cov_label)
        if candidates:
            nc.next_rain_eta_min, ci, tr = candidates[0]
            if (nc.rain_end_eta_min is not None
                    and nc.next_rain_eta_min > nc.rain_end_eta_min):
                nc.gap_min = nc.next_rain_eta_min - nc.rain_end_eta_min
    else:
        nc.rain_duration_min = 0.0
        candidates = _find_upwind_candidates(calib, tracks, approaches, label_array,
                                             t_lat, t_lon, rt)
        if candidates:
            nc.next_rain_eta_min, ci, tr = candidates[0]
    if nc.raining_now:
        parts = [f"Raining {nc.rain_duration_min:.0f} min."]
        if nc.rain_end_eta_min is not None:
            parts.append(f"Ends ~{nc.rain_end_eta_min:.0f} min.")
        if nc.gap_min is not None:
            parts.append(f"Gap ~{nc.gap_min:.0f} min, next ~{nc.next_rain_eta_min:.0f} min.")
        elif nc.next_rain_eta_min is not None:
            parts.append(f"Next cell ~{nc.next_rain_eta_min:.0f} min (no gap).")
        nc.summary = " ".join(parts)
    else:
        nc.summary = (f"Dry. Next rain ~{nc.next_rain_eta_min:.0f} min."
                      if nc.next_rain_eta_min is not None else
                      "Dry. No approaching rain.")
    return nc


# ----------------------------------------------------------------------------
# Pljusak client + full weather parsing
# ----------------------------------------------------------------------------
def _parse_opt_float(row, idx, lo=None, hi=None):
    if len(row) <= idx:
        return None
    val = row[idx]
    if val is None:
        return None
    if isinstance(val, str):
        s = val.strip().replace(",", ".")
        if not s or s in ("-", "—", "null", "None", "NaN", "nan"):
            return None
        try:
            f = float(s)
        except ValueError:
            return None
    else:
        try:
            f = float(val)
        except (TypeError, ValueError):
            return None
    if not math.isfinite(f):
        return None
    if lo is not None and f < lo:
        return None
    if hi is not None and f > hi:
        return None
    return f


def _parse_wind_dir_deg(raw_dir):
    if raw_dir is None:
        return None, None
    s = str(raw_dir).strip().upper()
    if not s or s in ("-", "—", "TIHO", "CALM", "C", "VRB", "0"):
        return s if s and s != "-" else None, None
    if s in COMPASS_16_DEG:
        return s, COMPASS_16_DEG[s]
    try:
        deg = float(s)
        if math.isfinite(deg) and 0.0 <= deg <= 360.0:
            deg = deg % 360.0
            return bearing_label(deg), deg
    except ValueError:
        pass
    return s, None


@dataclass
class StationReading:
    slug: str
    name: str
    lat: float
    lon: float
    precip_1h: float
    time_str: str
    elevation_m: Optional[float] = None
    temp_c: Optional[float] = None
    temp_trend: Optional[float] = None
    pressure_hpa: Optional[float] = None
    pressure_trend: Optional[float] = None
    humidity_pct: Optional[float] = None
    wind_dir_str: Optional[str] = None
    wind_dir_deg: Optional[float] = None
    wind_speed_kmh: Optional[float] = None
    wind_gust_kmh: Optional[float] = None
    precip_month: Optional[float] = None
    precip_today: Optional[float] = None
    precip_24h: Optional[float] = None
    precip_6h: Optional[float] = None
    max_temp_c: Optional[float] = None
    min_temp_c: Optional[float] = None
    uv: Optional[float] = None
    temp_24h_c: Optional[float] = None


@dataclass
class StationAnalysis:
    reading: StationReading
    dist_km: float
    age_min: Optional[float]
    slope_mmh: Optional[float]
    trend: str
    rain_rate_mmh: Optional[float] = None


@dataclass
class LocalWeather:
    radius_km: float = 5.0
    station_count: int = 0
    nearest_station: Optional[str] = None
    nearest_dist_km: Optional[float] = None
    temp_c: Optional[float] = None
    temp_trend: Optional[float] = None
    humidity_pct: Optional[float] = None
    pressure_hpa: Optional[float] = None
    pressure_trend: Optional[float] = None
    wind_speed_kmh: Optional[float] = None
    wind_gust_kmh: Optional[float] = None
    wind_dir_deg: Optional[int] = None
    wind_dir_cardinal: Optional[str] = None
    precip_today_mm: Optional[float] = None
    precip_1h_mm: Optional[float] = None
    precip_6h_mm: Optional[float] = None
    precip_24h_mm: Optional[float] = None
    rain_rate_mmh: Optional[float] = None


class PljusakClient:
    def __init__(self, url=PLJUSAK_URL, timeout=30):
        self.url = url
        self.timeout = timeout
        self.history = {}
        self.last_fetch_utc = None
        self.last_readings = []
        self.last_error = None

    def fetch(self, now_utc):
        try:
            r = requests.get(self.url, headers=PLJUSAK_HEADERS, timeout=self.timeout)
            r.raise_for_status()
            m = re.search(r"var\s+podaci\s*=\s*(\[.*?\]);\s*</script>", r.text, re.S)
            if not m:
                self.last_error = "regex miss"
                return False
            rows = json.loads(m.group(1))
            readings = []
            for row in rows:
                if not row or row[0] == "kamera":
                    continue
                lat = _parse_opt_float(row, 2, -90.0, 90.0)
                lon = _parse_opt_float(row, 3, -180.0, 180.0)
                if lat is None or lon is None:
                    continue

                elev_m = _parse_opt_float(row, 4, -50.0, 5000.0)
                ts = str(row[10]).strip() if len(row) > 10 and row[10] is not None else ""
                temp_c = _parse_opt_float(row, 12, -55.0, 60.0)
                temp_trend = _parse_opt_float(row, 13, -30.0, 30.0)
                pressure_hpa = _parse_opt_float(row, 14, 930.0, 1070.0)
                pressure_trend = _parse_opt_float(row, 15, -30.0, 30.0)
                humidity_pct = _parse_opt_float(row, 16, 1.0, 100.0)

                raw_dir = row[17] if len(row) > 17 else None
                wind_dir_str, wind_dir_deg = _parse_wind_dir_deg(raw_dir)

                wind_ms = _parse_opt_float(row, 18, 0.0, 90.0)
                wind_speed_kmh = (wind_ms * 3.6) if wind_ms is not None else None

                precip_month = _parse_opt_float(row, 19, 0.0, 5000.0)
                precip_today = _parse_opt_float(row, 20, 0.0, 1500.0)
                precip_24h = _parse_opt_float(row, 21, 0.0, 1500.0)
                p1h_opt = _parse_opt_float(row, 22, 0.0, 500.0)
                precip_1h = p1h_opt if p1h_opt is not None else 0.0
                precip_6h = _parse_opt_float(row, 23, 0.0, 1000.0)

                max_temp_c = _parse_opt_float(row, 24, -55.0, 60.0)
                min_temp_c = _parse_opt_float(row, 25, -55.0, 60.0)

                gust_ms = _parse_opt_float(row, 26, 0.0, 110.0)
                wind_gust_kmh = (gust_ms * 3.6) if gust_ms is not None else None

                uv = _parse_opt_float(row, 27, 0.0, 25.0)
                temp_24h_c = _parse_opt_float(row, 28, -55.0, 60.0)
                slug = str(row[29]).strip() if len(row) > 29 and row[29] is not None else ""
                name = str(row[1]).strip() if len(row) > 1 and row[1] is not None else ""

                readings.append(StationReading(
                    slug=slug, name=name, lat=lat, lon=lon,
                    precip_1h=precip_1h, time_str=ts,
                    elevation_m=elev_m, temp_c=temp_c, temp_trend=temp_trend,
                    pressure_hpa=pressure_hpa, pressure_trend=pressure_trend,
                    humidity_pct=humidity_pct, wind_dir_str=wind_dir_str,
                    wind_dir_deg=wind_dir_deg, wind_speed_kmh=wind_speed_kmh,
                    wind_gust_kmh=wind_gust_kmh, precip_month=precip_month,
                    precip_today=precip_today, precip_24h=precip_24h,
                    precip_6h=precip_6h, max_temp_c=max_temp_c,
                    min_temp_c=min_temp_c, uv=uv, temp_24h_c=temp_24h_c,
                ))
            self.last_readings = readings
            self.last_fetch_utc = now_utc
            self.last_error = None
            for st in readings:
                h = self.history.setdefault(st.slug, [])
                h.append((now_utc, st.precip_1h, st.precip_today))
                if len(h) > 60:
                    del h[:len(h) - 60]
            return True
        except Exception as e:
            self.last_error = str(e)
            return False

    def analyze_nearby(self, t_lat, t_lon, radius_km, display_tz, max_age_min=15):
        out = []
        for st in self.last_readings:
            dist = haversine_km(st.lat, st.lon, t_lat, t_lon)
            if radius_km is not None and dist > radius_km:
                continue
            age = _station_age_min(st.time_str, self.last_fetch_utc, display_tz)
            slope, trend = self.slope_mmh(st.slug, window_min=12)
            rate = self.rain_rate_mmh(st.slug, window_min=12)
            out.append(StationAnalysis(st, dist, age, slope, trend, rain_rate_mmh=rate))
        out.sort(key=lambda a: a.dist_km)
        return out

    def get_all_stations(self, t_lat, t_lon, display_tz):
        return self.analyze_nearby(t_lat, t_lon, radius_km=None, display_tz=display_tz)

    def slope_mmh(self, slug, window_min=12):
        h = self.history.get(slug)
        if not h or len(h) < 2:
            return None, "insufficient"
        latest_ts, latest_mm, _ = h[-1]
        earliest = None
        for entry in reversed(h):
            if (latest_ts - entry[0]).total_seconds() >= window_min * 60:
                earliest = entry
                break
        if not earliest:
            return None, "insufficient"
        dt_h = (latest_ts - earliest[0]).total_seconds() / 3600.0
        if dt_h <= 0:
            return None, "insufficient"
        slope = (latest_mm - earliest[1]) / dt_h
        if slope > 50 or slope < -50:
            return None, "unknown"
        if slope > 0.5:
            return slope, "rising"
        if slope < -0.5:
            return slope, "falling"
        return slope, "stable"

    def rain_rate_mmh(self, slug, window_min=12):
        h = self.history.get(slug)
        if not h or len(h) < 2:
            return None
        latest_ts, latest_1h, latest_today = h[-1]
        earliest = None
        for entry in reversed(h[:-1]):
            if (latest_ts - entry[0]).total_seconds() >= window_min * 60:
                earliest = entry
                break
        if earliest is None:
            earliest = h[0]
        dt_h = (latest_ts - earliest[0]).total_seconds() / 3600.0
        if dt_h < (3.0 / 60.0):
            return None

        earliest_1h, earliest_today = earliest[1], earliest[2]
        if (latest_today is not None and earliest_today is not None
                and latest_today >= earliest_today):
            rate = (latest_today - earliest_today) / dt_h
        else:
            rate = max(0.0, (latest_1h - earliest_1h) / dt_h)

        if not math.isfinite(rate) or rate < 0.0 or rate > 250.0:
            return None
        return rate


def summarize_local_weather(stations, radius_km, max_age_min):
    lw = LocalWeather(radius_km=float(radius_km))
    fresh = [
        s for s in stations
        if s.dist_km <= radius_km
        and s.age_min is not None
        and s.age_min <= max_age_min
    ]
    lw.station_count = len(fresh)
    if not fresh:
        return lw

    nearest = min(fresh, key=lambda s: s.dist_km)
    lw.nearest_station = nearest.reading.name or nearest.reading.slug
    lw.nearest_dist_km = round(nearest.dist_km, 1)

    def _med(vals, digits=1):
        clean = [v for v in vals if v is not None and math.isfinite(v)]
        if not clean:
            return None
        m = float(np.median(clean))
        return round(m, digits) if digits is not None else m

    lw.temp_c = _med([s.reading.temp_c for s in fresh], 1)
    lw.temp_trend = _med([s.reading.temp_trend for s in fresh], 2)
    lw.humidity_pct = _med([s.reading.humidity_pct for s in fresh], 0)
    if lw.humidity_pct is not None:
        lw.humidity_pct = int(round(lw.humidity_pct))
    lw.pressure_hpa = _med([s.reading.pressure_hpa for s in fresh], 1)
    lw.pressure_trend = _med([s.reading.pressure_trend for s in fresh], 2)
    lw.wind_speed_kmh = _med([s.reading.wind_speed_kmh for s in fresh], 1)

    gusts = [s.reading.wind_gust_kmh for s in fresh if s.reading.wind_gust_kmh is not None]
    if gusts:
        lw.wind_gust_kmh = round(max(gusts), 1)

    sin_sum = 0.0
    cos_sum = 0.0
    dir_count = 0
    for s in fresh:
        deg = s.reading.wind_dir_deg
        if deg is None:
            continue
        spd = s.reading.wind_speed_kmh
        w = max(0.5, float(spd)) if spd is not None else 1.0
        rad = math.radians(deg)
        sin_sum += w * math.sin(rad)
        cos_sum += w * math.cos(rad)
        dir_count += 1
    if dir_count > 0 and math.hypot(sin_sum, cos_sum) > 1e-6:
        mean_deg = math.degrees(math.atan2(sin_sum, cos_sum)) % 360.0
        deg_int = int(round(mean_deg)) % 360
        lw.wind_dir_deg = deg_int
        lw.wind_dir_cardinal = bearing_label(deg_int)

    lw.precip_today_mm = _med([s.reading.precip_today for s in fresh], 1)
    lw.precip_1h_mm = _med([s.reading.precip_1h for s in fresh], 1)
    lw.precip_6h_mm = _med([s.reading.precip_6h for s in fresh], 1)
    lw.precip_24h_mm = _med([s.reading.precip_24h for s in fresh], 1)
    lw.rain_rate_mmh = _med([s.rain_rate_mmh for s in fresh], 1)
    return lw


def _station_age_min(time_str, fetch_utc, display_tz):
    if not time_str or not fetch_utc:
        return None
    try:
        parts = time_str.split(":")
        h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
    except (ValueError, IndexError):
        return None
    fetch_local = fetch_utc.astimezone(display_tz)
    dt = datetime(fetch_local.year, fetch_local.month, fetch_local.day,
                  h, m, s, tzinfo=display_tz)
    if dt > fetch_local + timedelta(minutes=2):
        dt -= timedelta(days=1)
    age = (fetch_local - dt).total_seconds() / 60.0
    return None if (age < 0 or age > 24 * 60) else age


# ----------------------------------------------------------------------------
# Combined analysis
# ----------------------------------------------------------------------------
@dataclass
class CombinedAnalysis:
    radar_raining: bool = False
    raining_now: bool = False
    confirmed: bool = False
    confidence: str = "dry"
    nearest_station: Optional[StationAnalysis] = None
    intensity_mmh: Optional[float] = None
    station_trend: str = "none"
    upstream_trend: str = "none"
    nearby_stations: list = field(default_factory=list)
    wet_stations: list = field(default_factory=list)
    station_count_wet: int = 0
    station_count_fresh: int = 0
    station_count_stale: int = 0
    wet_fraction: Optional[float] = None
    station_median_mm: Optional[float] = None


def combine_analysis(radar_raining, stations, tracks, approaches, calib,
                     t_lat, t_lon, confirm_radius, confirm_mm, confirm_age,
                     confirm_frac, confirm_min_count, rt):
    ca = CombinedAnalysis(radar_raining=radar_raining)
    fresh = [s for s in stations
             if s.dist_km <= confirm_radius
             and s.age_min is not None and s.age_min <= confirm_age]
    stale = [s for s in stations
             if s.dist_km <= confirm_radius
             and (s.age_min is None or s.age_min > confirm_age)]
    wet = [s for s in fresh if s.reading.precip_1h >= confirm_mm]
    ca.nearby_stations = fresh
    ca.wet_stations = wet
    ca.station_count_wet = len(wet)
    ca.station_count_fresh = len(fresh)
    ca.station_count_stale = len(stale)
    ca.wet_fraction = (len(wet) / len(fresh)) if fresh else None
    ca.nearest_station = min(wet, key=lambda s: s.dist_km) if wet else None
    rising_slopes = [s.slope_mmh for s in wet
                     if s.trend == "rising"
                     and s.slope_mmh is not None and s.slope_mmh > 0]
    if rising_slopes:
        ca.intensity_mmh = float(np.median(rising_slopes))
    elif wet:
        ca.intensity_mmh = 0.0
    if wet:
        ca.station_median_mm = float(np.median([s.reading.precip_1h for s in wet]))
    counts = {"rising": 0, "falling": 0, "stable": 0}
    for s in fresh:
        if s.trend in counts:
            counts[s.trend] += 1
    if any(counts.values()):
        ca.station_trend = max(counts, key=counts.get)
    else:
        ca.station_trend = "none"
    wet_frac = ca.wet_fraction or 0.0
    station_says_rain = (ca.station_count_fresh > 0
                         and ca.station_count_wet >= confirm_min_count
                         and wet_frac >= confirm_frac)
    ca.raining_now = station_says_rain or radar_raining
    if radar_raining and station_says_rain:
        ca.confirmed = True; ca.confidence = "confirmed"
    elif radar_raining:
        ca.confidence = "radar_only"
    elif station_says_rain:
        ca.confidence = "station_only"
    else:
        ca.confidence = "dry"
    mv = []
    for tr, a in zip(tracks, approaches):
        if tr.speed_kmh < MIN_SPEED_KMH:
            continue
        if not is_upwind(calib, tr.cluster, t_lat, t_lon, tr.vx_kmh, tr.vy_kmh):
            continue
        mv.append((tr.vx_kmh, tr.vy_kmh))
    if mv:
        vx, vy = max(mv, key=lambda v: math.hypot(*v))
        speed = math.hypot(vx, vy)
        ux, uy = vx / speed, vy / speed
        up = []
        for st in stations:
            if st.dist_km > 100:
                continue
            coslat = math.cos(math.radians(t_lat))
            dx = (st.reading.lon - t_lon) * 111.32 * coslat
            dy = (st.reading.lat - t_lat) * 111.32
            proj = dx * ux + dy * uy
            if -100 < proj < -5:
                up.append(st)
        counts2 = {"rising": 0, "falling": 0, "stable": 0}
        for st in up:
            if st.trend in counts2:
                counts2[st.trend] += 1
        if counts2["rising"] > max(counts2["falling"], counts2["stable"]):
            ca.upstream_trend = "rising"
        elif counts2["falling"] > max(counts2["rising"], counts2["stable"]):
            ca.upstream_trend = "falling"
        elif counts2["stable"] > 0 or counts2["rising"] > 0:
            ca.upstream_trend = "mixed"
        else:
            ca.upstream_trend = "unknown"
    return ca


# ----------------------------------------------------------------------------
# Satellite compositing + DHMZ SRI rendering
# ----------------------------------------------------------------------------
def _basemap_cache_path(data_dir, style, lat, lon, radius_km, view_rot, size_px):
    if not data_dir:
        return None
    safe_style = style.replace("/", "_")
    key = f"{safe_style}_{lat:.4f}_{lon:.4f}_{radius_km:.0f}_{view_rot:.1f}_{size_px}"
    return os.path.join(data_dir, "basemap_cache", key + ".png")


def fetch_basemap(style, token, lat, lon, radius_km, view_rot, size_px, data_dir):
    if not token or not style:
        return None
    cache_path = _basemap_cache_path(data_dir, style, lat, lon, radius_km, view_rot, size_px)
    if cache_path and os.path.exists(cache_path):
        try:
            return Image.open(cache_path).convert("RGB")
        except Exception:
            pass

    coslat = max(0.01, math.cos(math.radians(lat)))
    try:
        zoom = math.log2(size_px * 360.0 * 111.32 * coslat / (512.0 * 2 * radius_km))
    except ValueError:
        zoom = 8.0
    zoom = max(0.0, min(22.0, zoom))

    url = (f"https://api.mapbox.com/styles/v1/{style}/static/"
           f"{lon:.5f},{lat:.5f},{zoom:.2f},{view_rot:.1f},0/"
           f"{size_px}x{size_px}?access_token={token}")

    last_err = None
    for attempt in (1, 2):
        try:
            r = requests.get(url, timeout=(10, 30))
            r.raise_for_status()
            img = Image.open(io.BytesIO(r.content)).convert("RGB")
            break
        except Exception as e:
            last_err = e
            if attempt == 1:
                time.sleep(2)
                continue
            print(f"basemap: fetch failed after {attempt} attempts: {last_err}",
                  file=sys.stderr)
            return None

    if cache_path:
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            img.save(cache_path, format="PNG", optimize=True)
        except Exception as e:
            print(f"basemap: cache save failed: {e}", file=sys.stderr)
    return img


def _apply_basemap_color_control(pil_img, rt):
    sat = float(rt.get("basemap_saturation", 1.0))
    bri = float(rt.get("basemap_brightness", 1.0))
    con = float(rt.get("basemap_contrast", 1.0))
    if abs(sat - 1.0) > 1e-3:
        pil_img = ImageEnhance.Color(pil_img).enhance(sat)
    if abs(con - 1.0) > 1e-3:
        pil_img = ImageEnhance.Contrast(pil_img).enhance(con)
    if abs(bri - 1.0) > 1e-3:
        pil_img = ImageEnhance.Brightness(pil_img).enhance(bri)
    return pil_img


def _colorize_dhmz_sri(mask, mmh):
    H, W = mask.shape
    color = np.zeros((H, W, 3), dtype=np.uint8)
    if not mask.any():
        return color
    vals = np.clip(mmh, 0.2, 250.0)
    for lo, hi, rgb_col, _ in DHMZ_SRI_SCALE:
        sel = mask & (vals >= lo) & (vals < hi)
        if sel.any():
            color[sel] = rgb_col
    below = mask & (vals < DHMZ_SRI_SCALE[0][0])
    if below.any():
        color[below] = DHMZ_SRI_SCALE[0][2]
    return color


def _radar_to_grayscale(mask, mmh):
    H, W = mask.shape
    gray = np.zeros((H, W), dtype=np.uint8)
    if mask.any():
        clipped = np.clip(mmh, 0.1, 250.0)
        logmmh = np.log10(clipped)
        norm = np.clip((logmmh + 1.0) / 3.4, 0.0, 1.0)
        gray[mask] = (40.0 + norm[mask] * 180.0).astype(np.uint8)
    return gray


def _sample_radar_at_view(gray, radar_mask, radar_mmh, calib, target_lat, target_lon,
                          radius_km, view_rot, out_w, out_h):
    H, W = out_h, out_w
    xs = np.arange(W, dtype=np.float32) - W / 2.0
    ys = -(np.arange(H, dtype=np.float32) - H / 2.0)
    X, Y = np.meshgrid(xs, ys)
    kpp_x = 2.0 * radius_km / W
    kpp_y = 2.0 * radius_km / H
    dx_km = X * kpp_x
    dy_km = Y * kpp_y

    rot_rad = math.radians(view_rot)
    c, s = math.cos(rot_rad), math.sin(rot_rad)
    world_dx = dx_km * c + dy_km * s
    world_dy = -dx_km * s + dy_km * c

    lat = target_lat + world_dy / 111.32
    coslat = math.cos(math.radians(target_lat))
    lon = target_lon + world_dx / (111.32 * coslat)
    inv = calib._inv
    lat_c = calib.coef_lat[2]; lon_c = calib.coef_lon[2]
    radar_x = inv[0, 0] * (lat - lat_c) + inv[0, 1] * (lon - lon_c) + calib.map_region.x0
    radar_y = inv[1, 0] * (lat - lat_c) + inv[1, 1] * (lon - lon_c) + calib.map_region.y0
    rx = np.round(radar_x).astype(np.int32)
    ry = np.round(radar_y).astype(np.int32)
    valid = ((rx >= 0) & (rx < radar_mask.shape[1])
             & (ry >= 0) & (ry < radar_mask.shape[0]))
    gray_out = np.zeros((H, W), dtype=np.uint8)
    mmh_out = np.zeros((H, W), dtype=np.float32)
    mask_out = np.zeros((H, W), dtype=bool)
    gray_out[valid] = gray[ry[valid], rx[valid]]
    mmh_out[valid] = radar_mmh[ry[valid], rx[valid]]
    mask_out[valid] = radar_mask[ry[valid], rx[valid]]
    return gray_out, mask_out, mmh_out


def _latlon_to_overlay_px(calib, target_lat, target_lon, radius_km, size_px,
                           view_rot, lat, lon):
    coslat = math.cos(math.radians(target_lat))
    dx_km = (lon - target_lon) * 111.32 * coslat
    dy_km = (lat - target_lat) * 111.32

    rot_rad = math.radians(view_rot)
    c, s = math.cos(rot_rad), math.sin(rot_rad)
    rx_km = dx_km * c - dy_km * s
    ry_km = dx_km * s + dy_km * c

    kmpp = 2.0 * radius_km / size_px
    px = size_px / 2.0 + rx_km / kmpp
    py = size_px / 2.0 - ry_km / kmpp
    return px, py


def _draw_filled_circle(rgb, cx, cy, radius, color, alpha=0.55):
    r = int(radius)
    if r <= 0: return
    cx, cy = int(round(cx)), int(round(cy))
    H, W = rgb.shape[:2]
    x0, x1 = max(0, cx - r), min(W, cx + r + 1)
    y0, y1 = max(0, cy - r), min(H, cy + r + 1)
    if x0 >= x1 or y0 >= y1: return
    yy, xx = np.ogrid[y0:y1, x0:x1]
    m = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    if not m.any(): return
    if alpha >= 1.0:
        rgb[y0:y1, x0:x1][m] = color
    else:
        bg = rgb[y0:y1, x0:x1][m].astype(np.float32)
        fg = np.array(color, dtype=np.float32)
        rgb[y0:y1, x0:x1][m] = (bg * (1 - alpha) + fg * alpha).astype(np.uint8)


def _draw_circle_outline(rgb, cx, cy, radius, color, thickness=1):
    r = int(radius)
    if r <= 0: return
    cx, cy = int(round(cx)), int(round(cy))
    H, W = rgb.shape[:2]
    n = max(48, int(2 * math.pi * r))
    for k in range(n):
        ang = 2 * math.pi * k / n
        for d in range(thickness):
            px = int(cx + (r - d) * math.cos(ang))
            py = int(cy + (r - d) * math.sin(ang))
            if 0 <= px < W and 0 <= py < H:
                rgb[py, px] = color


def _draw_coast_overlay(rgb, calib, target_lat, target_lon, radius_km, view_rot, size_px):
    coast_color = (200, 200, 200)
    for (la1, lo1), (la2, lo2) in zip(ADRIATIC_COAST, ADRIATIC_COAST[1:]):
        steps = max(2, int(max(abs(la2 - la1), abs(lo2 - lo1)) / 0.03))
        prev_px = prev_py = None
        for k in range(steps + 1):
            t = k / steps
            la = la1 + (la2 - la1) * t
            lo = lo1 + (lo2 - lo1) * t
            px, py = _latlon_to_overlay_px(calib, target_lat, target_lon,
                                           radius_km, size_px, view_rot, la, lo)
            if 0 <= px < size_px and 0 <= py < size_px:
                if prev_px is not None:
                    dx = px - prev_px; dy = py - prev_py
                    n_line = int(max(1, math.hypot(dx, dy)))
                    for s in range(n_line + 1):
                        fp = s / n_line
                        lx = int(round(prev_px + dx * fp))
                        ly = int(round(prev_py + dy * fp))
                        if 0 <= lx < size_px and 0 <= ly < size_px:
                            rgb[ly, lx] = coast_color
                prev_px, prev_py = px, py
            else:
                prev_px = prev_py = None


def _draw_text_outlined(draw, x, y, text, fill, outline):
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0: continue
            draw.text((x + dx, y + dy), text, fill=outline)
    draw.text((x, y), text, fill=fill)


def _draw_arrow(draw, x0, y0, x1, y1, color, outline=(0, 0, 0)):
    dx = x1 - x0
    dy = y1 - y0
    length = math.hypot(dx, dy)
    if length < 1.5:
        draw.ellipse([x0 - 2.5, y0 - 2.5, x0 + 2.5, y0 + 2.5],
                     fill=color, outline=outline)
        return

    ux, uy = dx / length, dy / length
    px, py = -uy, ux

    head_len = min(11.0, max(5.0, length * 0.35))
    head_w = head_len * 0.55

    bx = x1 - ux * head_len
    by = y1 - uy * head_len

    draw.line([(x0, y0), (bx, by)], fill=outline, width=4)
    tip_out = (x1 + ux * 1.5, y1 + uy * 1.5)
    left_out = (bx - ux * 1.0 + px * (head_w + 1.5), by - uy * 1.0 + py * (head_w + 1.5))
    right_out = (bx - ux * 1.0 - px * (head_w + 1.5), by - uy * 1.0 - py * (head_w + 1.5))
    draw.polygon([tip_out, left_out, right_out], fill=outline)

    draw.line([(x0, y0), (bx, by)], fill=color, width=2)
    tip = (x1, y1)
    left = (bx + px * head_w, by + py * head_w)
    right = (bx - px * head_w, by - py * head_w)
    draw.polygon([tip, left, right], fill=color)
    draw.ellipse([x0 - 2.5, y0 - 2.5, x0 + 2.5, y0 + 2.5],
                 fill=color, outline=outline)


def _draw_compass(img, cx, cy, view_rot, size=24):
    draw = ImageDraw.Draw(img)
    draw.ellipse([cx - size, cy - size, cx + size, cy + size],
                 outline=(255, 255, 255), width=2)
    angle_rad = math.radians(view_rot)
    nx = cx - size * 0.75 * math.sin(angle_rad)
    ny = cy - size * 0.75 * math.cos(angle_rad)
    draw.line([cx, cy, nx, ny], fill=(255, 255, 255), width=2)
    draw.ellipse([nx - 2, ny - 2, nx + 2, ny + 2], fill=(255, 255, 255))
    _draw_text_outlined(draw, nx + 3, ny - 5, "N", (255, 255, 255), (0, 0, 0))


def _draw_scale_bar(img, cx, cy, radius_km, size_px, view_rot):
    draw = ImageDraw.Draw(img)
    kmpp = 2 * radius_km / size_px
    bar_km = 50 if radius_km >= 200 else 25 if radius_km >= 100 else 10
    bar_px = bar_km / kmpp
    x0 = cx
    y0 = cy
    draw.line([x0, y0, x0 + bar_px, y0], fill=(255, 255, 255), width=2)
    draw.line([x0, y0 - 4, x0, y0 + 4], fill=(255, 255, 255), width=2)
    draw.line([x0 + bar_px, y0 - 4, x0 + bar_px, y0 + 4], fill=(255, 255, 255), width=2)
    _draw_text_outlined(draw, x0 + bar_px + 4, y0 - 6, f"{bar_km} km",
                        (255, 255, 255), (0, 0, 0))


def _draw_legend(pil_img, confirm_mm, has_shadow=False):
    W, H = pil_img.size
    box_w, box_h = 238, 74
    x0 = W - box_w - 10
    y0 = H - box_h - 10
    if x0 < 0 or y0 < 0:
        return

    overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)
    odraw.rounded_rectangle(
        [x0, y0, x0 + box_w, y0 + box_h],
        radius=5,
        fill=(12, 16, 22, 185),
        outline=(110, 120, 135, 200),
        width=1,
    )
    pil_rgba = pil_img.convert("RGBA")
    pil_rgba = Image.alpha_composite(pil_rgba, overlay)
    pil_img.paste(pil_rgba.convert("RGB"))

    draw = ImageDraw.Draw(pil_img)

    r1_y = y0 + 11
    draw.ellipse([x0 + 7, r1_y - 4, x0 + 15, r1_y + 4],
                 fill=(255, 255, 255), outline=(0, 0, 0))
    draw.text((x0 + 18, r1_y - 6), "Dry", fill=(235, 235, 235))

    rx = x0 + 44
    draw.ellipse([rx, r1_y - 4, rx + 8, r1_y + 4],
                 fill=(235, 35, 35), outline=(0, 0, 0))
    draw.text((rx + 11, r1_y - 6), f"≥{confirm_mm:g}mm", fill=(255, 190, 190))

    ax = x0 + 102
    _draw_arrow(draw, ax, r1_y, ax + 14, r1_y, (110, 235, 255))
    draw.text((ax + 17, r1_y - 6), "1h", fill=(195, 240, 255))
    if has_shadow:
        sx = ax + 36
        _draw_arrow(draw, sx, r1_y, sx + 14, r1_y, (220, 90, 255))
        draw.text((sx + 17, r1_y - 6), "Shadow", fill=(235, 175, 255))

    draw.text((x0 + 8, y0 + 25), "DHMZ SRI (mm/h):", fill=(200, 215, 230))
    bar_x0 = x0 + 8
    bar_y0 = y0 + 40
    bar_w = box_w - 16
    bar_h = 10
    n_bins = len(DHMZ_SRI_SCALE)
    step_w = bar_w / n_bins
    for i, (_, _, col, _) in enumerate(DHMZ_SRI_SCALE):
        bx0 = int(round(bar_x0 + i * step_w))
        bx1 = int(round(bar_x0 + (i + 1) * step_w))
        draw.rectangle([bx0, bar_y0, bx1, bar_y0 + bar_h], fill=col)
    draw.rectangle([bar_x0, bar_y0, bar_x0 + bar_w, bar_y0 + bar_h],
                   outline=(160, 170, 180))

    ticks = [(0, "0.2"), (2, "1"), (4, "3"), (7, "10"), (10, "30"), (13, "70+")]
    for idx, lbl in ticks:
        tx = int(round(bar_x0 + idx * step_w))
        draw.text((tx, bar_y0 + bar_h + 2), lbl, fill=(220, 220, 220))


def compose_map(radar_mask, radar_mmh, gray, calib, target_lat, target_lon,
                radius_km, view_rot, size_px, basemap_img, basemap_kind,
                tracks, approaches, approaching_flags, threat_flags,
                stations, show_cities, intensity_mode, sri_thresholds, rt,
                confirm_mm=0.2, shadow_tracks=None):
    gray_out, mask_out, mmh_out = _sample_radar_at_view(
        gray, radar_mask, radar_mmh, calib, target_lat, target_lon,
        radius_km, view_rot, size_px, size_px)

    if basemap_img is not None:
        try:
            base_pil = basemap_img.convert("RGB")
            if base_pil.size != (size_px, size_px):
                base_pil = base_pil.resize((size_px, size_px), Image.LANCZOS)
            base_pil = _apply_basemap_color_control(base_pil, rt)
            rgb = np.array(base_pil, dtype=np.uint8).copy()
        except Exception:
            rgb = np.stack([gray_out, gray_out, gray_out], axis=-1)
            basemap_kind = "greyscale"
    else:
        rgb = np.stack([gray_out, gray_out, gray_out], axis=-1)

    if mask_out.any():
        color = _colorize_dhmz_sri(mask_out, mmh_out)
        if basemap_kind == "mapbox":
            alpha = 0.72
            rgb[mask_out] = (rgb[mask_out].astype(np.float32) * (1 - alpha)
                             + color[mask_out].astype(np.float32) * alpha).astype(np.uint8)
        else:
            rgb[mask_out] = color[mask_out]

    if basemap_kind != "mapbox":
        _draw_coast_overlay(rgb, calib, target_lat, target_lon,
                            radius_km, view_rot, size_px)

    tpx, tpy = _latlon_to_overlay_px(calib, target_lat, target_lon,
                                     radius_km, size_px, view_rot, target_lat, target_lon)
    _draw_circle_outline(rgb, tpx, tpy, 6, (255, 255, 255), thickness=1)

    for st in stations:
        px, py = _latlon_to_overlay_px(calib, target_lat, target_lon,
                                       radius_km, size_px, view_rot,
                                       st.reading.lat, st.reading.lon)
        if 0 <= px < size_px and 0 <= py < size_px:
            is_raining = st.reading.precip_1h >= confirm_mm
            if is_raining:
                color = (235, 35, 35)
                r = 5
            else:
                color = (255, 255, 255)
                r = 4
            _draw_filled_circle(rgb, px, py, r, color, alpha=0.95)
            _draw_circle_outline(rgb, px, py, r, (0, 0, 0), thickness=1)

    for i, (tr, a, tf) in enumerate(zip(tracks, approaches, threat_flags)):
        if not approaching_flags[i]:
            continue
        clat, clon = calib.image_pixel_to_latlon(tr.cluster.cx, tr.cluster.cy)
        px, py = _latlon_to_overlay_px(calib, target_lat, target_lon,
                                       radius_km, size_px, view_rot, clat, clon)
        if 0 <= px < size_px and 0 <= py < size_px:
            pix_r = max(8.0, a.cluster_radius_km / (2 * radius_km / size_px))
            color = (220, 60, 60) if tf else (230, 140, 40)
            _draw_circle_outline(rgb, px, py, pix_r, color, thickness=2)

    pil = Image.fromarray(rgb.astype(np.uint8))
    draw = ImageDraw.Draw(pil)

    has_sh_vec = False
    if shadow_tracks:
        for strk in shadow_tracks:
            if strk.speed_kmh < MIN_SPEED_KMH:
                continue
            clat, clon = calib.image_pixel_to_latlon(strk.cluster.cx, strk.cluster.cy)
            px0, py0 = _latlon_to_overlay_px(calib, target_lat, target_lon,
                                             radius_km, size_px, view_rot, clat, clon)
            coslat = max(0.01, math.cos(math.radians(clat)))
            lat_1h = clat + strk.vy_kmh / 111.32
            lon_1h = clon + strk.vx_kmh / (111.32 * coslat)
            px1, py1 = _latlon_to_overlay_px(calib, target_lat, target_lon,
                                             radius_km, size_px, view_rot, lat_1h, lon_1h)
            if ((0 <= px0 < size_px and 0 <= py0 < size_px)
                    or (0 <= px1 < size_px and 0 <= py1 < size_px)):
                _draw_arrow(draw, px0, py0, px1, py1, (220, 90, 255))
                has_sh_vec = True

    for i, (tr, a, af, tf) in enumerate(zip(tracks, approaches, approaching_flags, threat_flags)):
        if tr.speed_kmh < MIN_SPEED_KMH:
            continue
        clat, clon = calib.image_pixel_to_latlon(tr.cluster.cx, tr.cluster.cy)
        px0, py0 = _latlon_to_overlay_px(calib, target_lat, target_lon,
                                         radius_km, size_px, view_rot, clat, clon)
        coslat = max(0.01, math.cos(math.radians(clat)))
        lat_1h = clat + tr.vy_kmh / 111.32
        lon_1h = clon + tr.vx_kmh / (111.32 * coslat)
        px1, py1 = _latlon_to_overlay_px(calib, target_lat, target_lon,
                                         radius_km, size_px, view_rot, lat_1h, lon_1h)
        if ((0 <= px0 < size_px and 0 <= py0 < size_px)
                or (0 <= px1 < size_px and 0 <= py1 < size_px)):
            if tf:
                arrow_color = (245, 55, 55)
            elif af:
                arrow_color = (255, 175, 40)
            else:
                arrow_color = (110, 235, 255)
            _draw_arrow(draw, px0, py0, px1, py1, arrow_color)

    if show_cities:
        for label, name, clat, clon in CITIES:
            px, py = _latlon_to_overlay_px(calib, target_lat, target_lon,
                                           radius_km, size_px, view_rot, clat, clon)
            if 0 <= px < size_px and 0 <= py < size_px:
                _draw_text_outlined(draw, px + 3, py - 6, label,
                                    (255, 255, 255), (0, 0, 0))

    _draw_compass(pil, size_px - 34, 34, view_rot)
    _draw_scale_bar(pil, 14, size_px - 18, radius_km, size_px, view_rot)
    _draw_legend(pil, confirm_mm, has_shadow=has_sh_vec)
    return pil


def render_composed_png(img, calib, target_lat, target_lon, radius_km,
                        view_rot, size_px, basemap_img, basemap_kind,
                        tracks, approaches, approaching_flags, threat_flags,
                        stations, show_cities,
                        intensity_mode, sri_legend, legend_mask,
                        sri_thresholds, rt, confirm_mm=0.2, header_text=None,
                        shadow_tracks=None):
    try:
        img.seek(max(0, getattr(img, "n_frames", 1) - 1))
    except Exception:
        pass
    frame = img.convert("RGB")
    mask, mmh = analyze_frame(frame, calib.map_region, intensity_mode,
                              sri_legend, legend_mask)
    gray = _radar_to_grayscale(mask, mmh)
    pil = compose_map(mask, mmh, gray, calib, target_lat, target_lon,
                      radius_km, view_rot, size_px, basemap_img, basemap_kind,
                      tracks, approaches, approaching_flags, threat_flags,
                      stations, show_cities,
                      intensity_mode, sri_thresholds, rt, confirm_mm=confirm_mm,
                      shadow_tracks=shadow_tracks)
    if header_text:
        draw = ImageDraw.Draw(pil)
        _draw_text_outlined(draw, 10, 8, header_text, (255, 220, 60), (0, 0, 0))
    buf = io.BytesIO()
    pil.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def render_composed_gif(img, calib, target_lat, target_lon, radius_km,
                        view_rot, size_px, basemap_img, basemap_kind,
                        tracks, tracks_by_id, split_frames_by_id,
                        frame_interval_min, approaches, approaching_flags,
                        threat_flags, stations, show_cities,
                        intensity_mode, sri_legend, legend_mask,
                        sri_thresholds, rt, confirm_mm=0.2,
                        n_unique_frames=None,
                        frame_duration_ms=400, max_frames=30):
    nf = n_unique_frames if n_unique_frames is not None else getattr(img, "n_frames", 1)
    start = max(0, nf - max_frames)
    split_frames_by_id = split_frames_by_id or {}
    base_win = float(rt.get("velocity_window_min", VELOCITY_WINDOW_MIN))
    adaptive_win = bool(rt.get("velocity_adaptive", True))
    out = []
    try:
        for fidx in range(start, nf):
            try:
                img.seek(fidx)
            except Exception:
                continue
            frame = img.convert("RGB")
            mask, mmh = analyze_frame(frame, calib.map_region, intensity_mode,
                                      sri_legend, legend_mask)
            gray = _radar_to_grayscale(mask, mmh)

            frame_tracks = []
            frame_approaches = []
            frame_appr_flags = []
            frame_threat_flags = []
            min_obs_needed = min(rt["min_duration_frames"], fidx + 1)

            for cid, obs in tracks_by_id.items():
                obs_up_to = [o for o in obs if o[0] <= fidx]
                if not obs_up_to or obs_up_to[-1][0] != fidx:
                    continue
                cluster_here = obs_up_to[-1][1]
                if cluster_here.size < rt["min_size_px"]:
                    continue
                if len(obs_up_to) < min_obs_needed:
                    continue
                vx_kmh, vy_kmh = _estimate_velocity_kmh(
                    obs_up_to, calib, frame_interval_min,
                    split_frames_by_id.get(cid), base_win, adaptive_win)
                tr_copy = Track(
                    cluster=PixelCluster(
                        cluster_here.cx, cluster_here.cy,
                        cluster_here.size, cluster_here.label,
                        cluster_here.max_mmh, cluster_here.mean_mmh),
                    cid=cid,
                    vx_kmh=vx_kmh,
                    vy_kmh=vy_kmh,
                    n_observations=len(obs_up_to),
                    span_frames=obs_up_to[-1][0] - obs_up_to[0][0])
                a_here = assess(tr_copy, target_lat, target_lon, calib)
                frame_tracks.append(tr_copy)
                frame_approaches.append(a_here)
                frame_appr_flags.append(refine_approaching(tr_copy, a_here, rt))
                frame_threat_flags.append(refine_threat(tr_copy, a_here, rt))

            pil = compose_map(mask, mmh, gray, calib, target_lat, target_lon,
                              radius_km, view_rot, size_px, basemap_img, basemap_kind,
                              frame_tracks, frame_approaches,
                              frame_appr_flags, frame_threat_flags,
                              stations, show_cities,
                              intensity_mode, sri_thresholds, rt,
                              confirm_mm=confirm_mm)
            out.append(pil)
    finally:
        try: img.seek(0)
        except Exception: pass
    if not out:
        return None
    buf = io.BytesIO()
    out[0].save(buf, format="GIF", save_all=True, append_images=out[1:],
                duration=frame_duration_ms, loop=0, optimize=True)
    return buf.getvalue()


# ----------------------------------------------------------------------------
# Parallel Live Shadow Runner (runs on every cycle without applying to HA/live)
# ----------------------------------------------------------------------------
def run_parallel_shadow_cycle(frame_masks_mmh, calib, target_lat, target_lon,
                              eff_interval, now_utc, ca_raining_now,
                              prev_rain_state, acc_store, profile_store,
                              replay_engine, rt):
    if not rt.get("shadow_enabled", True) or not profile_store:
        return {"enabled": False}

    sh_name = rt.get("shadow_profile") or ""
    sh_prof = profile_store.get(sh_name) if sh_name else None
    if not sh_prof:
        sh_prof = profile_store.best_profile()
    if not sh_prof:
        return {"enabled": False, "note": "no shadow profile available"}

    sh_name = sh_prof["name"]
    sh_rt = dict(rt)
    sh_rt.update(sh_prof.get("params") or {})

    (_, _, sh_raw_tracks, n_frames, sh_rain_history,
     sh_label_array, sh_tracks_by_id, _, sh_cluster_hits) = track_frame_sequence(
        frame_masks_mmh, calib, eff_interval, target_lat, target_lon, rt=sh_rt)

    sh_tracks = [
        t for t in sh_raw_tracks
        if t.n_observations >= min(sh_rt["min_duration_frames"], n_frames)
        and t.cluster.size >= sh_rt["min_size_px"]
    ]
    sh_approaches = [assess(t, target_lat, target_lon, calib) for t in sh_tracks]
    sh_appr_flags = [refine_approaching(t, a, sh_rt) for t, a in zip(sh_tracks, sh_approaches)]
    sh_threat_flags = [refine_threat(t, a, sh_rt) for t, a in zip(sh_tracks, sh_approaches)]
    sh_nc = compute_nowcast(calib, target_lat, target_lon, sh_tracks, sh_approaches,
                            sh_rain_history, sh_label_array, eff_interval, sh_rt)

    sh_appr_items = [(tr, a) for tr, a, af in zip(sh_tracks, sh_approaches, sh_appr_flags) if af]
    sh_threat_items = [(tr, a) for tr, a, tf in zip(sh_tracks, sh_approaches, sh_threat_flags) if tf]
    sh_appr_items.sort(key=lambda x: x[1].eta_min if x[1].eta_min is not None else 9999)
    sh_threat_items.sort(key=lambda x: x[1].eta_min if x[1].eta_min is not None else 9999)

    shadow_tag = f"shadow:{sh_name}"
    if acc_store and acc_store.enabled:
        now_iso = now_utc.isoformat()
        sh_snap = {k: sh_rt[k] for k in _TUNABLE_KEYS if k in sh_rt}
        if ca_raining_now and not prev_rain_state:
            acc_store.observe(now_iso, "rain_start", key="target", tag=shadow_tag)
        if not ca_raining_now and prev_rain_state:
            acc_store.observe(now_iso, "rain_end", key="target", tag=shadow_tag)
        if not ca_raining_now and sh_nc.next_rain_eta_min is not None:
            acc_store.predict(now_iso, "rain_start", sh_nc.next_rain_eta_min,
                              key="target", tag=shadow_tag, settings=sh_snap,
                              horizon_min=sh_nc.next_rain_eta_min)
        if ca_raining_now and sh_nc.rain_end_eta_min is not None:
            acc_store.predict(now_iso, "rain_end", sh_nc.rain_end_eta_min,
                              key="target", tag=shadow_tag, settings=sh_snap,
                              horizon_min=sh_nc.rain_end_eta_min)
        for cid, hit_frame in sh_cluster_hits.items():
            obs = sh_tracks_by_id.get(cid, [])
            first_f = obs[0][0] if obs else 0
            key = f"sh:{first_f}:{cid}"
            if key in acc_store.observed_cluster_arrivals:
                continue
            acc_store.observed_cluster_arrivals.add(key)
            acc_store.observe(now_iso, "cluster_arrive", key=key, tag=shadow_tag,
                              extra={"cid": cid, "hit_frame": hit_frame})
        for tr, a, af, tf in zip(sh_tracks, sh_approaches, sh_appr_flags, sh_threat_flags):
            if not af or a.eta_min is None:
                continue
            obs = sh_tracks_by_id.get(tr.cid, [])
            first_f = obs[0][0] if obs else 0
            key = f"sh:{first_f}:{tr.cid}"
            acc_store.predict(now_iso, "cluster_arrive", a.eta_min,
                              key=key, tag=shadow_tag, settings=sh_snap, horizon_min=a.eta_min)
            if tf:
                acc_store.predict(now_iso, "threat", a.eta_min,
                                  key=key, tag=shadow_tag, settings=sh_snap, horizon_min=a.eta_min)

    live_shadow_stats = None
    live_champ_stats = None
    if acc_store and acc_store.enabled:
        live_shadow_stats = acc_store.stats(tag=shadow_tag)
        live_champ_stats = acc_store.stats(tag=rt.get("settings_tag", ""))

    return {
        "enabled": True,
        "profile_name": sh_name,
        "stage": sh_prof.get("stage", "initial"),
        "params": sh_prof.get("params") or {},
        "offline_metrics": sh_prof.get("metrics"),
        "autotune_enabled": bool(rt.get("shadow_autotune_enabled", True)),
        "autotune_hours": float(rt.get("shadow_autotune_hours", 6.0)),
        "autotune_status": replay_engine.last_autotune_note if replay_engine else "idle",
        "last_autotune_utc": replay_engine.last_autotune_utc if replay_engine else None,
        "live": {
            "approaching_count": len(sh_appr_items),
            "threat_count": len(sh_threat_items),
            "next_rain_eta_min": round(sh_nc.next_rain_eta_min, 0) if sh_nc.next_rain_eta_min is not None else None,
            "rain_end_eta_min": round(sh_nc.rain_end_eta_min, 0) if sh_nc.rain_end_eta_min is not None else None,
            "summary": sh_nc.summary,
            "threats": [
                {"label": cluster_label(i), "size_px": tr.cluster.size,
                 "max_mmh": round(tr.cluster.max_mmh, 1),
                 "miss_km": round(a.closest_km, 1) if a.closest_km is not None else None,
                 "eta_min": round(a.eta_min, 0) if a.eta_min is not None else None}
                for i, (tr, a) in enumerate(sh_threat_items)
            ],
        },
        "live_accuracy": {
            "champion": live_champ_stats.get("by_kind") if live_champ_stats else {},
            "challenger": live_shadow_stats.get("by_kind") if live_shadow_stats else {},
            "challenger_scores": live_shadow_stats.get("total_scores", 0) if live_shadow_stats else 0,
        },
        "tracks": sh_tracks,
    }


# ----------------------------------------------------------------------------
# HTTP server (Live Dashboard + Separate Offline Replay & Shadow Tuner GUI)
# ----------------------------------------------------------------------------
_INDEX_HTML = """<!DOCTYPE html><html><head><title>DHMZ Radar Tracker</title>
<style>
body{background:#111;color:#ddd;font-family:-apple-system,BlinkMacSystemFont,sans-serif;margin:0;padding:20px}
h1{color:#fff;margin-top:0;display:flex;align-items:center;gap:12px;flex-wrap:wrap}.layout{display:flex;gap:20px;flex-wrap:wrap}
.left{flex:1 1 470px;max-width:580px}.right{flex:2 1 600px}
.section{background:#1a1a1a;border:1px solid #333;border-radius:4px;padding:12px;margin-bottom:12px}
.section.shadow-card{border-color:#53357a;background:#18151f}
.section h3{margin:0 0 8px 0;font-size:13px;text-transform:uppercase;color:#5cf;letter-spacing:0.5px;display:flex;justify-content:space-between;align-items:center}
.section.shadow-card h3{color:#c9f}
.section h4{margin:8px 0 4px 0;font-size:12px;color:#5cf}
.field{display:flex;align-items:center;margin-bottom:6px;font-size:13px}
.field label{flex:1;color:#ccc}.field input[type=number]{width:90px;background:#0a0a0a;color:#ddd;border:1px solid #333;padding:4px;border-radius:3px}
.field input[type=text]{flex:0 0 160px;background:#0a0a0a;color:#ddd;border:1px solid #333;padding:4px;border-radius:3px}
.field select{width:130px;background:#0a0a0a;color:#ddd;border:1px solid #333;padding:4px;border-radius:3px}
.field input[type=checkbox]{accent-color:#5cf;width:18px;height:18px}
.actions{margin-top:8px;display:flex;gap:8px;flex-wrap:wrap}
button{background:#2a5c8a;color:#fff;border:0;padding:7px 12px;border-radius:3px;cursor:pointer;font-size:12px}
button:hover{background:#3a7cba}button.secondary{background:#444}
button.secondary:hover{background:#555}button.danger{background:#8a2a2a}
button.danger:hover{background:#ba3a3a}button.purple{background:#663399}
button.purple:hover{background:#7d42b8}button.green{background:#246b42}
button.green:hover{background:#2f8a55}
.status{background:#1a1a1a;border:1px solid #333;border-radius:4px;padding:12px;margin-bottom:12px;font-size:13px}
.status .row{display:flex;margin-bottom:4px}.status .label{color:#888;flex:0 0 160px}
.wx-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}
.wx-box{background:#121519;border:1px solid #282c34;border-radius:4px;padding:7px 8px}
.wx-k{font-size:11px;color:#888;text-transform:uppercase}
.wx-v{font-size:15px;font-weight:600;color:#fff;margin-top:2px;font-variant-numeric:tabular-nums}
.wx-sub{font-size:11px;color:#9aa}
.threat-item{color:#f66;font-weight:bold}
table.acc{width:100%;border-collapse:collapse;font-size:12px}
table.acc th,table.acc td{padding:4px 6px;text-align:left;border-bottom:1px solid #252830}
table.acc th{color:#5cf;font-weight:normal;text-transform:uppercase;font-size:11px}
table.acc td.num{text-align:right;font-variant-numeric:tabular-nums}
.better{color:#5e8;font-weight:600}.worse{color:#f77}
img{max-width:100%;border:1px solid #333;display:block;margin-bottom:12px}
a{color:#5cf}.hint{color:#778;font-size:11px;margin-top:4px}
.cfgpath{font-family:monospace;color:#7c7}
.badge{background:#244866;color:#9df;padding:3px 8px;border-radius:3px;font-size:12px;text-decoration:none}
.pill-champ{background:#1b3b4b;color:#6ef;padding:1px 6px;border-radius:3px;font-size:11px}
.pill-shad{background:#3a2254;color:#d8f;padding:1px 6px;border-radius:3px;font-size:11px}
</style></head><body>
<h1>DHMZ Radar Tracker <a class="badge" href="/replay">Offline Replay & 2-Stage Tuner GUI →</a></h1>
<p><a href="/threat.gif">/threat.gif</a> · <a href="/threat.png">/threat.png</a> · <a href="/status">/status</a> · <a href="/settings">/settings</a> · <a href="/config">/config</a> · <a href="/archive">/archive</a> · <a href="/profiles">/profiles</a> · <a href="/accuracy">/accuracy</a></p>
<div class="layout">
<div class="left">
<div class="status" id="status">Loading...</div>

<div class="section shadow-card">
  <h3><span>Live Parallel Shadow Challenger</span><span id="sh_badge" class="pill-shad">—</span></h3>
  <div id="shadow_box">Loading parallel shadow comparison...</div>
  <div class="actions" style="margin-top:8px">
    <select id="sh_picker" onchange="selectShadowProfile()" style="flex:1"></select>
    <button class="purple" onclick="triggerTuneNow()">Auto-Tune Now</button>
    <button class="green" onclick="promoteActiveShadow()">Promote → Live</button>
  </div>
</div>

<div class="section"><h3>Local Weather (Pljusak)</h3>
<div id="local_wx">Loading...</div>
</div>
<div class="section"><h3>Storm Frame Archive</h3>
<div id="archive_box">Loading...</div>
</div>
<div class="section"><h3>Accuracy (Live Outcomes)</h3>
<div id="accuracy">Loading...</div>
<div class="actions" style="margin-top:8px">
<button class="secondary" onclick="replayAccuracy()">Replay scores</button>
<button class="danger" onclick="resetAccuracy()">Reset scores</button>
</div></div>
<div class="section"><h3>Sensors (HA)</h3>
<div id="sensors">Loading...</div>
</div>
<div class="section"><h3>Runtime settings (hot)</h3>
<div class="field"><label>Velocity window (min)</label><input type="number" id="velocity_window_min" min="10" max="90" step="5"></div>
<div class="field"><label>Adaptive velocity window</label><input type="checkbox" id="velocity_adaptive"></div>
<div class="field"><label>Base approach radius (km)</label><input type="number" id="approaching_radius_km" min="1" max="500" step="5"></div>
<div class="field"><label>Cluster size weight</label><input type="number" id="cluster_radius_weight" min="0" max="5" step="0.1"></div>
<div class="field"><label>Hard angle cap (°)</label><input type="number" id="only_approach_deg" min="0" max="180" step="5"></div>
<div class="field"><label>Threat radius (km)</label><input type="number" id="threat_radius_km" min="1" max="500" step="5"></div>
<div class="field"><label>Threat angle (°)</label><input type="number" id="threat_angle_deg" min="0" max="180" step="5"></div>
<div class="field"><label>Threat min duration (fr)</label><input type="number" id="threat_min_duration" min="1" max="50" step="1"></div>
<div class="field"><label>Threat min SRI (mm/h)</label><input type="number" id="threat_min_sri_mmh" min="0" max="500" step="0.5"></div>
<div class="field"><label>Parallel shadow runner</label><input type="checkbox" id="shadow_enabled"></div>
<div class="field"><label>Show shadow vectors on map</label><input type="checkbox" id="shadow_overlay_live"></div>
<div class="field"><label>Background auto-tuner</label><input type="checkbox" id="shadow_autotune_enabled"></div>
<div class="field"><label>Auto-tune interval (hours)</label><input type="number" id="shadow_autotune_hours" min="0.5" max="168" step="0.5"></div>
<div class="field"><label>Map radius (km)</label><input type="number" id="radius_km" min="20" max="1000" step="10"></div>
<div class="field"><label>Local weather radius (km)</label><input type="number" id="weather_radius_km" min="0.5" max="100" step="0.5"></div>
<div class="field"><label>Archive storm frames</label><input type="checkbox" id="archive_enabled"></div>
<div class="field"><label>Archive radius (km)</label><input type="number" id="archive_radius_km" min="20" max="500" step="10"></div>
<div class="field"><label>Archive min precip (px)</label><input type="number" id="archive_min_pixels" min="10" max="5000" step="10"></div>
<div class="field"><label>Min cell size (px)</label><input type="number" id="min_size_px" min="5" max="5000" step="5"></div>
<div class="field"><label>Min duration (fr)</label><input type="number" id="min_duration_frames" min="1" max="50" step="1"></div>
<div class="field"><label>View rotation: auto (coast)</label><input type="checkbox" id="view_rotation_auto"></div>
<div class="field"><label>View rotation (° CW, up)</label><input type="number" id="view_rotation" min="0" max="360" step="5"></div>
<div class="field"><label>Basemap</label>
<select id="basemap">
<option value="auto">auto</option>
<option value="mapbox">mapbox</option>
<option value="greyscale">greyscale</option>
<option value="none">none</option>
</select></div>
<div class="field"><label>Basemap saturation (0=grey)</label><input type="number" id="basemap_saturation" min="0" max="2" step="0.1"></div>
<div class="field"><label>Basemap brightness</label><input type="number" id="basemap_brightness" min="0.2" max="2" step="0.05"></div>
<div class="field"><label>Basemap contrast</label><input type="number" id="basemap_contrast" min="0.2" max="2" step="0.05"></div>
<div class="field"><label>Cities</label><input type="checkbox" id="show_cities"></div>
<div class="field"><label>Poll interval (s)</label><input type="number" id="poll_s" min="30" max="7200" step="30"></div>
<div class="field"><label>Image scale</label><input type="number" id="web_scale" min="0.1" max="4" step="0.1"></div>
<div class="field"><label>Settings tag</label><input type="text" id="settings_tag" placeholder="baseline, iter-01"></div>
<div class="hint">Saved to config.json. Applies next cycle.</div>
<div class="actions">
<button onclick="saveAll(false)">Save</button>
<button onclick="saveAll(true)">Save & refresh</button>
</div></div>
</div>
<div class="right">
<img id="g" src="/threat.gif">
<img id="p" src="/threat.png">
</div>
</div>
<script>
const RT_FIELDS=['velocity_window_min','velocity_adaptive','approaching_radius_km','cluster_radius_weight','only_approach_deg','threat_radius_km','threat_angle_deg','threat_min_duration','threat_min_sri_mmh','shadow_enabled','shadow_overlay_live','shadow_autotune_enabled','shadow_autotune_hours','radius_km','weather_radius_km','archive_enabled','archive_radius_km','archive_min_pixels','min_size_px','min_duration_frames','view_rotation_auto','view_rotation','basemap','basemap_saturation','basemap_brightness','basemap_contrast','show_cities','poll_s','web_scale','settings_tag'];
const MAIN_SENSORS=['raining_now','raining_confirmed','rain_confidence','rain_duration_min','rain_end_eta_min','next_rain_eta_min','gap_min','summary','radar_intensity_mmh','nearest_station_mm','intensity_mmh','station_trend','upstream_trend','local_temp_c','local_humidity','local_pressure_hpa','local_wind_speed_kmh','local_wind_gust_kmh','local_wind_dir_deg','local_wind_dir','local_rain_today_mm','local_rain_1h_mm','local_rain_rate_mmh'];
const DIAG_SENSORS=['coverage_map','nowcast_error_min','data_age_min','frame_timestamp','cell_count','nearest_cell_km','nearest_cell_eta_min','nearest_station_km','nearest_station_age_min','station_count_wet','station_count_fresh','station_wet_fraction','station_median_mm','local_weather_stations'];
let currentShadowName='';
function setField(k,v){const el=document.getElementById(k);if(!el)return;if(el.type==='checkbox')el.checked=!!v;else el.value=v;}
function getField(k){const el=document.getElementById(k);if(!el)return null;if(el.type==='checkbox')return el.checked;if(k==='settings_tag'||k==='basemap')return el.value;return parseFloat(el.value);}
function loadSettings(){fetch('/settings').then(r=>r.json()).then(s=>{RT_FIELDS.forEach(k=>{if(k in s)setField(k,s[k]);});}).catch(e=>{});}
function saveAll(refresh){const body={};RT_FIELDS.forEach(k=>body[k]=getField(k));
 fetch('/settings',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})
 .then(r=>r.json()).then(d=>{if(d.ok&&refresh)return fetch('/refresh',{method:'POST'});})
 .then(()=>{bumpImages();loadStatus();loadAccuracy();});}
function fmtVal(v){if(v===null||v===undefined)return '—';if(typeof v==='boolean')return v?'yes':'no';return String(v);}

function refreshProfilePicker(selectedName){
  fetch('/profiles').then(r=>r.json()).then(d=>{
    const sel=document.getElementById('sh_picker');
    const profs=d.profiles||[];
    sel.innerHTML='';
    profs.forEach(p=>{
      const opt=document.createElement('option');
      opt.value=p.name;
      const loss=p.metrics&&p.metrics.composite_loss!=null?` [loss ${p.metrics.composite_loss}]`:'';
      opt.textContent=`Shadow: ${p.name}${loss}`;
      sel.appendChild(opt);
    });
    if(selectedName)sel.value=selectedName;
  }).catch(()=>{});
}

function selectShadowProfile(){
  const name=document.getElementById('sh_picker').value;
  if(!name)return;
  fetch('/shadow/select',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name})})
  .then(()=>fetch('/refresh',{method:'POST'}))
  .then(()=>{setTimeout(()=>{loadStatus();bumpImages();},1500);});
}

function triggerTuneNow(){
  document.getElementById('shadow_box').innerHTML='<div class="hint">Running 2-stage auto-tuner across archived storms in background...</div>';
  fetch('/shadow/tune_now',{method:'POST'}).then(r=>r.json()).then(d=>{
    setTimeout(()=>{loadStatus();refreshProfilePicker();bumpImages();},2000);
  });
}

function promoteActiveShadow(){
  if(!currentShadowName)return;
  if(!confirm(`Promote shadow challenger "${currentShadowName}" to live config.json? (Your baseline remains saved in profiles.json)`))return;
  fetch('/profiles/promote',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:currentShadowName})})
  .then(r=>r.json()).then(d=>{
    if(d.ok){loadSettings();loadStatus();alert(`Promoted "${currentShadowName}" to active live config!`);}
  });
}

function loadStatus(){fetch('/status').then(r=>r.json()).then(s=>{const el=document.getElementById('status');if(!s||!s.updated){el.textContent='No data yet — waiting for first cycle';return;}
 const threatsHtml=(s.threats||[]).map(t=>`<div class="threat-item">• ${t.label} ${t.size_px}px, ${t.max_mmh!=null?t.max_mmh+'mm/h, ':''}miss ${t.miss_km}km, ETA ${t.eta_min!=null?t.eta_min+'min':'?'}</div>`).join('');
 el.innerHTML=`<div class="row"><span class="label">Target</span><span>${s.target||'?'}</span></div>
 <div class="row"><span class="label">Active Champion</span><span><span class="pill-champ">${s.settings_tag||'baseline'}</span></span></div>
 <div class="row"><span class="label">Updated</span><span>${s.updated}</span></div>
 <div class="row"><span class="label">Confidence</span><span>${s.confidence||'?'}</span></div>
 <div class="row"><span class="label">Raining</span><span>${s.raining_now?'yes':'no'}</span></div>
 <div class="row"><span class="label">Approaching / Threats</span><span>${s.approaching_count||0} / ${s.threat_count||0}</span></div>
 <div class="row"><span class="label">View rotation</span><span>${s.view_rotation_used!=null?s.view_rotation_used.toFixed(0)+'°'+(s.view_rotation_auto?' (auto)':'') :'—'}</span></div>
 <div class="row"><span class="label">Basemap</span><span>${s.basemap_used||'—'}</span></div>
 <div class="row"><span class="label">Config</span><span class="cfgpath">${s.config_path||'(not persisted)'}</span></div>
 ${threatsHtml}`;

 // Parallel Shadow Card
 const sh=s.shadow||{};
 const shBox=document.getElementById('shadow_box');
 const shBadge=document.getElementById('sh_badge');
 if(!sh.enabled){
   shBadge.textContent='disabled';
   shBox.innerHTML='<div class="hint">Parallel shadow runner disabled or no profile initialized yet.</div>';
 }else{
   currentShadowName=sh.profile_name||'';
   shBadge.textContent=`${sh.profile_name} (${sh.stage||'initial'})`;
   refreshProfilePicker(currentShadowName);
   const live=sh.live||{};
   const off=sh.offline_metrics||{};
   const acc=sh.live_accuracy||{};
   const cStart=(acc.champion&&acc.champion.rain_start)?acc.champion.rain_start.mean_abs_error_min:null;
   const sStart=(acc.challenger&&acc.challenger.rain_start)?acc.challenger.rain_start.mean_abs_error_min:null;
   const cCell=(acc.champion&&acc.champion.cluster_arrive)?acc.champion.cluster_arrive.mean_abs_error_min:null;
   const sCell=(acc.challenger&&acc.challenger.cluster_arrive)?acc.challenger.cluster_arrive.mean_abs_error_min:null;
   const sens=s.sensors||{};
   shBox.innerHTML=`<table class="acc">
     <tr><th>Metric</th><th class="num">Champion (${s.settings_tag||'live'})</th><th class="num">Shadow (${sh.profile_name})</th></tr>
     <tr><td>Approaching cells</td><td class="num">${s.approaching_count||0}</td><td class="num">${live.approaching_count||0}</td></tr>
     <tr><td>Active threats</td><td class="num">${s.threat_count||0}</td><td class="num">${live.threat_count||0}</td></tr>
     <tr><td>Next rain ETA</td><td class="num">${sens.next_rain_eta_min!=null?sens.next_rain_eta_min+' min':'—'}</td><td class="num">${live.next_rain_eta_min!=null?live.next_rain_eta_min+' min':'—'}</td></tr>
     <tr><td>Rain end ETA</td><td class="num">${sens.rain_end_eta_min!=null?sens.rain_end_eta_min+' min':'—'}</td><td class="num">${live.rain_end_eta_min!=null?live.rain_end_eta_min+' min':'—'}</td></tr>
     <tr><td>Live Rain Start MAE</td><td class="num">${cStart!=null?cStart.toFixed(1)+' min':'—'}</td><td class="num">${sStart!=null?sStart.toFixed(1)+' min':'—'}</td></tr>
     <tr><td>Live Cell Arrival MAE</td><td class="num">${cCell!=null?cCell.toFixed(1)+' min':'—'}</td><td class="num">${sCell!=null?sCell.toFixed(1)+' min':'—'}</td></tr>
     <tr><td>Walk-Fwd 30m Miss / F1</td><td class="num">—</td><td class="num">${off.pos_miss_30m_km!=null?off.pos_miss_30m_km+' km':'—'} / ${off.f1!=null?off.f1:'—'}</td></tr>
   </table>
   <div class="hint" style="margin-top:6px">Shadow summary: ${live.summary||'—'}</div>
   <div class="hint">Auto-tuner: ${sh.autotune_enabled?'every '+sh.autotune_hours+'h':'off'} · status: ${sh.autotune_status||'idle'} (${acc.challenger_scores||0} live shadow scores)</div>`;
 }

 const arch=s.archive||{};
 const archEl=document.getElementById('archive_box');
 if(!arch.archive_dir){
   archEl.innerHTML='<div class="hint">Persistence disabled (no data-dir)</div>';
 }else{
   archEl.innerHTML=`<div class="hint">Status: ${arch.enabled?'Active':'Paused'} (radius ≤ ${arch.radius_km||200} km) · Episodes: <b>${arch.total_episodes||0}</b> · Frames: <b>${arch.total_frames||0}</b></div>
   <div class="hint">Last episode: <span class="cfgpath">${arch.last_episode||'none'}</span> · <a href="/replay">Open Offline Replay & Shadow Tuner</a></div>`;
 }
 const wx=s.local_weather||{};
 const wxEl=document.getElementById('local_wx');
 if(!wx.station_count){
   wxEl.innerHTML=`<div class="hint">No fresh Pljusak stations within ${wx.radius_km||5} km</div>`;
 }else{
   const windTxt=wx.wind_speed_kmh!=null?`${wx.wind_speed_kmh} km/h`:'—';
   const gustTxt=wx.wind_gust_kmh!=null?`gust ${wx.wind_gust_kmh}`:'';
   const dirTxt=wx.wind_dir_deg!=null?`${wx.wind_dir_cardinal} (${wx.wind_dir_deg}°)`:'—';
   const pTrend=wx.pressure_trend!=null?(wx.pressure_trend>0?'+':'')+wx.pressure_trend+' hPa':'';
   const tTrend=wx.temp_trend!=null?(wx.temp_trend>0?'+':'')+wx.temp_trend+'°/h':'';
   wxEl.innerHTML=`<div class="wx-grid">
     <div class="wx-box"><div class="wx-k">Temp</div><div class="wx-v">${wx.temp_c!=null?wx.temp_c+' °C':'—'}</div><div class="wx-sub">${tTrend||' '}</div></div>
     <div class="wx-box"><div class="wx-k">Humidity</div><div class="wx-v">${wx.humidity_pct!=null?wx.humidity_pct+' %':'—'}</div><div class="wx-sub"> </div></div>
     <div class="wx-box"><div class="wx-k">Pressure</div><div class="wx-v">${wx.pressure_hpa!=null?wx.pressure_hpa+' hPa':'—'}</div><div class="wx-sub">${pTrend||' '}</div></div>
     <div class="wx-box"><div class="wx-k">Wind</div><div class="wx-v">${windTxt}</div><div class="wx-sub">${gustTxt||' '}</div></div>
     <div class="wx-box"><div class="wx-k">Heading</div><div class="wx-v">${dirTxt}</div><div class="wx-sub"> </div></div>
     <div class="wx-box"><div class="wx-k">Rain (Today / 1h)</div><div class="wx-v">${wx.precip_today_mm!=null?wx.precip_today_mm:'—'} / ${wx.precip_1h_mm!=null?wx.precip_1h_mm:'—'} mm</div><div class="wx-sub">${wx.rain_rate_mmh!=null?wx.rain_rate_mmh+' mm/h':' '}</div></div>
   </div>
   <div class="hint" style="margin-top:6px">${wx.station_count} fresh station(s) ≤ ${wx.radius_km} km · nearest: ${wx.nearest_station||'?'} (${wx.nearest_dist_km} km)</div>`;
 }
 const sensors=s.sensors||{};
 let html='<table class="acc"><tr><th>Sensor</th><th>Value</th></tr>';
 for(const k of MAIN_SENSORS){html+=`<tr><td>${k}</td><td class="num">${fmtVal(sensors[k])}</td></tr>`;}
 html+='</table><h4>Diagnostic</h4><table class="acc">';
 for(const k of DIAG_SENSORS){html+=`<tr><td>${k}</td><td class="num">${fmtVal(sensors[k])}</td></tr>`;}
 html+='</table>';
 document.getElementById('sensors').innerHTML=html;
}).catch(e=>{});}
function loadAccuracy(){fetch('/accuracy').then(r=>r.json()).then(a=>{const el=document.getElementById('accuracy');
 if(!a){el.innerHTML='<div class="hint">No accuracy data</div>';return;}
 let html='<div class="hint">Preds '+a.total_predictions+' · Outcomes '+a.total_outcomes+' · Scores '+a.total_scores+' · Superseded '+(a.superseded_predictions||0)+'</div>';
 const pend=a.pending_by_kind||{};
 if(Object.keys(pend).length){html+='<div class="hint">Pending: '+Object.entries(pend).map(([k,v])=>k+':'+v).join(', ')+'</div>';}
 if(a.total_scores>0){
   const order=['cluster_arrive','rain_start','rain_end','threat'];
   html+='<table class="acc"><tr><th>Kind</th><th>N</th><th>Bias</th><th>MAE</th><th>p90</th></tr>';
   for(const k of order){const v=a.by_kind[k];if(!v||!v.n)continue;
     const bias=v.median_error_min!=null?(v.median_error_min>0?'+':'')+v.median_error_min.toFixed(0):'—';
     html+=`<tr><td>${k}</td><td class="num">${v.n}</td><td class="num">${bias}</td><td class="num">${v.mean_abs_error_min.toFixed(1)}</td><td class="num">${v.p90_abs_error_min.toFixed(1)}</td></tr>`;}
   html+='</table>';
   const tags=Object.keys(a.by_tag||{});
   if(tags.length){html+='<h4>By tag (Champion vs Shadow)</h4><table class="acc"><tr><th>Tag</th><th>N</th><th>Bias</th><th>MAE</th></tr>';
     for(const t of tags){const v=a.by_tag[t];if(!v||!v.n)continue;
       const bias=v.median_error_min!=null?(v.median_error_min>0?'+':'')+v.median_error_min.toFixed(0):'—';
       html+=`<tr><td>${t||'(untagged)'}</td><td class="num">${v.n}</td><td class="num">${bias}</td><td class="num">${v.mean_abs_error_min.toFixed(1)}</td></tr>`;}
     html+='</table>';}
 }
 el.innerHTML=html;
}).catch(e=>{});}
function replayAccuracy(){if(!confirm('Re-score all predictions?'))return;fetch('/accuracy/replay',{method:'POST'}).then(()=>loadAccuracy());}
function resetAccuracy(){if(!confirm('Clear all scores?'))return;fetch('/accuracy/reset',{method:'POST'}).then(()=>loadAccuracy());}
function bumpImages(){const t=Date.now();document.getElementById('g').src='/threat.gif?t='+t;document.getElementById('p').src='/threat.png?t='+t;}
loadSettings();loadStatus();loadAccuracy();
setInterval(loadStatus,10000);setInterval(loadAccuracy,30000);setInterval(bumpImages,60000);
</script></body></html>"""


_REPLAY_HTML = """<!DOCTYPE html><html><head><title>Offline Replay & Shadow Tuner — DHMZ Radar</title>
<style>
body{background:#0e1014;color:#dde;font-family:-apple-system,BlinkMacSystemFont,sans-serif;margin:0;padding:18px}
h1{color:#fff;margin:0 0 12px 0;font-size:20px;display:flex;align-items:center;gap:12px}
.badge{background:#233547;color:#7cf;padding:3px 8px;border-radius:3px;font-size:12px;text-decoration:none}
.grid{display:flex;gap:18px;flex-wrap:wrap}
.col-left{flex:1 1 420px;max-width:480px}
.col-mid{flex:1 1 480px;max-width:560px}
.col-right{flex:1 1 520px}
.card{background:#161a20;border:1px solid #2b323d;border-radius:5px;padding:12px;margin-bottom:14px}
.card h3{margin:0 0 8px 0;font-size:12px;text-transform:uppercase;color:#6cf;letter-spacing:0.6px}
.row{display:flex;align-items:center;gap:8px;margin-bottom:7px;font-size:13px}
.row label{flex:1;color:#bcc}
input[type=number],select,input[type=text]{background:#0b0d10;color:#eee;border:1px solid #333b48;padding:4px 6px;border-radius:3px;font-size:12px}
input[type=number]{width:80px}
select{min-width:150px}
input[type=checkbox]{accent-color:#6cf;width:16px;height:16px}
button{background:#265b8c;color:#fff;border:0;padding:7px 12px;border-radius:3px;cursor:pointer;font-size:12px;font-weight:500}
button:hover{background:#3477b5}
button.purple{background:#663399}button.purple:hover{background:#7d42b8}
button.green{background:#246b42}button.green:hover{background:#2f8a55}
button.secondary{background:#383f4a}button.secondary:hover{background:#4a5362}
button:disabled{opacity:0.45;cursor:wait}
table{width:100%;border-collapse:collapse;font-size:12px}
th,td{padding:5px 6px;text-align:left;border-bottom:1px solid #242a34}
th{color:#7cf;font-size:11px;text-transform:uppercase;font-weight:500}
td.num{text-align:right;font-variant-numeric:tabular-nums}
.better{color:#5e8;font-weight:600}.worse{color:#f77}
.leg-cyan{color:#6ef}.leg-mag{color:#d6f}
.scrubber{display:flex;align-items:center;gap:8px;margin-top:8px}
.scrubber input[type=range]{flex:1;accent-color:#6cf}
img#replay_img{width:100%;max-width:640px;border:1px solid #2b323d;border-radius:4px;display:block;background:#080a0d}
.hint{color:#778;font-size:11px;margin-top:4px;line-height:1.4}
</style></head><body>
<h1>Offline Storm Replay & Shadow Auto-Tuner <a class="badge" href="/">← Live Dashboard</a></h1>

<div class="grid">
  <div class="col-left">
    <div class="card">
      <h3>1. Archived Storm Episode</h3>
      <div class="row">
        <label>Episode</label>
        <select id="ep_select" onchange="onEpisodeChange()"></select>
        <button class="secondary" onclick="loadEpisodes()">Reload</button>
      </div>
      <div id="ep_meta" class="hint">Loading archived episodes...</div>
      <div class="scrubber">
        <button class="secondary" onclick="stepFrame(-1)">◀</button>
        <button id="play_btn" class="secondary" onclick="togglePlay()">▶ Play</button>
        <button class="secondary" onclick="stepFrame(1)">▶</button>
        <input type="range" id="frame_slider" min="0" max="0" value="0" oninput="renderFrame()">
        <span id="frame_lbl" style="font-size:12px;min-width:52px;text-align:right">0/0</span>
      </div>
      <div class="hint" style="margin-top:6px">
        Vectors: <span class="leg-cyan">■ Baseline</span> vs <span class="leg-mag">■ Shadow Candidate</span> (1h projected track)
      </div>
    </div>

    <img id="replay_img" alt="Archived radar frame">
  </div>

  <div class="col-mid">
    <div class="card">
      <h3>2. Shadow Runner Parameters</h3>
      <div class="row"><label>Load profile into Shadow</label>
        <select id="profile_select" onchange="loadProfileIntoShadow()"></select>
      </div>
      <div class="row"><label>Velocity window (min)</label><input type="number" id="sh_velocity_window_min" min="10" max="90" step="5"></div>
      <div class="row"><label>Adaptive velocity window</label><input type="checkbox" id="sh_velocity_adaptive"></div>
      <div class="row"><label>Min cell size (px)</label><input type="number" id="sh_min_size_px" min="5" max="5000" step="5"></div>
      <div class="row"><label>Min duration (frames)</label><input type="number" id="sh_min_duration_frames" min="1" max="50" step="1"></div>
      <div class="row"><label>Base approach radius (km)</label><input type="number" id="sh_approaching_radius_km" min="1" max="500" step="2"></div>
      <div class="row"><label>Cluster size weight</label><input type="number" id="sh_cluster_radius_weight" min="0" max="5" step="0.1"></div>
      <div class="row"><label>Hard angle cap (°)</label><input type="number" id="sh_only_approach_deg" min="0" max="180" step="5"></div>
      <div class="row"><label>Threat radius (km)</label><input type="number" id="sh_threat_radius_km" min="1" max="500" step="2"></div>
      <div class="row"><label>Threat angle (°)</label><input type="number" id="sh_threat_angle_deg" min="0" max="180" step="5"></div>
      <div class="row"><label>Threat min duration (fr)</label><input type="number" id="sh_threat_min_duration" min="1" max="50" step="1"></div>
      <div class="row"><label>Threat min SRI (mm/h)</label><input type="number" id="sh_threat_min_sri_mmh" min="0" max="500" step="0.5"></div>
      <div class="row" style="margin-top:8px">
        <button onclick="runShadowCompare()">Compare Shadow vs Baseline</button>
        <button id="tune_btn" class="purple" onclick="runTwoStageTune()">Run 2-Stage Auto-Tuner</button>
      </div>
      <div id="tune_status" class="hint">
        Stage 1 tunes kinematics (speed, heading, 30m miss). Stage 2 refines approach/threat cone geometry (F1 & ETA). Every run is saved as a new profile iteration so baseline is preserved.
      </div>
    </div>
  </div>

  <div class="col-right">
    <div class="card">
      <h3>3. Walk-Forward Speed, Heading & Alert Scorecard</h3>
      <div id="scorecard_box" class="hint">Click "Compare Shadow vs Baseline" or "Run 2-Stage Auto-Tuner" to score across archived storms.</div>
      <div class="row" style="margin-top:10px">
        <input type="text" id="save_name" placeholder="Profile name (e.g. iter-01)" style="flex:1">
        <button class="green" onclick="saveCurrentShadow()">Save Shadow as Profile</button>
      </div>
    </div>

    <div class="card">
      <h3>4. Saved Parameter Iterations (History)</h3>
      <div id="profiles_box" class="hint">Loading saved profiles...</div>
    </div>
  </div>
</div>

<script>
const KEYS=['velocity_window_min','velocity_adaptive','min_size_px','min_duration_frames','approaching_radius_km','cluster_radius_weight','only_approach_deg','threat_radius_km','threat_angle_deg','threat_min_duration','threat_min_sri_mmh'];
let episodes=[];
let profiles=[];
let playTimer=null;
let lastShadowMetrics=null;

function getShadowParams(){
  const p={};
  KEYS.forEach(k=>{
    const el=document.getElementById('sh_'+k);
    if(!el)return;
    p[k]=el.type==='checkbox'?el.checked:parseFloat(el.value);
  });
  return p;
}

function setShadowParams(p){
  KEYS.forEach(k=>{
    const el=document.getElementById('sh_'+k);
    if(!el||!(k in p))return;
    if(el.type==='checkbox')el.checked=!!p[k];
    else el.value=p[k];
  });
}

function loadEpisodes(){
  fetch('/archive?n=50').then(r=>r.json()).then(d=>{
    episodes=d.episodes||[];
    const sel=document.getElementById('ep_select');
    sel.innerHTML='';
    if(!episodes.length){
      document.getElementById('ep_meta').textContent='No archived storm episodes yet. Wait for storms within archive radius.';
      return;
    }
    episodes.forEach(ep=>{
      const opt=document.createElement('option');
      opt.value=ep.episode_id;
      opt.textContent=`${ep.episode_id} (${ep.frame_count} fr, peak ${ep.peak_sri_mmh||'?'} mm/h${ep.target_hit?' · HIT':''})`;
      sel.appendChild(opt);
    });
    onEpisodeChange();
  });
}

function onEpisodeChange(){
  const epId=document.getElementById('ep_select').value;
  const ep=episodes.find(e=>e.episode_id===epId);
  const slider=document.getElementById('frame_slider');
  if(!ep)return;
  const maxIdx=Math.max(0,(ep.frame_count||1)-1);
  slider.max=maxIdx;
  slider.value=maxIdx;
  document.getElementById('ep_meta').textContent=`Start: ${ep.start_utc||'?'} · End: ${ep.end_utc||'?'} · Peak SRI: ${ep.peak_sri_mmh||0} mm/h`;
  renderFrame();
}

function stepFrame(delta){
  const slider=document.getElementById('frame_slider');
  const next=Math.max(0,Math.min(parseInt(slider.max),parseInt(slider.value)+delta));
  slider.value=next;
  renderFrame();
}

function togglePlay(){
  const btn=document.getElementById('play_btn');
  if(playTimer){clearInterval(playTimer);playTimer=null;btn.textContent='▶ Play';return;}
  btn.textContent='⏸ Pause';
  playTimer=setInterval(()=>{
    const slider=document.getElementById('frame_slider');
    let v=parseInt(slider.value)+1;
    if(v>parseInt(slider.max))v=0;
    slider.value=v;
    renderFrame();
  },450);
}

function renderFrame(){
  const epId=document.getElementById('ep_select').value;
  const slider=document.getElementById('frame_slider');
  const idx=parseInt(slider.value)||0;
  const total=parseInt(slider.max)+1;
  document.getElementById('frame_lbl').textContent=`${idx+1}/${total}`;
  if(!epId)return;
  const sh=encodeURIComponent(JSON.stringify(getShadowParams()));
  document.getElementById('replay_img').src=`/replay/frame?episode=${encodeURIComponent(epId)}&idx=${idx}&shadow=${sh}&t=${Date.now()}`;
}

function loadProfiles(){
  fetch('/profiles').then(r=>r.json()).then(d=>{
    profiles=d.profiles||[];
    const sel=document.getElementById('profile_select');
    sel.innerHTML='';
    profiles.forEach(p=>{
      const opt=document.createElement('option');
      opt.value=p.name;
      const loss=p.metrics&&p.metrics.composite_loss!=null?` [loss ${p.metrics.composite_loss}]`:'';
      opt.textContent=`${p.name} (${p.stage})${loss}`;
      sel.appendChild(opt);
    });
    if(profiles.length){
      const latest=profiles[profiles.length-1];
      sel.value=latest.name;
      setShadowParams(latest.params);
      renderFrame();
    }
    renderProfilesTable();
  });
}

function loadProfileIntoShadow(){
  const name=document.getElementById('profile_select').value;
  const p=profiles.find(x=>x.name===name);
  if(p){setShadowParams(p.params);renderFrame();}
}

function renderProfilesTable(){
  const box=document.getElementById('profiles_box');
  if(!profiles.length){box.textContent='No profiles saved.';return;}
  let html='<table><tr><th>Profile</th><th>Win</th><th>Adapt</th><th>30m Miss</th><th>Hdg MAE</th><th>F1</th><th>Loss</th><th>Actions</th></tr>';
  profiles.forEach(p=>{
    const m=p.metrics||{};
    const pr=p.params||{};
    html+=`<tr>
      <td><b>${p.name}</b><br><span class="hint">${p.stage}</span></td>
      <td class="num">${pr.velocity_window_min||'—'}m</td>
      <td>${pr.velocity_adaptive?'on':'off'}</td>
      <td class="num">${m.pos_miss_30m_km!=null?m.pos_miss_30m_km+' km':'—'}</td>
      <td class="num">${m.heading_mae_deg!=null?m.heading_mae_deg+'°':'—'}</td>
      <td class="num">${m.f1!=null?m.f1:'—'}</td>
      <td class="num">${m.composite_loss!=null?m.composite_loss:'—'}</td>
      <td>
        <button class="secondary" onclick="applyProfile('${p.name}')">Load</button>
        <button class="purple" onclick="setAsLiveShadow('${p.name}')">Set Shadow</button>
        <button class="green" onclick="promoteProfile('${p.name}')">Adopt Live</button>
      </td>
    </tr>`;
  });
  html+='</table>';
  box.innerHTML=html;
}

function applyProfile(name){
  const p=profiles.find(x=>x.name===name);
  if(p){setShadowParams(p.params);document.getElementById('profile_select').value=name;renderFrame();}
}

function setAsLiveShadow(name){
  fetch('/shadow/select',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name})})
  .then(r=>r.json()).then(d=>{
    alert(d.ok?`"${name}" is now running in parallel as the Live Shadow Challenger!`:`Error: ${d.error}`);
  });
}

function promoteProfile(name){
  if(!confirm(`Promote profile "${name}" to live config.json? (Baseline remains safely stored in profiles.json)`))return;
  fetch('/profiles/promote',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name})})
  .then(r=>r.json()).then(d=>{
    alert(d.ok?`Promoted "${name}" to live config.json!`:`Error: ${d.error}`);
  });
}

function fmtDelta(baseVal, shVal, lowerIsBetter, unit=''){
  if(baseVal==null||shVal==null)return '—';
  const diff=shVal-baseVal;
  if(Math.abs(diff)<1e-3)return `${shVal}${unit}`;
  const good=lowerIsBetter?(diff<0):(diff>0);
  const sign=diff>0?'+':'';
  return `<span class="${good?'better':'worse'}">${shVal}${unit} (${sign}${diff.toFixed(2)})</span>`;
}

function renderScorecard(base, sh){
  lastShadowMetrics=sh;
  const rows=[
    ['30-min Track Miss (km)', base.pos_miss_30m_km, sh.pos_miss_30m_km, true, ' km'],
    ['Heading MAE (°)', base.heading_mae_deg, sh.heading_mae_deg, true, '°'],
    ['Speed MAE (km/h)', base.speed_mae_kmh, sh.speed_mae_kmh, true, ' km/h'],
    ['Avg Adaptive Window', base.avg_eff_window_min, sh.avg_eff_window_min, false, ' min'],
    ['Alert Precision', base.precision, sh.precision, false, ''],
    ['Alert Recall', base.recall, sh.recall, false, ''],
    ['Alert F1 Score', base.f1, sh.f1, false, ''],
    ['Arrival ETA MAE (min)', base.eta_mae_min, sh.eta_mae_min, true, ' min'],
    ['Composite Loss (lower=better)', base.composite_loss, sh.composite_loss, true, ''],
  ];
  let html=`<div class="hint" style="margin-bottom:6px">Evaluated ${sh.episodes_evaluated} storm episode(s), ${sh.kinematic_samples} walk-forward steps</div>`;
  html+='<table><tr><th>Metric</th><th class="num">Baseline</th><th class="num">Shadow</th></tr>';
  rows.forEach(([lbl,bv,sv,low,u])=>{
    html+=`<tr><td>${lbl}</td><td class="num">${bv!=null?bv+u:'—'}</td><td class="num">${fmtDelta(bv,sv,low,u)}</td></tr>`;
  });
  html+='</table>';
  document.getElementById('scorecard_box').innerHTML=html;
}

function runShadowCompare(){
  document.getElementById('tune_status').textContent='Running walk-forward evaluation...';
  fetch('/replay/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({shadow:getShadowParams()})})
  .then(r=>r.json()).then(d=>{
    if(d.error){document.getElementById('tune_status').textContent='Error: '+d.error;return;}
    renderScorecard(d.baseline,d.shadow);
    renderFrame();
    document.getElementById('tune_status').textContent='Walk-forward comparison complete.';
  });
}

function runTwoStageTune(){
  const btn=document.getElementById('tune_btn');
  btn.disabled=true;
  document.getElementById('tune_status').textContent='Running Stage 1 (kinematics) & Stage 2 (alert cone) across archived storms...';
  fetch('/replay/autotune',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({base:getShadowParams(),save:true})})
  .then(r=>r.json()).then(d=>{
    btn.disabled=false;
    if(!d.ok){document.getElementById('tune_status').textContent='Tuner error: '+(d.error||'failed');return;}
    setShadowParams(d.tuned_params);
    renderScorecard(d.baseline_metrics,d.tuned_metrics);
    loadProfiles();
    renderFrame();
    document.getElementById('tune_status').textContent=`Saved new tuned iteration "${d.saved_profile}" (loss ${d.baseline_metrics.composite_loss} → ${d.tuned_metrics.composite_loss}).`;
  }).catch(e=>{btn.disabled=false;});
}

function saveCurrentShadow(){
  const name=document.getElementById('save_name').value.trim();
  fetch('/profiles',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:name||null,params:getShadowParams(),metrics:lastShadowMetrics,stage:'manual'})})
  .then(r=>r.json()).then(()=>{
    document.getElementById('save_name').value='';
    loadProfiles();
  });
}

loadEpisodes();
loadProfiles();
</script></body></html>"""


class _VizHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = self.path.split("?")[0]
        qs = self.path.split("?", 1)[1] if "?" in self.path else ""
        params = {}
        for part in qs.split("&"):
            if "=" in part:
                k, v = part.split("=", 1)
                params[k] = v
        if path == "/":
            self._html(_INDEX_HTML)
        elif path == "/replay":
            self._html(_REPLAY_HTML)
        elif path == "/replay/frame":
            self._replay_frame(params)
        elif path == "/threat.png":
            self._bytes("png", "image/png")
        elif path == "/threat.gif":
            self._bytes("gif", "image/gif")
        elif path == "/status":
            self._status()
        elif path == "/settings":
            self._json(_snapshot())
        elif path == "/config":
            self._config_get()
        elif path == "/archive":
            self._archive(params)
        elif path == "/profiles":
            self._profiles_get()
        elif path == "/accuracy":
            self._accuracy(params)
        elif path == "/accuracy/predictions":
            self._accuracy_predictions(params)
        elif path == "/accuracy/events":
            self._accuracy_events(params)
        else:
            self.send_response(404); self.end_headers()

    def do_POST(self):
        path = self.path.split("?")[0]
        if path == "/settings":
            self._settings_post()
        elif path == "/config":
            self._config_post()
        elif path == "/replay/run":
            self._replay_run()
        elif path == "/replay/autotune":
            self._replay_autotune()
        elif path == "/shadow/select":
            self._shadow_select()
        elif path == "/shadow/tune_now":
            _FORCE_TUNE.set()
            self._json({"ok": True, "status": "triggered"})
        elif path == "/profiles":
            self._profiles_post()
        elif path == "/profiles/promote":
            self._profiles_promote()
        elif path == "/refresh":
            _FORCE_REFRESH.set()
            self.send_response(200); self.send_header("Content-Length", "0"); self.end_headers()
        elif path == "/accuracy/reset":
            acc = _VIZ_STATE.get("acc_store")
            if acc: acc.reset_scores()
            self.send_response(200); self.send_header("Content-Length", "0"); self.end_headers()
        elif path == "/accuracy/replay":
            acc = _VIZ_STATE.get("acc_store")
            if acc: acc.replay()
            self.send_response(200); self.send_header("Content-Length", "0"); self.end_headers()
        else:
            self.send_response(404); self.end_headers()

    def do_DELETE(self):
        path = self.path.split("?")[0]
        if path == "/config":
            cfg_path = _VIZ_STATE.get("config_path")
            if cfg_path:
                for p in (cfg_path, cfg_path + ".bak"):
                    try: os.remove(p)
                    except FileNotFoundError: pass
            self.send_response(200); self.send_header("Content-Length", "0"); self.end_headers()
        else:
            self.send_response(404); self.end_headers()

    def _read_json_body(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            return json.loads(self.rfile.read(length)) if length > 0 else {}
        except Exception:
            return None

    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers(); self.wfile.write(body)

    def _bytes(self, key, ctype):
        with _VIZ_STATE["lock"]:
            data = _VIZ_STATE[key]
        if data is None:
            self.send_response(503); self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", "9"); self.end_headers()
            self.wfile.write(b"not ready"); return
        self.send_response(200); self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache, max-age=0")
        self.end_headers(); self.wfile.write(data)

    def _status(self):
        with _VIZ_STATE["lock"]:
            status = dict(_VIZ_STATE["status"])
        status["config_path"] = _VIZ_STATE.get("config_path")
        self._json(status)

    def _archive(self, params):
        arch = _VIZ_STATE.get("archiver")
        if not arch:
            self._json({"enabled": False, "episodes": []}); return
        limit = int(params.get("n", 30))
        res = arch.summary()
        res["episodes"] = arch.list_episodes(limit=limit)
        self._json(res)

    def _profiles_get(self):
        ps = _VIZ_STATE.get("profile_store")
        if not ps:
            self._json({"profiles": []}); return
        self._json({"profiles": ps.list_all()})

    def _profiles_post(self):
        ps = _VIZ_STATE.get("profile_store")
        body = self._read_json_body()
        if not ps or body is None:
            self._json({"ok": False, "error": "invalid request"}, 400); return
        name = body.get("name") or ps.next_iter_name()
        params = body.get("params") or _snapshot()
        metrics = body.get("metrics")
        stage = body.get("stage") or "manual"
        saved = ps.save_profile(name, params, metrics=metrics, stage=stage)
        self._json({"ok": True, "profile": saved})

    def _shadow_select(self):
        ps = _VIZ_STATE.get("profile_store")
        body = self._read_json_body()
        if not ps or not body or "name" not in body:
            self._json({"ok": False, "error": "missing profile name"}, 400); return
        if not ps.get(body["name"]):
            self._json({"ok": False, "error": "profile not found"}, 404); return
        with _RUNTIME_LOCK:
            _RUNTIME["shadow_profile"] = body["name"]
            _RUNTIME["shadow_enabled"] = True
        _persist_runtime_to_disk()
        self._json({"ok": True, "shadow_profile": body["name"]})

    def _profiles_promote(self):
        ps = _VIZ_STATE.get("profile_store")
        body = self._read_json_body()
        if not ps or not body or "name" not in body:
            self._json({"ok": False, "error": "missing profile name"}, 400); return
        prof = ps.get(body["name"])
        if not prof:
            self._json({"ok": False, "error": "profile not found"}, 404); return
        ps.ensure_baseline(_snapshot())
        with _RUNTIME_LOCK:
            for k, v in prof["params"].items():
                if k in _RUNTIME_TYPES:
                    _RUNTIME[k] = _coerce(k, v)
            _RUNTIME["settings_tag"] = prof["name"]
            current = dict(_RUNTIME)
        _persist_runtime_to_disk()
        self._json({"ok": True, "active_profile": prof["name"], "settings": current})

    def _replay_run(self):
        eng = _VIZ_STATE.get("replay_engine")
        ps = _VIZ_STATE.get("profile_store")
        body = self._read_json_body()
        if not eng or body is None:
            self._json({"error": "replay engine unavailable"}, 503); return
        base_prof = ps.get("baseline") if ps else None
        base_params = base_prof["params"] if base_prof else _snapshot()
        shadow_params = dict(base_params)
        shadow_params.update(body.get("shadow") or {})
        ep_ids = body.get("episodes")
        base_res = eng.evaluate_params(base_params, episode_ids=ep_ids)
        shad_res = eng.evaluate_params(shadow_params, episode_ids=ep_ids)
        if ps and base_prof and base_prof.get("metrics") is None:
            ps.save_profile("baseline", base_params, metrics=base_res, parent=None, stage="initial")
        self._json({"baseline": base_res, "shadow": shad_res})

    def _replay_autotune(self):
        eng = _VIZ_STATE.get("replay_engine")
        ps = _VIZ_STATE.get("profile_store")
        body = self._read_json_body() or {}
        if not eng:
            self._json({"ok": False, "error": "replay engine unavailable"}, 503); return
        res = _execute_shadow_autotune(
            eng, ps, set_active_challenger=bool(body.get("adopt_shadow", True)))
        self._json(res)

    def _replay_frame(self, params):
        import urllib.parse
        eng = _VIZ_STATE.get("replay_engine")
        ps = _VIZ_STATE.get("profile_store")
        if not eng:
            self.send_response(503); self.end_headers(); return
        ep_id = urllib.parse.unquote(params.get("episode", ""))
        idx = int(params.get("idx", 0))
        data = eng.get_episode_data(ep_id)
        if not data:
            self.send_response(404); self.end_headers(); return
        idx = max(0, min(len(data) - 1, idx))
        prefix_mmh = [(d[1], d[2]) for d in data[:idx + 1]]

        base_prof = ps.get("baseline") if ps else None
        base_rt = _snapshot()
        if base_prof:
            base_rt.update(base_prof["params"])

        shadow_rt = dict(base_rt)
        if "shadow" in params:
            try:
                sh_raw = json.loads(urllib.parse.unquote(params["shadow"]))
                for k, v in sh_raw.items():
                    if k in _RUNTIME_TYPES:
                        shadow_rt[k] = _coerce(k, v)
            except Exception:
                pass

        (_, _, b_tracks, _, _, _, _, _, _) = track_frame_sequence(
            prefix_mmh, eng.calib, eng.frame_interval_min,
            eng.target_lat, eng.target_lon, rt=base_rt)
        b_tracks = [t for t in b_tracks
                    if t.n_observations >= min(base_rt["min_duration_frames"], idx + 1)
                    and t.cluster.size >= base_rt["min_size_px"]]
        b_appr = [assess(t, eng.target_lat, eng.target_lon, eng.calib) for t in b_tracks]
        b_af = [refine_approaching(t, a, base_rt) for t, a in zip(b_tracks, b_appr)]
        b_tf = [refine_threat(t, a, base_rt) for t, a in zip(b_tracks, b_appr)]

        (_, _, s_tracks, _, _, _, _, _, _) = track_frame_sequence(
            prefix_mmh, eng.calib, eng.frame_interval_min,
            eng.target_lat, eng.target_lon, rt=shadow_rt)
        s_tracks = [t for t in s_tracks
                    if t.n_observations >= min(shadow_rt["min_duration_frames"], idx + 1)
                    and t.cluster.size >= shadow_rt["min_size_px"]]

        ts, mask, mmh, _, fname = data[idx]
        gray = _radar_to_grayscale(mask, mmh)
        rt_cur = _snapshot()

        # Always point North (0.0°) and use archive_radius_km for archived replay frames
        archive_rot = 0.0
        archive_rad = float(rt_cur.get("archive_radius_km", 200.0))

        static_cur = _static_snapshot()
        size_px = static_cur.get("basemap_size") or 640
        bm_img = None
        bm_kind = "greyscale"
        if static_cur.get("mapbox_token") and rt_cur.get("basemap") in ("auto", "mapbox"):
            bm_img = fetch_basemap(
                static_cur.get("mapbox_style") or "mapbox/satellite-v9",
                static_cur["mapbox_token"], eng.target_lat, eng.target_lon,
                archive_rad, archive_rot, size_px, _VIZ_STATE.get("data_dir"))
            if bm_img is not None:
                bm_kind = "mapbox"

        pil = compose_map(
            mask, mmh, gray, eng.calib, eng.target_lat, eng.target_lon,
            archive_rad, archive_rot, size_px, bm_img, bm_kind,
            b_tracks, b_appr, b_af, b_tf, [], rt_cur["show_cities"],
            eng.intensity_mode, DEFAULT_SRI_THRESHOLDS, rt_cur,
            shadow_tracks=s_tracks)
        draw = ImageDraw.Draw(pil)
        ts_lbl = ts.strftime("%Y-%m-%d %H:%MZ") if ts else fname
        _draw_text_outlined(draw, 10, 8, f"REPLAY {ep_id} | {ts_lbl} ({idx+1}/{len(data)})",
                            (255, 220, 60), (0, 0, 0))
        buf = io.BytesIO()
        pil.save(buf, format="PNG", optimize=True)
        out_bytes = buf.getvalue()
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(out_bytes)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(out_bytes)

    def _config_get(self):
        cfg_path = _VIZ_STATE.get("config_path")
        if cfg_path and os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            if cfg.get("static", {}).get("mapbox_token"):
                cfg["static"]["mapbox_token"] = "***"
            self._json(cfg)
        else:
            self._json({"version": 1, "note": "no config persisted",
                        "runtime": _snapshot(), "static": _static_snapshot()})

    def _config_post(self):
        new_cfg = self._read_json_body()
        if not isinstance(new_cfg, dict):
            self._json({"ok": False, "error": "bad request"}, 400); return
        cfg_path = _VIZ_STATE.get("config_path")
        if not cfg_path:
            self._json({"ok": False, "error": "no config path"}, 400); return
        existing = _load_json_file(cfg_path) or {}
        now = datetime.now(timezone.utc).isoformat()
        if new_cfg.get("static", {}).get("mapbox_token") == "***":
            new_cfg["static"]["mapbox_token"] = existing.get("static", {}).get("mapbox_token")
        new_cfg.setdefault("version", 1)
        new_cfg.setdefault("created_at", existing.get("created_at", now))
        new_cfg["updated_at"] = now
        if _save_json_file(cfg_path, new_cfg):
            self._json({"ok": True, "restart_required": True, "path": cfg_path})
        else:
            self._json({"ok": False, "error": "save failed"}, 500)

    def _accuracy(self, params):
        acc = _VIZ_STATE.get("acc_store")
        if not acc:
            self._json({"error": "disabled"}, 503); return
        self._json(acc.stats(since_hours=float(params["since_hours"]) if "since_hours" in params else None,
                             tag=params.get("tag")))

    def _accuracy_predictions(self, params):
        acc = _VIZ_STATE.get("acc_store")
        if not acc:
            self._json({"error": "disabled"}, 503); return
        self._json(acc.list_predictions(limit=int(params.get("n", 100))))

    def _accuracy_events(self, params):
        acc = _VIZ_STATE.get("acc_store")
        if not acc:
            self._json({"error": "disabled"}, 503); return
        self._json(acc.list_outcomes(limit=int(params.get("n", 100))))

    def _settings_post(self):
        new = self._read_json_body()
        if not isinstance(new, dict):
            self._json({"ok": False, "error": "bad json"}, 400); return
        with _RUNTIME_LOCK:
            for k, v in new.items():
                if k in _RUNTIME_TYPES:
                    _RUNTIME[k] = _coerce(k, v)
            current = dict(_RUNTIME)
        _persist_runtime_to_disk()
        self._json({"ok": True, "settings": current})

    def _html(self, s):
        body = s.encode("utf-8")
        self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers(); self.wfile.write(body)

    def log_message(self, *a):
        pass


def start_web(port):
    srv = ThreadingHTTPServer(("0.0.0.0", port), _VizHandler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv


# ----------------------------------------------------------------------------
# Rotation helper
# ----------------------------------------------------------------------------
def get_effective_rotation(rt, target_lat, target_lon, basemap_north_up=None):
    if rt["view_rotation_auto"]:
        sat_rot = detect_coast_rotation_from_satellite(basemap_north_up)
        if sat_rot is not None:
            return True, sat_rot
        rot, _ = compute_coast_rotation(target_lat, target_lon)
        if rot is not None:
            return True, rot
    return False, float(rt["view_rotation"])


# ----------------------------------------------------------------------------
# Accuracy store
# ----------------------------------------------------------------------------
def _safe_fromiso(s):
    try:
        dt = datetime.fromisoformat(s)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return datetime.min.replace(tzinfo=timezone.utc)


def _prune_old(rows, ts_key, keep_days):
    if not rows:
        return rows
    cutoff = datetime.now(timezone.utc) - timedelta(days=keep_days)
    return [r for r in rows if _safe_fromiso(r.get(ts_key, "")) >= cutoff]


class AccuracyStore:
    def __init__(self, data_dir, enabled=True):
        self.enabled = enabled and bool(data_dir)
        self.data_dir = data_dir
        self.lock = threading.Lock()
        self.predictions = []
        self.outcomes = []
        self.scores = []
        self.observed_cluster_arrivals = set()
        if self.enabled:
            os.makedirs(data_dir, exist_ok=True)
            self._load()

    def _path(self, name):
        return os.path.join(self.data_dir, name)

    def _load_jsonl(self, fn):
        p = self._path(fn)
        if not os.path.exists(p): return []
        out = []
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try: out.append(json.loads(line))
                    except Exception: pass
        return out

    def _append_jsonl(self, fn, row):
        with open(self._path(fn), "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _write_jsonl(self, fn, rows):
        p = self._path(fn); tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(tmp, p)

    def _load(self):
        self.predictions = _prune_old(self._load_jsonl("predictions.jsonl"), "ts", 14)
        self.outcomes = _prune_old(self._load_jsonl("outcomes.jsonl"), "ts", 14)
        self.scores = _prune_old(self._load_jsonl("scores.jsonl"), "scored_at", 60)
        self._write_jsonl("predictions.jsonl", self.predictions)
        self._write_jsonl("outcomes.jsonl", self.outcomes)
        self._write_jsonl("scores.jsonl", self.scores)
        for o in self.outcomes:
            if o.get("kind") == "cluster_arrive" and o.get("key"):
                self.observed_cluster_arrivals.add(o["key"])

    def predict(self, ts_iso, kind, predicted_min, key="", tag="",
                settings=None, horizon_min=None):
        if not self.enabled or predicted_min is None: return
        with self.lock:
            for p in reversed(self.predictions):
                if p.get("scored") or p.get("superseded"): continue
                if p.get("kind") == kind and p.get("key", "") == key and p.get("tag", "") == tag:
                    p["superseded"] = True; break
            t0 = _safe_fromiso(ts_iso)
            t_pred = (t0 + timedelta(minutes=float(predicted_min))).isoformat()
            row = {"id": f"{ts_iso}:{kind}:{key}:{tag}", "ts": ts_iso, "kind": kind,
                   "key": key, "tag": tag, "predicted_min": round(float(predicted_min), 1),
                   "predicted_at": t_pred,
                   "horizon_min": round(float(horizon_min), 1) if horizon_min is not None else None,
                   "settings": settings or {}, "scored": False, "superseded": False}
            self.predictions.append(row); self._append_jsonl("predictions.jsonl", row)

    def observe(self, ts_iso, kind, key="", tag="", extra=None):
        if not self.enabled: return
        with self.lock:
            row = {"ts": ts_iso, "kind": kind, "key": key, "tag": tag}
            if extra: row["extra"] = extra
            self.outcomes.append(row); self._append_jsonl("outcomes.jsonl", row)
            self._match_against_locked(row)

    def _match_against_locked(self, outcome):
        kind = outcome["kind"]
        key = outcome.get("key", "")
        out_tag = outcome.get("tag", "")
        t_actual = _safe_fromiso(outcome["ts"])
        best_p, best_dt = None, float("inf")
        for p in self.predictions:
            if p.get("scored") or p.get("superseded"): continue
            if p.get("kind") != kind or p.get("key", "") != key: continue
            if out_tag and p.get("tag", "") != out_tag: continue
            if (not out_tag) and str(p.get("tag", "")).startswith("shadow:"): continue
            t0 = _safe_fromiso(p["ts"])
            if t0 >= t_actual: continue
            horizon = p.get("horizon_min") or p.get("predicted_min") or 30.0
            tol_min = max(MATCH_TOLERANCE_MIN, horizon * MATCH_TOLERANCE_FACTOR)
            dt_from_issue = (t_actual - t0).total_seconds() / 60.0
            if dt_from_issue > tol_min * 2: continue
            t_pred = _safe_fromiso(p["predicted_at"])
            err = abs((t_actual - t_pred).total_seconds() / 60.0)
            if err <= tol_min and err < best_dt:
                best_dt = err; best_p = p
        if best_p is not None:
            best_p["scored"] = True
            t0 = _safe_fromiso(best_p["ts"])
            actual_min = (t_actual - t0).total_seconds() / 60.0
            err = actual_min - best_p["predicted_min"]
            sc = {"scored_at": outcome["ts"], "prediction_id": best_p["id"],
                  "kind": kind, "key": key, "tag": best_p.get("tag", ""),
                  "predicted_min": best_p["predicted_min"],
                  "actual_min": round(actual_min, 1),
                  "error_min": round(err, 1), "abs_error_min": round(abs(err), 1),
                  "settings": best_p.get("settings", {})}
            self.scores.append(sc); self._append_jsonl("scores.jsonl", sc)
            self._write_jsonl("predictions.jsonl", self.predictions)

    def replay(self):
        if not self.enabled: return
        with self.lock:
            for p in self.predictions: p["scored"] = False
            self.scores = []
            for o in self.outcomes: self._match_against_locked(o)
            self._write_jsonl("predictions.jsonl", self.predictions)
            self._write_jsonl("scores.jsonl", self.scores)

    def reset_scores(self):
        if not self.enabled: return
        with self.lock:
            self.predictions = []; self.outcomes = []; self.scores = []
            self.observed_cluster_arrivals = set()
            self._write_jsonl("predictions.jsonl", []); self._write_jsonl("outcomes.jsonl", [])
            self._write_jsonl("scores.jsonl", [])

    def stats(self, since_hours=None, tag=None):
        with self.lock:
            rows = list(self.scores)
            preds = list(self.predictions)
            outs = list(self.outcomes)
        if since_hours is not None:
            cutoff = datetime.now(timezone.utc) - timedelta(hours=since_hours)
            rows = [r for r in rows if _safe_fromiso(r["scored_at"]) >= cutoff]
        if tag is not None:
            rows = [r for r in rows if r.get("tag", "") == tag]

        def summarize(sub):
            if not sub: return {"n": 0}
            errs = [r["error_min"] for r in sub]; aerrs = [r["abs_error_min"] for r in sub]
            return {"n": len(sub), "median_error_min": round(float(np.median(errs)), 1),
                    "mean_error_min": round(float(np.mean(errs)), 1),
                    "mean_abs_error_min": round(float(np.mean(aerrs)), 1),
                    "p90_abs_error_min": round(float(np.percentile(aerrs, 90)), 1)}

        by_kind = {}
        for k in {r["kind"] for r in rows}:
            by_kind[k] = summarize([r for r in rows if r["kind"] == k])
        by_tag = {}
        for t in {r.get("tag", "") for r in rows}:
            by_tag[t] = summarize([r for r in rows if r.get("tag", "") == t])
        pending = {}
        for p in preds:
            if not p.get("scored") and not p.get("superseded"):
                pending[p["kind"]] = pending.get(p["kind"], 0) + 1
        superseded = sum(1 for p in preds if p.get("superseded"))
        return {"total_predictions": len(preds), "total_outcomes": len(outs),
                "total_scores": len(rows), "superseded_predictions": superseded,
                "pending_by_kind": pending, "overall": summarize(rows),
                "by_kind": by_kind, "by_tag": by_tag}

    def list_predictions(self, limit=100):
        with self.lock: return list(reversed(self.predictions[-limit:]))

    def list_outcomes(self, limit=100):
        with self.lock: return list(reversed(self.outcomes[-limit:]))

    def last_mae_min(self, kind=None):
        with self.lock:
            rows = [r for r in self.scores if (kind is None or r["kind"] == kind)]
        return round(float(np.mean([r["abs_error_min"] for r in rows[-20:]])), 1) if rows else None


# ----------------------------------------------------------------------------
# MQTT bridge
# ----------------------------------------------------------------------------
SENSOR_DEFS = [
    ("raining_now",           "Raining now",             "binary", "moisture", None, None, None),
    ("raining_confirmed",     "Raining confirmed",       "binary", "moisture", None, None, None),
    ("rain_confidence",       "Rain confidence",         "sensor", None, None, "mdi:radar", None),
    ("rain_duration_min",     "Rain duration",           "sensor", "duration", "min", None, None),
    ("rain_end_eta_min",      "Rain end ETA",            "sensor", "duration", "min", "mdi:weather-partly-rainy", None),
    ("next_rain_eta_min",     "Next rain ETA",           "sensor", "duration", "min", "mdi:weather-rainy", None),
    ("gap_min",               "Dry gap",                 "sensor", "duration", "min", None, None),
    ("summary",               "Nowcast summary",         "sensor", None, None, "mdi:text", None),
    ("radar_intensity_mmh",   "Radar intensity",         "sensor", "precipitation_intensity", "mm/h", "mdi:weather-pouring", None),
    ("nearest_station_mm",    "Nearest wet station",     "sensor", "precipitation", "mm", "mdi:weather-rainy", None),
    ("intensity_mmh",         "Station rain intensity",  "sensor", "precipitation_intensity", "mm/h", None, None),
    ("station_trend",         "Station trend",           "sensor", None, None, "mdi:trending-up", None),
    ("upstream_trend",        "Upstream trend",          "sensor", None, None, "mdi:weather-windy", None),
    ("local_temp_c",          "Local temperature",       "sensor", "temperature", "°C", None, None),
    ("local_humidity",        "Local humidity",          "sensor", "humidity", "%", None, None),
    ("local_pressure_hpa",    "Local pressure",          "sensor", "atmospheric_pressure", "hPa", None, None),
    ("local_wind_speed_kmh",  "Local wind speed",        "sensor", "wind_speed", "km/h", None, None),
    ("local_wind_gust_kmh",   "Local wind gust",         "sensor", "wind_speed", "km/h", "mdi:weather-windy", None),
    ("local_wind_dir_deg",    "Local wind bearing",      "sensor", "wind_direction", "°", None, None),
    ("local_wind_dir",        "Local wind direction",    "sensor", None, None, "mdi:compass-outline", None),
    ("local_rain_today_mm",   "Local rain today",        "sensor", "precipitation", "mm", "mdi:weather-rainy", None),
    ("local_rain_1h_mm",      "Local rain 1h",           "sensor", "precipitation", "mm", "mdi:weather-rainy", None),
    ("local_rain_rate_mmh",   "Local rain rate",         "sensor", "precipitation_intensity", "mm/h", "mdi:weather-pouring", None),
    ("coverage_map",          "Coverage map",            "camera", None, None, None, "diagnostic"),
    ("nowcast_error_min",     "Nowcast error",           "sensor", "duration", "min", "mdi:target", "diagnostic"),
    ("data_age_min",          "Radar data age",          "sensor", "duration", "min", None, "diagnostic"),
    ("frame_timestamp",       "Radar frame time",        "sensor", "timestamp", None, None, "diagnostic"),
    ("cell_count",            "Approaching cells",       "sensor", None, None, None, "diagnostic"),
    ("nearest_cell_km",       "Nearest cell distance",   "sensor", "distance", "km", None, "diagnostic"),
    ("nearest_cell_eta_min",  "Nearest cell ETA",        "sensor", "duration", "min", None, "diagnostic"),
    ("nearest_station_km",    "Nearest station distance","sensor", "distance", "km", None, "diagnostic"),
    ("nearest_station_age_min","Nearest station age",    "sensor", "duration", "min", None, "diagnostic"),
    ("station_count_wet",     "Wet stations",            "sensor", None, None, None, "diagnostic"),
    ("station_count_fresh",   "Fresh stations",          "sensor", None, None, None, "diagnostic"),
    ("station_wet_fraction",  "Station wet fraction",    "sensor", None, "%", None, "diagnostic"),
    ("station_median_mm",     "Station median rain",     "sensor", "precipitation", "mm", None, "diagnostic"),
    ("local_weather_stations","Local weather stations",  "sensor", None, None, "mdi:weather-station", "diagnostic"),
]
SENSOR_KEYS = [s[0] for s in SENSOR_DEFS]


class HAMQTTBridge:
    def __init__(self, host, port=1883, user=None, password=None,
                 discovery_prefix="homeassistant", state_prefix="dhmz_radar",
                 device_id="dhmz_radar_default", device_name="DHMZ Radar",
                 enabled_sensors=None, retain=True, tls=False,
                 threat_topic=None, web_url=None):
        if not HAVE_MQTT:
            raise RuntimeError("paho-mqtt not installed")
        self.host = host; self.port = port
        self.discovery_prefix = discovery_prefix; self.state_prefix = state_prefix
        self.device_id = device_id; self.device_name = device_name
        self.enabled_sensors = (set(SENSOR_KEYS) if enabled_sensors is None
                                else set(enabled_sensors))
        self.retain = retain
        self.threat_topic = threat_topic or f"{state_prefix}/{device_id}/threat"
        self.avail_topic = f"{state_prefix}/{device_id}/status"
        self.web_url = web_url
        self._discovered = False
        self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2,
                                   client_id=f"{device_id}-{os.getpid()}")
        if user: self._client.username_pw_set(user, password)
        if tls: self._client.tls_set()
        self._client.will_set(self.avail_topic, "offline", qos=1, retain=True)
        self._client.on_connect = self._on_connect

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        self._client.publish(self.avail_topic, "online", qos=1, retain=True)
        if not self._discovered:
            self.publish_discovery()

    def connect(self):
        self._client.connect(self.host, self.port, keepalive=60)
        self._client.loop_start()

    def close(self):
        try:
            self._client.publish(self.avail_topic, "offline", qos=1, retain=True)
            self._client.loop_stop(); self._client.disconnect()
        except Exception:
            pass

    def _device_block(self):
        d = {"identifiers": [self.device_id], "name": self.device_name,
             "manufacturer": "DHMZ / pljusak.com", "model": "Radar + station tracker"}
        if self.web_url: d["configuration_url"] = self.web_url
        return d

    def _state_topic(self, key):
        return f"{self.state_prefix}/{self.device_id}/{key}/state"

    def _attr_topic(self, key):
        return f"{self.state_prefix}/{self.device_id}/{key}/attributes"

    def publish_discovery(self):
        dev = self._device_block()
        for key, name, kind, dev_cls, unit, icon, ent_cat in SENSOR_DEFS:
            uid = f"{self.device_id}_{key}"
            if kind == "camera":
                comp = "camera"
            elif kind == "binary":
                comp = "binary_sensor"
            else:
                comp = "sensor"
            disc_topic = f"{self.discovery_prefix}/{comp}/{self.device_id}/{key}/config"
            if key not in self.enabled_sensors:
                self._client.publish(disc_topic, "", qos=1, retain=True)
                continue
            payload = {"name": name, "unique_id": uid, "device": dev,
                       "availability_topic": self.avail_topic,
                       "payload_available": "online", "payload_not_available": "offline"}
            if kind == "camera":
                payload["topic"] = f"{self.state_prefix}/{self.device_id}/{key}/image"
            elif kind == "binary":
                payload["state_topic"] = self._state_topic(key)
                payload["payload_on"] = "ON"; payload["payload_off"] = "OFF"
            else:
                payload["state_topic"] = self._state_topic(key)
            if dev_cls: payload["device_class"] = dev_cls
            if unit: payload["unit_of_measurement"] = unit
            if icon: payload["icon"] = icon
            if ent_cat: payload["entity_category"] = ent_cat
            if key in ("raining_now", "raining_confirmed", "summary", "local_temp_c"):
                payload["json_attributes_topic"] = self._attr_topic(key)
            self._client.publish(disc_topic, json.dumps(payload), qos=1, retain=True)
        self._discovered = True

    def publish_state(self, state_dict, attrs_by_key=None, image_bytes=None):
        attrs_by_key = attrs_by_key or {}
        for key, name, kind, *_ in SENSOR_DEFS:
            if key not in self.enabled_sensors: continue
            if kind == "camera":
                if image_bytes:
                    self._client.publish(f"{self.state_prefix}/{self.device_id}/{key}/image",
                                         image_bytes, qos=0, retain=self.retain)
                continue
            val = state_dict.get(key)
            if kind == "binary":
                payload = "ON" if val else "OFF"
            elif val is None:
                payload = "unknown"
            else:
                payload = str(val)
            self._client.publish(self._state_topic(key), payload, qos=0, retain=self.retain)
            if key in attrs_by_key:
                self._client.publish(self._attr_topic(key),
                                     json.dumps(attrs_by_key[key], default=str),
                                     qos=0, retain=self.retain)

    def publish_threat(self, payload):
        self._client.publish(self.threat_topic, json.dumps(payload, default=str),
                             qos=1, retain=False)


# ----------------------------------------------------------------------------
# CLI + main
# ----------------------------------------------------------------------------
def parse_sri_thresholds(s):
    parts = [float(x) for x in s.split(",")]
    if len(parts) != 3:
        raise ValueError("--sri-thresholds takes 3 numbers")
    return tuple(parts)


def parse_region(s):
    parts = [int(x) for x in s.split(",")]
    if len(parts) != 4:
        raise ValueError("--timestamp-region takes 4 ints")
    return tuple(parts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="/data")
    p.add_argument("--config", default=None)
    p.add_argument("--location", default=None)
    p.add_argument("--lat", type=float, default=None)
    p.add_argument("--lon", type=float, default=None)
    p.add_argument("--calibration", default=None)
    p.add_argument("--calibration-inline", default=None)
    p.add_argument("--radius", type=float, default=None)
    p.add_argument("--weather-radius", type=float, default=None)
    p.add_argument("--approaching-radius", type=float, default=None)
    p.add_argument("--cluster-radius-weight", type=float, default=None)
    p.add_argument("--only-approach", type=float, default=None)
    p.add_argument("--threat-radius", type=float, default=None)
    p.add_argument("--threat-angle", type=float, default=None)
    p.add_argument("--threat-min-duration", type=int, default=None)
    p.add_argument("--threat-min-sri", type=float, default=None)
    p.add_argument("--min-duration", type=int, default=None)
    p.add_argument("--min-size", type=int, default=None)
    p.add_argument("--velocity-window", type=float, default=None)
    p.add_argument("--velocity-adaptive", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--archive-enabled", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--archive-radius", type=float, default=None)
    p.add_argument("--archive-min-pixels", type=int, default=None)
    p.add_argument("--shadow-enabled", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--shadow-profile", default=None)
    p.add_argument("--shadow-overlay-live", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--shadow-autotune-enabled", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--shadow-autotune-hours", type=float, default=None)
    p.add_argument("--radar-intensity", choices=("simple", "sri"), default=None)
    p.add_argument("--sri-legend-region", default=None)
    p.add_argument("--sri-legend-orientation", choices=("horizontal", "vertical"), default=None)
    p.add_argument("--sri-legend-padding", type=int, default=None)
    p.add_argument("--sri-breaks", default=None)
    p.add_argument("--sri-thresholds", default=None)
    p.add_argument("--frame-interval-min", type=float, default=None)
    p.add_argument("--timestamp-region", default=None)
    p.add_argument("--timestamp-enabled", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--timestamp-tz", default=None)
    p.add_argument("--timestamp-source", choices=("first", "last"), default=None)
    p.add_argument("--display-tz", default=None)
    p.add_argument("--pljusak-enabled", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--pljusak-url", default=None)
    p.add_argument("--pljusak-interval", type=int, default=None)
    p.add_argument("--station-confirm-radius", type=float, default=None)
    p.add_argument("--station-confirm-min-mm", type=float, default=None)
    p.add_argument("--station-confirm-max-age-min", type=float, default=None)
    p.add_argument("--station-confirm-fraction", type=float, default=None)
    p.add_argument("--station-confirm-min-count", type=int, default=None)
    p.add_argument("--poll", type=int, default=None)
    p.add_argument("--view-rotation-auto", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--view-rotation", type=float, default=None)
    p.add_argument("--basemap", choices=("auto", "mapbox", "greyscale", "none"), default=None)
    p.add_argument("--basemap-saturation", type=float, default=None)
    p.add_argument("--basemap-brightness", type=float, default=None)
    p.add_argument("--basemap-contrast", type=float, default=None)
    p.add_argument("--mapbox-token", default=None)
    p.add_argument("--mapbox-style", default=None)
    p.add_argument("--basemap-size", type=int, default=None)
    p.add_argument("--cities", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--web-enabled", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--web-port", type=int, default=None)
    p.add_argument("--web-scale", type=float, default=None)
    p.add_argument("--mqtt-enabled", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--mqtt-host", default=None)
    p.add_argument("--mqtt-port", type=int, default=None)
    p.add_argument("--mqtt-user", default=os.environ.get("MQTT_USER"))
    p.add_argument("--mqtt-password", default=os.environ.get("MQTT_PASSWORD"))
    p.add_argument("--mqtt-tls", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--ha-discovery-prefix", default=None)
    p.add_argument("--ha-state-prefix", default=None)
    p.add_argument("--ha-device-id", default=None)
    p.add_argument("--ha-device-name", default=None)
    p.add_argument("--ha-sensors", default=None)
    p.add_argument("--mqtt-no-retain", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--ha-threat-topic", default=None)
    p.add_argument("--jsonl-dump", default=None)
    p.add_argument("--accuracy-enabled", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--settings-tag", default=None)
    p.add_argument("--once", action="store_true")
    p.add_argument("--list-cities", action="store_true")
    p.add_argument("--dump-sri-calibration", action="store_true")
    args = p.parse_args()

    if args.list_cities:
        for label, name, lat, lon in CITIES:
            print(f"  {label:<4} {name:<18} ({lat:.4f}, {lon:.4f})")
        return

    data_dir = args.data_dir or None
    cfg_path = args.config or (os.path.join(data_dir, "config.json") if data_dir else None)
    existing_cfg = _load_json_file(cfg_path)
    merged_runtime, merged_static = _merge_config(args, existing_cfg)
    with _RUNTIME_LOCK:
        _RUNTIME.update(merged_runtime)
    _apply_static(merged_static)
    _VIZ_STATE["config_path"] = cfg_path
    _VIZ_STATE["data_dir"] = data_dir
    if cfg_path:
        now_iso = datetime.now(timezone.utc).isoformat()
        _save_json_file(cfg_path, {
            "version": 1,
            "created_at": (existing_cfg or {}).get("created_at", now_iso),
            "updated_at": now_iso,
            "runtime": _snapshot(),
            "static": _static_snapshot(),
        })
        print(f"config: {cfg_path}")

    st = _static_snapshot()
    if st.get("location"):
        r = resolve_city(st["location"])
        if not r:
            p.error(f"Unknown city '{st['location']}'")
        target_name, target_lat, target_lon = r
    elif st.get("lat") is not None and st.get("lon") is not None:
        target_name = f"({st['lat']:.4f},{st['lon']:.4f})"
        target_lat, target_lon = st["lat"], st["lon"]
    else:
        p.error("Provide --location or --lat/--lon (or set in config.json)")

    sri_thresholds = parse_sri_thresholds(st["sri_thresholds"])
    display_tz, display_tz_name = get_zoneinfo(st["display_tz"])
    ts_tz = display_tz if st["timestamp_tz"] is None else get_zoneinfo(st["timestamp_tz"])[0]
    ts_region = (None if not st["timestamp_enabled"]
                 else parse_region(st["timestamp_region"]))

    sri_legend = None
    legend_mask = None
    if st["radar_intensity"] == "sri":
        if not st["sri_legend_region"]:
            p.error("--radar-intensity sri requires --sri-legend-region")
        if not st["sri_breaks"]:
            p.error("--radar-intensity sri requires --sri-breaks")
        breaks = [float(x) for x in st["sri_breaks"].split(",")]
        sri_legend = SRILegend(
            region=parse_region(st["sri_legend_region"]),
            orientation=st["sri_legend_orientation"],
            breaks=breaks, padding_px=st["sri_legend_padding"])

    acc_store = AccuracyStore(data_dir, enabled=bool(st["accuracy_enabled"] and data_dir))
    _VIZ_STATE["acc_store"] = acc_store

    profile_store = ProfileStore(data_dir)
    profile_store.ensure_baseline(_snapshot())
    _VIZ_STATE["profile_store"] = profile_store

    web_url = None
    if st["web_enabled"]:
        start_web(st["web_port"])
        web_url = f"http://localhost:{st['web_port']}"
        print(f"Web UI: {web_url} (Replay & Shadow Tuner: {web_url}/replay)")

    mqtt_bridge = None
    if st["mqtt_enabled"] and st["mqtt_host"]:
        device_id = st["mqtt_device_id"] or f"dhmz_radar_{city_label(target_name)}"
        enabled_sensors = None
        if st["mqtt_sensors"] != "all":
            enabled_sensors = [x.strip() for x in st["mqtt_sensors"].split(",") if x.strip()]
        mqtt_bridge = HAMQTTBridge(
            host=st["mqtt_host"], port=st["mqtt_port"],
            user=args.mqtt_user, password=args.mqtt_password,
            discovery_prefix=st["mqtt_discovery_prefix"],
            state_prefix=st["mqtt_state_prefix"],
            device_id=device_id, device_name=st["mqtt_device_name"],
            enabled_sensors=enabled_sensors, retain=not st["mqtt_no_retain"],
            tls=st["mqtt_tls"], threat_topic=st["mqtt_threat_topic"],
            web_url=web_url)
        mqtt_bridge.connect()
        print(f"HA MQTT: {st['mqtt_host']}:{st['mqtt_port']} ({device_id})")

    pljusak = PljusakClient(url=st["pljusak_url"]) if st["pljusak_enabled"] else None

    calib = None
    archiver = None
    replay_engine = None
    tuner_started = False
    basemap_img = None
    basemap_north_up = None
    basemap_kind = "greyscale"
    last_basemap_key = None
    last_frame_ts = None
    prev_rain_state = False
    last_threat_sig = set()

    print(f"Target: {target_name} ({target_lat:.4f}N, {target_lon:.4f}E)")
    print(f"Display timezone: {display_tz_name}")

    try:
        while True:
            t_start = time.monotonic()
            now_utc = datetime.now(timezone.utc)
            rt = _snapshot()

            if pljusak:
                if (pljusak.last_fetch_utc is None
                        or (now_utc - pljusak.last_fetch_utc).total_seconds() >= st["pljusak_interval"]):
                    pljusak.fetch(now_utc)

            try:
                img = fetch_image(GIF_URL)
            except Exception as e:
                print(f"Radar fetch error: {e}", file=sys.stderr)
                if args.once: break
                _FORCE_REFRESH.wait(timeout=min(60, rt["poll_s"]))
                _FORCE_REFRESH.clear()
                continue

            if calib is None:
                calib = load_calibration(st["calibration"], st["calibration_inline"],
                                         image_size=img.size)
                if sri_legend:
                    x0, y0, x1, y1 = sri_legend.region
                    legend_mask = np.zeros((img.size[1], img.size[0]), dtype=bool)
                    legend_mask[max(0, y0):min(img.size[1], y1),
                                max(0, x0):min(img.size[0], x1)] = True
                if data_dir:
                    archiver = StormArchiver(data_dir, calib, target_lat, target_lon, target_name)
                    _VIZ_STATE["archiver"] = archiver
                    replay_engine = OfflineReplayEngine(
                        archiver, calib, target_lat, target_lon,
                        st["frame_interval_min"], st["radar_intensity"], sri_legend)
                    _VIZ_STATE["replay_engine"] = replay_engine
                    if not tuner_started:
                        tuner_started = True
                        threading.Thread(
                            target=_shadow_autotune_worker,
                            args=(replay_engine, profile_store, archiver),
                            daemon=True).start()

            if sri_legend and not sri_legend.is_ready():
                try:
                    img.seek(max(0, getattr(img, "n_frames", 1) - 1))
                except Exception:
                    pass
                sri_legend.calibrate(img.convert("RGB"))
                if args.dump_sri_calibration:
                    sri_legend.print_samples()
                    return

            frame_ts, raw_ocr, _ = read_frame_timestamp(
                img, ts_region, ts_tz, source=st["timestamp_source"])
            if frame_ts is not None:
                last_frame_ts = frame_ts
            eff_interval = st["frame_interval_min"]
            data_age = ((now_utc - last_frame_ts).total_seconds() / 60.0
                        if last_frame_ts else None)

            (last_mask, last_mmh, raw_tracks, n_frames, rain_history,
             label_array, tracks_by_id, split_frames_by_id,
             cluster_first_target_hit, frame_masks_mmh) = track_animation(
                img, calib, eff_interval, target_lat, target_lon,
                st["radar_intensity"], sri_legend, legend_mask, rt=rt)

            tracks = [t for t in raw_tracks
                      if t.n_observations >= min(rt["min_duration_frames"], n_frames)
                      and t.cluster.size >= rt["min_size_px"]]
            approaches = [assess(t, target_lat, target_lon, calib) for t in tracks]
            approaching_flags = [refine_approaching(t, a, rt) for t, a in zip(tracks, approaches)]
            threat_flags = [refine_threat(t, a, rt) for t, a in zip(tracks, approaches)]

            nc = compute_nowcast(calib, target_lat, target_lon, tracks, approaches,
                                 rain_history, label_array, eff_interval, rt)

            stations = []
            local_wx = LocalWeather(radius_km=float(rt.get("weather_radius_km", 5.0)))
            if pljusak and pljusak.last_readings:
                all_stations = pljusak.get_all_stations(target_lat, target_lon, display_tz)
                stations = [s for s in all_stations if s.dist_km <= rt["radius_km"]]
                local_wx = summarize_local_weather(
                    all_stations,
                    radius_km=float(rt.get("weather_radius_km", 5.0)),
                    max_age_min=float(st["station_confirm_max_age_min"]),
                )

            ca = combine_analysis(
                nc.raining_now, stations, tracks, approaches, calib,
                target_lat, target_lon,
                st["station_confirm_radius"], st["station_confirm_min_mm"],
                st["station_confirm_max_age_min"], st["station_confirm_fraction"],
                st["station_confirm_min_count"], rt)

            # Archive active storm frames for offline replay & auto-tuning
            if archiver:
                archiver.process_animation(
                    img, frame_masks_mmh, last_frame_ts, eff_interval, now_utc, rt,
                    local_weather=local_wx, target_raining=ca.raining_now)

            # Parallel Live Shadow Runner (never modifies live state or HA sensors)
            shadow_report = run_parallel_shadow_cycle(
                frame_masks_mmh, calib, target_lat, target_lon, eff_interval,
                now_utc, ca.raining_now, prev_rain_state, acc_store,
                profile_store, replay_engine, rt)

            # Basemap fetch (north-up first when auto-rotation is on, then rotated)
            size_px = st["basemap_size"] or 640
            bm_choice = rt["basemap"]
            want_mapbox = (bm_choice == "mapbox") or (bm_choice == "auto" and bool(st["mapbox_token"]))
            style = st["mapbox_style"] or "mapbox/satellite-v9"

            if want_mapbox and st["mapbox_token"] and rt["view_rotation_auto"]:
                nu_key = (style, round(target_lat, 4), round(target_lon, 4),
                          round(rt["radius_km"], 0), 0.0, size_px)
                if nu_key != last_basemap_key or basemap_north_up is None:
                    basemap_north_up = fetch_basemap(
                        style, st["mapbox_token"], target_lat, target_lon,
                        rt["radius_km"], 0.0, size_px, data_dir)

            rot_auto_used, view_rot_used = get_effective_rotation(
                rt, target_lat, target_lon, basemap_north_up=basemap_north_up)

            bm_key = (bm_choice, style, round(target_lat, 4), round(target_lon, 4),
                      round(rt["radius_km"], 0), round(view_rot_used, 1), size_px)
            if bm_key != last_basemap_key:
                last_basemap_key = bm_key
                basemap_img = None
                basemap_kind = "greyscale"
                if want_mapbox and st["mapbox_token"]:
                    if abs(view_rot_used) < 0.05 and basemap_north_up is not None:
                        basemap_img = basemap_north_up
                    else:
                        basemap_img = fetch_basemap(
                            style, st["mapbox_token"], target_lat, target_lon,
                            rt["radius_km"], view_rot_used, size_px, data_dir)
                    if basemap_img is not None:
                        basemap_kind = "mapbox"
                    elif bm_choice == "mapbox":
                        print("basemap: mapbox requested but failed, falling back to greyscale",
                              file=sys.stderr)
                if basemap_img is None and bm_choice == "none":
                    basemap_kind = "none"

            # Accuracy recording (Champion)
            if acc_store.enabled:
                now_iso = now_utc.isoformat()
                tag = rt.get("settings_tag", "")
                rt_snap = {k: rt[k] for k in ("approaching_radius_km", "cluster_radius_weight",
                                              "only_approach_deg", "threat_radius_km",
                                              "threat_angle_deg", "threat_min_duration",
                                              "threat_min_sri_mmh", "min_size_px",
                                              "min_duration_frames", "velocity_window_min",
                                              "velocity_adaptive")}
                if ca.raining_now and not prev_rain_state:
                    acc_store.observe(now_iso, "rain_start", key="target", tag=tag)
                if not ca.raining_now and prev_rain_state:
                    acc_store.observe(now_iso, "rain_end", key="target", tag=tag)
                if not ca.raining_now and nc.next_rain_eta_min is not None:
                    acc_store.predict(now_iso, "rain_start", nc.next_rain_eta_min,
                                      key="target", tag=tag, settings=rt_snap,
                                      horizon_min=nc.next_rain_eta_min)
                if ca.raining_now and nc.rain_end_eta_min is not None:
                    acc_store.predict(now_iso, "rain_end", nc.rain_end_eta_min,
                                      key="target", tag=tag, settings=rt_snap,
                                      horizon_min=nc.rain_end_eta_min)
                for cid, hit_frame in cluster_first_target_hit.items():
                    obs = tracks_by_id.get(cid, [])
                    first_f = obs[0][0] if obs else 0
                    key = f"{first_f}:{cid}"
                    if key in acc_store.observed_cluster_arrivals: continue
                    acc_store.observed_cluster_arrivals.add(key)
                    acc_store.observe(now_iso, "cluster_arrive", key=key, tag=tag,
                                      extra={"cid": cid, "hit_frame": hit_frame})
                for tr, a, af, tf in zip(tracks, approaches, approaching_flags, threat_flags):
                    if not af or a.eta_min is None: continue
                    obs = tracks_by_id.get(tr.cid, [])
                    first_f = obs[0][0] if obs else 0
                    key = f"{first_f}:{tr.cid}"
                    acc_store.predict(now_iso, "cluster_arrive", a.eta_min,
                                      key=key, tag=tag, settings=rt_snap, horizon_min=a.eta_min)
                    if tf:
                        acc_store.predict(now_iso, "threat", a.eta_min,
                                          key=key, tag=tag, settings=rt_snap, horizon_min=a.eta_min)
            prev_rain_state = ca.raining_now

            # Render outputs (include purple shadow vectors if enabled)
            out_size = max(128, int(round(size_px * rt["web_scale"])))
            shadow_overlay_tracks = (
                shadow_report.get("tracks")
                if (rt.get("shadow_enabled", True) and rt.get("shadow_overlay_live", True))
                else None
            )
            png_bytes = render_composed_png(
                img, calib, target_lat, target_lon, rt["radius_km"],
                view_rot_used, out_size, basemap_img, basemap_kind,
                tracks, approaches, approaching_flags, threat_flags,
                stations, rt["show_cities"],
                st["radar_intensity"], sri_legend, legend_mask,
                sri_thresholds, rt, confirm_mm=st["station_confirm_min_mm"],
                shadow_tracks=shadow_overlay_tracks)
            gif_bytes = render_composed_gif(
                img, calib, target_lat, target_lon, rt["radius_km"],
                view_rot_used, out_size, basemap_img, basemap_kind,
                tracks, tracks_by_id, split_frames_by_id, eff_interval,
                approaches, approaching_flags, threat_flags,
                stations, rt["show_cities"],
                st["radar_intensity"], sri_legend, legend_mask,
                sri_thresholds, rt, confirm_mm=st["station_confirm_min_mm"],
                n_unique_frames=len(frame_masks_mmh))

            px_t, py_t = calib.latlon_to_image_pixel(target_lat, target_lon)
            xi, yi = int(round(px_t)), int(round(py_t))
            radar_mmh = (round(float(last_mmh[yi, xi]), 1)
                         if (0 <= yi < last_mmh.shape[0] and 0 <= xi < last_mmh.shape[1]
                             and last_mask[yi, xi])
                         else 0.0)

            appr_items = [(tr, a) for tr, a, af in zip(tracks, approaches, approaching_flags) if af]
            threat_items = [(tr, a) for tr, a, tf in zip(tracks, approaches, threat_flags) if tf]
            appr_items.sort(key=lambda x: x[1].eta_min if x[1].eta_min is not None else 9999)
            threat_items.sort(key=lambda x: x[1].eta_min if x[1].eta_min is not None else 9999)

            nearest_cell_km = round(appr_items[0][1].dist_km, 1) if appr_items else None
            nearest_cell_eta = (round(appr_items[0][1].eta_min, 0)
                                if (appr_items and appr_items[0][1].eta_min is not None) else None)
            nearest_st_km = round(stations[0].dist_km, 1) if stations else None
            nearest_st_age = (round(stations[0].age_min, 1)
                              if (stations and stations[0].age_min is not None) else None)

            state_dict = {
                "raining_now": ca.raining_now,
                "raining_confirmed": ca.confirmed,
                "rain_confidence": ca.confidence,
                "rain_duration_min": round(nc.rain_duration_min, 0) if nc.rain_duration_min is not None else None,
                "rain_end_eta_min": round(nc.rain_end_eta_min, 0) if nc.rain_end_eta_min is not None else None,
                "next_rain_eta_min": round(nc.next_rain_eta_min, 0) if nc.next_rain_eta_min is not None else None,
                "gap_min": round(nc.gap_min, 0) if nc.gap_min is not None else None,
                "summary": nc.summary,
                "radar_intensity_mmh": radar_mmh,
                "nearest_station_mm": round(ca.nearest_station.reading.precip_1h, 1) if ca.nearest_station else None,
                "intensity_mmh": round(ca.intensity_mmh, 1) if ca.intensity_mmh is not None else None,
                "station_trend": ca.station_trend,
                "upstream_trend": ca.upstream_trend,
                "local_temp_c": local_wx.temp_c,
                "local_humidity": local_wx.humidity_pct,
                "local_pressure_hpa": local_wx.pressure_hpa,
                "local_wind_speed_kmh": local_wx.wind_speed_kmh,
                "local_wind_gust_kmh": local_wx.wind_gust_kmh,
                "local_wind_dir_deg": local_wx.wind_dir_deg,
                "local_wind_dir": local_wx.wind_dir_cardinal,
                "local_rain_today_mm": local_wx.precip_today_mm,
                "local_rain_1h_mm": local_wx.precip_1h_mm,
                "local_rain_rate_mmh": local_wx.rain_rate_mmh,
                "nowcast_error_min": acc_store.last_mae_min(),
                "data_age_min": round(data_age, 1) if data_age is not None else None,
                "frame_timestamp": last_frame_ts.isoformat() if last_frame_ts else None,
                "cell_count": len(appr_items),
                "nearest_cell_km": nearest_cell_km,
                "nearest_cell_eta_min": nearest_cell_eta,
                "nearest_station_km": nearest_st_km,
                "nearest_station_age_min": nearest_st_age,
                "station_count_wet": ca.station_count_wet,
                "station_count_fresh": ca.station_count_fresh,
                "station_wet_fraction": round(ca.wet_fraction * 100, 0) if ca.wet_fraction is not None else None,
                "station_median_mm": round(ca.station_median_mm, 1) if ca.station_median_mm is not None else None,
                "local_weather_stations": local_wx.station_count,
            }
            attrs_by_key = {
                "raining_now": {"radar_raining": ca.radar_raining, "confidence": ca.confidence},
                "raining_confirmed": {
                    "wet_stations": [s.reading.name for s in ca.wet_stations],
                    "fresh_count": ca.station_count_fresh, "stale_count": ca.station_count_stale,
                },
                "summary": {
                    "threats": [
                        {"label": cluster_label(i), "size_px": tr.cluster.size,
                         "max_mmh": round(tr.cluster.max_mmh, 1),
                         "miss_km": round(a.closest_km, 1) if a.closest_km is not None else None,
                         "eta_min": round(a.eta_min, 0) if a.eta_min is not None else None}
                        for i, (tr, a) in enumerate(threat_items)
                    ],
                },
                "local_temp_c": {
                    "radius_km": local_wx.radius_km,
                    "station_count": local_wx.station_count,
                    "nearest_station": local_wx.nearest_station,
                    "nearest_dist_km": local_wx.nearest_dist_km,
                    "temp_trend_c_h": local_wx.temp_trend,
                    "pressure_trend_hpa_h": local_wx.pressure_trend,
                    "precip_6h_mm": local_wx.precip_6h_mm,
                    "precip_24h_mm": local_wx.precip_24h_mm,
                },
            }

            if mqtt_bridge:
                mqtt_bridge.publish_state(state_dict, attrs_by_key=attrs_by_key,
                                          image_bytes=png_bytes)
                cur_sig = {(tr.cid, round(a.eta_min or 0, -1)) for tr, a in threat_items}
                if cur_sig != last_threat_sig:
                    last_threat_sig = cur_sig
                    if threat_items:
                        mqtt_bridge.publish_threat({
                            "ts": now_utc.isoformat(), "target": target_name,
                            "count": len(threat_items),
                            "threats": attrs_by_key["summary"]["threats"],
                        })

            shadow_status_public = {k: v for k, v in shadow_report.items() if k != "tracks"}

            with _VIZ_STATE["lock"]:
                _VIZ_STATE["png"] = png_bytes
                _VIZ_STATE["gif"] = gif_bytes
                _VIZ_STATE["updated"] = now_utc.astimezone(display_tz).strftime("%Y-%m-%d %H:%M:%S %Z")
                _VIZ_STATE["status"] = {
                    "target": target_name,
                    "settings_tag": rt.get("settings_tag", "baseline"),
                    "updated": _VIZ_STATE["updated"],
                    "confidence": ca.confidence,
                    "raining_now": ca.raining_now,
                    "approaching_count": len(appr_items),
                    "threat_count": len(threat_items),
                    "view_rotation_used": view_rot_used,
                    "view_rotation_auto": rot_auto_used,
                    "basemap_used": basemap_kind,
                    "local_weather": {
                        "radius_km": local_wx.radius_km,
                        "station_count": local_wx.station_count,
                        "nearest_station": local_wx.nearest_station,
                        "nearest_dist_km": local_wx.nearest_dist_km,
                        "temp_c": local_wx.temp_c,
                        "temp_trend": local_wx.temp_trend,
                        "humidity_pct": local_wx.humidity_pct,
                        "pressure_hpa": local_wx.pressure_hpa,
                        "pressure_trend": local_wx.pressure_trend,
                        "wind_speed_kmh": local_wx.wind_speed_kmh,
                        "wind_gust_kmh": local_wx.wind_gust_kmh,
                        "wind_dir_deg": local_wx.wind_dir_deg,
                        "wind_dir_cardinal": local_wx.wind_dir_cardinal,
                        "precip_today_mm": local_wx.precip_today_mm,
                        "precip_1h_mm": local_wx.precip_1h_mm,
                        "precip_6h_mm": local_wx.precip_6h_mm,
                        "precip_24h_mm": local_wx.precip_24h_mm,
                        "rain_rate_mmh": local_wx.rain_rate_mmh,
                    },
                    "archive": archiver.summary() if archiver else {"enabled": False},
                    "shadow": shadow_status_public,
                    "sensors": state_dict,
                    "threats": attrs_by_key["summary"]["threats"],
                }

            sh_info = (f" | shadow[{shadow_report['profile_name']}]="
                       f"{shadow_report['live']['approaching_count']}ap/"
                       f"{shadow_report['live']['threat_count']}th"
                       if shadow_report.get("enabled") else "")
            print(f"[{now_utc.astimezone(display_tz).strftime('%H:%M:%S')}] "
                  f"{ca.confidence} | {nc.summary} | "
                  f"approaching={len(appr_items)} threats={len(threat_items)}{sh_info}")

            if args.once:
                break
            elapsed = time.monotonic() - t_start
            wait_s = max(5.0, rt["poll_s"] - elapsed)
            _FORCE_REFRESH.wait(timeout=wait_s)
            _FORCE_REFRESH.clear()
    except KeyboardInterrupt:
        pass
    finally:
        if mqtt_bridge:
            mqtt_bridge.close()


if __name__ == "__main__":
    main()
