#!/usr/bin/env python3
"""
DHMZ radar nowcaster — local, flow-based, replay-first.

Design:
  * Scope is local (default 120 km focus radius). We are not trying to nowcast
    the whole Adriatic; DHMZ's model does that. We nowcast the 0–2 h window at
    one point (or a small target region), where radar extrapolation beats
    persistence.
  * Motion is estimated blockwise by cross-correlation (TREC-style), producing
    a per-pixel flow field. No centroids are used for velocity, ETA, or
    direction. Centroids survive only as label anchors for the map.
  * ETA is computed by extrapolating every precip pixel forward and asking
    "when do enough pixels cross the target boundary?". This is shape-aware by
    construction: convex, concave, crescent, elongated, split — all handled
    correctly, with no area-equivalent radius approximations.
  * Every archived storm frame has its replay PNG precomputed at archive time.
    Replay is a static file fetch. No on-the-fly rendering, no lag.
  * The frame buffer is append-only and probed by the DHMZ static PNG
    (kompozit-stat.png). Cold start decodes the full GIF; steady state fetches
    one PNG per cycle.

Endpoints:
  /                  live dashboard
  /replay            offline replay (fast — reads precomputed PNGs)
  /threat.png        current frame composite
  /threat.gif        animated composite
  /status            JSON: sensors, nowcast, local_weather, archive
  /settings          GET/POST hot-swappable runtime
  /config            GET/POST/DELETE persisted config
  /archive           GET episodes + frame counts
  /replay/frame      GET PNG (precomputed, or on-demand fallback)
  /accuracy          GET stats
  /accuracy/predictions, /accuracy/events
  /accuracy/reset, /accuracy/replay
  /refresh           POST force refresh
"""

import argparse
import hashlib
import io
import json
import math
import os
import re
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

import numpy as np
import requests
import yaml
from PIL import Image, ImageDraw, ImageEnhance, ImageSequence
from scipy import ndimage
from scipy.signal import correlate2d
from scipy.spatial import cKDTree

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


# ============================================================================
# SECTION 1 — Constants
# ============================================================================

GIF_URL = "https://vrijeme.hr/anim_kompozit.gif"
STATIC_PNG_URL = "https://vrijeme.hr/kompozit-stat.png"
RADAR_HEADERS = {"User-Agent": "Mozilla/5.0 (dhmz-radar-nowcaster/2.0)"}
PLJUSAK_URL = "https://pljusak.com/"
PLJUSAK_HEADERS = {
    "User-Agent": "Mozilla/5.0 (pljusak-station-puller/1.0)",
    "Referer": "https://pljusak.com/",
    "Accept": "text/html,application/xhtml+xml",
}

MIN_SAT = 0.45
MIN_VAL = 0.25
MIN_CLUSTER_PIXELS = 15
MIN_SPEED_KMH = 2.0

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
DEFAULT_CALIBRATION_JSON = (
    '{"map":{"x0":0,"y0":0,"x1":720,"y1":720},'
    '"points":['
    '{"pixel":[343,226],"city":"Zagreb"},'
    '{"pixel":[385,490],"city":"Split"},'
    '{"pixel":[566,258],"city":"Osijek"}'
    ']}'
)

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


# ============================================================================
# SECTION 2 — Runtime (hot-swappable) + Static (restart-required) config
# ============================================================================

_RUNTIME = {
    "focus_radius_km": 120.0,
    "threat_radius_km": 20.0,
    "arrival_pixel_threshold": 20,
    "departure_pixel_threshold": 20,
    "max_horizon_min": 120.0,
    "step_min": 5.0,
    "min_size_px": 20,
    "min_sri_mmh": 0.0,
    # Flow field — bumped averaging to 8 pairs = 40 min (was 5 = 25 min)
    "flow_downsample": 2,
    "flow_block_px": 24,
    "flow_search_px": 10,
    "flow_average_pairs": 8,
    "flow_smooth_sigma": 2.0,
    # Display
    "fit_check_overlay": False,
    "arrow_min_speed_kmh": 5.0,   # NEW — display filter, arrows only shown above this
    # View
    "radius_km": 80.0,
    "weather_radius_km": 5.0,
    "view_rotation_auto": True,
    "view_rotation": 0.0,
    "basemap": "auto",
    "basemap_saturation": 1.0,
    "basemap_brightness": 1.0,
    "basemap_contrast": 1.0,
    "show_cities": True,
    "web_scale": 1.0,
    # Archive
    "archive_enabled": True,
    "archive_radius_km": 100.0,
    "archive_min_pixels": 20,
    "archive_precompute_replay": True,
    # Loop
    "poll_s": 300,
    "settings_tag": "local",
}
_RUNTIME_TYPES = {
    "focus_radius_km": (float, 20.0, 1000.0),
    "threat_radius_km": (float, 1.0, 500.0),
    "arrival_pixel_threshold": (int, 1, 5000),
    "departure_pixel_threshold": (int, 1, 5000),
    "max_horizon_min": (float, 30.0, 360.0),
    "step_min": (float, 1.0, 30.0),
    "min_size_px": (int, 5, 5000),
    "min_sri_mmh": (float, 0.0, 500.0),
    "flow_downsample": (int, 1, 6),
    "flow_block_px": (int, 6, 64),
    "flow_search_px": (int, 2, 32),
    "flow_average_pairs": (int, 1, 12),
    "flow_smooth_sigma": (float, 0.0, 8.0),
    "fit_check_overlay": (bool, None, None),
    "arrow_min_speed_kmh": (float, 0.0, 30.0),
    "radius_km": (float, 20.0, 1000.0),
    "weather_radius_km": (float, 0.5, 100.0),
    "view_rotation_auto": (bool, None, None),
    "view_rotation": (float, 0.0, 360.0),
    "basemap": (str, None, None),
    "basemap_saturation": (float, 0.0, 2.0),
    "basemap_brightness": (float, 0.2, 2.0),
    "basemap_contrast": (float, 0.2, 2.0),
    "show_cities": (bool, None, None),
    "web_scale": (float, 0.1, 4.0),
    "archive_enabled": (bool, None, None),
    "archive_radius_km": (float, 20.0, 500.0),
    "archive_min_pixels": (int, 10, 5000),
    "archive_precompute_replay": (bool, None, None),
    "poll_s": (int, 30, 7200),
    "settings_tag": (str, None, None),
}

_STATIC_DEFAULTS = {
    "location": None, "lat": None, "lon": None,
    "calibration_inline": DEFAULT_CALIBRATION_JSON, "calibration": None,
    "radar_intensity": "simple",
    "sri_legend_region": None, "sri_legend_orientation": "horizontal",
    "sri_legend_padding": 1, "sri_breaks": None, "sri_thresholds": "2.5,10,30",
    "frame_interval_min": 5.0,
    "timestamp_region": DEFAULT_TIMESTAMP_REGION_STR,
    "timestamp_enabled": True, "timestamp_tz": None, "timestamp_source": "last",
    "static_png_enabled": True, "static_timestamp_region": None,
    "display_tz": DEFAULT_DISPLAY_TZ,
    "pljusak_enabled": True, "pljusak_url": PLJUSAK_URL, "pljusak_interval": 60,
    "station_confirm_radius": 15.0, "station_confirm_min_mm": 0.2,
    "station_confirm_max_age_min": 15.0, "station_confirm_fraction": 0.5,
    "station_confirm_min_count": 1,
    "web_enabled": True, "web_port": 8080,
    "mqtt_enabled": True, "mqtt_host": None, "mqtt_port": 1883, "mqtt_tls": False,
    "mqtt_discovery_prefix": "homeassistant", "mqtt_state_prefix": "dhmz_radar",
    "mqtt_device_id": None, "mqtt_device_name": "DHMZ Radar",
    "mqtt_sensors": "all", "mqtt_no_retain": False, "mqtt_threat_topic": None,
    "jsonl_dump": None, "accuracy_enabled": True,
    "mapbox_token": None, "mapbox_style": "mapbox/satellite-v9", "basemap_size": 640,
}

_RUNTIME = dict(_RUNTIME)
_STATIC = dict(_STATIC_DEFAULTS)
_RUNTIME_LOCK = threading.Lock()
_STATIC_LOCK = threading.Lock()
_FORCE_REFRESH = threading.Event()

_VIZ_STATE = {
    "png": None, "gif": None, "updated": None, "status": {},
    "lock": threading.Lock(),
}

_REPLAY_CACHE = OrderedDict()
_REPLAY_CACHE_MAX = 300
_REPLAY_CACHE_LOCK = threading.Lock()


# ============================================================================
# SECTION 3 — Config helpers
# ============================================================================

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
    existing = _load_json_file(cfg_path) or {}
    now = datetime.now(timezone.utc).isoformat()
    _save_json_file(cfg_path, {
        "version": 2,
        "created_at": existing.get("created_at", now),
        "updated_at": now,
        "runtime": _snapshot(),
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

    # CLI overrides
    for name in ("focus_radius_km", "threat_radius_km", "arrival_pixel_threshold",
                 "departure_pixel_threshold", "max_horizon_min", "step_min",
                 "min_size_px", "min_sri_mmh", "flow_downsample", "flow_block_px",
                 "flow_search_px", "flow_average_pairs", "radius_km",
                 "weather_radius_km", "view_rotation", "basemap",
                 "basemap_saturation", "basemap_brightness", "basemap_contrast",
                 "web_scale", "archive_radius_km", "archive_min_pixels",
                 "poll_s", "settings_tag"):
        v = getattr(cli_args, name, None)
        if v is not None:
            runtime[name] = _coerce(name, v)

    for name in ("view_rotation_auto", "show_cities", "archive_enabled",
                 "archive_precompute_replay"):
        v = getattr(cli_args, name, None)
        if v is not None:
            runtime[name] = bool(v)

    for name in ("calibration", "calibration_inline", "radar_intensity",
                 "sri_legend_region", "sri_legend_orientation", "sri_legend_padding",
                 "sri_breaks", "sri_thresholds", "frame_interval_min",
                 "timestamp_region", "timestamp_tz", "timestamp_source",
                 "static_timestamp_region", "display_tz", "pljusak_url",
                 "pljusak_interval", "station_confirm_radius",
                 "station_confirm_min_mm", "station_confirm_max_age_min",
                 "station_confirm_fraction", "station_confirm_min_count",
                 "web_port", "mqtt_host", "mqtt_port", "ha_discovery_prefix",
                 "ha_state_prefix", "ha_device_id", "ha_device_name",
                 "ha_sensors", "ha_threat_topic", "jsonl_dump",
                 "mapbox_token", "mapbox_style", "basemap_size", "location"):
        v = getattr(cli_args, name, None)
        if v is not None:
            key = {
                "ha_discovery_prefix": "mqtt_discovery_prefix",
                "ha_state_prefix": "mqtt_state_prefix",
                "ha_device_id": "mqtt_device_id",
                "ha_device_name": "mqtt_device_name",
                "ha_sensors": "mqtt_sensors",
                "ha_threat_topic": "mqtt_threat_topic",
            }.get(name, name)
            static[key] = v

    for name in ("web_enabled", "pljusak_enabled", "timestamp_enabled",
                 "mqtt_enabled", "accuracy_enabled", "mqtt_no_retain",
                 "static_png_enabled", "mqtt_tls"):
        v = getattr(cli_args, name, None)
        if v is not None:
            key = "mqtt_tls" if name == "mqtt_tls" else name
            static[key] = bool(v)

    if cli_args.lat is not None:
        static["lat"] = cli_args.lat
        static["location"] = None
    if cli_args.lon is not None:
        static["lon"] = cli_args.lon
        static["location"] = None
    if cli_args.location is not None:
        static["location"] = cli_args.location
        static["lat"] = None
        static["lon"] = None

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


# ============================================================================
# SECTION 4 — Cities, timezone
# ============================================================================

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


def cluster_label(idx):
    return CLUSTER_DIGITS[idx] if 0 <= idx < len(CLUSTER_DIGITS) else f"#{idx + 1}"


def bearing_label(deg):
    dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    return dirs[int((deg + 11.25) % 360 // 22.5)]


# ============================================================================
# SECTION 5 — Coast rotation
# ============================================================================

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
    total_grad = float(np.sum(np.hypot(gx, gy) * w))
    if total_grad <= 1e-6 or (math.hypot(sx, sy) / total_grad) < 0.15:
        return None
    return math.degrees(math.atan2(sx, -sy)) % 360.0


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
        t = max(0.0, min(1.0, (tx * sx + ty * sy) / seg_sq))
        dist = math.hypot(tx - sx * t, ty - sy * t)
        if dist < best_dist:
            best_dist = dist
            best_bearing = math.degrees(math.atan2(sx, sy)) % 360
    if best_dist > max_dist_km or best_bearing is None:
        return None, best_dist
    return (best_bearing + 90.0) % 360.0, best_dist


def get_effective_rotation(rt, target_lat, target_lon, basemap_north_up=None):
    if rt["view_rotation_auto"]:
        rot = detect_coast_rotation_from_satellite(basemap_north_up)
        if rot is not None:
            return True, rot
        rot, _ = compute_coast_rotation(target_lat, target_lon)
        if rot is not None:
            return True, rot
    return False, float(rt["view_rotation"])


# ============================================================================
# SECTION 6 — OCR
# ============================================================================

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


def read_frame_timestamp_from_pil(frame, region, tz):
    """OCR a single already-loaded PIL frame's timestamp region."""
    if not HAVE_TESSERACT or region is None or frame is None:
        return None, None
    try:
        crop = frame.convert("RGB").crop(region)
        w, h = crop.size
        if w < 10 or h < 5:
            return None, None
        crop = crop.resize((w * 3, h * 3), Image.LANCZOS)
        gray = np.array(crop.convert("L"))
        t = _otsu(gray)
        binary = ((gray > t) * 255).astype(np.uint8)
        if binary.mean() < 128:
            binary = 255 - binary
        text = pytesseract.image_to_string(
            Image.fromarray(binary), config="--psm 7").strip()
        return parse_timestamp_text(text, tz), text
    except Exception:
        return None, None


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
        dt, txt = read_frame_timestamp_from_pil(frame, region, tz)
        return dt, txt, idx
    except Exception:
        return None, None, 0
    finally:
        try:
            img.seek(0)
        except Exception:
            pass


# ============================================================================
# SECTION 7 — Geometry + calibration
# ============================================================================

def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


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


# ============================================================================
# SECTION 8 — SRI legend
# ============================================================================

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


# ============================================================================
# SECTION 9 — Radar image analysis
# ============================================================================

def fetch_image(url, timeout=30):
    r = requests.get(url, headers=RADAR_HEADERS, timeout=timeout)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content))


def fetch_static_png(timeout=30):
    r = requests.get(STATIC_PNG_URL, headers=RADAR_HEADERS, timeout=timeout)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGB")


def _dedupe_trailing_static_frames(frames):
    """Drop trailing frames pixel-identical to the last frame. Keeps the FIRST
    frame of the trailing run so indices stay contiguous from zero."""
    if len(frames) < 2:
        return frames
    last_hash = hashlib.md5(frames[-1].tobytes()).hexdigest()
    keep_from = len(frames) - 1
    while keep_from > 0:
        if hashlib.md5(frames[keep_from - 1].tobytes()).hexdigest() != last_hash:
            break
        keep_from -= 1
    return frames[:keep_from + 1]


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


# ============================================================================
# SECTION 10 — Flow field estimation (blockwise cross-correlation)
# ============================================================================

def _subpixel_peak(corr, py, px):
    """Refine an integer correlation peak to sub-pixel via parabolic fit.

    For slow storms (< 1 px per frame of motion), the integer argmax flips
    between adjacent offsets frame-to-frame, which makes velocity direction
    swing wildly. Parabolic fitting resolves the peak to ~0.05 px, which
    when combined with temporal averaging gives stable velocities even for
    sub-pixel motion.
    """
    sub_dy = 0.0
    sub_dx = 0.0
    if 0 < py < corr.shape[0] - 1:
        c0 = float(corr[py - 1, px])
        c1 = float(corr[py, px])
        c2 = float(corr[py + 1, px])
        denom = c0 - 2.0 * c1 + c2
        if abs(denom) > 1e-9:
            sub_dy = 0.5 * (c0 - c2) / denom
    if 0 < px < corr.shape[1] - 1:
        c0 = float(corr[py, px - 1])
        c1 = float(corr[py, px])
        c2 = float(corr[py, px + 1])
        denom = c0 - 2.0 * c1 + c2
        if abs(denom) > 1e-9:
            sub_dx = 0.5 * (c0 - c2) / denom
    # Parabolic fit is only valid very close to the peak
    sub_dy = max(-0.5, min(0.5, sub_dy))
    sub_dx = max(-0.5, min(0.5, sub_dx))
    return py + sub_dy, px + sub_dx


def estimate_flow_field(prev_mask, curr_mask, block_px=16, search_px=8, downsample=2):
    """Blockwise TREC-style motion field between two consecutive masks.

    Returns flow (H, W, 2) where flow[y, x] = (dx_px, dy_px) is the per-frame
    displacement of the block containing (y, x), in FULL-resolution pixels
    (float, sub-pixel precision).

    Image convention: y increases downward, x increases rightward. So dy > 0
    means motion is southward; dx > 0 means motion is eastward.
    """
    H, W = curr_mask.shape
    if downsample > 1:
        prev_small = prev_mask[::downsample, ::downsample].astype(np.float32)
        curr_small = curr_mask[::downsample, ::downsample].astype(np.float32)
    else:
        prev_small = prev_mask.astype(np.float32)
        curr_small = curr_mask.astype(np.float32)

    Hs, Ws = curr_small.shape
    ny = Hs // block_px
    nx = Ws // block_px
    if ny == 0 or nx == 0:
        return np.zeros((H, W, 2), dtype=np.float32)

    pad = search_px
    curr_pad = np.pad(curr_small, pad, mode='constant')

    best_dy = np.zeros((ny, nx), dtype=np.float32)
    best_dx = np.zeros((ny, nx), dtype=np.float32)

    for by in range(ny):
        y0 = by * block_px
        for bx in range(nx):
            x0 = bx * block_px
            p = prev_small[y0:y0 + block_px, x0:x0 + block_px]
            if p.sum() < 5:
                continue
            search = curr_pad[y0:y0 + block_px + 2 * pad,
                              x0:x0 + block_px + 2 * pad]
            if search.shape[0] < block_px + 1 or search.shape[1] < block_px + 1:
                continue
            corr = correlate2d(search, p, mode='valid')
            idx = np.unravel_index(np.argmax(corr), corr.shape)
            sub_dy, sub_dx = _subpixel_peak(corr, idx[0], idx[1])
            best_dy[by, bx] = sub_dy - pad
            best_dx[by, bx] = sub_dx - pad

    flow_low = np.zeros((ny * block_px, nx * block_px, 2), dtype=np.float32)
    for by in range(ny):
        for bx in range(nx):
            flow_low[by * block_px:(by + 1) * block_px,
                     bx * block_px:(bx + 1) * block_px, 0] = best_dx[by, bx]
            flow_low[by * block_px:(by + 1) * block_px,
                     bx * block_px:(bx + 1) * block_px, 1] = best_dy[by, bx]

    flow_full = np.repeat(np.repeat(flow_low, downsample, axis=0),
                          downsample, axis=1)
    if flow_full.shape[0] < H or flow_full.shape[1] < W:
        padded = np.zeros((H, W, 2), dtype=np.float32)
        padded[:flow_full.shape[0], :flow_full.shape[1]] = flow_full
        flow_full = padded

    flow_full *= downsample
    return flow_full[:H, :W]


def combined_flow(flows, frame_idx, average_pairs=8, smooth_sigma=2.0):
    """Temporal mean of the last `average_pairs` flow pairs, then spatially
    smoothed with a Gaussian.

    average_pairs=8 at 5-min cadence = 40 min lookback, matching the upper
    end of the old adaptive velocity window. The mean of N pairwise flows is
    mathematically equivalent to a least-squares slope fit through N+1
    positions when motion is constant — this is why averaging is sufficient
    for translation, provided the pairwise flows themselves are unbiased
    (which sub-pixel fitting ensures).
    """
    if not flows or frame_idx <= 0:
        return None
    start = max(0, frame_idx - average_pairs)
    window = flows[start:frame_idx]
    if not window:
        return None
    mean = np.mean(np.stack(window, axis=0), axis=0)
    if smooth_sigma > 0:
        mean = np.stack([
            ndimage.gaussian_filter(mean[..., 0], sigma=smooth_sigma, mode='nearest'),
            ndimage.gaussian_filter(mean[..., 1], sigma=smooth_sigma, mode='nearest'),
        ], axis=-1).astype(np.float32)
    return mean


def flow_to_kmh(flow_field, calib, frame_interval_min, x, y):
    y_i = max(0, min(flow_field.shape[0] - 1, int(y)))
    x_i = max(0, min(flow_field.shape[1] - 1, int(x)))
    dx_px = float(flow_field[y_i, x_i, 0])
    dy_px = float(flow_field[y_i, x_i, 1])
    kmpp = calib.km_per_pixel(float(x), float(y))
    interval_h = frame_interval_min / 60.0
    return dx_px * kmpp / interval_h, -dy_px * kmpp / interval_h


def cluster_flow_velocity(flow_field, label_array, label, calib,
                          frame_interval_min):
    """Robust velocity for a cluster: MEAN flow across all its pixels.

    Mean (not median) matches the old system's centroid-displacement velocity.
    For a translating storm, mean(fx) is the centroid's x-velocity, which is
    what the old linear-fit method recovered. Median jumps between motion
    modes for clusters that contain both leading and trailing edges (mixed
    motion), which was a source of arrow swinging.

    Returns (vx_kmh, vy_kmh, coherence) where coherence ∈ [0, 1] is the
    circular variance of per-pixel flow directions inside the cluster.
    Coherence 1.0 = all pixels move the same way (compact, well-defined storm).
    Coherence 0.0 = pixels move in random directions (deforming complex).
    Suppress the arrow when coherence < 0.4.
    """
    pixels = np.argwhere(label_array == label)
    if len(pixels) < 4:
        return 0.0, 0.0, 0.0
    ys = np.clip(pixels[:, 0], 0, flow_field.shape[0] - 1)
    xs = np.clip(pixels[:, 1], 0, flow_field.shape[1] - 1)
    fx = flow_field[ys, xs, 0]
    fy = flow_field[ys, xs, 1]
    active = (np.abs(fx) + np.abs(fy)) > 0.01
    if int(active.sum()) < 4:
        return 0.0, 0.0, 0.0
    # MEAN not median — mass-weighted centroid velocity matches old behavior
    dx_px = float(fx[active].mean())
    dy_px = float(fy[active].mean())
    angles = np.arctan2(fy[active], fx[active])
    mean_cos = float(np.cos(angles).mean())
    mean_sin = float(np.sin(angles).mean())
    coherence = math.hypot(mean_cos, mean_sin)
    kmpp = calib.km_per_pixel(float(xs.mean()), float(ys.mean()))
    interval_h = frame_interval_min / 60.0
    vx_kmh = dx_px * kmpp / interval_h
    vy_kmh = -dy_px * kmpp / interval_h
    return vx_kmh, vy_kmh, coherence


# ============================================================================
# SECTION 11 — Pixel-extrapolation ETA (shape-aware)
# ============================================================================

def _pixels_to_km_offset(calib, pixels_yx, target_lat, target_lon):
    """Convert pixel coordinates to (dx_km_east, dy_km_north) relative to target."""
    ys = pixels_yx[:, 0].astype(np.float64)
    xs = pixels_yx[:, 1].astype(np.float64)
    lat = (calib.coef_lat[0] * (xs - calib.map_region.x0)
           + calib.coef_lat[1] * (ys - calib.map_region.y0)
           + calib.coef_lat[2])
    lon = (calib.coef_lon[0] * (xs - calib.map_region.x0)
           + calib.coef_lon[1] * (ys - calib.map_region.y0)
           + calib.coef_lon[2])
    coslat = math.cos(math.radians(target_lat))
    return ((lon - target_lon) * 111.32 * coslat,
            (lat - target_lat) * 111.32)


def _pixels_to_kmh(calib, pixels_yx, flow_field, frame_interval_min):
    ys = pixels_yx[:, 0].astype(np.int64)
    xs = pixels_yx[:, 1].astype(np.int64)
    ys = np.clip(ys, 0, flow_field.shape[0] - 1)
    xs = np.clip(xs, 0, flow_field.shape[1] - 1)
    dx_px = flow_field[ys, xs, 0]
    dy_px = flow_field[ys, xs, 1]
    kmpp = calib.km_per_pixel(float(xs.mean()), float(ys.mean()))
    interval_h = frame_interval_min / 60.0
    return dx_px * kmpp / interval_h, -dy_px * kmpp / interval_h


def predict_arrival_pixels(calib, pixels_yx, flow_field, target_lat, target_lon,
                           frame_interval_min, target_radius_km=20.0,
                           arrival_px=20, max_horizon_min=120.0, step_min=5.0):
    """Shape-aware ETA. Extrapolates pixels forward and returns the first t at
    which >= arrival_px pixels are inside target_radius_km.

    Returns ETA in minutes (0.0 if already inside), or None if no arrival
    within max_horizon_min.
    """
    if len(pixels_yx) == 0:
        return None
    dx, dy = _pixels_to_km_offset(calib, pixels_yx, target_lat, target_lon)
    r2 = target_radius_km ** 2
    if int((dx * dx + dy * dy <= r2).sum()) >= arrival_px:
        return 0.0
    vx, vy = _pixels_to_kmh(calib, pixels_yx, flow_field, frame_interval_min)
    if float(np.hypot(vx, vy).max()) < MIN_SPEED_KMH:
        return None
    t = step_min
    while t <= max_horizon_min:
        th = t / 60.0
        ex = dx + vx * th
        ey = dy + vy * th
        if int((ex * ex + ey * ey <= r2).sum()) >= arrival_px:
            return float(t)
        t += step_min
    return None


def predict_departure_pixels(calib, pixels_yx, flow_field, target_lat, target_lon,
                             frame_interval_min, target_radius_km=20.0,
                             departure_px=20, max_horizon_min=120.0, step_min=5.0):
    """Symmetric to arrival: first t at which fewer than departure_px pixels
    remain inside target_radius_km."""
    if len(pixels_yx) == 0:
        return 0.0
    dx, dy = _pixels_to_km_offset(calib, pixels_yx, target_lat, target_lon)
    r2 = target_radius_km ** 2
    if int((dx * dx + dy * dy <= r2).sum()) < departure_px:
        return 0.0
    vx, vy = _pixels_to_kmh(calib, pixels_yx, flow_field, frame_interval_min)
    t = step_min
    while t <= max_horizon_min:
        th = t / 60.0
        ex = dx + vx * th
        ey = dy + vy * th
        if int((ex * ex + ey * ey <= r2).sum()) < departure_px:
            return float(t)
        t += step_min
    return None


# ============================================================================
# SECTION 12 — Clustering (label carrier only; no velocity math)
# ============================================================================

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
    cents = ndimage.center_of_mass(mask, label_array, idx)
    if mmh is not None:
        max_vals = ndimage.maximum(mmh, label_array, idx)
        mean_vals = ndimage.mean(mmh, label_array, idx)
    else:
        max_vals = np.zeros(n, dtype=np.float32)
        mean_vals = np.zeros(n, dtype=np.float32)
    out = []
    for lbl, sz, (cy, cx), mx_v, mn_v in zip(idx, sizes, cents, max_vals, mean_vals):
        if sz < min_pixels:
            continue
        if not (math.isfinite(cx) and math.isfinite(cy)):
            continue
        out.append(PixelCluster(float(cx), float(cy), int(sz), int(lbl),
                                float(mx_v), float(mn_v)))
    return out, label_array


# ============================================================================
# SECTION 13 — Nowcast (rain start/end at target)
# ============================================================================

@dataclass
class Nowcast:
    raining_now: bool = False
    rain_duration_min: Optional[float] = None
    rain_end_eta_min: Optional[float] = None
    next_rain_eta_min: Optional[float] = None
    gap_min: Optional[float] = None
    summary: str = ""
    next_cell_label: Optional[int] = None
    next_cell_size: Optional[int] = None


@dataclass
class ClusterArrival:
    cluster: PixelCluster
    eta_min: Optional[float]
    vx_kmh: float
    vy_kmh: float
    coherence: float = 1.0


def analyze_frame_state(mask, mmh, flow, calib, target_lat, target_lon, rt, st):
    """Compute clusters, per-cluster arrival ETAs, and nowcast for ONE frame."""
    clusters, label_array = clusters_from_mask(mask, int(rt["min_size_px"]), mmh=mmh)
    arrivals: list[ClusterArrival] = []
    arrival_thr = int(rt["arrival_pixel_threshold"])
    target_rad = float(rt["threat_radius_km"])
    step = float(rt["step_min"])
    horizon = float(rt["max_horizon_min"])
    fi = float(st["frame_interval_min"])

    for c in clusters:
        pixels = np.argwhere(label_array == c.label)
        if len(pixels) == 0:
            continue
        eta = predict_arrival_pixels(
            calib, pixels, flow, target_lat, target_lon, fi,
            target_radius_km=target_rad, arrival_px=arrival_thr,
            max_horizon_min=horizon, step_min=step)
        vx, vy, coherence = cluster_flow_velocity(
            flow, label_array, c.label, calib, fi)
        arrivals.append(ClusterArrival(
            cluster=c, eta_min=eta,
            vx_kmh=float(vx), vy_kmh=float(vy),
            coherence=float(coherence)))

    nc = Nowcast()
    px, py = calib.latlon_to_image_pixel(target_lat, target_lon)
    xi, yi = int(round(px)), int(round(py))
    in_view = (0 <= yi < mask.shape[0] and 0 <= xi < mask.shape[1])
    nc.raining_now = bool(in_view and mask[yi, xi])

    if nc.raining_now:
        cov_label = int(label_array[yi, xi]) if in_view else 0
        if cov_label > 0:
            pixels = np.argwhere(label_array == cov_label)
            nc.rain_end_eta_min = predict_departure_pixels(
                calib, pixels, flow, target_lat, target_lon, fi,
                target_radius_km=target_rad,
                departure_px=int(rt["departure_pixel_threshold"]),
                max_horizon_min=horizon, step_min=step)
        best = None
        for a in arrivals:
            if a.cluster.label == cov_label:
                continue
            if a.eta_min is not None and (best is None or a.eta_min < best.eta_min):
                best = a
        if best is not None:
            nc.next_rain_eta_min = best.eta_min
            nc.next_cell_label = best.cluster.label
            nc.next_cell_size = best.cluster.size
        if nc.next_rain_eta_min is not None and nc.rain_end_eta_min is not None:
            if nc.next_rain_eta_min > nc.rain_end_eta_min:
                nc.gap_min = nc.next_rain_eta_min - nc.rain_end_eta_min
    else:
        best = None
        for a in arrivals:
            if a.eta_min is not None and (best is None or a.eta_min < best.eta_min):
                best = a
        if best is not None:
            nc.next_rain_eta_min = best.eta_min
            nc.next_cell_label = best.cluster.label
            nc.next_cell_size = best.cluster.size

    if nc.raining_now:
        parts = ["Raining now."]
        if nc.rain_end_eta_min is not None:
            parts.append(f"Ends ~{nc.rain_end_eta_min:.0f} min.")
        if nc.gap_min is not None:
            parts.append(f"Gap ~{nc.gap_min:.0f} min, next ~{nc.next_rain_eta_min:.0f} min.")
        elif nc.next_rain_eta_min is not None:
            parts.append(f"Next cell ~{nc.next_rain_eta_min:.0f} min.")
        nc.summary = " ".join(parts)
    else:
        if nc.next_rain_eta_min is not None:
            nc.summary = f"Dry. Rain in ~{nc.next_rain_eta_min:.0f} min."
        else:
            nc.summary = "Dry. No approach within horizon."
    return clusters, label_array, arrivals, nc


# ============================================================================
# SECTION 14 — Frame buffer (append-only, timestamp-keyed)
# ============================================================================

FRAME_BUFFER_MAX = 30
FRAME_BUFFER_GAP_TOL = 1.5


def init_frame_buffer_from_gif(img, ts_region, ts_tz):
    """Cold start: decode GIF, OCR each frame, dedupe by timestamp."""
    frames = [f.convert("RGB") for f in ImageSequence.Iterator(img)]
    buf = []
    seen = set()
    for f in frames:
        dt, _ = read_frame_timestamp_from_pil(f, ts_region, ts_tz)
        if dt is None or dt in seen:
            continue
        seen.add(dt)
        buf.append((dt, f))
    buf.sort(key=lambda x: x[0])
    return buf[-FRAME_BUFFER_MAX:]


def append_static_png_to_buffer(static_img, buf, ts_region, ts_tz,
                                ref_size=None, interval_min=5.0):
    """Steady state: OCR the static PNG; append if newer."""
    dt, _ = read_frame_timestamp_from_pil(static_img, ts_region, ts_tz)
    if dt is None:
        return "ocr_fail", None
    img = static_img.convert("RGB")
    if ref_size is not None and img.size != ref_size:
        img = img.resize(ref_size, Image.LANCZOS)
    if not buf:
        buf.append((dt, img))
        return "seeded", None
    delta_min = (dt - buf[-1][0]).total_seconds() / 60.0
    if delta_min <= 0:
        return "stale", delta_min
    if delta_min > interval_min * FRAME_BUFFER_GAP_TOL:
        return "gap", delta_min
    buf.append((dt, img))
    if len(buf) > FRAME_BUFFER_MAX:
        buf.pop(0)
    return "appended", delta_min


def buffer_to_pixels(buf, ts_region, ts_tz):
    """Dedupe by frame hash to remove any mid-sequence duplicates."""
    out = []
    seen = set()
    for dt, img in buf:
        h = hashlib.md5(img.tobytes()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        out.append((dt, img))
    return out


# ============================================================================
# SECTION 15 — Storm archiver (with precomputed replay PNGs)
# ============================================================================

def _replay_params_hash(rt, st):
    payload = {
        "threat_radius_km": rt["threat_radius_km"],
        "arrival_pixel_threshold": rt["arrival_pixel_threshold"],
        "departure_pixel_threshold": rt["departure_pixel_threshold"],
        "radius_km": rt["radius_km"],
        "archive_radius_km": rt["archive_radius_km"],
        "show_cities": rt["show_cities"],
        "basemap": rt["basemap"],
        "web_scale": rt["web_scale"],
        "flow_smooth_sigma": rt.get("flow_smooth_sigma", 2.0),
        "flow_average_pairs": rt.get("flow_average_pairs", 5),
        "intensity": st["radar_intensity"],
        "mapbox_style": st.get("mapbox_style"),
        "basemap_size": st.get("basemap_size"),
    }
    return hashlib.md5(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:12]


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
        self.total_replays_saved = 0
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
        replay_count = 0
        for ep in episodes:
            ep_dir = os.path.join(self.root_dir, ep)
            try:
                for f in os.listdir(ep_dir):
                    if ".replay-" in f:
                        replay_count += 1
                    elif f.endswith(".png"):
                        count += 1
            except OSError:
                pass
        self.total_frames_saved = count
        self.total_replays_saved = replay_count

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
        if self.root_dir and os.path.isdir(self.root_dir):
            for d in sorted(os.listdir(self.root_dir)):
                ep_dir = os.path.join(self.root_dir, d)
                if not os.path.isdir(ep_dir):
                    continue
                meta = _load_json_file(os.path.join(ep_dir, "event.json"))
                start_dt = _safe_fromiso(meta.get("start_utc")) if meta else self._parse_episode_start(d)
                end_dt = _safe_fromiso(meta.get("end_utc")) if meta else start_dt
                if start_dt == datetime.min.replace(tzinfo=timezone.utc):
                    continue
                lo = start_dt - timedelta(minutes=self.EPISODE_GAP_MIN)
                hi = end_dt + timedelta(minutes=self.EPISODE_GAP_MIN)
                if lo <= frame_ts_utc <= hi:
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

    def process_animation(self, frames, frame_times, flows, frame_masks_mmh,
                          rt, st, calib, replay_renderer=None,
                          local_weather=None, target_raining=False):
        if not self.root_dir or not rt.get("archive_enabled", True):
            return 0
        radius_km = float(rt.get("archive_radius_km", 100.0))
        min_px = int(rt.get("archive_min_pixels", 20))
        n_frames = len(frame_masks_mmh)
        if n_frames == 0:
            return 0
        params_hash = _replay_params_hash(rt, st)
        precompute = bool(rt.get("archive_precompute_replay", True)) and replay_renderer is not None
        saved_this_cycle = 0
        smooth_sigma = float(rt.get("flow_smooth_sigma", 2.0))

        with self.lock:
            for idx in range(n_frames):
                mask, mmh = frame_masks_mmh[idx]
                h, w = mask.shape
                rmask = self._get_radius_mask(h, w, radius_km)
                local_precip = mask & rmask
                px_count = int(local_precip.sum())
                if px_count < min_px:
                    continue

                frame_ts = frame_times[idx].replace(second=0, microsecond=0)
                fname = frame_ts.strftime("%Y%m%d_%H%M%SZ.png")
                ep_id, ep_dir, is_new_ep = self._resolve_episode_dir(frame_ts)
                if is_new_ep:
                    self.total_episodes += 1
                fpath = os.path.join(ep_dir, fname)
                replay_path = fpath.replace(".png", f".replay-{params_hash}.png")

                if not os.path.exists(fpath):
                    try:
                        frames[idx].convert("P", palette=Image.ADAPTIVE, colors=64).save(
                            fpath + ".tmp", format="PNG", optimize=True)
                        os.replace(fpath + ".tmp", fpath)
                        saved_this_cycle += 1
                        self.total_frames_saved += 1
                    except Exception as e:
                        print(f"archive: raw save error {e}", file=sys.stderr)
                        continue

                if precompute and not os.path.exists(replay_path):
                    try:
                        flow = combined_flow(flows, idx,
                                             int(rt["flow_average_pairs"]),
                                             smooth_sigma=smooth_sigma)
                        if flow is None:
                            flow = np.zeros((h, w, 2), dtype=np.float32)
                        png = replay_renderer(frames[idx], mask, mmh, flow,
                                              calib, frame_ts, idx, n_frames, ep_id)
                        with open(replay_path + ".tmp", "wb") as fh:
                            fh.write(png)
                        os.replace(replay_path + ".tmp", replay_path)
                        self.total_replays_saved += 1
                    except Exception as e:
                        print(f"archive: replay precompute error {e}", file=sys.stderr)

                peak_mmh = float(mmh[local_precip].max()) if px_count > 0 else 0.0
                mean_mmh = float(mmh[local_precip].mean()) if px_count > 0 else 0.0
                self._update_episode_meta(
                    ep_dir, ep_id, frame_ts, radius_km, px_count,
                    peak_mmh, mean_mmh, params_hash,
                    local_weather if idx == n_frames - 1 else None,
                    target_raining if idx == n_frames - 1 else False)
                self.last_episode_id = ep_id
                self.last_saved_utc = frame_ts.isoformat()

        return saved_this_cycle

    def _update_episode_meta(self, ep_dir, ep_id, frame_ts, radius_km,
                             px_count, peak_mmh, mean_mmh, params_hash,
                             local_weather, target_raining):
        meta_path = os.path.join(ep_dir, "event.json")
        meta = _load_json_file(meta_path) or {
            "episode_id": ep_id, "target": self.target_name,
            "target_lat": self.target_lat, "target_lon": self.target_lon,
            "archive_radius_km": radius_km,
            "start_utc": frame_ts.isoformat(), "end_utc": frame_ts.isoformat(),
            "frame_count": 0, "max_precip_pixels": 0, "peak_sri_mmh": 0.0,
            "target_hit": False, "replay_params_hash": params_hash,
        }
        meta["replay_params_hash"] = params_hash
        ts_iso = frame_ts.isoformat()
        if ts_iso < meta["start_utc"]:
            meta["start_utc"] = ts_iso
        if ts_iso > meta["end_utc"]:
            meta["end_utc"] = ts_iso
        try:
            meta["frame_count"] = sum(
                1 for f in os.listdir(ep_dir)
                if f.endswith(".png") and ".replay-" not in f
            )
        except OSError:
            meta["frame_count"] = meta.get("frame_count", 0) + 1
        meta["max_precip_pixels"] = max(int(meta.get("max_precip_pixels", 0)), px_count)
        meta["peak_sri_mmh"] = round(max(float(meta.get("peak_sri_mmh", 0.0)), peak_mmh), 1)
        if target_raining:
            meta["target_hit"] = True
        if local_weather and getattr(local_weather, "station_count", 0) > 0:
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
        # FIX: was ".replay.png" which misses ".replay-abc123.png"
        files = sorted(f for f in os.listdir(ep_dir)
                       if f.endswith(".png") and ".replay-" not in f)
        out = []
        for fn in files:
            try:
                ts = datetime.strptime(fn, "%Y%m%d_%H%M%SZ.png").replace(tzinfo=timezone.utc)
            except ValueError:
                ts = None
            out.append((ts, os.path.join(ep_dir, fn), fn))
        return out

    def find_replay_png(self, episode_id, filename, params_hash):
        if not self.root_dir:
            return None
        safe_id = os.path.basename(episode_id)
        safe_fn = os.path.basename(filename)
        candidate = os.path.join(self.root_dir, safe_id,
                                 safe_fn.replace(".png", f".replay-{params_hash}.png"))
        return candidate if os.path.exists(candidate) else None

    def summary(self):
        with self.lock:
            return {
                "enabled": bool(self.root_dir and _snapshot().get("archive_enabled", True)),
                "archive_dir": self.root_dir,
                "radius_km": _snapshot().get("archive_radius_km", 100.0),
                "total_episodes": self.total_episodes,
                "total_frames": self.total_frames_saved,
                "total_replays": self.total_replays_saved,
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
                # FIX: same filter fix as load_episode_frames
                raw_files = sorted(f for f in os.listdir(ep_dir)
                                   if f.endswith(".png") and ".replay-" not in f)
                if meta:
                    meta["frame_count"] = len(raw_files)
                    meta["frames"] = raw_files
                    out.append(meta)
                else:
                    out.append({"episode_id": d, "frame_count": len(raw_files),
                                "frames": raw_files})
        return out


# ============================================================================
# SECTION 16 — Replay engine
# ============================================================================

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
        self._frame_cache = {}
        self._flow_cache = {}
        self.lock = threading.Lock()

    def get_episode_data(self, episode_id):
        if not self.archiver:
            return []
        entries = self.archiver.load_episode_frames(episode_id)
        if not entries:
            return []
        with self.lock:
            cached = self._frame_cache.get(episode_id)
            if cached and len(cached) == len(entries):
                return cached
        decoded = []
        for ts, fpath, fname in entries:
            try:
                with Image.open(fpath) as im:
                    mask, mmh = analyze_frame(im, self.calib.map_region,
                                              self.intensity_mode,
                                              self.sri_legend, None)
                decoded.append((ts, mask, mmh, fpath, fname))
            except Exception as e:
                print(f"replay: decode error {fpath}: {e}", file=sys.stderr)
        with self.lock:
            if len(self._frame_cache) > 12:
                self._frame_cache.clear()
            self._frame_cache[episode_id] = decoded
        return decoded

    def get_flows(self, episode_id, data):
        """Flow fields between consecutive frames, cached per episode length."""
        if len(data) < 2:
            return []
        cache_key = (episode_id, len(data))
        with self.lock:
            cached = self._flow_cache.get(cache_key)
            if cached is not None:
                return cached
        rt = _snapshot()
        flows = []
        for i in range(len(data) - 1):
            flows.append(estimate_flow_field(
                data[i][1], data[i + 1][1],
                block_px=int(rt["flow_block_px"]),
                search_px=int(rt["flow_search_px"]),
                downsample=int(rt["flow_downsample"])))
        with self.lock:
            if len(self._flow_cache) > 6:
                self._flow_cache.clear()
            self._flow_cache[cache_key] = flows
        return flows


# ============================================================================
# SECTION 17 — Compositing / rendering
# ============================================================================

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
            print(f"basemap: fetch failed: {last_err}", file=sys.stderr)
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


def _sample_raw_at_view(raw_rgb, calib, target_lat, target_lon,
                        radius_km, view_rot, out_w, out_h):
    """Sample a raw RGB frame (DHMZ's own colours) at output view coordinates.

    Returns (H, W, 3) uint8 array, plus a valid mask indicating which output
    pixels fell inside the raw frame's bounds.
    """
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
    valid = ((rx >= 0) & (rx < raw_rgb.shape[1])
             & (ry >= 0) & (ry < raw_rgb.shape[0]))
    out = np.zeros((H, W, 3), dtype=np.uint8)
    out[valid] = raw_rgb[ry[valid], rx[valid]]
    return out, valid


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
    return size_px / 2.0 + rx_km / kmpp, size_px / 2.0 - ry_km / kmpp


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
    draw.polygon([(x1, y1), (bx + px * head_w, by + py * head_w),
                  (bx - px * head_w, by - py * head_w)], fill=color)
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
    draw.line([cx, cy, cx + bar_px, cy], fill=(255, 255, 255), width=2)
    draw.line([cx, cy - 4, cx, cy + 4], fill=(255, 255, 255), width=2)
    draw.line([cx + bar_px, cy - 4, cx + bar_px, cy + 4], fill=(255, 255, 255), width=2)
    _draw_text_outlined(draw, cx + bar_px + 4, cy - 6, f"{bar_km} km",
                        (255, 255, 255), (0, 0, 0))


def _draw_legend(pil_img, confirm_mm):
    W, H = pil_img.size
    box_w, box_h = 238, 74
    x0 = W - box_w - 10
    y0 = H - box_h - 10
    if x0 < 0 or y0 < 0:
        return
    overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)
    odraw.rounded_rectangle([x0, y0, x0 + box_w, y0 + box_h], radius=5,
                            fill=(12, 16, 22, 185), outline=(110, 120, 135, 200), width=1)
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
    draw.text((ax + 17, r1_y - 6), "flow", fill=(195, 240, 255))
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
                clusters, arrivals, stations, show_cities,
                intensity_mode, rt, confirm_mm=0.2, raw_frame=None):
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

    # Fit-check overlay: raw DHMZ frame at 30% opacity, drawn on top of the
    # composite so DHMZ's own coastlines and land/sea contrast are visible for
    # comparison with the satellite basemap. Only rendered when the runtime
    # toggle is on.
    if raw_frame is not None and rt.get("fit_check_overlay", False):
        raw_rgb = np.asarray(raw_frame.convert("RGB"), dtype=np.uint8)
        raw_view, _ = _sample_raw_at_view(
            raw_rgb, calib, target_lat, target_lon, radius_km, view_rot,
            size_px, size_px)
        alpha_raw = 0.30
        rgb = (rgb.astype(np.float32) * (1 - alpha_raw)
               + raw_view.astype(np.float32) * alpha_raw).astype(np.uint8)

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
            color = (235, 35, 35) if is_raining else (255, 255, 255)
            r = 5 if is_raining else 4
            _draw_filled_circle(rgb, px, py, r, color, alpha=0.95)
            _draw_circle_outline(rgb, px, py, r, (0, 0, 0), thickness=1)

    pil = Image.fromarray(rgb.astype(np.uint8))
    draw = ImageDraw.Draw(pil)

    for a in arrivals:
        if a.coherence < 0.4:
            continue
        if math.hypot(a.vx_kmh, a.vy_kmh) < MIN_SPEED_KMH:
            continue
        c = a.cluster
        clat, clon = calib.image_pixel_to_latlon(c.cx, c.cy)
        px0, py0 = _latlon_to_overlay_px(calib, target_lat, target_lon,
                                         radius_km, size_px, view_rot, clat, clon)
        coslat = max(0.01, math.cos(math.radians(clat)))
        lat_1h = clat + a.vy_kmh / 111.32
        lon_1h = clon + a.vx_kmh / (111.32 * coslat)
        px1, py1 = _latlon_to_overlay_px(calib, target_lat, target_lon,
                                         radius_km, size_px, view_rot, lat_1h, lon_1h)
        if not ((0 <= px0 < size_px and 0 <= py0 < size_px)
                or (0 <= px1 < size_px and 0 <= py1 < size_px)):
            continue
        if a.eta_min is not None and a.eta_min <= 30:
            arrow_color = (245, 55, 55)
        elif a.eta_min is not None:
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
    _draw_legend(pil, confirm_mm)
    return pil


def render_composed_png(frame, calib, target_lat, target_lon, radius_km,
                        view_rot, size_px, basemap_img, basemap_kind,
                        clusters, arrivals, stations, show_cities,
                        intensity_mode, sri_legend, legend_mask,
                        rt, confirm_mm=0.2, header_text=None):
    frame = frame.convert("RGB")
    mask, mmh = analyze_frame(frame, calib.map_region, intensity_mode,
                              sri_legend, legend_mask)
    gray = _radar_to_grayscale(mask, mmh)
    pil = compose_map(mask, mmh, gray, calib, target_lat, target_lon,
                      radius_km, view_rot, size_px, basemap_img, basemap_kind,
                      clusters, arrivals, stations, show_cities,
                      intensity_mode, rt, confirm_mm=confirm_mm,
                      raw_frame=frame)
    if header_text:
        draw = ImageDraw.Draw(pil)
        _draw_text_outlined(draw, 10, 8, header_text, (255, 220, 60), (0, 0, 0))
    buf = io.BytesIO()
    pil.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def render_composed_gif(frames, flows, calib, target_lat, target_lon, radius_km,
                        view_rot, size_px, basemap_img, basemap_kind,
                        stations, show_cities, intensity_mode, sri_legend,
                        legend_mask, rt, st, confirm_mm=0.2,
                        frame_duration_ms=400, max_frames=30):
    nf = len(frames)
    start = max(0, nf - max_frames)
    smooth_sigma = float(rt.get("flow_smooth_sigma", 2.0))
    out = []
    for fidx in range(start, nf):
        frame = frames[fidx].convert("RGB")
        mask, mmh = analyze_frame(frame, calib.map_region, intensity_mode,
                                  sri_legend, legend_mask)
        gray = _radar_to_grayscale(mask, mmh)
        flow = combined_flow(flows, fidx, int(rt["flow_average_pairs"]),
                             smooth_sigma=smooth_sigma)
        if flow is None:
            flow = np.zeros((*mask.shape, 2), dtype=np.float32)
        _, _, arrivals, _ = analyze_frame_state(mask, mmh, flow, calib,
                                                target_lat, target_lon, rt, st)
        pil = compose_map(mask, mmh, gray, calib, target_lat, target_lon,
                          radius_km, view_rot, size_px, basemap_img, basemap_kind,
                          [a.cluster for a in arrivals], arrivals, stations,
                          show_cities, intensity_mode, rt, confirm_mm=confirm_mm,
                          raw_frame=frame)
        out.append(pil)
    if not out:
        return None
    buf = io.BytesIO()
    out[0].save(buf, format="GIF", save_all=True, append_images=out[1:],
                duration=frame_duration_ms, loop=0, optimize=True)
    return buf.getvalue()


# ============================================================================
# SECTION 18 — Accuracy store
# ============================================================================

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

    def _expire_locked(self, now_utc):
        for p in self.predictions:
            if p.get("scored") or p.get("superseded") or p.get("expired"):
                continue
            try:
                t0 = _safe_fromiso(p["ts"])
            except Exception:
                continue
            horizon = p.get("horizon_min") or p.get("predicted_min") or 30.0
            tol = max(MATCH_TOLERANCE_MIN, horizon * MATCH_TOLERANCE_FACTOR)
            if (now_utc - t0).total_seconds() / 60.0 > horizon + tol * 2:
                p["expired"] = True

    def predict(self, ts_iso, kind, predicted_min, key="", tag="",
                settings=None, horizon_min=None):
        if not self.enabled or predicted_min is None: return
        with self.lock:
            self._expire_locked(_safe_fromiso(ts_iso))
            for p in reversed(self.predictions):
                if p.get("scored") or p.get("superseded") or p.get("expired"):
                    continue
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
            if p.get("scored") or p.get("superseded") or p.get("expired"): continue
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
            self._write_jsonl("predictions.jsonl", [])
            self._write_jsonl("outcomes.jsonl", [])
            self._write_jsonl("scores.jsonl", [])

    def stats(self, since_hours=None, tag=None):
        with self.lock:
            self._expire_locked(datetime.now(timezone.utc))
            rows = list(self.scores)
            preds = list(self.predictions)
            outs = list(self.outcomes)
        if since_hours is not None:
            cutoff = datetime.now(timezone.utc) - timedelta(hours=since_hours)
            rows = [r for r in rows if _safe_fromiso(r["scored_at"]) >= cutoff]
        if tag is not None:
            rows = [r for r in rows if r.get("tag", "") == tag]

        def summarize(sub):
            if not sub:
                return {"n": 0}
            errs = [r["error_min"] for r in sub]
            aerrs = [r["abs_error_min"] for r in sub]
            return {"n": len(sub),
                    "median_error_min": round(float(np.median(errs)), 1),
                    "mean_error_min": round(float(np.mean(errs)), 1),
                    "mean_abs_error_min": round(float(np.mean(aerrs)), 1),
                    "median_abs_error_min": round(float(np.median(aerrs)), 1),
                    "p90_abs_error_min": round(float(np.percentile(aerrs, 90)), 1)}

        by_kind = {}
        for k in {r["kind"] for r in rows}:
            by_kind[k] = summarize([r for r in rows if r["kind"] == k])
        by_tag = {}
        for t in {r.get("tag", "") for r in rows}:
            by_tag[t] = summarize([r for r in rows if r.get("tag", "") == t])
        pending = {}
        expired_count = 0
        for p in preds:
            if p.get("scored") or p.get("superseded"):
                continue
            if p.get("expired"):
                expired_count += 1
                continue
            pending[p["kind"]] = pending.get(p["kind"], 0) + 1
        superseded = sum(1 for p in preds if p.get("superseded"))
        return {"total_predictions": len(preds), "total_outcomes": len(outs),
                "total_scores": len(rows), "superseded_predictions": superseded,
                "expired_predictions": expired_count,
                "pending_by_kind": pending, "overall": summarize(rows),
                "by_kind": by_kind, "by_tag": by_tag}

    def list_predictions(self, limit=100):
        with self.lock: return list(reversed(self.predictions[-limit:]))

    def list_outcomes(self, limit=100):
        with self.lock: return list(reversed(self.outcomes[-limit:]))

    def last_mae_min(self, kind="rain_start"):
        with self.lock:
            rows = [r for r in self.scores if (kind is None or r["kind"] == kind)]
        if not rows:
            return None
        recent = rows[-50:]
        return round(float(np.median([r["abs_error_min"] for r in recent])), 1)


# ============================================================================
# SECTION 19 — Pljusak client
# ============================================================================

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
                    min_temp_c=min_temp_c, uv=uv, temp_24h_c=temp_24h_c))
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
    fresh = [s for s in stations
             if s.dist_km <= radius_km
             and s.age_min is not None
             and s.age_min <= max_age_min]
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


# ============================================================================
# SECTION 20 — Combined analysis
# ============================================================================

@dataclass
class CombinedAnalysis:
    radar_raining: bool = False
    raining_now: bool = False
    confirmed: bool = False
    confidence: str = "dry"
    nearest_station: Optional[StationAnalysis] = None
    intensity_mmh: Optional[float] = None
    station_trend: str = "none"
    nearby_stations: list = field(default_factory=list)
    wet_stations: list = field(default_factory=list)
    station_count_wet: int = 0
    station_count_fresh: int = 0
    station_count_stale: int = 0
    wet_fraction: Optional[float] = None
    station_median_mm: Optional[float] = None


def combine_analysis(radar_raining, stations, t_lat, t_lon,
                     confirm_radius, confirm_mm, confirm_age,
                     confirm_frac, confirm_min_count):
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
    return ca


# ============================================================================
# SECTION 21 — MQTT
# ============================================================================

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
    ("cluster_count",         "Precip clusters",         "sensor", None, None, None, "diagnostic"),
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
             "manufacturer": "DHMZ / pljusak.com", "model": "Radar nowcaster"}
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
            comp = {"camera": "camera", "binary": "binary_sensor"}.get(kind, "sensor")
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
            payload = "ON" if (kind == "binary" and val) else \
                      "OFF" if (kind == "binary") else \
                      ("unknown" if val is None else str(val))
            self._client.publish(self._state_topic(key), payload, qos=0, retain=self.retain)
            if key in attrs_by_key:
                self._client.publish(self._attr_topic(key),
                                     json.dumps(attrs_by_key[key], default=str),
                                     qos=0, retain=self.retain)

    def publish_threat(self, payload):
        self._client.publish(self.threat_topic, json.dumps(payload, default=str),
                             qos=1, retain=False)

# ============================================================================
# SECTION 23 — Web UI
#   23.1  _INDEX_HTML       dashboard template
#   23.2  _REPLAY_HTML      replay template
#   23.3  _VizHandler       HTTP handler (routes + per-endpoint bodies)
#   23.4  start_web         server bootstrap
# ============================================================================

# ---------------------------------------------------------------- 23.1 -----
_INDEX_HTML = """<!DOCTYPE html><html><head><title>DHMZ Radar Nowcaster</title>
<style>
body{background:#111;color:#ddd;font-family:-apple-system,BlinkMacSystemFont,sans-serif;margin:0;padding:20px}
h1{color:#fff;margin-top:0;display:flex;align-items:center;gap:12px;flex-wrap:wrap}.layout{display:flex;gap:20px;flex-wrap:wrap}
.left{flex:1 1 470px;max-width:580px}.right{flex:2 1 600px}
.section{background:#1a1a1a;border:1px solid #333;border-radius:4px;padding:12px;margin-bottom:12px}
.section h3{margin:0 0 8px 0;font-size:13px;text-transform:uppercase;color:#5cf;letter-spacing:0.5px;display:flex;justify-content:space-between;align-items:center}
.field{display:flex;align-items:center;margin-bottom:6px;font-size:13px}
.field label{flex:1;color:#ccc}.field input[type=number]{width:90px;background:#0a0a0a;color:#ddd;border:1px solid #333;padding:4px;border-radius:3px}
.field input[type=text]{flex:0 0 160px;background:#0a0a0a;color:#ddd;border:1px solid #333;padding:4px;border-radius:3px}
.field select{width:130px;background:#0a0a0a;color:#ddd;border:1px solid #333;padding:4px;border-radius:3px}
.field input[type=checkbox]{accent-color:#5cf;width:18px;height:18px}
.actions{margin-top:8px;display:flex;gap:8px;flex-wrap:wrap}
button{background:#2a5c8a;color:#fff;border:0;padding:7px 12px;border-radius:3px;cursor:pointer;font-size:12px}
button:hover{background:#3a7cba}button.secondary{background:#444}
button.secondary:hover{background:#555}button.danger{background:#8a2a2a}
button.danger:hover{background:#ba3a3a}
.status{background:#1a1a1a;border:1px solid #333;border-radius:4px;padding:12px;margin-bottom:12px;font-size:13px}
.status .row{display:flex;margin-bottom:4px}.status .label{color:#888;flex:0 0 160px}
.wx-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}
.wx-box{background:#121519;border:1px solid #282c34;border-radius:4px;padding:7px 8px}
.wx-k{font-size:11px;color:#888;text-transform:uppercase}
.wx-v{font-size:15px;font-weight:600;color:#fff;margin-top:2px;font-variant-numeric:tabular-nums}
.wx-sub{font-size:11px;color:#9aa}
table.acc{width:100%;border-collapse:collapse;font-size:12px}
table.acc th,table.acc td{padding:4px 6px;text-align:left;border-bottom:1px solid #252830}
table.acc th{color:#5cf;font-weight:normal;text-transform:uppercase;font-size:11px}
table.acc td.num{text-align:right;font-variant-numeric:tabular-nums}
img{max-width:100%;border:1px solid #333;display:block;margin-bottom:12px}
a{color:#5cf}.hint{color:#778;font-size:11px;margin-top:4px}
.cfgpath{font-family:monospace;color:#7c7}
.badge{background:#244866;color:#9df;padding:3px 8px;border-radius:3px;font-size:12px;text-decoration:none}
</style></head><body>
<h1>DHMZ Radar Nowcaster <a class="badge" href="/replay">Offline Replay →</a></h1>
<p><a href="/threat.gif">/threat.gif</a> · <a href="/threat.png">/threat.png</a> · <a href="/status">/status</a> · <a href="/settings">/settings</a> · <a href="/config">/config</a> · <a href="/archive">/archive</a> · <a href="/accuracy">/accuracy</a></p>
<div class="layout">
<div class="left">
<div class="status" id="status">Loading...</div>
<div class="section"><h3>Local Weather (Pljusak)</h3>
<div id="local_wx">Loading...</div>
</div>
<div class="section"><h3>Storm Frame Archive</h3>
<div id="archive_box">Loading...</div>
</div>
<div class="section"><h3>Accuracy</h3>
<div id="accuracy">Loading...</div>
<div class="actions" style="margin-top:8px">
<button class="secondary" onclick="replayAccuracy()">Replay scores</button>
<button class="danger" onclick="resetAccuracy()">Reset scores</button>
</div></div>
<div class="section"><h3>Sensors (HA)</h3>
<div id="sensors">Loading...</div>
</div>
<div class="section"><h3>Runtime settings (hot)</h3>
<div class="field"><label>Focus radius (km)</label><input type="number" id="focus_radius_km" min="20" max="1000" step="10"></div>
<div class="field"><label>Threat radius (km)</label><input type="number" id="threat_radius_km" min="1" max="500" step="5"></div>
<div class="field"><label>Arrival pixel threshold</label><input type="number" id="arrival_pixel_threshold" min="1" max="5000" step="5"></div>
<div class="field"><label>Departure pixel threshold</label><input type="number" id="departure_pixel_threshold" min="1" max="5000" step="5"></div>
<div class="field"><label>Max horizon (min)</label><input type="number" id="max_horizon_min" min="30" max="360" step="10"></div>
<div class="field"><label>Min cell size (px)</label><input type="number" id="min_size_px" min="5" max="5000" step="5"></div>
<div class="field"><label>Map radius (km)</label><input type="number" id="radius_km" min="20" max="1000" step="10"></div>
<div class="field"><label>Local weather radius (km)</label><input type="number" id="weather_radius_km" min="0.5" max="100" step="0.5"></div>
<div class="field"><label>Archive frames</label><input type="checkbox" id="archive_enabled"></div>
<div class="field"><label>Archive radius (km)</label><input type="number" id="archive_radius_km" min="20" max="500" step="10"></div>
<div class="field"><label>Precompute replay PNGs</label><input type="checkbox" id="archive_precompute_replay"></div>
<div class="field"><label>Fit check overlay (raw DHMZ @ 30%)</label><input type="checkbox" id="fit_check_overlay"></div>
<div class="field"><label>View rotation: auto</label><input type="checkbox" id="view_rotation_auto"></div>
<div class="field"><label>View rotation (° CW, up)</label><input type="number" id="view_rotation" min="0" max="360" step="5"></div>
<div class="field"><label>Basemap</label>
<select id="basemap">
<option value="auto">auto</option><option value="mapbox">mapbox</option>
<option value="greyscale">greyscale</option><option value="none">none</option>
</select></div>
<div class="field"><label>Basemap saturation</label><input type="number" id="basemap_saturation" min="0" max="2" step="0.1"></div>
<div class="field"><label>Basemap brightness</label><input type="number" id="basemap_brightness" min="0.2" max="2" step="0.05"></div>
<div class="field"><label>Basemap contrast</label><input type="number" id="basemap_contrast" min="0.2" max="2" step="0.05"></div>
<div class="field"><label>Cities</label><input type="checkbox" id="show_cities"></div>
<div class="field"><label>Poll interval (s)</label><input type="number" id="poll_s" min="30" max="7200" step="30"></div>
<div class="field"><label>Image scale</label><input type="number" id="web_scale" min="0.1" max="4" step="0.1"></div>
<div class="field"><label>Settings tag</label><input type="text" id="settings_tag"></div>
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
const RT_FIELDS=['focus_radius_km','threat_radius_km','arrival_pixel_threshold','departure_pixel_threshold','max_horizon_min','min_size_px','radius_km','weather_radius_km','archive_enabled','archive_radius_km','archive_precompute_replay','fit_check_overlay','view_rotation_auto','view_rotation','basemap','basemap_saturation','basemap_brightness','basemap_contrast','show_cities','poll_s','web_scale','settings_tag'];
const MAIN_SENSORS=['raining_now','raining_confirmed','rain_confidence','rain_duration_min','rain_end_eta_min','next_rain_eta_min','gap_min','summary','radar_intensity_mmh','nearest_station_mm','intensity_mmh','station_trend','local_temp_c','local_humidity','local_pressure_hpa','local_wind_speed_kmh','local_wind_gust_kmh','local_wind_dir','local_rain_today_mm','local_rain_1h_mm','local_rain_rate_mmh'];
const DIAG_SENSORS=['nowcast_error_min','data_age_min','frame_timestamp','cluster_count','nearest_cell_eta_min','nearest_station_km','nearest_station_age_min','station_count_wet','station_count_fresh','station_wet_fraction','station_median_mm','local_weather_stations'];
function setField(k,v){const el=document.getElementById(k);if(!el)return;if(el.type==='checkbox')el.checked=!!v;else el.value=v;}
function getField(k){const el=document.getElementById(k);if(!el)return null;if(el.type==='checkbox')return el.checked;if(k==='settings_tag'||k==='basemap')return el.value;return parseFloat(el.value);}
function loadSettings(){fetch('/settings').then(r=>r.json()).then(s=>{RT_FIELDS.forEach(k=>{if(k in s)setField(k,s[k]);});});}
function saveAll(refresh){const body={};RT_FIELDS.forEach(k=>body[k]=getField(k));
 fetch('/settings',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})
 .then(r=>r.json()).then(d=>{if(d.ok&&refresh)return fetch('/refresh',{method:'POST'});})
 .then(()=>{bumpImages();loadStatus();});}
function fmtVal(v){if(v===null||v===undefined)return '—';if(typeof v==='boolean')return v?'yes':'no';return String(v);}
function loadStatus(){fetch('/status').then(r=>r.json()).then(s=>{const el=document.getElementById('status');if(!s||!s.updated){el.textContent='No data yet';return;}
 el.innerHTML=`<div class="row"><span class="label">Target</span><span>${s.target||'?'}</span></div>
 <div class="row"><span class="label">Updated</span><span>${s.updated}</span></div>
 <div class="row"><span class="label">Confidence</span><span>${s.confidence||'?'}</span></div>
 <div class="row"><span class="label">Raining</span><span>${s.raining_now?'yes':'no'}</span></div>
 <div class="row"><span class="label">Nowcast</span><span>${s.summary||'—'}</span></div>
 <div class="row"><span class="label">View rotation</span><span>${s.view_rotation_used!=null?s.view_rotation_used.toFixed(0)+'°'+(s.view_rotation_auto?' (auto)':'') :'—'}</span></div>
 <div class="row"><span class="label">Basemap</span><span>${s.basemap_used||'—'}</span></div>
 <div class="row"><span class="label">Config</span><span class="cfgpath">${s.config_path||'(not persisted)'}</span></div>`;
 const arch=s.archive||{};
 const archEl=document.getElementById('archive_box');
 if(!arch.archive_dir){archEl.innerHTML='<div class="hint">Persistence disabled (no data-dir)</div>';}
 else{archEl.innerHTML=`<div class="hint">Status: ${arch.enabled?'Active':'Paused'} (≤${arch.radius_km||100} km) · Episodes: <b>${arch.total_episodes||0}</b> · Frames: <b>${arch.total_frames||0}</b> · Replays: <b>${arch.total_replays||0}</b></div>
 <div class="hint">Last episode: <span class="cfgpath">${arch.last_episode||'none'}</span> · <a href="/replay">Open Replay</a></div>`;}
 const wx=s.local_weather||{};
 const wxEl=document.getElementById('local_wx');
 if(!wx.station_count){wxEl.innerHTML=`<div class="hint">No fresh Pljusak stations within ${wx.radius_km||5} km</div>`;}
 else{const windTxt=wx.wind_speed_kmh!=null?`${wx.wind_speed_kmh} km/h`:'—';
  const gustTxt=wx.wind_gust_kmh!=null?`gust ${wx.wind_gust_kmh}`:'';
  const dirTxt=wx.wind_dir_deg!=null?`${wx.wind_dir_cardinal} (${wx.wind_dir_deg}°)`:'—';
  const pTrend=wx.pressure_trend!=null?(wx.pressure_trend>0?'+':'')+wx.pressure_trend+' hPa':'';
  const tTrend=wx.temp_trend!=null?(wx.temp_trend>0?'+':'')+wx.temp_trend+'°/h':'';
  wxEl.innerHTML=`<div class="wx-grid">
   <div class="wx-box"><div class="wx-k">Temp</div><div class="wx-v">${wx.temp_c!=null?wx.temp_c+' °C':'—'}</div><div class="wx-sub">${tTrend||' '}</div></div>
   <div class="wx-box"><div class="wx-k">Humidity</div><div class="wx-v">${wx.humidity_pct!=null?wx.humidity_pct+' %':'—'}</div></div>
   <div class="wx-box"><div class="wx-k">Pressure</div><div class="wx-v">${wx.pressure_hpa!=null?wx.pressure_hpa+' hPa':'—'}</div><div class="wx-sub">${pTrend||' '}</div></div>
   <div class="wx-box"><div class="wx-k">Wind</div><div class="wx-v">${windTxt}</div><div class="wx-sub">${gustTxt||' '}</div></div>
   <div class="wx-box"><div class="wx-k">Heading</div><div class="wx-v">${dirTxt}</div></div>
   <div class="wx-box"><div class="wx-k">Rain (Today / 1h)</div><div class="wx-v">${wx.precip_today_mm!=null?wx.precip_today_mm:'—'} / ${wx.precip_1h_mm!=null?wx.precip_1h_mm:'—'} mm</div><div class="wx-sub">${wx.rain_rate_mmh!=null?wx.rain_rate_mmh+' mm/h':' '}</div></div>
  </div>
  <div class="hint" style="margin-top:6px">${wx.station_count} fresh ≤ ${wx.radius_km} km · nearest: ${wx.nearest_station||'?'} (${wx.nearest_dist_km} km)</div>`;}
 const sensors=s.sensors||{};
 let html='<table class="acc"><tr><th>Sensor</th><th>Value</th></tr>';
 for(const k of MAIN_SENSORS){html+=`<tr><td>${k}</td><td class="num">${fmtVal(sensors[k])}</td></tr>`;}
 html+='</table><h4>Diagnostic</h4><table class="acc">';
 for(const k of DIAG_SENSORS){html+=`<tr><td>${k}</td><td class="num">${fmtVal(sensors[k])}</td></tr>`;}
 html+='</table>';
 document.getElementById('sensors').innerHTML=html;
});}
function loadAccuracy(){fetch('/accuracy').then(r=>r.json()).then(a=>{const el=document.getElementById('accuracy');
 if(!a){el.innerHTML='<div class="hint">No accuracy data</div>';return;}
 let html='<div class="hint">Preds '+a.total_predictions+' · Outcomes '+a.total_outcomes+' · Scores '+a.total_scores+' · Expired '+(a.expired_predictions||0)+' · Superseded '+(a.superseded_predictions||0)+'</div>';
 const pend=a.pending_by_kind||{};
 if(Object.keys(pend).length){html+='<div class="hint">Pending: '+Object.entries(pend).map(([k,v])=>k+':'+v).join(', ')+'</div>';}
 if(a.total_scores>0){const order=['cluster_arrive','rain_start','rain_end','threat'];
  html+='<table class="acc"><tr><th>Kind</th><th>N</th><th>Median err</th><th>Median abs</th><th>Mean abs</th><th>p90 abs</th></tr>';
  for(const k of order){const v=a.by_kind[k];if(!v||!v.n)continue;
   const bias=v.median_error_min!=null?(v.median_error_min>0?'+':'')+v.median_error_min.toFixed(0):'—';
   html+=`<tr><td>${k}</td><td class="num">${v.n}</td><td class="num">${bias}</td><td class="num">${v.median_abs_error_min.toFixed(1)}</td><td class="num">${v.mean_abs_error_min.toFixed(1)}</td><td class="num">${v.p90_abs_error_min.toFixed(1)}</td></tr>`;}
  html+='</table>';
  const tags=Object.keys(a.by_tag||{});
  if(tags.length){html+='<h4>By tag</h4><table class="acc"><tr><th>Tag</th><th>N</th><th>Median abs</th></tr>';
   for(const t of tags){const v=a.by_tag[t];if(!v||!v.n)continue;
    html+=`<tr><td>${t||'(untagged)'}</td><td class="num">${v.n}</td><td class="num">${v.median_abs_error_min.toFixed(1)}</td></tr>`;}
   html+='</table>';}}
 el.innerHTML=html;
});}
function replayAccuracy(){if(!confirm('Re-score all predictions?'))return;fetch('/accuracy/replay',{method:'POST'}).then(()=>loadAccuracy());}
function resetAccuracy(){if(!confirm('Clear all scores?'))return;fetch('/accuracy/reset',{method:'POST'}).then(()=>loadAccuracy());}
function bumpImages(){const t=Date.now();document.getElementById('g').src='/threat.gif?t='+t;document.getElementById('p').src='/threat.png?t='+t;}
loadSettings();loadStatus();loadAccuracy();
setInterval(loadStatus,10000);setInterval(loadAccuracy,30000);setInterval(bumpImages,60000);
</script></body></html>"""


# ---------------------------------------------------------------- 23.2 -----
_REPLAY_HTML = """<!DOCTYPE html><html><head><title>Offline Replay — DHMZ Radar</title>
<style>
body{background:#0e1014;color:#dde;font-family:-apple-system,BlinkMacSystemFont,sans-serif;margin:0;padding:18px}
h1{color:#fff;margin:0 0 12px 0;font-size:20px;display:flex;align-items:center;gap:12px}
.badge{background:#233547;color:#7cf;padding:3px 8px;border-radius:3px;font-size:12px;text-decoration:none}
.grid{display:flex;gap:18px;flex-wrap:wrap}
.col-left{flex:1 1 520px;max-width:760px}
.col-right{flex:1 1 320px;max-width:420px}
.card{background:#161a20;border:1px solid #2b323d;border-radius:5px;padding:12px;margin-bottom:14px}
.card h3{margin:0 0 8px 0;font-size:12px;text-transform:uppercase;color:#6cf;letter-spacing:0.6px}
.row{display:flex;align-items:center;gap:8px;margin-bottom:7px;font-size:13px}
.row label{flex:1;color:#bcc}
button{background:#265b8c;color:#fff;border:0;padding:7px 12px;border-radius:3px;cursor:pointer;font-size:12px;font-weight:500}
button:hover{background:#3477b5}
button.secondary{background:#383f4a}button.secondary:hover{background:#4a5362}
table{width:100%;border-collapse:collapse;font-size:12px}
th,td{padding:5px 6px;text-align:left;border-bottom:1px solid #242a34}
th{color:#7cf;font-size:11px;text-transform:uppercase;font-weight:500}
td.num{text-align:right;font-variant-numeric:tabular-nums}
.scrubber{display:flex;align-items:center;gap:8px;margin-top:8px}
.scrubber input[type=range]{flex:1;accent-color:#6cf}
img#replay_img{width:100%;max-width:720px;border:1px solid #2b323d;border-radius:4px;display:block;background:#080a0d}
.hint{color:#778;font-size:11px;margin-top:4px;line-height:1.4}
</style></head><body>
<h1>Offline Replay <a class="badge" href="/">← Live Dashboard</a></h1>

<div class="grid">
  <div class="col-left">
    <div class="card">
      <h3>Archived Storm Episode</h3>
      <div class="row">
        <label>Episode</label>
        <select id="ep_select" onchange="onEpisodeChange()" style="flex:1;background:#0b0d10;color:#eee;border:1px solid #333b48;padding:4px 6px;border-radius:3px"></select>
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
        Images are precomputed at archive time — should load instantly.
      </div>
    </div>
    <img id="replay_img" alt="Archived radar frame">
  </div>

  <div class="col-right">
    <div class="card">
      <h3>Episodes</h3>
      <div id="episodes_box" class="hint">Loading...</div>
    </div>
    <div class="card">
      <h3>Tips</h3>
      <div class="hint">
        Play runs at ~2 fps. Frames with precomputed replays load instantly.
        Toggle fit-check overlay on the dashboard to see the raw DHMZ frame
        semi-transparently.
      </div>
    </div>
  </div>
</div>

<script>
let episodes=[];
let playTimer=null;

function loadEpisodes(){
  fetch('/archive?n=50').then(r=>r.json()).then(d=>{
    episodes=d.episodes||[];
    const sel=document.getElementById('ep_select');
    sel.innerHTML='';
    if(!episodes.length){
      document.getElementById('ep_meta').textContent='No archived storm episodes yet.';
      document.getElementById('episodes_box').textContent='No episodes.';
      return;
    }
    episodes.forEach(ep=>{
      const opt=document.createElement('option');
      opt.value=ep.episode_id;
      opt.textContent=`${ep.episode_id} (${ep.frame_count} fr, peak ${ep.peak_sri_mmh||'?'} mm/h${ep.target_hit?' · HIT':''})`;
      sel.appendChild(opt);
    });
    let html='<table><tr><th>Episode</th><th class="num">Frames</th><th class="num">Peak</th><th>Hit</th></tr>';
    episodes.forEach(ep=>{
      html+=`<tr><td>${ep.episode_id}</td><td class="num">${ep.frame_count||0}</td><td class="num">${ep.peak_sri_mmh||'—'}</td><td>${ep.target_hit?'✓':'—'}</td></tr>`;
    });
    html+='</table>';
    document.getElementById('episodes_box').innerHTML=html;
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
  // ?t= forces a fresh request (server-side cache handles speed)
  document.getElementById('replay_img').src=`/replay/frame?episode=${encodeURIComponent(epId)}&idx=${idx}&t=${Date.now()}`;
}

loadEpisodes();
</script></body></html>"""


# ---------------------------------------------------------------- 23.3 -----
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

    def _replay_frame(self, params):
        import urllib.parse
        eng = _VIZ_STATE.get("replay_engine")
        arch = _VIZ_STATE.get("archiver")
        if not eng:
            self.send_response(503); self.end_headers(); return
        ep_id = urllib.parse.unquote(params.get("episode", ""))
        try:
            idx = int(params.get("idx", 0))
        except (ValueError, TypeError):
            idx = 0

        data = eng.get_episode_data(ep_id)
        if not data:
            self.send_response(404); self.end_headers(); return
        idx = max(0, min(len(data) - 1, idx))
        rt = _snapshot()
        st = _static_snapshot()

        # Fast path: precomputed replay PNG for this exact params hash
        if arch is not None:
            _, fpath, fname = data[idx]
            ph = _replay_params_hash(rt, st)
            precomputed = arch.find_replay_png(ep_id, fname, ph)
            if precomputed is not None:
                try:
                    with open(precomputed, "rb") as fh:
                        out = fh.read()
                    self.send_response(200)
                    self.send_header("Content-Type", "image/png")
                    self.send_header("Content-Length", str(len(out)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers(); self.wfile.write(out)
                    return
                except Exception as e:
                    print(f"replay: precomputed read error {e}", file=sys.stderr)

        # Slow path: render on demand, with in-memory cache
        cache_key = (ep_id, idx, _replay_params_hash(rt, st))
        cached = _replay_cache_get(cache_key)
        if cached is not None:
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(cached)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers(); self.wfile.write(cached)
            return

        flows = eng.get_flows(ep_id, data)
        flow = combined_flow(flows, idx, int(rt["flow_average_pairs"]),
                             smooth_sigma=float(rt.get("flow_smooth_sigma", 2.0)))
        if flow is None:
            m_shape = data[idx][1].shape
            flow = np.zeros((*m_shape, 2), dtype=np.float32)

        ts, mask, mmh, fpath, _ = data[idx]

        # Load the raw frame if fit-check overlay is requested
        raw_img = None
        if rt.get("fit_check_overlay"):
            try:
                raw_img = Image.open(fpath).convert("RGB")
            except Exception as e:
                print(f"replay: raw frame load error {e}", file=sys.stderr)

        try:
            png = build_replay_png(
                raw_img, mask, mmh, flow, eng.calib, ts, idx, len(data), ep_id,
                eng.target_lat, eng.target_lon, rt, st,
                eng.sri_legend, None,
                stations=[],
                data_dir=_VIZ_STATE.get("data_dir"))
        except Exception as e:
            print(f"replay: on-demand render error {e}", file=sys.stderr)
            self.send_response(500); self.end_headers(); return

        _replay_cache_put(cache_key, png)
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(png)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers(); self.wfile.write(png)

    def _config_get(self):
        cfg_path = _VIZ_STATE.get("config_path")
        if cfg_path and os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            if cfg.get("static", {}).get("mapbox_token"):
                cfg["static"]["mapbox_token"] = "***"
            self._json(cfg)
        else:
            self._json({"version": 2, "note": "no config persisted",
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
        new_cfg.setdefault("version", 2)
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


# ---------------------------------------------------------------- 23.4 -----
def start_web(port):
    srv = ThreadingHTTPServer(("0.0.0.0", port), _VizHandler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv

# ============================================================================
# SECTION 22 — Replay rendering (shared between archive precompute and on-demand)
# ============================================================================

def build_replay_png(frame, mask, mmh, flow, calib, frame_ts, idx, total,
                     ep_id, target_lat, target_lon, rt, st, sri_legend, legend_mask,
                     stations=None, data_dir=None):
    """Render a single replay frame to PNG bytes. Used by archiver precompute and
    by the /replay/frame fallback path."""
    gray = _radar_to_grayscale(mask, mmh)
    # Per-frame cluster/arrival state
    _, _, arrivals, _ = analyze_frame_state(mask, mmh, flow, calib,
                                            target_lat, target_lon, rt, st)
    archive_rad = float(rt.get("archive_radius_km", 100.0))
    size_px = st.get("basemap_size") or 640
    bm_img = None
    bm_kind = "greyscale"
    if st.get("mapbox_token") and rt.get("basemap") in ("auto", "mapbox"):
        bm_img = fetch_basemap(
            st.get("mapbox_style") or "mapbox/satellite-v9",
            st["mapbox_token"], target_lat, target_lon,
            archive_rad, 0.0, size_px, data_dir)
        if bm_img is not None:
            bm_kind = "mapbox"
    pil = compose_map(mask, mmh, gray, calib, target_lat, target_lon,
                      archive_rad, 0.0, size_px, bm_img, bm_kind,
                      [a.cluster for a in arrivals], arrivals, stations or [],
                      rt["show_cities"], st["radar_intensity"], rt)
    draw = ImageDraw.Draw(pil)
    ts_lbl = frame_ts.strftime("%Y-%m-%d %H:%MZ") if frame_ts else "?"
    header = f"REPLAY {ep_id} | {ts_lbl} ({idx + 1}/{total})"
    if st.get("radar_intensity") == "sri":
        pass  # legend already drawn
    _draw_text_outlined(draw, 10, 8, header, (255, 220, 60), (0, 0, 0))
    buf = io.BytesIO()
    pil.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _replay_cache_get(key):
    with _REPLAY_CACHE_LOCK:
        v = _REPLAY_CACHE.get(key)
        if v is not None:
            _REPLAY_CACHE.move_to_end(key)
        return v


def _replay_cache_put(key, data):
    with _REPLAY_CACHE_LOCK:
        _REPLAY_CACHE[key] = data
        _REPLAY_CACHE.move_to_end(key)
        while len(_REPLAY_CACHE) > _REPLAY_CACHE_MAX:
            _REPLAY_CACHE.popitem(last=False)

         
# ============================================================================
# SECTION 24 — CLI + main loop
#   24.1  parse_sri_thresholds, parse_region
#   24.2  build_arg_parser
#   24.3  bootstrap_from_args
#   24.4  run_loop
#   24.5  main
# ============================================================================

# ---------------------------------------------------------------- 24.1 -----
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


# ---------------------------------------------------------------- 24.2 -----
def build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="/data")
    p.add_argument("--config", default=None)
    p.add_argument("--location", default=None)
    p.add_argument("--lat", type=float, default=None)
    p.add_argument("--lon", type=float, default=None)
    p.add_argument("--calibration", default=None)
    p.add_argument("--calibration-inline", default=None)
    p.add_argument("--focus-radius", type=float, dest="focus_radius_km", default=None)
    p.add_argument("--threat-radius", type=float, dest="threat_radius_km", default=None)
    p.add_argument("--arrival-pixel-threshold", type=int, dest="arrival_pixel_threshold", default=None)
    p.add_argument("--departure-pixel-threshold", type=int, dest="departure_pixel_threshold", default=None)
    p.add_argument("--max-horizon", type=float, dest="max_horizon_min", default=None)
    p.add_argument("--min-size", type=int, dest="min_size_px", default=None)
    p.add_argument("--radius", type=float, dest="radius_km", default=None)
    p.add_argument("--weather-radius", type=float, dest="weather_radius_km", default=None)
    p.add_argument("--archive-radius", type=float, dest="archive_radius_km", default=None)
    p.add_argument("--archive-min-pixels", type=int, dest="archive_min_pixels", default=None)
    p.add_argument("--archive-enabled", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--archive-precompute-replay", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--fit-check-overlay", action=argparse.BooleanOptionalAction, dest="fit_check_overlay", default=None)
    p.add_argument("--flow-downsample", type=int, dest="flow_downsample", default=None)
    p.add_argument("--flow-block-px", type=int, dest="flow_block_px", default=None)
    p.add_argument("--flow-search-px", type=int, dest="flow_search_px", default=None)
    p.add_argument("--flow-average-pairs", type=int, dest="flow_average_pairs", default=None)
    p.add_argument("--flow-smooth-sigma", type=float, dest="flow_smooth_sigma", default=None)
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
    p.add_argument("--static-png-enabled", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--static-timestamp-region", default=None)
    p.add_argument("--display-tz", default=None)
    p.add_argument("--pljusak-enabled", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--pljusak-url", default=None)
    p.add_argument("--pljusak-interval", type=int, default=None)
    p.add_argument("--station-confirm-radius", type=float, default=None)
    p.add_argument("--station-confirm-min-mm", type=float, default=None)
    p.add_argument("--station-confirm-max-age-min", type=float, default=None)
    p.add_argument("--station-confirm-fraction", type=float, default=None)
    p.add_argument("--station-confirm-min-count", type=int, default=None)
    p.add_argument("--poll", type=int, dest="poll_s", default=None)
    p.add_argument("--view-rotation-auto", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--view-rotation", type=float, default=None)
    p.add_argument("--basemap", choices=("auto", "mapbox", "greyscale", "none"), default=None)
    p.add_argument("--basemap-saturation", type=float, default=None)
    p.add_argument("--basemap-brightness", type=float, default=None)
    p.add_argument("--basemap-contrast", type=float, default=None)
    p.add_argument("--mapbox-token", default=None)
    p.add_argument("--mapbox-style", default=None)
    p.add_argument("--basemap-size", type=int, default=None)
    p.add_argument("--cities", action=argparse.BooleanOptionalAction, dest="show_cities", default=None)
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
    return p


# ---------------------------------------------------------------- 24.3 -----
def bootstrap_from_args(args):
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
            "version": 2,
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
            raise SystemExit(f"Unknown city '{st['location']}'")
        target_name, target_lat, target_lon = r
    elif st.get("lat") is not None and st.get("lon") is not None:
        target_name = f"({st['lat']:.4f},{st['lon']:.4f})"
        target_lat, target_lon = st["lat"], st["lon"]
    else:
        raise SystemExit("Provide --location or --lat/--lon (or set in config.json)")

    display_tz, display_tz_name = get_zoneinfo(st["display_tz"])
    ts_tz = display_tz if st["timestamp_tz"] is None else get_zoneinfo(st["timestamp_tz"])[0]
    ts_region = (None if not st["timestamp_enabled"] else parse_region(st["timestamp_region"]))
    static_ts_region = (parse_region(st["static_timestamp_region"])
                        if st.get("static_timestamp_region") else ts_region)

    sri_legend = None
    legend_mask = None
    if st["radar_intensity"] == "sri":
        if not st["sri_legend_region"]:
            raise SystemExit("--radar-intensity sri requires --sri-legend-region")
        if not st["sri_breaks"]:
            raise SystemExit("--radar-intensity sri requires --sri-breaks")
        breaks = [float(x) for x in st["sri_breaks"].split(",")]
        sri_legend = SRILegend(
            region=parse_region(st["sri_legend_region"]),
            orientation=st["sri_legend_orientation"],
            breaks=breaks, padding_px=st["sri_legend_padding"])

    acc_store = AccuracyStore(data_dir, enabled=bool(st["accuracy_enabled"] and data_dir))
    _VIZ_STATE["acc_store"] = acc_store

    web_url = None
    if st["web_enabled"]:
        start_web(st["web_port"])
        web_url = f"http://localhost:{st['web_port']}"
        print(f"Web UI: {web_url}")

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

    print(f"Target: {target_name} ({target_lat:.4f}N, {target_lon:.4f}E)")
    print(f"Display timezone: {display_tz_name}")

    return {
        "data_dir": data_dir,
        "target_name": target_name,
        "target_lat": target_lat,
        "target_lon": target_lon,
        "display_tz": display_tz,
        "ts_tz": ts_tz,
        "ts_region": ts_region,
        "static_ts_region": static_ts_region,
        "sri_legend": sri_legend,
        "legend_mask": legend_mask,
        "acc_store": acc_store,
        "mqtt_bridge": mqtt_bridge,
        "pljusak": pljusak,
    }


# ---------------------------------------------------------------- 24.4 -----
def run_loop(ctx, args):
    data_dir = ctx["data_dir"]
    target_name = ctx["target_name"]
    target_lat = ctx["target_lat"]
    target_lon = ctx["target_lon"]
    display_tz = ctx["display_tz"]
    ts_tz = ctx["ts_tz"]
    ts_region = ctx["ts_region"]
    static_ts_region = ctx["static_ts_region"]
    sri_legend = ctx["sri_legend"]
    legend_mask = ctx["legend_mask"]
    acc_store = ctx["acc_store"]
    mqtt_bridge = ctx["mqtt_bridge"]
    pljusak = ctx["pljusak"]

    calib = None
    archiver = None
    replay_engine = None
    basemap_img = None
    basemap_north_up = None
    basemap_kind = "greyscale"
    last_basemap_key = None
    last_frame_ts = None
    prev_rain_state = False
    last_threat_sig = set()
    frame_buffer = []
    gif_ref_size = None
    ocr_source = "init"
    current_rain_history = []

    try:
        while True:
            t_start = time.monotonic()
            now_utc = datetime.now(timezone.utc)
            rt = _snapshot()
            st = _static_snapshot()

            if pljusak:
                if (pljusak.last_fetch_utc is None
                        or (now_utc - pljusak.last_fetch_utc).total_seconds() >= st["pljusak_interval"]):
                    pljusak.fetch(now_utc)

            need_gif = (not frame_buffer) or args.once
            gif_img = None
            if need_gif:
                try:
                    gif_img = fetch_image(GIF_URL)
                except Exception as e:
                    print(f"Radar fetch error: {e}", file=sys.stderr)
                    if args.once: break
                    _FORCE_REFRESH.wait(timeout=min(60, rt["poll_s"]))
                    _FORCE_REFRESH.clear()
                    continue

            if calib is None:
                if gif_img is None:
                    try:
                        gif_img = fetch_image(GIF_URL)
                    except Exception as e:
                        print(f"Radar fetch error (calib): {e}", file=sys.stderr)
                        if args.once: break
                        _FORCE_REFRESH.wait(timeout=min(60, rt["poll_s"]))
                        _FORCE_REFRESH.clear()
                        continue
                calib = load_calibration(st["calibration"], st["calibration_inline"],
                                         image_size=gif_img.size)
                gif_ref_size = gif_img.size
                if sri_legend:
                    x0, y0, x1, y1 = sri_legend.region
                    legend_mask = np.zeros((gif_img.size[1], gif_img.size[0]), dtype=bool)
                    legend_mask[max(0, y0):min(gif_img.size[1], y1),
                                max(0, x0):min(gif_img.size[0], x1)] = True
                if data_dir:
                    archiver = StormArchiver(data_dir, calib, target_lat, target_lon, target_name)
                    _VIZ_STATE["archiver"] = archiver
                    replay_engine = OfflineReplayEngine(
                        archiver, calib, target_lat, target_lon,
                        st["frame_interval_min"], st["radar_intensity"], sri_legend)
                    _VIZ_STATE["replay_engine"] = replay_engine

            if sri_legend and not sri_legend.is_ready():
                if gif_img is not None:
                    try:
                        gif_img.seek(max(0, getattr(gif_img, "n_frames", 1) - 1))
                    except Exception:
                        pass
                    sri_legend.calibrate(gif_img.convert("RGB"))
                elif frame_buffer:
                    sri_legend.calibrate(frame_buffer[-1][1])
                if args.dump_sri_calibration:
                    sri_legend.print_samples()
                    return

            if not frame_buffer and gif_img is not None:
                frame_buffer = init_frame_buffer_from_gif(gif_img, ts_region, ts_tz)
                ocr_source = "gif_init"
            else:
                static_img = None
                if st.get("static_png_enabled", True):
                    try:
                        static_img = fetch_static_png()
                    except Exception as e:
                        print(f"static PNG fetch error: {e}", file=sys.stderr)
                if static_img is not None:
                    status, detail = append_static_png_to_buffer(
                        static_img, frame_buffer, static_ts_region, ts_tz,
                        ref_size=gif_ref_size,
                        interval_min=st["frame_interval_min"])
                    if status == "gap":
                        print(f"frame buffer gap ({detail:.1f} min) → reinit", file=sys.stderr)
                        if gif_img is None:
                            try:
                                gif_img = fetch_image(GIF_URL)
                            except Exception as e:
                                print(f"Radar fetch error (gap): {e}", file=sys.stderr)
                                continue
                        frame_buffer = init_frame_buffer_from_gif(gif_img, ts_region, ts_tz)
                        ocr_source = "gif_reinit"
                    elif status == "ocr_fail":
                        ocr_source = "static_ocr_fail"
                    elif status in ("appended", "seeded"):
                        ocr_source = "static"
                    else:
                        ocr_source = "static_stale"
                else:
                    if gif_img is None:
                        try:
                            gif_img = fetch_image(GIF_URL)
                        except Exception as e:
                            print(f"Radar fetch error (fallback): {e}", file=sys.stderr)
                            if args.once: break
                            _FORCE_REFRESH.wait(timeout=min(60, rt["poll_s"]))
                            _FORCE_REFRESH.clear()
                            continue
                    dt, _, _ = read_frame_timestamp(gif_img, ts_region, ts_tz, source="last")
                    if dt is not None and (not frame_buffer or dt > frame_buffer[-1][0]):
                        gif_img.seek(getattr(gif_img, "n_frames", 1) - 1)
                        f = gif_img.convert("RGB")
                        if gif_ref_size and f.size != gif_ref_size:
                            f = f.resize(gif_ref_size, Image.LANCZOS)
                        frame_buffer.append((dt, f))
                        if len(frame_buffer) > FRAME_BUFFER_MAX:
                            frame_buffer.pop(0)
                        ocr_source = "gif_tail"
                    else:
                        ocr_source = "gif_tail_stale"

            if not frame_buffer:
                print("frame buffer empty — skipping cycle", file=sys.stderr)
                if args.once: break
                _FORCE_REFRESH.wait(timeout=min(60, rt["poll_s"]))
                _FORCE_REFRESH.clear()
                continue

            last_frame_ts = frame_buffer[-1][0]
            eff_interval = st["frame_interval_min"]
            data_age = ((now_utc - last_frame_ts).total_seconds() / 60.0
                        if last_frame_ts else None)

            frames_list = [f for (_, f) in frame_buffer]
            times_list = [t for (t, _) in frame_buffer]
            masks_mmh = [analyze_frame(f, calib.map_region, st["radar_intensity"],
                                       sri_legend, legend_mask) for f in frames_list]

            flows = []
            for i in range(len(frames_list) - 1):
                flows.append(estimate_flow_field(
                    masks_mmh[i][0], masks_mmh[i + 1][0],
                    block_px=int(rt["flow_block_px"]),
                    search_px=int(rt["flow_search_px"]),
                    downsample=int(rt["flow_downsample"])))

            current_mask, current_mmh = masks_mmh[-1]
            h_c, w_c = current_mask.shape
            focus_mask = calib.radius_mask(h_c, w_c, target_lat, target_lon,
                                           float(rt["focus_radius_km"]))
            focus_mask_c = current_mask & focus_mask
            current_flow = combined_flow(
                flows, len(flows), int(rt["flow_average_pairs"]),
                smooth_sigma=float(rt.get("flow_smooth_sigma", 2.0)))
            if current_flow is None:
                current_flow = np.zeros((h_c, w_c, 2), dtype=np.float32)

            clusters, label_array, arrivals, nc = analyze_frame_state(
                focus_mask_c, current_mmh, current_flow, calib,
                target_lat, target_lon, rt, st)

            px, py = calib.latlon_to_image_pixel(target_lat, target_lon)
            xi, yi = int(round(px)), int(round(py))
            if 0 <= yi < h_c and 0 <= xi < w_c:
                raining_at_target = bool(current_mask[yi, xi])
            else:
                raining_at_target = False
            current_rain_history.append(raining_at_target)
            if len(current_rain_history) > 30:
                current_rain_history.pop(0)
            if nc.raining_now:
                streak = 0
                for r in reversed(current_rain_history):
                    if r: streak += 1
                    else: break
                nc.rain_duration_min = streak * eff_interval

            stations = []
            local_wx = LocalWeather(radius_km=float(rt.get("weather_radius_km", 5.0)))
            if pljusak and pljusak.last_readings:
                all_stations = pljusak.get_all_stations(target_lat, target_lon, display_tz)
                stations = [s for s in all_stations if s.dist_km <= rt["focus_radius_km"]]
                local_wx = summarize_local_weather(
                    all_stations,
                    radius_km=float(rt.get("weather_radius_km", 5.0)),
                    max_age_min=float(st["station_confirm_max_age_min"]))

            ca = combine_analysis(
                nc.raining_now, stations, target_lat, target_lon,
                st["station_confirm_radius"], st["station_confirm_min_mm"],
                st["station_confirm_max_age_min"], st["station_confirm_fraction"],
                st["station_confirm_min_count"])

            if archiver:
                def _renderer_for_archive(frame, mask, mmh, flow, calib_, frame_ts,
                                          idx, total, ep_id):
                    return build_replay_png(
                        frame, mask, mmh, flow, calib_, frame_ts, idx, total, ep_id,
                        target_lat, target_lon, rt, st, sri_legend, legend_mask,
                        stations=stations, data_dir=data_dir)
                try:
                    archiver.process_animation(
                        frames_list, times_list, flows, masks_mmh, rt, st,
                        calib, replay_renderer=_renderer_for_archive,
                        local_weather=local_wx, target_raining=ca.raining_now)
                except Exception as e:
                    print(f"archive: error {e}", file=sys.stderr)

            if acc_store.enabled:
                now_iso = now_utc.isoformat()
                tag = rt.get("settings_tag", "")
                rt_snap = {"threat_radius_km": rt["threat_radius_km"],
                           "arrival_pixel_threshold": rt["arrival_pixel_threshold"],
                           "focus_radius_km": rt["focus_radius_km"],
                           "min_size_px": rt["min_size_px"]}
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
            prev_rain_state = ca.raining_now

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
                if basemap_img is None and bm_choice == "none":
                    basemap_kind = "none"

            out_size = max(128, int(round(size_px * rt["web_scale"])))
            png_bytes = render_composed_png(
                frames_list[-1], calib, target_lat, target_lon, rt["radius_km"],
                view_rot_used, out_size, basemap_img, basemap_kind,
                clusters, arrivals, stations, rt["show_cities"],
                st["radar_intensity"], sri_legend, legend_mask,
                rt, confirm_mm=st["station_confirm_min_mm"])
            gif_bytes = render_composed_gif(
                frames_list, flows, calib, target_lat, target_lon, rt["radius_km"],
                view_rot_used, out_size, basemap_img, basemap_kind,
                stations, rt["show_cities"], st["radar_intensity"],
                sri_legend, legend_mask, rt, st,
                confirm_mm=st["station_confirm_min_mm"])

            radar_mmh = (round(float(current_mmh[yi, xi]), 1)
                         if (0 <= yi < current_mmh.shape[0] and 0 <= xi < current_mmh.shape[1]
                             and current_mask[yi, xi])
                         else 0.0)
            nearest_cell_eta = nc.next_rain_eta_min
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
                "nowcast_error_min": acc_store.last_mae_min(kind="rain_start"),
                "data_age_min": round(data_age, 1) if data_age is not None else None,
                "frame_timestamp": last_frame_ts.isoformat() if last_frame_ts else None,
                "cluster_count": len(clusters),
                "nearest_cell_eta_min": round(nearest_cell_eta, 0) if nearest_cell_eta is not None else None,
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
                "summary": {"nowcast": nc.summary,
                            "clusters": [{"label": cluster_label(i), "size_px": a.cluster.size,
                                          "max_mmh": round(a.cluster.max_mmh, 1),
                                          "eta_min": round(a.eta_min, 0) if a.eta_min is not None else None,
                                          "coherence": round(a.coherence, 2)}
                                         for i, a in enumerate(arrivals)]},
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
                cur_sig = {(i, round(a.eta_min or 0, -1)) for i, a in enumerate(arrivals)
                           if a.eta_min is not None}
                if cur_sig != last_threat_sig:
                    last_threat_sig = cur_sig
                    if arrivals:
                        mqtt_bridge.publish_threat({
                            "ts": now_utc.isoformat(), "target": target_name,
                            "count": len([a for a in arrivals if a.eta_min is not None]),
                            "cells": attrs_by_key["summary"]["clusters"],
                        })

            with _VIZ_STATE["lock"]:
                _VIZ_STATE["png"] = png_bytes
                _VIZ_STATE["gif"] = gif_bytes
                _VIZ_STATE["updated"] = now_utc.astimezone(display_tz).strftime("%Y-%m-%d %H:%M:%S %Z")
                _VIZ_STATE["status"] = {
                    "target": target_name,
                    "settings_tag": rt.get("settings_tag", "local"),
                    "updated": _VIZ_STATE["updated"],
                    "confidence": ca.confidence,
                    "raining_now": ca.raining_now,
                    "summary": nc.summary,
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
                    "sensors": state_dict,
                }

            print(f"[{now_utc.astimezone(display_tz).strftime('%H:%M:%S')}] "
                  f"{ca.confidence} | {nc.summary} | "
                  f"buf={len(frame_buffer)} src={ocr_source} clusters={len(clusters)}")

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


# ---------------------------------------------------------------- 24.5 -----
def main():
    p = build_arg_parser()
    args = p.parse_args()
    if args.list_cities:
        for label, name, lat, lon in CITIES:
            print(f"  {label:<4} {name:<18} ({lat:.4f}, {lon:.4f})")
        return
    ctx = bootstrap_from_args(args)
    run_loop(ctx, args)


if __name__ == "__main__":
    main()
