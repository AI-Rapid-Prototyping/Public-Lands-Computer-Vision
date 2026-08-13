from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import tempfile
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional
from zoneinfo import ZoneInfo

os.environ.pop("OPENSSL_FORCE_FIPS_MODE", None)

import cv2
import numpy as np
import requests
from bs4 import BeautifulSoup
from ultralytics import YOLO


try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    # Databricks notebook cells do not define __file__.
    SCRIPT_DIR = Path.cwd().resolve()


def _default_output_dir() -> Path:
    name = SCRIPT_DIR.name.upper()
    if name == "GRCA":
        return SCRIPT_DIR
    return SCRIPT_DIR / "GRCA"


DEFAULT_WEBCAM_PAGE_URL = (
    "https://www.nps.gov/media/webcam/view.htm?id=9B5FC6BA-9FE6-EC6B-61637825D562D367&r=/grca/learn/photosmultimedia/webcams.htm"
)
DEFAULT_FALLBACK_IMAGE_URL = "https://www.nps.gov/webcams-grca/camera.jpg"
DEFAULT_MODEL_PATH = "yolov8x.pt"
DEFAULT_OUTPUT_DIR = _default_output_dir()
DEFAULT_LANES_PATH = DEFAULT_OUTPUT_DIR / "grca_lanes.json"
DEFAULT_FEED_OUTPUT_PATH = DEFAULT_OUTPUT_DIR / "grca_latest_feed.txt"
DEFAULT_ANNOTATED_IMAGE_OUTPUT_PATH = DEFAULT_OUTPUT_DIR / "grca_latest_annotated.jpg"
DEFAULT_ARCHIVE_OUTPUT_PATH = DEFAULT_OUTPUT_DIR / "archivefeed.csv"
DEFAULT_TRACKER_STATE_PATH = DEFAULT_OUTPUT_DIR / "grca_vehicle_tracker_state.json"
DEFAULT_GITHUB_REPOSITORY = "AI-Rapid-Prototyping/Public-Lands-Computer-Vision"
DEFAULT_GITHUB_BRANCH = "main"
DEFAULT_GITHUB_JSON_PATH = "GRCA/grca_vehicle_count_latest.json"
DEFAULT_GITHUB_FEED_PATH = "GRCA/grca_latest_feed.txt"
DEFAULT_GITHUB_IMAGE_PATH = "GRCA/grca_latest_annotated.jpg"
DEFAULT_GITHUB_ARCHIVE_CSV_PATH = "GRCA/archivefeed.csv"
DEFAULT_GITHUB_TRACKER_STATE_PATH = "GRCA/grca_vehicle_tracker_state.json"
DEFAULT_ARCHIVE_MAX_ROWS = 4999
DEFAULT_ARCHIVE_ROTATION_DIRNAME = "archive"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
DEFAULT_OUTPUT_PATH = DEFAULT_OUTPUT_DIR / "grca_vehicle_count_latest.json"
DEFAULT_ENV_PATH = SCRIPT_DIR / ".env"
VEHICLE_CLASSES = [2, 3, 5, 7]
TRACK_MAX_MISSED_SECONDS = 240
TRACK_INACTIVE_RETENTION_SECONDS = 900
TRACK_MATCH_SCORE_THRESHOLD = 0.45
TRACK_REACTIVATION_SCORE_THRESHOLD = 0.50
TRACK_HIST_BINS = (6, 4, 4)
ARIZONA_TIME_ZONE = ZoneInfo("America/Phoenix")


@dataclass(frozen=True)
class Config:
    webcam_page_url: str = DEFAULT_WEBCAM_PAGE_URL
    fallback_image_url: str = DEFAULT_FALLBACK_IMAGE_URL
    model_path: str = DEFAULT_MODEL_PATH
    lanes_path: Path = DEFAULT_LANES_PATH
    output_path: Path = DEFAULT_OUTPUT_PATH
    feed_output_path: Path = DEFAULT_FEED_OUTPUT_PATH
    annotated_image_output_path: Path = DEFAULT_ANNOTATED_IMAGE_OUTPUT_PATH
    archive_output_path: Path = DEFAULT_ARCHIVE_OUTPUT_PATH
    tracker_state_output_path: Path = DEFAULT_TRACKER_STATE_PATH
    github_repository: str = DEFAULT_GITHUB_REPOSITORY
    github_branch: str = DEFAULT_GITHUB_BRANCH
    github_json_path: str = DEFAULT_GITHUB_JSON_PATH
    github_feed_path: str = DEFAULT_GITHUB_FEED_PATH
    github_image_path: str = DEFAULT_GITHUB_IMAGE_PATH
    github_archive_csv_path: str = DEFAULT_GITHUB_ARCHIVE_CSV_PATH
    github_tracker_state_path: str = DEFAULT_GITHUB_TRACKER_STATE_PATH
    github_token: Optional[str] = None
    publish_to_github: bool = False
    confidence: float = 0.50
    iou: float = 0.45
    image_size: int = 1280
    user_agent: str = DEFAULT_USER_AGENT
    archive_max_rows: int = DEFAULT_ARCHIVE_MAX_ROWS


@dataclass
class DetectionBox:
    class_id: int
    class_name: str
    confidence: float
    xyxy: list[float]


@dataclass(frozen=True)
class LaneDefinition:
    lane_id: str
    label: str
    polygon: list[list[float]]


@dataclass
class RunResult:
    status: str
    timestamp_utc: str
    webcam_page_url: str
    image_url: Optional[str] = None
    model_path: str = DEFAULT_MODEL_PATH
    vehicle_count: int = 0
    daily_vehicle_count: int = 0
    hourly_vehicle_count: int = 0
    lane_1_count: int = 0
    lane_2_count: int = 0
    lane_3_count: int = 0
    lane_4_count: int = 0
    in_line_count: int = 0
    detections: list[DetectionBox] = field(default_factory=list)
    message: Optional[str] = None
    error: Optional[str] = None


def load_config() -> Config:
    """Load runtime settings from environment variables."""

    load_env_file()
    output_path = normalize_grca_path(Path(os.getenv("OUTPUT_PATH", str(DEFAULT_OUTPUT_PATH))).expanduser())
    feed_output_path = normalize_grca_path(Path(os.getenv("FEED_OUTPUT_PATH", str(DEFAULT_FEED_OUTPUT_PATH))).expanduser())
    annotated_image_output_path = normalize_grca_path(Path(os.getenv("ANNOTATED_IMAGE_OUTPUT_PATH", str(DEFAULT_ANNOTATED_IMAGE_OUTPUT_PATH))).expanduser())
    archive_output_path = normalize_grca_path(Path(os.getenv("ARCHIVE_OUTPUT_PATH", str(DEFAULT_ARCHIVE_OUTPUT_PATH))).expanduser())
    tracker_state_output_path = normalize_grca_path(Path(os.getenv("TRACKER_STATE_PATH", str(DEFAULT_TRACKER_STATE_PATH))).expanduser())
    github_token = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
    publish_to_github = os.getenv("PUBLISH_TO_GITHUB", "").strip().lower() in {"1", "true", "yes", "on"}
    if github_token and not os.getenv("PUBLISH_TO_GITHUB"):
        publish_to_github = True
    return Config(
        webcam_page_url=os.getenv("WEBCAM_PAGE_URL", DEFAULT_WEBCAM_PAGE_URL),
        fallback_image_url=os.getenv("FALLBACK_IMAGE_URL", DEFAULT_FALLBACK_IMAGE_URL),
        model_path=os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH),
        lanes_path=normalize_grca_path(Path(os.getenv("LANES_PATH", str(DEFAULT_LANES_PATH))).expanduser()),
        output_path=output_path,
        feed_output_path=feed_output_path,
        annotated_image_output_path=annotated_image_output_path,
        archive_output_path=archive_output_path,
        tracker_state_output_path=tracker_state_output_path,
        github_repository=os.getenv("GITHUB_REPOSITORY", DEFAULT_GITHUB_REPOSITORY),
        github_branch=os.getenv("GITHUB_BRANCH", DEFAULT_GITHUB_BRANCH),
        github_json_path=os.getenv("GITHUB_JSON_PATH", DEFAULT_GITHUB_JSON_PATH),
        github_feed_path=os.getenv("GITHUB_FEED_PATH", DEFAULT_GITHUB_FEED_PATH),
        github_image_path=os.getenv("GITHUB_IMAGE_PATH", DEFAULT_GITHUB_IMAGE_PATH),
        github_archive_csv_path=os.getenv("GITHUB_ARCHIVE_CSV_PATH", DEFAULT_GITHUB_ARCHIVE_CSV_PATH),
        github_tracker_state_path=os.getenv("GITHUB_TRACKER_STATE_PATH", DEFAULT_GITHUB_TRACKER_STATE_PATH),
        github_token=github_token,
        publish_to_github=publish_to_github,
        confidence=float(os.getenv("YOLO_CONFIDENCE", "0.50")),
        iou=float(os.getenv("YOLO_IOU", "0.45")),
        image_size=int(os.getenv("YOLO_IMAGE_SIZE", "1280")),
        user_agent=os.getenv("USER_AGENT", DEFAULT_USER_AGENT),
        archive_max_rows=int(os.getenv("ARCHIVE_MAX_ROWS", str(DEFAULT_ARCHIVE_MAX_ROWS))),
    )


def build_headers(user_agent: str) -> dict[str, str]:
    return {
        "User-Agent": user_agent,
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def arizona_day_key(moment: Optional[datetime] = None) -> str:
    moment = moment or datetime.now(timezone.utc)
    return moment.astimezone(ARIZONA_TIME_ZONE).date().isoformat()


def arizona_hour_key(moment: Optional[datetime] = None) -> str:
    moment = moment or datetime.now(timezone.utc)
    return moment.astimezone(ARIZONA_TIME_ZONE).strftime("%Y-%m-%dT%H")


def timestamp_arizona_day_key(timestamp_value: Any) -> Optional[str]:
    if not timestamp_value:
        return None

    try:
        parsed = datetime.fromisoformat(str(timestamp_value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return arizona_day_key(parsed)


def normalize_grca_path(path: Path) -> Path:
    raw = str(path)
    raw = raw.replace("/GRCA/GRCA", "/GRCA")
    raw = raw.replace("\\GRCA\\GRCA", "\\GRCA")
    return Path(raw)


def load_env_file(env_path: Path = DEFAULT_ENV_PATH) -> None:
    """Load simple KEY=VALUE pairs from a local .env file if present."""

    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        if not key or key in os.environ:
            continue

        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]

        os.environ[key] = value


def resolve_image_url(page_html: bytes, fallback_image_url: str) -> str:
    soup = BeautifulSoup(page_html, "html.parser")
    preferred_selectors = [
        "img.WebcamPreview__CoverImage",
        "img.refreshWebcamPreview",
        "a.WebcamPreview__CoverLink img",
    ]
    for selector in preferred_selectors:
        for img in soup.select(selector):
            for attr in ("src", "data-src", "data-original", "data-lazy-src"):
                src = img.get(attr, "")
                if not src:
                    continue
                if src.startswith("/"):
                    return "https://www.nps.gov" + src
                if src.startswith("//"):
                    return "https:" + src
                return src

    for img in soup.find_all("img"):
        src = img.get("src", "") or img.get("data-src", "") or img.get("data-original", "") or img.get("data-lazy-src", "")
        if not src:
            continue
        if any(keyword in src.lower() for keyword in ("webcam", "camera", "cctv", "az511", "webcams-")):
            if src.startswith("/"):
                return "https://www.nps.gov" + src
            if src.startswith("//"):
                return "https:" + src
            return src
    return fallback_image_url


def cache_busted_url(url: str) -> str:
    """Add a request-specific cache-buster without changing the base image URL."""

    parsed = urlsplit(url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query["_"] = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, urlencode(query), parsed.fragment))


def fetch_bytes(url: str, headers: dict[str, str], timeout: int = 10) -> requests.Response:
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response


def download_image(image_url: str, headers: dict[str, str]) -> np.ndarray:
    response = fetch_bytes(cache_busted_url(image_url), headers=headers)
    image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
    img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("OpenCV could not decode the downloaded image")
    return img


@lru_cache(maxsize=1)
def load_model(model_path: str) -> YOLO:
    return YOLO(model_path)


def serialize_detections(result: Any) -> list[DetectionBox]:
    if not result.boxes:
        return []

    names = result.names or {}
    detections: list[DetectionBox] = []
    for box in result.boxes:
        class_id = int(box.cls[0]) if getattr(box, "cls", None) is not None else -1
        confidence = float(box.conf[0]) if getattr(box, "conf", None) is not None else 0.0
        coords = [float(value) for value in box.xyxy[0].tolist()]
        detections.append(
            DetectionBox(
                class_id=class_id,
                class_name=str(names.get(class_id, class_id)),
                confidence=confidence,
                xyxy=coords,
            )
        )
    return detections


def load_lane_definitions(path: Path) -> list[LaneDefinition]:
    if not path.exists():
        raise FileNotFoundError(f"Lane file not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    lanes_raw = payload.get("lanes", [])
    if not isinstance(lanes_raw, list):
        raise ValueError("Lane JSON must contain a list under 'lanes'")

    lanes: list[LaneDefinition] = []
    for index, raw_lane in enumerate(lanes_raw, start=1):
        if not isinstance(raw_lane, dict):
            continue
        polygon_raw = raw_lane.get("polygon", [])
        polygon: list[list[float]] = []
        for point in polygon_raw:
            if isinstance(point, (list, tuple)) and len(point) == 2:
                polygon.append([float(point[0]), float(point[1])])
        if len(polygon) < 3:
            continue
        lanes.append(
            LaneDefinition(
                lane_id=str(raw_lane.get("lane_id") or f"lane_{index}"),
                label=str(raw_lane.get("label") or f"Lane {index}"),
                polygon=polygon,
            )
        )

    if not lanes:
        raise ValueError(f"No valid lane polygons found in {path}")
    return lanes


def point_in_polygon(point: tuple[float, float], polygon: list[list[float]]) -> bool:
    x, y = point
    inside = False
    n = len(polygon)
    if n < 3:
        return False

    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        intersects = ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi)
        if intersects:
            inside = not inside
        j = i
    return inside


def detection_anchor_point(detection: DetectionBox) -> tuple[float, float]:
    x1, y1, x2, y2 = detection.xyxy
    height = y2 - y1
    y = y2 - max(6.0, 0.1 * height)
    return (x1 + x2) / 2.0, y


def lane_detection_counts(detections: list[DetectionBox], lanes: list[LaneDefinition]) -> dict[str, int]:
    counts = {lane.label: 0 for lane in lanes}
    for detection in detections:
        anchor = detection_anchor_point(detection)
        for lane in lanes:
            if point_in_polygon(anchor, lane.polygon):
                counts[lane.label] = counts.get(lane.label, 0) + 1
                break
    return counts


def iou_xyxy(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union_area = area_a + area_b - inter_area
    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def bbox_center(xyxy: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = xyxy
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def bbox_size(xyxy: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = xyxy
    return max(0.0, x2 - x1), max(0.0, y2 - y1)


def bbox_area(xyxy: list[float]) -> float:
    width, height = bbox_size(xyxy)
    return width * height


def bbox_diagonal(xyxy: list[float]) -> float:
    width, height = bbox_size(xyxy)
    return float(np.hypot(width, height))


def clip_xyxy_to_image(xyxy: list[float], image_width: int, image_height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = xyxy
    left = max(0, min(image_width, int(round(x1))))
    top = max(0, min(image_height, int(round(y1))))
    right = max(0, min(image_width, int(round(x2))))
    bottom = max(0, min(image_height, int(round(y2))))
    return left, top, right, bottom


def detection_appearance_hist(image: Optional[np.ndarray], xyxy: list[float]) -> list[float]:
    if image is None:
        return []

    image_height, image_width = image.shape[:2]
    left, top, right, bottom = clip_xyxy_to_image(xyxy, image_width, image_height)
    if right <= left or bottom <= top:
        return []

    crop = image[top:bottom, left:right]
    if crop.size == 0:
        return []

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, list(TRACK_HIST_BINS), [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, None)
    return [float(value) for value in hist.flatten().tolist()]


def histogram_similarity(hist_a: list[float], hist_b: list[float]) -> float:
    if not hist_a or not hist_b or len(hist_a) != len(hist_b):
        return 0.0

    array_a = np.asarray(hist_a, dtype=np.float32)
    array_b = np.asarray(hist_b, dtype=np.float32)
    if array_a.size == 0 or array_b.size == 0:
        return 0.0

    score = cv2.compareHist(array_a, array_b, cv2.HISTCMP_CORREL)
    if not np.isfinite(score):
        return 0.0
    return float(max(0.0, min(1.0, (score + 1.0) / 2.0)))


def box_size_similarity(box_a: list[float], box_b: list[float]) -> float:
    area_a = bbox_area(box_a)
    area_b = bbox_area(box_b)
    if area_a <= 0.0 or area_b <= 0.0:
        return 0.0
    return float(max(0.0, 1.0 - (abs(area_a - area_b) / max(area_a, area_b))))


def lane_label_for_detection(detection: DetectionBox, lanes: list[LaneDefinition]) -> Optional[str]:
    anchor = detection_anchor_point(detection)
    for lane in lanes:
        if point_in_polygon(anchor, lane.polygon):
            return lane.label
    return None


def lane_bonus(track_lane_label: Optional[str], detection_lane_label: Optional[str], lane_index_map: dict[str, int]) -> float:
    if not track_lane_label or not detection_lane_label:
        return 0.0

    track_index = lane_index_map.get(track_lane_label)
    detection_index = lane_index_map.get(detection_lane_label)
    if track_index is None or detection_index is None:
        return 0.0

    lane_distance = abs(track_index - detection_index)
    if lane_distance == 0:
        return 0.10
    if lane_distance == 1:
        return 0.05
    return 0.0


def blend_histograms(existing_hist: list[float], new_hist: list[float]) -> list[float]:
    if not new_hist:
        return list(existing_hist)
    if not existing_hist:
        return [float(value) for value in new_hist]
    if len(existing_hist) != len(new_hist):
        return [float(value) for value in new_hist]

    existing = np.asarray(existing_hist, dtype=np.float32)
    incoming = np.asarray(new_hist, dtype=np.float32)
    blended = (0.7 * existing) + (0.3 * incoming)
    total = float(blended.sum())
    if total > 0.0:
        blended /= total
    return [float(value) for value in blended.tolist()]


@dataclass
class VehicleTrack:
    track_id: int
    class_id: int
    class_name: str
    first_seen_ts: float
    last_seen_ts: float
    confidence: float
    xyxy: list[float]
    prev_seen_ts: Optional[float] = None
    prev_xyxy: Optional[list[float]] = None
    appearance_hist: list[float] = field(default_factory=list)
    lane_label: Optional[str] = None
    missed_count: int = 0
    inactive_since_ts: Optional[float] = None

    def predicted_xyxy(self, observed_ts: float) -> list[float]:
        if self.prev_xyxy is None or self.prev_seen_ts is None:
            return list(self.xyxy)
        if self.last_seen_ts <= self.prev_seen_ts:
            return list(self.xyxy)

        elapsed = max(0.0, observed_ts - self.last_seen_ts)
        history_delta = max(1e-3, self.last_seen_ts - self.prev_seen_ts)
        prev_center_x, prev_center_y = bbox_center(self.prev_xyxy)
        last_center_x, last_center_y = bbox_center(self.xyxy)
        velocity_x = (last_center_x - prev_center_x) / history_delta
        velocity_y = (last_center_y - prev_center_y) / history_delta
        predicted_center_x = last_center_x + (velocity_x * elapsed)
        predicted_center_y = last_center_y + (velocity_y * elapsed)
        prev_width, prev_height = bbox_size(self.prev_xyxy)
        last_width, last_height = bbox_size(self.xyxy)
        width = max(8.0, (prev_width + last_width) / 2.0)
        height = max(8.0, (prev_height + last_height) / 2.0)
        return [
            predicted_center_x - (width / 2.0),
            predicted_center_y - (height / 2.0),
            predicted_center_x + (width / 2.0),
            predicted_center_y + (height / 2.0),
        ]

    def update_from_detection(
        self,
        detection: DetectionBox,
        observed_ts: float,
        appearance_hist: list[float],
        lane_label: Optional[str],
    ) -> None:
        self.prev_seen_ts = self.last_seen_ts
        self.prev_xyxy = list(self.xyxy)
        self.xyxy = list(detection.xyxy)
        self.last_seen_ts = observed_ts
        self.class_id = detection.class_id
        self.class_name = detection.class_name
        self.confidence = detection.confidence
        self.appearance_hist = blend_histograms(self.appearance_hist, appearance_hist)
        self.lane_label = lane_label
        self.missed_count = 0
        self.inactive_since_ts = None

    @classmethod
    def from_snapshot(cls, payload: dict[str, Any]) -> "VehicleTrack":
        prev_xyxy_raw = payload.get("prev_xyxy")
        appearance_hist_raw = payload.get("appearance_hist") or []
        return cls(
            track_id=int(payload.get("track_id", 0)),
            class_id=int(payload.get("class_id", -1)),
            class_name=str(payload.get("class_name", "unknown")),
            first_seen_ts=float(payload.get("first_seen_ts", datetime.now(timezone.utc).timestamp())),
            last_seen_ts=float(payload.get("last_seen_ts", datetime.now(timezone.utc).timestamp())),
            confidence=float(payload.get("confidence", 0.0)),
            xyxy=[float(value) for value in payload.get("xyxy", [0.0, 0.0, 0.0, 0.0])],
            prev_seen_ts=float(payload["prev_seen_ts"]) if payload.get("prev_seen_ts") is not None else None,
            prev_xyxy=[float(value) for value in prev_xyxy_raw] if isinstance(prev_xyxy_raw, list) else None,
            appearance_hist=[float(value) for value in appearance_hist_raw if isinstance(value, (int, float))],
            lane_label=str(payload.get("lane_label")) if payload.get("lane_label") else None,
            missed_count=int(payload.get("missed_count", 0)),
            inactive_since_ts=float(payload["inactive_since_ts"]) if payload.get("inactive_since_ts") is not None else None,
        )

    def snapshot(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrackerSummary:
    daily_vehicle_count: int
    hourly_vehicle_count: int
    active_vehicle_count: int


class VehicleTracker:
    def __init__(
        self,
        *,
        day_key: Optional[str] = None,
        next_track_id: int = 1,
        daily_vehicle_count: int = 0,
        hourly_vehicle_counts: Optional[dict[str, int]] = None,
        active_tracks: Optional[dict[int, VehicleTrack]] = None,
        inactive_tracks: Optional[dict[int, VehicleTrack]] = None,
        last_observed_ts: float = 0.0,
    ) -> None:
        self._day_key = day_key
        self._next_track_id = next_track_id
        self._daily_vehicle_count = daily_vehicle_count
        self._hourly_vehicle_counts = dict(hourly_vehicle_counts or {})
        self._active_tracks = active_tracks or {}
        self._inactive_tracks = inactive_tracks or {}
        self._last_observed_ts = last_observed_ts

    @property
    def day_key(self) -> Optional[str]:
        return self._day_key

    @property
    def daily_vehicle_count(self) -> int:
        return self._daily_vehicle_count

    @property
    def last_observed_ts(self) -> float:
        return self._last_observed_ts

    def reconcile_daily_count(self, minimum_count: int = 0) -> int:
        track_ids = set(self._active_tracks) | set(self._inactive_tracks)
        highest_track_id = max(track_ids, default=0)
        self._daily_vehicle_count = max(
            0,
            self._daily_vehicle_count,
            int(minimum_count),
            self._next_track_id - 1,
            highest_track_id,
        )
        return self._daily_vehicle_count

    def reconcile_counters_from(self, other: "VehicleTracker") -> None:
        if self._day_key != other._day_key:
            return

        for hour_key, count in other._hourly_vehicle_counts.items():
            self._hourly_vehicle_counts[hour_key] = max(
                self._hourly_vehicle_counts.get(hour_key, 0),
                count,
            )
        self.reconcile_daily_count(other.daily_vehicle_count)

    @staticmethod
    def _track_iou(track: VehicleTrack, detection: DetectionBox) -> float:
        return iou_xyxy(track.xyxy, detection.xyxy)

    def ensure_day(self, day_key: str) -> None:
        if self._day_key == day_key:
            return
        self._day_key = day_key
        self._next_track_id = 1
        self._daily_vehicle_count = 0
        self._hourly_vehicle_counts = {}
        self._active_tracks = {}
        self._inactive_tracks = {}
        self._last_observed_ts = 0.0

    def _prune_inactive_tracks(self, observed_ts: float) -> None:
        expired_track_ids = [
            track_id
            for track_id, track in self._inactive_tracks.items()
            if track.inactive_since_ts is not None and observed_ts - track.inactive_since_ts > TRACK_INACTIVE_RETENTION_SECONDS
        ]
        for track_id in expired_track_ids:
            self._inactive_tracks.pop(track_id, None)

    def _promote_stale_tracks(self, observed_ts: float) -> None:
        stale_track_ids = [
            track_id
            for track_id, track in self._active_tracks.items()
            if observed_ts - track.last_seen_ts >= TRACK_MAX_MISSED_SECONDS
        ]
        for track_id in stale_track_ids:
            track = self._active_tracks.pop(track_id, None)
            if track is None:
                continue
            if track.inactive_since_ts is None:
                track.inactive_since_ts = track.last_seen_ts
            self._inactive_tracks[track_id] = track

    def _score_track_match(
        self,
        track: VehicleTrack,
        detection: DetectionBox,
        observed_ts: float,
        detection_hist: list[float],
        detection_lane_label: Optional[str],
        lane_index_map: dict[str, int],
    ) -> float:
        if track.class_id != detection.class_id:
            return -1.0

        predicted_xyxy = track.predicted_xyxy(observed_ts)
        iou_score = iou_xyxy(predicted_xyxy, detection.xyxy)
        predicted_center_x, predicted_center_y = bbox_center(predicted_xyxy)
        detection_center_x, detection_center_y = bbox_center(detection.xyxy)
        distance = float(np.hypot(predicted_center_x - detection_center_x, predicted_center_y - detection_center_y))
        predicted_diag = max(bbox_diagonal(predicted_xyxy), 1.0)
        detection_diag = max(bbox_diagonal(detection.xyxy), 1.0)
        search_radius = max(90.0, predicted_diag * 3.0, detection_diag * 3.0, 50.0 + (2.5 * max(0.0, observed_ts - track.last_seen_ts)))
        center_score = 1.0 / (1.0 + (distance / search_radius))
        size_score = box_size_similarity(predicted_xyxy, detection.xyxy)
        appearance_score = histogram_similarity(track.appearance_hist, detection_hist)
        bonus = lane_bonus(track.lane_label, detection_lane_label, lane_index_map)

        if iou_score <= 0.01 and center_score < 0.15 and appearance_score < 0.15:
            return -1.0

        score = (
            (0.22 * iou_score)
            + (0.25 * center_score)
            + (0.38 * appearance_score)
            + (0.10 * size_score)
            + bonus
        )
        if track.inactive_since_ts is not None:
            score *= 0.95
        return score

    def _greedy_match_pool(
        self,
        tracks: dict[int, VehicleTrack],
        detections_meta: list[dict[str, Any]],
        observed_ts: float,
        lane_index_map: dict[str, int],
        threshold: float,
    ) -> tuple[list[tuple[int, int]], set[int]]:
        track_items = list(tracks.items())
        candidate_matches: list[tuple[float, int, int]] = []
        for track_index, (_track_id, track) in enumerate(track_items):
            for detection_index, meta in enumerate(detections_meta):
                score = self._score_track_match(
                    track,
                    meta["detection"],
                    observed_ts,
                    meta["appearance_hist"],
                    meta["lane_label"],
                    lane_index_map,
                )
                if score >= threshold:
                    candidate_matches.append((score, track_index, detection_index))

        candidate_matches.sort(reverse=True)
        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        assignments: list[tuple[int, int]] = []

        for _score, track_index, detection_index in candidate_matches:
            if track_index in matched_tracks or detection_index in matched_detections:
                continue
            track_id, _track = track_items[track_index]
            assignments.append((track_id, detection_index))
            matched_tracks.add(track_index)
            matched_detections.add(detection_index)

        return assignments, matched_detections

    def _make_detection_meta(self, detections: list[DetectionBox], image: Optional[np.ndarray], lanes: list[LaneDefinition]) -> list[dict[str, Any]]:
        metadata: list[dict[str, Any]] = []
        for detection in detections:
            metadata.append(
                {
                    "detection": detection,
                    "lane_label": lane_label_for_detection(detection, lanes),
                    "appearance_hist": detection_appearance_hist(image, detection.xyxy),
                }
            )
        return metadata

    def update(
        self,
        detections: list[DetectionBox],
        image: Optional[np.ndarray],
        observed_ts: float,
        day_key: str,
        lanes: list[LaneDefinition],
    ) -> TrackerSummary:
        self.ensure_day(day_key)
        self._last_observed_ts = max(self._last_observed_ts, observed_ts)
        self._prune_inactive_tracks(observed_ts)
        self._promote_stale_tracks(observed_ts)

        current_hour_key = arizona_hour_key(datetime.fromtimestamp(observed_ts, timezone.utc))
        lane_index_map = {lane.label: index for index, lane in enumerate(lanes)}
        detections_meta = self._make_detection_meta(detections, image, lanes)

        active_assignments, matched_detection_indexes = self._greedy_match_pool(
            self._active_tracks,
            detections_meta,
            observed_ts,
            lane_index_map,
            TRACK_MATCH_SCORE_THRESHOLD,
        )

        for track_id, detection_index in active_assignments:
            track = self._active_tracks.get(track_id)
            meta = detections_meta[detection_index]
            if track is None:
                continue
            track.update_from_detection(
                meta["detection"],
                observed_ts,
                meta["appearance_hist"],
                meta["lane_label"],
            )

        remaining_indexes = [
            index
            for index in range(len(detections_meta))
            if index not in matched_detection_indexes
        ]
        remaining_meta = [detections_meta[index] for index in remaining_indexes]

        inactive_assignments, inactive_matched_indexes = self._greedy_match_pool(
            self._inactive_tracks,
            remaining_meta,
            observed_ts,
            lane_index_map,
            TRACK_REACTIVATION_SCORE_THRESHOLD,
        )

        for track_id, meta_index in inactive_assignments:
            track = self._inactive_tracks.pop(track_id, None)
            if track is None:
                continue
            meta = remaining_meta[meta_index]
            track.update_from_detection(
                meta["detection"],
                observed_ts,
                meta["appearance_hist"],
                meta["lane_label"],
            )
            self._active_tracks[track_id] = track

        remaining_after_inactive = [
            index
            for index in range(len(remaining_meta))
            if index not in inactive_matched_indexes
        ]

        for meta_index in remaining_after_inactive:
            meta = remaining_meta[meta_index]
            detection = meta["detection"]
            self._active_tracks[self._next_track_id] = VehicleTrack(
                track_id=self._next_track_id,
                class_id=detection.class_id,
                class_name=detection.class_name,
                first_seen_ts=observed_ts,
                last_seen_ts=observed_ts,
                confidence=detection.confidence,
                xyxy=list(detection.xyxy),
                prev_seen_ts=None,
                prev_xyxy=None,
                appearance_hist=list(meta["appearance_hist"]),
                lane_label=meta["lane_label"],
            )
            detection.track_id = self._next_track_id
            self._hourly_vehicle_counts[current_hour_key] = self._hourly_vehicle_counts.get(current_hour_key, 0) + 1
            self._daily_vehicle_count += 1
            self._next_track_id += 1

        for track_id, track in list(self._active_tracks.items()):
            if track.last_seen_ts == observed_ts:
                continue
            track.missed_count += 1
            if observed_ts - track.last_seen_ts >= TRACK_MAX_MISSED_SECONDS:
                track.inactive_since_ts = track.last_seen_ts
                self._inactive_tracks[track_id] = track
                self._active_tracks.pop(track_id, None)

        self.reconcile_daily_count(len(detections))
        return TrackerSummary(
            daily_vehicle_count=self._daily_vehicle_count,
            hourly_vehicle_count=self._hourly_vehicle_counts.get(current_hour_key, 0),
            active_vehicle_count=len(self._active_tracks),
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "day_key": self._day_key,
            "next_track_id": self._next_track_id,
            "daily_vehicle_count": self._daily_vehicle_count,
            "hourly_vehicle_counts": dict(self._hourly_vehicle_counts),
            "last_observed_ts": self._last_observed_ts,
            "active_tracks": [track.snapshot() for track in self._active_tracks.values()],
            "inactive_tracks": [track.snapshot() for track in self._inactive_tracks.values()],
        }

    @classmethod
    def from_snapshot(cls, payload: dict[str, Any]) -> "VehicleTracker":
        tracker = cls()

        day_key = payload.get("day_key")
        tracker._day_key = str(day_key) if day_key else None

        try:
            tracker._next_track_id = int(payload.get("next_track_id", 1))
        except (TypeError, ValueError):
            tracker._next_track_id = 1

        try:
            tracker._daily_vehicle_count = int(payload.get("daily_vehicle_count", 0))
        except (TypeError, ValueError):
            tracker._daily_vehicle_count = 0

        try:
            tracker._last_observed_ts = float(payload.get("last_observed_ts", 0.0))
        except (TypeError, ValueError):
            tracker._last_observed_ts = 0.0

        hourly_vehicle_counts_raw = payload.get("hourly_vehicle_counts") or {}
        if isinstance(hourly_vehicle_counts_raw, dict):
            for hour_key, value in hourly_vehicle_counts_raw.items():
                try:
                    tracker._hourly_vehicle_counts[str(hour_key)] = int(value)
                except (TypeError, ValueError):
                    continue

        active_tracks = payload.get("active_tracks") or []
        if isinstance(active_tracks, list):
            for raw_track in active_tracks:
                if not isinstance(raw_track, dict):
                    continue
                try:
                    track = VehicleTrack.from_snapshot(raw_track)
                    tracker._active_tracks[track.track_id] = track
                except (TypeError, ValueError, KeyError):
                    continue

        inactive_tracks = payload.get("inactive_tracks") or []
        if isinstance(inactive_tracks, list):
            for raw_track in inactive_tracks:
                if not isinstance(raw_track, dict):
                    continue
                try:
                    track = VehicleTrack.from_snapshot(raw_track)
                    if track.inactive_since_ts is None:
                        track.inactive_since_ts = track.last_seen_ts
                    if track.track_id in tracker._active_tracks:
                        continue
                    tracker._inactive_tracks[track.track_id] = track
                except (TypeError, ValueError, KeyError):
                    continue

        retained_tracks = list(tracker._active_tracks.values()) + list(tracker._inactive_tracks.values())
        if retained_tracks:
            tracker._last_observed_ts = max(
                tracker._last_observed_ts,
                max(track.last_seen_ts for track in retained_tracks),
            )
            tracker._next_track_id = max(
                tracker._next_track_id,
                max(track.track_id for track in retained_tracks) + 1,
            )
        tracker.reconcile_daily_count()
        return tracker


def tracker_to_json_text(tracker: VehicleTracker) -> str:
    return json.dumps(tracker.snapshot(), indent=2) + "\n"


def tracker_from_json_text(text: str) -> VehicleTracker:
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("Tracker state JSON must be an object")
    return VehicleTracker.from_snapshot(payload)


def tracker_state_fallback_path(output_path: Path) -> Path:
    return Path(tempfile.gettempdir()) / "grca" / output_path.name


def load_tracker_state_from_path(output_path: Path) -> Optional[VehicleTracker]:
    for candidate in (output_path, tracker_state_fallback_path(output_path)):
        if not candidate.exists():
            continue
        try:
            return tracker_from_json_text(candidate.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    return None


def write_tracker_state(tracker: VehicleTracker, output_path: Path) -> Path:
    writable_path = resolve_writable_output_path(output_path)
    writable_path.write_text(tracker_to_json_text(tracker), encoding="utf-8")
    return writable_path


def get_github_read_headers(token: Optional[str] = None) -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def load_tracker_state_from_github(config: Config) -> Optional[VehicleTracker]:
    headers = get_github_read_headers(config.github_token)
    url = github_contents_api_url(config.github_repository, config.github_tracker_state_path)

    try:
        response = requests.get(url, headers=headers, params={"ref": config.github_branch}, timeout=10)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        payload = response.json()
        content = payload.get("content")
        encoding = payload.get("encoding")
        if content and encoding == "base64":
            raw_text = base64.b64decode(content).decode("utf-8")
            return tracker_from_json_text(raw_text)
    except (requests.RequestException, UnicodeDecodeError, ValueError, json.JSONDecodeError):
        return None

    return None


def load_tracker_state(config: Config) -> VehicleTracker:
    candidates = [
        tracker
        for tracker in (
            load_tracker_state_from_path(config.tracker_state_output_path),
            load_tracker_state_from_github(config),
        )
        if tracker is not None
    ]
    if not candidates:
        return VehicleTracker()

    current_day_key = arizona_day_key()
    current_day_candidates = [tracker for tracker in candidates if tracker.day_key == current_day_key]
    eligible = current_day_candidates or candidates
    tracker = max(eligible, key=lambda candidate: candidate.last_observed_ts)
    for candidate in eligible:
        tracker.reconcile_counters_from(candidate)
    return tracker


def archive_daily_count_floor(rows: list[dict[str, Any]], day_key: str) -> int:
    count_floor = 0
    for row in rows:
        if timestamp_arizona_day_key(row.get("timestamp_utc")) != day_key:
            continue

        for field_name in ("daily_vehicle_count", "total_vehicles_detected", "vehicles_in_line"):
            try:
                value = int(float(row.get(field_name, 0)))
            except (TypeError, ValueError):
                continue
            count_floor = max(count_floor, value)
    return count_floor


def load_archive_daily_count_floor_from_path(output_path: Path, day_key: str) -> int:
    fallback_path = Path(tempfile.gettempdir()) / "grca" / output_path.name
    count_floor = 0
    candidates = [output_path, fallback_path]
    for current_path in (output_path, fallback_path):
        archive_dir = current_path.parent / DEFAULT_ARCHIVE_ROTATION_DIRNAME
        try:
            latest_archived_path = max(archive_dir.glob("*.csv"), default=None)
        except OSError:
            latest_archived_path = None
        if latest_archived_path is not None:
            candidates.append(latest_archived_path)

    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            with candidate.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            count_floor = max(count_floor, archive_daily_count_floor(rows, day_key))
        except (OSError, csv.Error):
            continue
    return count_floor


def load_archive_daily_count_floor_from_github(config: Config, day_key: str) -> int:
    headers = get_github_read_headers(config.github_token)
    count_floor = 0

    def _load_file_floor(repo_path: str) -> int:
        url = github_contents_api_url(config.github_repository, repo_path)
        try:
            response = requests.get(url, headers=headers, params={"ref": config.github_branch}, timeout=10)
            if response.status_code == 404:
                return 0
            response.raise_for_status()
            payload = response.json()
            content = payload.get("content")
            if content and payload.get("encoding") == "base64":
                csv_text = base64.b64decode(content).decode("utf-8")
                return archive_daily_count_floor(csv_text_to_rows(csv_text), day_key)
        except (requests.RequestException, UnicodeDecodeError, ValueError, json.JSONDecodeError, csv.Error):
            return 0
        return 0

    count_floor = _load_file_floor(config.github_archive_csv_path)

    archive_parent = config.github_archive_csv_path.rsplit("/", 1)[0] if "/" in config.github_archive_csv_path else ""
    archive_dir_path = "/".join(part for part in (archive_parent, DEFAULT_ARCHIVE_ROTATION_DIRNAME) if part)
    archive_dir_url = github_contents_api_url(config.github_repository, archive_dir_path)
    try:
        response = requests.get(archive_dir_url, headers=headers, params={"ref": config.github_branch}, timeout=10)
        if response.status_code == 404:
            return count_floor
        response.raise_for_status()
        entries = response.json()
        if not isinstance(entries, list):
            return count_floor
        archived_paths = sorted(
            (
                str(entry.get("path"))
                for entry in entries
                if isinstance(entry, dict)
                and entry.get("type") == "file"
                and str(entry.get("name", "")).lower().endswith(".csv")
                and entry.get("path")
            ),
            reverse=True,
        )
        if archived_paths:
            count_floor = max(count_floor, _load_file_floor(archived_paths[0]))
    except (requests.RequestException, ValueError, json.JSONDecodeError):
        return count_floor
    return count_floor


def load_archive_daily_count_floor(config: Config, day_key: str) -> int:
    local_floor = load_archive_daily_count_floor_from_path(config.archive_output_path, day_key)
    if not config.publish_to_github:
        return local_floor
    remote_floor = load_archive_daily_count_floor_from_github(config, day_key)
    return max(local_floor, remote_floor)


def publish_tracker_state_to_github(tracker: VehicleTracker, config: Config) -> tuple[bool, str]:
    if not config.github_token:
        return False, "Skipped tracker state publish: no GITHUB_TOKEN or GH_TOKEN set"

    return publish_text_to_github(
        tracker_to_json_text(tracker),
        config.github_repository,
        config.github_branch,
        config.github_tracker_state_path,
        config.github_token,
        "Update Grand Canyon vehicle tracker state",
    )


def run_detection(config: Config) -> RunResult:
    return run_detection_with_image(config)[0]


def run_detection_with_image(
    config: Config,
    tracker: Optional[VehicleTracker] = None,
    daily_count_floor: int = 0,
) -> tuple[RunResult, Optional[np.ndarray]]:
    tracker = tracker or load_tracker_state(config)
    return _run_detection_with_image(
        config,
        tracker=tracker,
        daily_count_floor=daily_count_floor,
        webcam_page_url=config.webcam_page_url,
        fallback_image_url=config.fallback_image_url,
        lanes_path=config.lanes_path,
    )


def _run_detection_with_image(
    config: Config,
    *,
    tracker: VehicleTracker,
    daily_count_floor: int,
    webcam_page_url: str,
    fallback_image_url: str,
    lanes_path: Path,
) -> tuple[RunResult, Optional[np.ndarray]]:
    headers = build_headers(config.user_agent)
    timestamp_dt = datetime.now(timezone.utc)
    timestamp = timestamp_dt.isoformat()
    day_key = arizona_day_key(timestamp_dt)

    try:
        image_url = fallback_image_url
        try:
            page_response = fetch_bytes(cache_busted_url(webcam_page_url), headers=headers)
            image_url = resolve_image_url(page_response.content, fallback_image_url)
        except Exception:
            image_url = fallback_image_url

        img = download_image(image_url, headers=headers)

        model = load_model(config.model_path)
        results = model.predict(
            img,
            classes=VEHICLE_CLASSES,
            conf=config.confidence,
            iou=config.iou,
            imgsz=config.image_size,
            verbose=False,
        )

        detections = serialize_detections(results[0]) if results else []
        lanes = load_lane_definitions(lanes_path)
        lane_counts = lane_detection_counts(detections, lanes)
        ordered_lane_counts = [lane_counts.get(lane.label, 0) for lane in lanes[:4]]
        while len(ordered_lane_counts) < 4:
            ordered_lane_counts.append(0)
        lane_1_count, lane_2_count, lane_3_count, lane_4_count = ordered_lane_counts[:4]
        in_line_count = sum(ordered_lane_counts[:4])
        tracker_summary = tracker.update(detections, img, timestamp_dt.timestamp(), day_key, lanes)
        daily_vehicle_count = tracker.reconcile_daily_count(
            max(daily_count_floor, len(detections), in_line_count)
        )
        hourly_vehicle_count = tracker_summary.hourly_vehicle_count
        result = RunResult(
            status="ok",
            timestamp_utc=timestamp,
            webcam_page_url=webcam_page_url,
            image_url=image_url,
            model_path=config.model_path,
            vehicle_count=len(detections),
            daily_vehicle_count=daily_vehicle_count,
            hourly_vehicle_count=hourly_vehicle_count,
            lane_1_count=lane_1_count,
            lane_2_count=lane_2_count,
            lane_3_count=lane_3_count,
            lane_4_count=lane_4_count,
            in_line_count=in_line_count,
            detections=detections,
            message=(
                f"Detected {len(detections)} vehicles anywhere in frame; "
                f"lane 1={lane_1_count}, lane 2={lane_2_count}, "
                f"lane 3={lane_3_count}, lane 4={lane_4_count}, "
                f"in line={in_line_count}, daily_vehicle_count={daily_vehicle_count}, "
                f"hourly_vehicle_count={hourly_vehicle_count}"
            ),
        )
        annotated_image = annotate_image(img, results, result)
        return (
            result,
            annotated_image,
        )
    except Exception as exc:
        tracker.reconcile_daily_count(daily_count_floor)
        return (
            RunResult(
                status="error",
                timestamp_utc=timestamp,
                webcam_page_url=webcam_page_url,
                model_path=config.model_path,
                daily_vehicle_count=tracker.daily_vehicle_count,
                hourly_vehicle_count=0,
                error=str(exc),
            ),
            None,
        )


def annotate_image(img: np.ndarray, results: Any, result: RunResult) -> np.ndarray:
    if not results:
        return img
    annotated = results[0].plot()
    overlay_lines = [
        result.timestamp_utc,
        f"total_vehicles_detected={result.vehicle_count}",
        f"daily_vehicle_count={result.daily_vehicle_count}",
        f"hourly_vehicle_count={result.hourly_vehicle_count}",
        f"vehicles_in_lane_1={result.lane_1_count}",
        f"vehicles_in_lane_2={result.lane_2_count}",
        f"vehicles_in_lane_3={result.lane_3_count}",
        f"vehicles_in_lane_4={result.lane_4_count}",
        f"vehicles_in_line={result.in_line_count}",
    ]
    y = 36
    for line in overlay_lines:
        cv2.putText(
            annotated,
            line,
            (16, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            line,
            (16, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 36
    return annotated


def result_to_dict(result: RunResult, output_schema: str) -> dict[str, Any]:
    payload = asdict(result)
    payload["detections"] = [asdict(detection) for detection in result.detections]
    payload["output_schema"] = output_schema
    return payload


def result_to_json_text(result: RunResult, output_schema: str) -> str:
    return json.dumps(result_to_dict(result, output_schema), indent=2) + "\n"


def result_to_feed_payload(result: RunResult, output_schema: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": result.status,
        "timestamp_utc": result.timestamp_utc,
        "total_vehicles_detected": result.vehicle_count,
        "daily_vehicle_count": result.daily_vehicle_count,
        "hourly_vehicle_count": result.hourly_vehicle_count,
        "vehicles_in_lane_1": result.lane_1_count,
        "vehicles_in_lane_2": result.lane_2_count,
        "vehicles_in_lane_3": result.lane_3_count,
        "vehicles_in_lane_4": result.lane_4_count,
        "vehicles_in_line": result.in_line_count,
        "output_schema": output_schema,
    }
    if result.message is not None:
        payload["message"] = result.message
    if result.error is not None:
        payload["error"] = result.error
    return payload


def result_to_feed_json_text(result: RunResult, output_schema: str) -> str:
    return json.dumps(result_to_feed_payload(result, output_schema), indent=2) + "\n"


def result_to_archive_row(result: RunResult) -> dict[str, Any]:
    return {
        "timestamp_utc": result.timestamp_utc,
        "total_vehicles_detected": result.vehicle_count,
        "daily_vehicle_count": result.daily_vehicle_count,
        "vehicles_in_lane_1": result.lane_1_count,
        "vehicles_in_lane_2": result.lane_2_count,
        "vehicles_in_lane_3": result.lane_3_count,
        "vehicles_in_lane_4": result.lane_4_count,
        "vehicles_in_line": result.in_line_count,
    }


def result_to_feed_text(result: RunResult, feed_title: str, output_schema: str) -> str:
    lines = [
        feed_title,
        f"status: {result.status}",
        f"timestamp_utc: {result.timestamp_utc}",
        f"total_vehicles_detected: {result.vehicle_count}",
        f"daily_vehicle_count: {result.daily_vehicle_count}",
        f"hourly_vehicle_count: {result.hourly_vehicle_count}",
        f"vehicles_in_lane_1: {result.lane_1_count}",
        f"vehicles_in_lane_2: {result.lane_2_count}",
        f"vehicles_in_lane_3: {result.lane_3_count}",
        f"vehicles_in_lane_4: {result.lane_4_count}",
        f"vehicles_in_line: {result.in_line_count}",
    ]

    if result.message:
        lines.append(f"message: {result.message}")

    if result.error:
        lines.append(f"error: {result.error}")

    if result.detections:
        lines.append("detections:")
        for detection in result.detections:
            lines.append(
                f"- {detection.class_name}"
                f" (class_id={detection.class_id}, confidence={detection.confidence:.3f})"
                f" xyxy={detection.xyxy}"
            )
    else:
        lines.append("detections: none")

    lines.append("lane_breakdown:")
    lines.append(f"- vehicles_in_lane_1: {result.lane_1_count}")
    lines.append(f"- vehicles_in_lane_2: {result.lane_2_count}")
    lines.append(f"- vehicles_in_lane_3: {result.lane_3_count}")
    lines.append(f"- vehicles_in_lane_4: {result.lane_4_count}")
    lines.append(f"- vehicles_in_line: {result.in_line_count}")

    lines.append(f"output_schema: {output_schema}")
    return "\n".join(lines) + "\n"


def result_to_image_commit_message(result: RunResult, camera_name: str) -> str:
    return (
        f"Update {camera_name} webcam image "
        f"{result.timestamp_utc} "
        f"vehicles={result.vehicle_count} "
        f"daily={result.daily_vehicle_count} "
        f"hourly={result.hourly_vehicle_count} "
        f"lane1={result.lane_1_count} "
        f"lane2={result.lane_2_count} "
        f"lane3={result.lane_3_count} "
        f"lane4={result.lane_4_count} "
        f"inline={result.in_line_count}"
    )


def write_result(result: RunResult, output_path: Path, output_schema: str = "grca_lane_vehicle_count_v2") -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(result_to_feed_json_text(result, output_schema), encoding="utf-8")
    return output_path


def write_feed(result: RunResult, output_path: Path, feed_title: str = "Grand Canyon South Entrance webcam vehicle feed", output_schema: str = "grca_lane_vehicle_feed_v2") -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    feed_text = result_to_feed_text(result, feed_title, output_schema)
    output_path.write_text(feed_text, encoding="utf-8")
    return output_path


def write_archive_csv(result: RunResult, output_path: Path, max_rows: int) -> Path:
    written_path, _archived_path = append_archive_csv(result, output_path, max_rows)
    return written_path


def write_annotated_image(img: np.ndarray, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), img):
        raise ValueError(f"Failed to write annotated image to {output_path}")
    return output_path


def archive_csv_header() -> list[str]:
    return [
        "timestamp_utc",
        "total_vehicles_detected",
        "daily_vehicle_count",
        "vehicles_in_lane_1",
        "vehicles_in_lane_2",
        "vehicles_in_lane_3",
        "vehicles_in_lane_4",
        "vehicles_in_line",
    ]


def archive_row_to_dict(result: RunResult) -> dict[str, Any]:
    return result_to_archive_row(result)


def archive_rows_to_csv_text(results: list[RunResult]) -> str:
    from io import StringIO

    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=archive_csv_header(), extrasaction="ignore")
    writer.writeheader()
    for result in results:
        writer.writerow(archive_row_to_dict(result))
    return buffer.getvalue()


def append_archive_csv(result: RunResult, output_path: Path, max_rows: int) -> tuple[Path, Optional[Path]]:
    return try_append_archive_rows(output_path, [result], max_rows)


def csv_text_to_rows(csv_text: str) -> list[dict[str, str]]:
    from io import StringIO

    if not csv_text.strip():
        return []
    return list(csv.DictReader(StringIO(csv_text)))


def rows_to_csv_text(rows: list[dict[str, Any]]) -> str:
    from io import StringIO

    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=archive_csv_header(), extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key) for key in archive_csv_header()})
    return buffer.getvalue()


def archive_csv_row_count(path: Path) -> int:
    if not path.exists() or path.stat().st_size <= 0:
        return 0

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return sum(1 for _ in reader)


def resolve_writable_output_path(output_path: Path) -> Path:
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path
    except (PermissionError, OSError):
        fallback_path = Path(tempfile.gettempdir()) / "grca" / output_path.name
        fallback_path.parent.mkdir(parents=True, exist_ok=True)
        return fallback_path


def maybe_rotate_archive_csv(output_path: Path, incoming_row_count: int, max_rows: int) -> tuple[Path, Optional[Path]]:
    writable_path = resolve_writable_output_path(output_path)
    if max_rows <= 0 or incoming_row_count <= 0:
        return writable_path, None

    current_row_count = archive_csv_row_count(writable_path)
    if current_row_count == 0 or current_row_count + incoming_row_count <= max_rows:
        return writable_path, None

    archive_dir = writable_path.parent / DEFAULT_ARCHIVE_ROTATION_DIRNAME
    archive_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    archived_path = archive_dir / f"{writable_path.stem}{timestamp}{writable_path.suffix}"
    suffix = 1
    while archived_path.exists():
        archived_path = archive_dir / f"{writable_path.stem}{timestamp}_{suffix}{writable_path.suffix}"
        suffix += 1
    writable_path.replace(archived_path)
    return writable_path, archived_path


def try_append_archive_rows(output_path: Path, results: list[RunResult], max_rows: int) -> tuple[Path, Optional[Path]]:
    writable_path, archived_path = maybe_rotate_archive_csv(output_path, len(results), max_rows)

    def _append(path: Path) -> None:
        file_exists = path.exists() and path.stat().st_size > 0
        with path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=archive_csv_header())
            if not file_exists:
                writer.writeheader()
            for result in results:
                writer.writerow(archive_row_to_dict(result))

    try:
        _append(writable_path)
        return writable_path, archived_path
    except (PermissionError, OSError):
        fallback_path = Path(tempfile.gettempdir()) / "grca" / writable_path.name
        fallback_path.parent.mkdir(parents=True, exist_ok=True)
        _append(fallback_path)
        return fallback_path, archived_path


def github_contents_api_url(repository: str, file_path: str) -> str:
    return f"https://api.github.com/repos/{repository}/contents/{file_path.lstrip('/')}"


def get_github_headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def publish_text_to_github(content_text: str, repository: str, branch: str, file_path: str, token: str, message: str) -> tuple[bool, str]:
    headers = get_github_headers(token)
    url = github_contents_api_url(repository, file_path)

    encoded_content = base64.b64encode(content_text.encode("utf-8")).decode("ascii")

    for attempt in range(1, 4):
        current_sha: Optional[str] = None
        try:
            response = requests.get(url, headers=headers, params={"ref": branch}, timeout=10)
            if response.status_code == 200:
                payload = response.json()
                current_sha = payload.get("sha")

                existing_content = payload.get("content")
                encoding = payload.get("encoding")
                if existing_content and encoding == "base64":
                    decoded = base64.b64decode(existing_content).decode("utf-8")
                    if decoded == content_text:
                        return True, f"GitHub file already up to date: {file_path}"
            elif response.status_code != 404:
                response.raise_for_status()
        except requests.RequestException as exc:
            return False, f"Failed to inspect GitHub file {file_path}: {exc}"

        body: dict[str, Any] = {
            "message": message,
            "content": encoded_content,
            "branch": branch,
        }
        if current_sha:
            body["sha"] = current_sha

        try:
            response = requests.put(url, headers=headers, json=body, timeout=20)
            response.raise_for_status()
            if attempt > 1:
                return True, f"Published {repository}/{file_path} on {branch} after refresh"
            return True, f"Published {repository}/{file_path} on {branch}"
        except requests.RequestException as exc:
            if getattr(exc.response, "status_code", None) == 409 and attempt < 3:
                continue
            if getattr(exc.response, "status_code", None) == 409:
                return False, f"Failed to publish GitHub file {file_path} after 3 conflict retries"
            return False, f"Failed to publish GitHub file {file_path}: {exc}"

    return False, f"Failed to publish GitHub file {file_path} after retries"


def publish_image_to_github(image: np.ndarray, repository: str, branch: str, file_path: str, token: str, message: str) -> tuple[bool, str]:
    suffix = Path(file_path).suffix.lower()
    ext = ".png" if suffix == ".png" else ".jpg"
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        return False, f"Failed to encode annotated image for {file_path}"

    image_bytes = encoded.tobytes()
    content_text = base64.b64encode(image_bytes).decode("ascii")

    headers = get_github_headers(token)
    url = github_contents_api_url(repository, file_path)

    for attempt in range(1, 4):
        current_sha: Optional[str] = None
        try:
            response = requests.get(url, headers=headers, params={"ref": branch}, timeout=10)
            if response.status_code == 200:
                payload = response.json()
                current_sha = payload.get("sha")
                existing_content = payload.get("content")
                encoding = payload.get("encoding")
                if existing_content and encoding == "base64":
                    existing_bytes = base64.b64decode(existing_content)
                    if existing_bytes == image_bytes:
                        return True, f"GitHub file already up to date: {file_path}"
            elif response.status_code != 404:
                response.raise_for_status()
        except requests.RequestException as exc:
            return False, f"Failed to inspect GitHub file {file_path}: {exc}"

        body: dict[str, Any] = {
            "message": message,
            "content": content_text,
            "branch": branch,
        }
        if current_sha:
            body["sha"] = current_sha

        try:
            response = requests.put(url, headers=headers, json=body, timeout=20)
            response.raise_for_status()
            if attempt > 1:
                return True, f"Published {repository}/{file_path} on {branch} after refresh"
            return True, f"Published {repository}/{file_path} on {branch}"
        except requests.RequestException as exc:
            if getattr(exc.response, "status_code", None) == 409 and attempt < 3:
                continue
            if getattr(exc.response, "status_code", None) == 409:
                return False, f"Failed to publish GitHub file {file_path} after 3 conflict retries"
            return False, f"Failed to publish GitHub file {file_path}: {exc}"

    return False, f"Failed to publish GitHub file {file_path} after retries"


def publish_feed_to_github(feed_text: str, config: Config, github_feed_path: str, message: str) -> tuple[bool, str]:
    if not config.github_token:
        return False, "Skipped GitHub publish: no GITHUB_TOKEN or GH_TOKEN set"

    return publish_text_to_github(
        feed_text,
        config.github_repository,
        config.github_branch,
        github_feed_path,
        config.github_token,
        message,
    )


def publish_json_to_github(json_text: str, config: Config, github_json_path: str, message: str) -> tuple[bool, str]:
    if not config.github_token:
        return False, "Skipped GitHub publish: no GITHUB_TOKEN or GH_TOKEN set"

    return publish_text_to_github(
        json_text,
        config.github_repository,
        config.github_branch,
        github_json_path,
        config.github_token,
        message,
    )


def publish_annotated_image_to_github(
    image: np.ndarray,
    config: Config,
    github_image_path: str,
    camera_name: str,
    result: RunResult,
) -> tuple[bool, str]:
    if not config.github_token:
        return False, "Skipped GitHub publish: no GITHUB_TOKEN or GH_TOKEN set"

    return publish_image_to_github(
        image,
        config.github_repository,
        config.github_branch,
        github_image_path,
        config.github_token,
        result_to_image_commit_message(result, camera_name),
    )


def publish_archive_csv_to_github(csv_text: str, config: Config, github_archive_csv_path: str, message: str) -> tuple[bool, str]:
    if not config.github_token:
        return False, "Skipped GitHub archive publish: no GITHUB_TOKEN or GH_TOKEN set"

    headers = get_github_headers(config.github_token)
    url = github_contents_api_url(config.github_repository, github_archive_csv_path)

    current_sha: Optional[str] = None
    existing_text = ""
    try:
        response = requests.get(url, headers=headers, params={"ref": config.github_branch}, timeout=10)
        if response.status_code == 200:
            payload = response.json()
            current_sha = payload.get("sha")
            existing_content = payload.get("content")
            encoding = payload.get("encoding")
            if existing_content and encoding == "base64":
                existing_text = base64.b64decode(existing_content).decode("utf-8")
        elif response.status_code != 404:
            response.raise_for_status()
    except requests.RequestException as exc:
        return False, f"Failed to inspect GitHub archive CSV {github_archive_csv_path}: {exc}"

    existing_rows = csv_text_to_rows(existing_text)
    incoming_rows = csv_text_to_rows(csv_text)

    if existing_rows and len(existing_rows) + len(incoming_rows) > config.archive_max_rows:
        archived_path = Path(github_archive_csv_path)
        archived_name = f"{archived_path.stem}{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}{archived_path.suffix}"
        archived_repo_path = str(archived_path.parent / DEFAULT_ARCHIVE_ROTATION_DIRNAME / archived_name).replace("\\", "/")

        archived_ok, archived_message = publish_text_to_github(
            existing_text,
            config.github_repository,
            config.github_branch,
            archived_repo_path,
            config.github_token,
            "Archive Grand Canyon archive CSV",
        )
        if not archived_ok:
            return False, archived_message

        combined_text = rows_to_csv_text(incoming_rows)
        body: dict[str, Any] = {
            "message": message,
            "content": base64.b64encode(combined_text.encode("utf-8")).decode("ascii"),
            "branch": config.github_branch,
        }
    else:
        combined_rows = existing_rows + incoming_rows
        combined_text = rows_to_csv_text(combined_rows)
        body = {
            "message": message,
            "content": base64.b64encode(combined_text.encode("utf-8")).decode("ascii"),
            "branch": config.github_branch,
        }

    if current_sha:
        body["sha"] = current_sha

    try:
        response = requests.put(url, headers=headers, json=body, timeout=20)
        response.raise_for_status()
    except requests.RequestException as exc:
        if getattr(exc.response, "status_code", None) == 409:
            try:
                refresh_response = requests.get(url, headers=headers, params={"ref": config.github_branch}, timeout=10)
                if refresh_response.status_code == 200:
                    refresh_payload = refresh_response.json()
                    refreshed_sha = refresh_payload.get("sha")
                    if refreshed_sha:
                        body["sha"] = refreshed_sha
                        retry_response = requests.put(url, headers=headers, json=body, timeout=20)
                        retry_response.raise_for_status()
                        return True, f"Published {config.github_repository}/{github_archive_csv_path} on {config.github_branch} after refresh"
            except requests.RequestException as retry_exc:
                return False, f"Failed to publish GitHub archive CSV {github_archive_csv_path} after refresh: {retry_exc}"
        return False, f"Failed to publish GitHub archive CSV {github_archive_csv_path}: {exc}"

    return True, f"Published {config.github_repository}/{github_archive_csv_path} on {config.github_branch}"


def publish_archive_csv_row_to_github(result: RunResult, config: Config, github_archive_csv_path: str, message: str) -> tuple[bool, str]:
    if not config.github_token:
        return False, "Skipped GitHub archive publish: no GITHUB_TOKEN or GH_TOKEN set"

    headers = get_github_headers(config.github_token)
    url = github_contents_api_url(config.github_repository, github_archive_csv_path)

    incoming_row = archive_row_to_dict(result)
    for attempt in range(1, 4):
        current_sha: Optional[str] = None
        existing_text = ""
        try:
            response = requests.get(url, headers=headers, params={"ref": config.github_branch}, timeout=10)
            if response.status_code == 200:
                payload = response.json()
                current_sha = payload.get("sha")
                existing_content = payload.get("content")
                encoding = payload.get("encoding")
                if existing_content and encoding == "base64":
                    existing_text = base64.b64decode(existing_content).decode("utf-8")
            elif response.status_code != 404:
                response.raise_for_status()
        except requests.RequestException as exc:
            return False, f"Failed to inspect GitHub archive CSV {github_archive_csv_path}: {exc}"

        existing_rows = csv_text_to_rows(existing_text)
        combined_rows = existing_rows + [incoming_row]
        if config.archive_max_rows > 0 and len(combined_rows) > config.archive_max_rows:
            combined_rows = combined_rows[-config.archive_max_rows :]

        body: dict[str, Any] = {
            "message": message,
            "content": base64.b64encode(rows_to_csv_text(combined_rows).encode("utf-8")).decode("ascii"),
            "branch": config.github_branch,
        }
        if current_sha:
            body["sha"] = current_sha

        try:
            response = requests.put(url, headers=headers, json=body, timeout=20)
            response.raise_for_status()
            return True, f"Published {config.github_repository}/{github_archive_csv_path} on {config.github_branch}"
        except requests.RequestException as exc:
            if getattr(exc.response, "status_code", None) == 409 and attempt < 3:
                continue
            if getattr(exc.response, "status_code", None) == 409:
                return False, f"Failed to publish GitHub archive CSV {github_archive_csv_path} after 3 conflict retries"
            return False, f"Failed to publish GitHub archive CSV {github_archive_csv_path}: {exc}"

    return False, f"Failed to publish GitHub archive CSV {github_archive_csv_path} after retries"


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Count vehicles in the Grand Canyon webcam image")
    parser.add_argument("--output", type=Path, default=None, help="Override output JSON path")
    parser.add_argument("--feed-output", type=Path, default=None, help="Override local feed text output path")
    parser.add_argument("--model", default=None, help="Override YOLO model path")
    parser.add_argument("--confidence", type=float, default=None, help="Override YOLO confidence threshold")
    parser.add_argument("--iou", type=float, default=None, help="Override YOLO IOU threshold")
    parser.add_argument("--imgsz", type=int, default=None, help="Override YOLO image size")
    args, _unknown = parser.parse_known_args(argv)
    return args


def build_config(args: argparse.Namespace) -> Config:
    config = load_config()
    return Config(
        webcam_page_url=config.webcam_page_url,
        fallback_image_url=config.fallback_image_url,
        model_path=args.model or config.model_path,
        lanes_path=config.lanes_path,
        output_path=args.output or config.output_path,
        feed_output_path=args.feed_output or config.feed_output_path,
        annotated_image_output_path=config.annotated_image_output_path,
        archive_output_path=config.archive_output_path,
        tracker_state_output_path=config.tracker_state_output_path,
        github_repository=config.github_repository,
        github_branch=config.github_branch,
        github_json_path=config.github_json_path,
        github_feed_path=config.github_feed_path,
        github_image_path=config.github_image_path,
        github_archive_csv_path=config.github_archive_csv_path,
        github_tracker_state_path=config.github_tracker_state_path,
        github_token=config.github_token,
        publish_to_github=config.publish_to_github,
        confidence=args.confidence if args.confidence is not None else config.confidence,
        iou=args.iou if args.iou is not None else config.iou,
        image_size=args.imgsz if args.imgsz is not None else config.image_size,
        user_agent=config.user_agent,
        archive_max_rows=config.archive_max_rows,
    )


def main() -> int:
    args = parse_args()
    config = build_config(args)
    day_key = arizona_day_key()
    tracker = load_tracker_state(config)
    tracker.ensure_day(day_key)
    daily_count_floor = load_archive_daily_count_floor(config, day_key)
    grca_result, grca_annotated_image = run_detection_with_image(
        config,
        tracker=tracker,
        daily_count_floor=daily_count_floor,
    )

    output_path = write_result(grca_result, config.output_path, "grca_lane_vehicle_count_v2")
    json_text = result_to_feed_json_text(grca_result, "grca_lane_vehicle_count_v2")
    feed_output_path = write_feed(grca_result, config.feed_output_path, "Grand Canyon South Entrance webcam vehicle feed", "grca_lane_vehicle_feed_v2")
    feed_text = result_to_feed_text(grca_result, "Grand Canyon South Entrance webcam vehicle feed", "grca_lane_vehicle_feed_v2")
    archive_output_path, archived_path = append_archive_csv(grca_result, config.archive_output_path, config.archive_max_rows)
    tracker_state_output_path = write_tracker_state(tracker, config.tracker_state_output_path)
    annotated_image_output_path = None
    if grca_annotated_image is not None:
        annotated_image_output_path = write_annotated_image(grca_annotated_image, config.annotated_image_output_path)

    publish_messages: list[str] = []
    publish_ok = True
    if config.publish_to_github:
        json_published, json_message = publish_json_to_github(json_text, config, config.github_json_path, "Update Grand Canyon vehicle count JSON")
        feed_published, feed_message = publish_feed_to_github(feed_text, config, config.github_feed_path, "Update Grand Canyon webcam feed")
        archive_published, archive_message = publish_archive_csv_row_to_github(grca_result, config, config.github_archive_csv_path, "Update Grand Canyon archive CSV")
        tracker_state_published, tracker_state_message = publish_tracker_state_to_github(tracker, config)
        image_published = True
        image_message = None
        if annotated_image_output_path is not None:
            image_published, image_message = publish_annotated_image_to_github(
                grca_annotated_image,
                config,
                config.github_image_path,
                "Grand Canyon South Entrance",
                grca_result,
            )
        if not archive_published:
            print(archive_message)
        if not tracker_state_published:
            print(tracker_state_message)
        if not image_published:
            print(image_message)
        publish_ok = json_published and feed_published and archive_published and tracker_state_published and image_published
        publish_messages.extend([json_message, feed_message])
        if archive_message:
            publish_messages.append(archive_message)
        if tracker_state_message:
            publish_messages.append(tracker_state_message)
        if image_message:
            publish_messages.append(image_message)

    print(json_text.rstrip())
    print(f"Wrote result to {output_path}")
    print(f"Wrote feed to {feed_output_path}")
    print(f"Updated archive to {archive_output_path}")
    print(f"Updated tracker state to {tracker_state_output_path}")
    print(f"Archive rows: {archive_csv_row_count(archive_output_path)}")
    if archived_path:
        print(f"Archived previous CSV to {archived_path}")
    if annotated_image_output_path:
        print(f"Wrote annotated image to {annotated_image_output_path}")
    for publish_message in publish_messages:
        print(publish_message)

    return 0 if grca_result.status == "ok" and publish_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
