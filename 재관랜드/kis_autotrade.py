from __future__ import annotations

import argparse
import io
import json
import math
import random
import shutil
import sys
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests


DEFAULT_CREDENTIALS_PATH = Path(__file__).resolve().parent / ".env"
CACHE_DIR = Path(".cache")
ORDER_LOCK_PATH = CACHE_DIR / "order_lock.json"
POSITION_STATE_PATH = CACHE_DIR / "position_state.json"
MASTER_CACHE_PATH = CACHE_DIR / "kis_master_codes.json"
SYMBOL_FAILURE_PATH = CACHE_DIR / "symbol_failures.json"
TRAINING_DATA_PATH = CACHE_DIR / "trade_training_data.jsonl"
MODEL_STATE_PATH = CACHE_DIR / "trade_model_state.json"
TRAINING_META_PATH = CACHE_DIR / "trade_training_meta.json"
PENDING_LABELS_PATH = CACHE_DIR / "pending_training_labels.json"
TRAINING_BACKUP_PREFIX = "trade_training_data.backup_"
TRAINING_BACKUP_MAX_FILES = 5
TRAINING_BACKUP_MIN_APPEND_ROWS = 200
TRAINING_BACKUP_MIN_SECONDS = 3600
KST = timezone(timedelta(hours=9))
AUTO_RETRAIN_MIN_NEW_LABELED_SAMPLES = 25
AUTO_RETRAIN_MIN_SECONDS = 300
TRAINING_LABEL_HORIZONS_MINUTES = (10, 30, 60)

MASTER_SOURCES = {
    "KOSPI": "https://new.real.download.dws.co.kr/common/master/kospi_code.mst.zip",
    "KOSDAQ": "https://new.real.download.dws.co.kr/common/master/kosdaq_code.mst.zip",
}
JUNK_NAME_TOKENS = (
    "스팩",
    "ETF",
    "ETN",
    "리츠",
    "REIT",
    "인버스",
    "레버리지",
    "ANKOR",
    "유전",
    "원유",
    "선물",
)


class KISError(RuntimeError):
    pass


@dataclass
class Credentials:
    app_key: str
    app_secret: str


@dataclass
class Account:
    cano: str
    acnt_prdt_cd: str


@dataclass
class Candidate:
    symbol: str
    name: str
    price: int
    change_rate: float
    volume: int
    score: float


@dataclass
class ModelWeights:
    intercept: float = 0.0
    price_weight: float = 0.0
    change_weight: float = 0.0
    volume_weight: float = 0.0
    bias_to_mid_momentum: float = 0.0
    sample_count: int = 0


class KISClient:
    def __init__(self, credentials: Credentials, account: Account, mode: str = "real") -> None:
        if mode not in {"real", "demo"}:
            raise ValueError("mode must be 'real' or 'demo'")
        self.credentials = credentials
        self.account = account
        self.mode = mode
        self.base_url = (
            "https://openapi.koreainvestment.com:9443"
            if mode == "real"
            else "https://openapivts.koreainvestment.com:29443"
        )
        CACHE_DIR.mkdir(exist_ok=True)
        self.session = requests.Session()
        self.session.trust_env = False
        self.access_token = self._load_cached_token()

    def _cache_path(self) -> Path:
        return CACHE_DIR / f"kis_token_{self.mode}.json"

    def _load_cached_token(self) -> str | None:
        cache_path = self._cache_path()
        if not cache_path.exists():
            return None
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            expires_at = datetime.fromisoformat(payload["expires_at"])
            if expires_at > datetime.now() + timedelta(minutes=1):
                return payload["access_token"]
        except (KeyError, ValueError, json.JSONDecodeError):
            return None
        return None

    def _save_token(self, access_token: str, expires_at: str | None) -> None:
        expiry = datetime.now() + timedelta(hours=23)
        if expires_at:
            try:
                expiry = datetime.fromisoformat(expires_at.replace(" ", "T"))
            except ValueError:
                pass
        payload = {"access_token": access_token, "expires_at": expiry.isoformat()}
        self._cache_path().write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def ensure_token(self) -> str:
        if self.access_token:
            return self.access_token

        response = self.session.post(
            f"{self.base_url}/oauth2/tokenP",
            headers={"content-type": "application/json"},
            json={
                "grant_type": "client_credentials",
                "appkey": self.credentials.app_key,
                "appsecret": self.credentials.app_secret,
            },
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()
        token = payload.get("access_token")
        if not token:
            raise KISError(f"토큰 발급 실패: {payload}")
        self.access_token = token
        self._save_token(token, payload.get("access_token_token_expired"))
        return token

    def _headers(self, tr_id: str, *, include_hashkey: dict[str, Any] | None = None) -> dict[str, str]:
        headers = {
            "content-type": "application/json",
            "authorization": f"Bearer {self.ensure_token()}",
            "appkey": self.credentials.app_key,
            "appsecret": self.credentials.app_secret,
            "tr_id": tr_id,
            "custtype": "P",
        }
        if include_hashkey is not None:
            headers["hashkey"] = self._issue_hashkey(include_hashkey)
        return headers

    def _issue_hashkey(self, payload: dict[str, Any]) -> str:
        response = self.session.post(
            f"{self.base_url}/uapi/hashkey",
            headers={
                "content-type": "application/json",
                "appkey": self.credentials.app_key,
                "appsecret": self.credentials.app_secret,
            },
            json=payload,
            timeout=15,
        )
        response.raise_for_status()
        body = response.json()
        hashkey = body.get("HASH")
        if not hashkey:
            raise KISError(f"hashkey 발급 실패: {body}")
        return hashkey

    def get_current_price(self, symbol: str, market: str = "J") -> dict[str, Any]:
        response = self.session.get(
            f"{self.base_url}/uapi/domestic-stock/v1/quotations/inquire-price",
            headers=self._headers("FHKST01010100"),
            params={"FID_COND_MRKT_DIV_CODE": market, "FID_INPUT_ISCD": symbol},
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("rt_cd") != "0":
            raise KISError(f"현재가 조회 실패: {payload.get('msg1', payload)}")
        return payload["output"]

    def get_buyable_cash(
        self,
        *,
        symbol: str,
        price: int,
        order_type: str = "00",
        include_cma: str = "N",
        include_overseas: str = "N",
    ) -> dict[str, Any]:
        tr_id = "TTTC8908R" if self.mode == "real" else "VTTC8908R"
        response = self.session.get(
            f"{self.base_url}/uapi/domestic-stock/v1/trading/inquire-psbl-order",
            headers=self._headers(tr_id),
            params={
                "CANO": self.account.cano,
                "ACNT_PRDT_CD": self.account.acnt_prdt_cd,
                "PDNO": symbol,
                "ORD_UNPR": str(price),
                "ORD_DVSN": order_type,
                "CMA_EVLU_AMT_ICLD_YN": include_cma,
                "OVRS_ICLD_YN": include_overseas,
            },
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("rt_cd") != "0":
            raise KISError(f"매수가능조회 실패: {payload.get('msg1', payload)}")
        return payload["output"]

    def get_stock_balance(self) -> dict[str, Any]:
        tr_id = "TTTC8434R" if self.mode == "real" else "VTTC8434R"
        response = self.session.get(
            f"{self.base_url}/uapi/domestic-stock/v1/trading/inquire-balance",
            headers=self._headers(tr_id),
            params={
                "CANO": self.account.cano,
                "ACNT_PRDT_CD": self.account.acnt_prdt_cd,
                "AFHR_FLPR_YN": "N",
                "OFL_YN": "",
                "INQR_DVSN": "02",
                "UNPR_DVSN": "01",
                "FUND_STTL_ICLD_YN": "N",
                "FNCG_AMT_AUTO_RDPT_YN": "N",
                "PRCS_DVSN": "00",
                "CTX_AREA_FK100": "",
                "CTX_AREA_NK100": "",
            },
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("rt_cd") != "0":
            raise KISError(f"잔고조회 실패: {payload.get('msg1', payload)}")
        return payload

    def place_cash_order(
        self,
        *,
        side: str,
        symbol: str,
        quantity: int,
        price: int,
        order_type: str = "00",
        exchange: str = "KRX",
    ) -> dict[str, Any]:
        if side not in {"buy", "sell"}:
            raise ValueError("side must be 'buy' or 'sell'")
        tr_id_map = {
            ("real", "buy"): "TTTC0012U",
            ("real", "sell"): "TTTC0011U",
            ("demo", "buy"): "VTTC0012U",
            ("demo", "sell"): "VTTC0011U",
        }
        payload = {
            "CANO": self.account.cano,
            "ACNT_PRDT_CD": self.account.acnt_prdt_cd,
            "PDNO": symbol,
            "ORD_DVSN": order_type,
            "ORD_QTY": str(quantity),
            "ORD_UNPR": str(price),
            "EXCG_ID_DVSN_CD": exchange,
            "SLL_TYPE": "",
            "CNDT_PRIC": "",
        }
        response = self.session.post(
            f"{self.base_url}/uapi/domestic-stock/v1/trading/order-cash",
            headers=self._headers(tr_id_map[(self.mode, side)], include_hashkey=payload),
            json=payload,
            timeout=15,
        )
        if not response.ok:
            raise KISError(f"주문 HTTP 오류 {response.status_code}: {response.text}")
        body = response.json()
        if body.get("rt_cd") != "0":
            raise KISError(f"주문 실패: {body.get('msg1', body)}")
        return body["output"]


def load_env_file(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if not path.exists():
        return mapping
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        mapping[key.strip().lower()] = value.strip().strip('"').strip("'")
    return mapping


def resolve_settings_path(path: Path = DEFAULT_CREDENTIALS_PATH) -> Path:
    if path.exists():
        return path
    base_dir = Path(__file__).resolve().parent
    for candidate in (base_dir / "??.env", base_dir / "??.txt"):
        if candidate.exists():
            return candidate
    return path


def load_credentials(path: Path = DEFAULT_CREDENTIALS_PATH) -> Credentials:
    settings_path = resolve_settings_path(path)
    if not settings_path.exists():
        raise FileNotFoundError(f"API ?? ??? ?? ? ????: {settings_path}")
    mapping = load_env_file(settings_path)
    app_key = (
        mapping.get("kis_app_key")
        or mapping.get("key")
        or mapping.get("app_key")
        or mapping.get("apikey")
    )
    app_secret = (
        mapping.get("kis_app_secret")
        or mapping.get("skey")
        or mapping.get("secret")
        or mapping.get("app_secret")
    )
    if not app_key or not app_secret:
        raise ValueError("?? ???? KIS_APP_KEY/KIS_APP_SECRET ?? ?? ?????.")
    return Credentials(app_key=app_key, app_secret=app_secret)


def load_app_settings(path: Path = DEFAULT_CREDENTIALS_PATH) -> dict[str, str]:
    return load_env_file(resolve_settings_path(path))

def parse_account(raw_account: str) -> Account:
    cleaned = raw_account.strip()
    if "-" not in cleaned:
        raise ValueError("계좌번호는 00000000-00 형식이어야 합니다.")
    cano, acnt_prdt_cd = cleaned.split("-", 1)
    if not (cano.isdigit() and acnt_prdt_cd.isdigit()):
        raise ValueError("계좌번호는 숫자와 하이픈만 사용해 주세요.")
    return Account(cano=cano, acnt_prdt_cd=acnt_prdt_cd)


def read_json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def write_json_file(path: Path, payload: dict[str, Any]) -> None:
    CACHE_DIR.mkdir(exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_order_locks() -> dict[str, Any]:
    return read_json_file(ORDER_LOCK_PATH)


def write_order_locks(payload: dict[str, Any]) -> None:
    write_json_file(ORDER_LOCK_PATH, payload)


def make_lock_key(*, mode: str, account: str, symbol: str, side: str) -> str:
    return f"{mode}:{account}:{symbol}:{side}"


def is_locked(lock_key: str, cooldown_seconds: int) -> bool:
    locks = read_order_locks()
    last_order_at = locks.get(lock_key)
    if not last_order_at:
        return False
    try:
        ordered_at = datetime.fromisoformat(last_order_at)
    except ValueError:
        return False
    return ordered_at > datetime.now() - timedelta(seconds=cooldown_seconds)


def lock_order(lock_key: str) -> None:
    locks = read_order_locks()
    locks[lock_key] = datetime.now().isoformat()
    write_order_locks(locks)


def read_positions() -> dict[str, Any]:
    return read_json_file(POSITION_STATE_PATH)


def write_positions(payload: dict[str, Any]) -> None:
    write_json_file(POSITION_STATE_PATH, payload)


def position_key(*, mode: str, account: str, symbol: str) -> str:
    return f"{mode}:{account}:{symbol}"


def get_position(*, mode: str, account: str, symbol: str) -> dict[str, Any] | None:
    return read_positions().get(position_key(mode=mode, account=account, symbol=symbol))


def save_position(*, mode: str, account: str, symbol: str, qty: int, entry_price: int) -> None:
    positions = read_positions()
    positions[position_key(mode=mode, account=account, symbol=symbol)] = {
        "qty": qty,
        "entry_price": entry_price,
        "saved_at": datetime.now(KST).isoformat(),
    }
    write_positions(positions)


def clear_position(*, mode: str, account: str, symbol: str) -> None:
    positions = read_positions()
    positions.pop(position_key(mode=mode, account=account, symbol=symbol), None)
    write_positions(positions)


def list_positions(*, mode: str, account: str) -> dict[str, dict[str, Any]]:
    prefix = f"{mode}:{account}:"
    positions = read_positions()
    return {key.split(":")[-1]: value for key, value in positions.items() if key.startswith(prefix)}


def invested_amount(*, mode: str, account: str) -> int:
    total = 0
    for position in list_positions(mode=mode, account=account).values():
        total += int(position.get("entry_price", 0)) * int(position.get("qty", 0))
    return total


def read_symbol_failures() -> dict[str, Any]:
    return read_json_file(SYMBOL_FAILURE_PATH)


def write_symbol_failures(payload: dict[str, Any]) -> None:
    write_json_file(SYMBOL_FAILURE_PATH, payload)


def sanitize_training_payload(payload: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, str):
            normalized = value.replace("\x00", " ").replace("\r", " ").replace("\n", " ").strip()
            if key == "name":
                normalized = normalized.replace("\"", "").strip()
            cleaned[key] = normalized
        else:
            cleaned[key] = value
    return cleaned


def list_training_backups() -> list[Path]:
    return sorted(CACHE_DIR.glob(f"{TRAINING_BACKUP_PREFIX}*.jsonl"))


def prune_training_backups(max_files: int = TRAINING_BACKUP_MAX_FILES) -> None:
    backups = list_training_backups()
    while len(backups) > max_files:
        oldest = backups.pop(0)
        try:
            oldest.unlink()
        except OSError:
            break


def maybe_backup_training_file(*, force: bool = False) -> Path | None:
    if not TRAINING_DATA_PATH.exists():
        return None
    meta = read_training_meta()
    append_count = int(meta.get("append_count_since_backup", 0) or 0)
    last_backup_at_raw = meta.get("last_backup_at")
    if not force:
        enough_rows = append_count >= TRAINING_BACKUP_MIN_APPEND_ROWS
        enough_time = False
        if last_backup_at_raw:
            try:
                last_backup_at = datetime.fromisoformat(str(last_backup_at_raw))
                enough_time = (datetime.now(KST) - last_backup_at).total_seconds() >= TRAINING_BACKUP_MIN_SECONDS
            except ValueError:
                enough_time = True
        if not last_backup_at_raw:
            enough_time = True
        if not enough_rows and not enough_time:
            return None
    backup_path = CACHE_DIR / f"{TRAINING_BACKUP_PREFIX}{datetime.now(KST).strftime('%Y%m%d_%H%M%S')}.jsonl"
    shutil.copy2(TRAINING_DATA_PATH, backup_path)
    meta["last_backup_at"] = datetime.now(KST).isoformat()
    meta["last_backup_path"] = str(backup_path)
    meta["append_count_since_backup"] = 0
    write_training_meta(meta)
    prune_training_backups()
    return backup_path


def append_training_row(payload: dict[str, Any]) -> None:
    CACHE_DIR.mkdir(exist_ok=True)
    payload = sanitize_training_payload(payload)
    with TRAINING_DATA_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    meta = read_training_meta()
    meta["append_count_since_backup"] = int(meta.get("append_count_since_backup", 0) or 0) + 1
    write_training_meta(meta)
    maybe_backup_training_file()


def normalize_training_row(row: dict[str, Any]) -> dict[str, Any] | None:
    required_keys = ("recorded_at", "side", "symbol", "price", "change_rate", "volume", "qty", "reason")
    for key in required_keys:
        if row.get(key) in (None, ""):
            return None

    cleaned = sanitize_training_payload(dict(row))
    cleaned["symbol"] = str(cleaned["symbol"]).strip()
    cleaned["side"] = str(cleaned["side"]).strip().lower()
    cleaned["reason"] = str(cleaned["reason"]).strip()
    cleaned["price"] = int(float(cleaned["price"]))
    cleaned["change_rate"] = float(cleaned["change_rate"])
    cleaned["volume"] = int(float(cleaned["volume"]))
    cleaned["qty"] = int(float(cleaned["qty"]))
    if "label_success" in cleaned and cleaned["label_success"] not in (None, ""):
        cleaned["label_success"] = float(cleaned["label_success"])
    return cleaned


def read_training_rows(path: Path | None = None) -> list[dict[str, Any]]:
    source_path = path or TRAINING_DATA_PATH
    if not source_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in source_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = normalize_training_row(json.loads(line))
            if row is not None:
                rows.append(row)
        except json.JSONDecodeError:
            continue
    return rows


def load_training_rows_for_model() -> list[dict[str, Any]]:
    rows = read_training_rows(TRAINING_DATA_PATH)
    for row in rows:
        row["data_source"] = row.get("data_source") or "real"
    return rows


def load_model_weights() -> ModelWeights:
    payload = read_json_file(MODEL_STATE_PATH)
    try:
        return ModelWeights(
            intercept=float(payload.get("intercept", 0.0)),
            price_weight=float(payload.get("price_weight", 0.0)),
            change_weight=float(payload.get("change_weight", 0.0)),
            volume_weight=float(payload.get("volume_weight", 0.0)),
            bias_to_mid_momentum=float(payload.get("bias_to_mid_momentum", 0.0)),
            sample_count=int(payload.get("sample_count", 0)),
        )
    except (TypeError, ValueError):
        return ModelWeights()


def save_model_weights(weights: ModelWeights) -> None:
    write_json_file(
        MODEL_STATE_PATH,
        {
            "intercept": weights.intercept,
            "price_weight": weights.price_weight,
            "change_weight": weights.change_weight,
            "volume_weight": weights.volume_weight,
            "bias_to_mid_momentum": weights.bias_to_mid_momentum,
            "sample_count": weights.sample_count,
            "updated_at": datetime.now(KST).isoformat(),
        },
    )


def read_training_meta() -> dict[str, Any]:
    return read_json_file(TRAINING_META_PATH)


def write_training_meta(payload: dict[str, Any]) -> None:
    write_json_file(TRAINING_META_PATH, payload)


def read_pending_labels() -> list[dict[str, Any]]:
    payload = read_json_file(PENDING_LABELS_PATH)
    items = payload.get("items", [])
    return items if isinstance(items, list) else []


def write_pending_labels(items: list[dict[str, Any]]) -> None:
    write_json_file(PENDING_LABELS_PATH, {"items": items})


def training_time_bucket(now: datetime | None = None) -> str:
    current = now or datetime.now(KST)
    return current.strftime("%H:%M")


def training_market_session(now: datetime | None = None) -> str:
    current = now or datetime.now(KST)
    if current.hour < 9:
        return "pre_open"
    if current.hour < 12:
        return "morning"
    if current.hour < 15 or (current.hour == 15 and current.minute <= 20):
        return "afternoon"
    return "after_close"


def training_weekday(now: datetime | None = None) -> int:
    current = now or datetime.now(KST)
    return current.weekday()


def training_minute_of_day(now: datetime | None = None) -> int:
    current = now or datetime.now(KST)
    return current.hour * 60 + current.minute


def parse_int_field(value: Any, default: int = 0) -> int:
    try:
        if value in (None, ""):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def parse_float_field(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def build_quote_feature_payload(quote: dict[str, Any], *, reference_price: int, qty: int) -> dict[str, Any]:
    current_price = parse_int_field(quote.get("stck_prpr"))
    open_price = parse_int_field(quote.get("stck_oprc"))
    high_price = parse_int_field(quote.get("stck_hgpr"))
    low_price = parse_int_field(quote.get("stck_lwpr"))
    prev_diff = parse_int_field(quote.get("prdy_vrss"))
    change_rate = parse_float_field(quote.get("prdy_ctrt"))
    volume = parse_int_field(quote.get("acml_vol"))
    traded_value = parse_int_field(quote.get("acml_tr_pbmn"))
    market_cap = parse_int_field(quote.get("hts_avls"))
    per = parse_float_field(quote.get("per"), default=-1.0)
    pbr = parse_float_field(quote.get("pbr"), default=-1.0)
    eps = parse_float_field(quote.get("eps"), default=-1.0)
    bps = parse_float_field(quote.get("bps"), default=-1.0)

    day_range = max(0, high_price - low_price)
    intraday_range_pct = ((high_price - low_price) / low_price * 100.0) if low_price > 0 else 0.0
    open_gap_pct = ((current_price - open_price) / open_price * 100.0) if open_price > 0 else 0.0
    price_vs_entry_pct = ((current_price - reference_price) / reference_price * 100.0) if reference_price > 0 else 0.0
    drawdown_from_high_pct = ((current_price - high_price) / high_price * 100.0) if high_price > 0 else 0.0
    rebound_from_low_pct = ((current_price - low_price) / low_price * 100.0) if low_price > 0 else 0.0
    price_position_in_range = ((current_price - low_price) / day_range) if day_range > 0 else 0.5

    return {
        "day_open": open_price,
        "day_high": high_price,
        "day_low": low_price,
        "previous_diff": prev_diff,
        "traded_value": traded_value,
        "market_cap": market_cap,
        "per": per,
        "pbr": pbr,
        "eps": eps,
        "bps": bps,
        "day_range": day_range,
        "intraday_range_pct": intraday_range_pct,
        "open_gap_pct": open_gap_pct,
        "price_vs_entry_pct": price_vs_entry_pct,
        "drawdown_from_high_pct": drawdown_from_high_pct,
        "rebound_from_low_pct": rebound_from_low_pct,
        "price_position_in_range": price_position_in_range,
        "turnover_value_per_share": (traded_value / volume) if volume > 0 else 0.0,
        "position_notional": current_price * max(1, qty),
    }


def sigmoid(value: float) -> float:
    clipped = max(-30.0, min(30.0, value))
    return 1.0 / (1.0 + math.exp(-clipped))


def extract_features(price: int, change_rate: float, volume: int) -> dict[str, float]:
    return {
        "price_norm": min(price, 50000) / 50000.0,
        "change_norm": change_rate / 30.0,
        "volume_norm": min(volume, 5_000_000) / 5_000_000.0,
        "mid_momentum": max(0.0, 1.0 - min(abs(change_rate - 4.0), 10.0) / 10.0),
    }


def score_with_model(weights: ModelWeights, price: int, change_rate: float, volume: int) -> float:
    features = extract_features(price, change_rate, volume)
    raw_score = (
        weights.intercept
        + (weights.price_weight * features["price_norm"])
        + (weights.change_weight * features["change_norm"])
        + (weights.volume_weight * features["volume_norm"])
        + (weights.bias_to_mid_momentum * features["mid_momentum"])
    )
    return sigmoid(raw_score)


def train_model_from_rows(rows: list[dict[str, Any]], *, epochs: int = 120, learning_rate: float = 0.08) -> ModelWeights:
    samples = [row for row in rows if row.get("side") == "buy" and isinstance(row.get("label_success"), (int, float))]
    if len(samples) < 8:
        return load_model_weights()

    weights = ModelWeights(sample_count=len(samples))
    for _ in range(epochs):
        grad_intercept = 0.0
        grad_price = 0.0
        grad_change = 0.0
        grad_volume = 0.0
        grad_mid = 0.0
        for row in samples:
            features = extract_features(
                int(row.get("price", 0) or 0),
                float(row.get("change_rate", 0.0) or 0.0),
                int(row.get("volume", 0) or 0),
            )
            prediction = sigmoid(
                weights.intercept
                + weights.price_weight * features["price_norm"]
                + weights.change_weight * features["change_norm"]
                + weights.volume_weight * features["volume_norm"]
                + weights.bias_to_mid_momentum * features["mid_momentum"]
            )
            error = prediction - float(row["label_success"])
            grad_intercept += error
            grad_price += error * features["price_norm"]
            grad_change += error * features["change_norm"]
            grad_volume += error * features["volume_norm"]
            grad_mid += error * features["mid_momentum"]

        scale = 1.0 / len(samples)
        weights.intercept -= learning_rate * grad_intercept * scale
        weights.price_weight -= learning_rate * grad_price * scale
        weights.change_weight -= learning_rate * grad_change * scale
        weights.volume_weight -= learning_rate * grad_volume * scale
        weights.bias_to_mid_momentum -= learning_rate * grad_mid * scale
    return weights


def retrain_model() -> ModelWeights:
    weights = train_model_from_rows(load_training_rows_for_model())
    save_model_weights(weights)
    meta = read_training_meta()
    meta.update(
        {
            "last_retrained_at": datetime.now(KST).isoformat(),
            "last_sample_count": weights.sample_count,
        }
    )
    write_training_meta(meta)
    return weights


def auto_retrain_if_needed(*, min_new_samples: int = AUTO_RETRAIN_MIN_NEW_LABELED_SAMPLES) -> ModelWeights | None:
    rows = load_training_rows_for_model()
    labeled_rows = [row for row in rows if isinstance(row.get("label_success"), (int, float))]
    current_labeled_count = len(labeled_rows)
    model_weights = load_model_weights()
    last_trained_count = int(model_weights.sample_count or 0)
    meta = read_training_meta()
    last_retrained_at_raw = meta.get("last_retrained_at")
    if last_retrained_at_raw:
        try:
            last_retrained_at = datetime.fromisoformat(last_retrained_at_raw)
            seconds_since_retrain = (datetime.now(KST) - last_retrained_at).total_seconds()
            if seconds_since_retrain < AUTO_RETRAIN_MIN_SECONDS:
                print(
                    f"자동 재학습 보류: 최근 재학습 후 {int(seconds_since_retrain)}초 경과 "
                    f"(최소 {AUTO_RETRAIN_MIN_SECONDS}초 필요)",
                    flush=True,
                )
                return None
        except ValueError:
            pass
    if current_labeled_count < max(8, last_trained_count + min_new_samples):
        print(
            f"자동 재학습 대기: labeled={current_labeled_count}, "
            f"last_trained={last_trained_count}, required_delta={min_new_samples}",
            flush=True,
        )
        return None
    print(
        f"자동 재학습 시작: labeled={current_labeled_count}, "
        f"last_trained={last_trained_count}",
        flush=True,
    )
    return retrain_model()


def record_trade_event(
    *,
    side: str,
    symbol: str,
    name: str,
    price: int,
    change_rate: float,
    volume: int,
    qty: int,
    reason: str,
    pnl_percent: float | None = None,
    label_success: float | None = None,
    model_score: float | None = None,
    candidate_score: float | None = None,
    entry_price: int | None = None,
    holding_minutes: float | None = None,
    tracked_invested: int | None = None,
    remaining_budget: int | None = None,
    tracked_position_count: int | None = None,
    stop_loss_percent: float | None = None,
    take_profit_percent: float | None = None,
    extra_features: dict[str, Any] | None = None,
) -> None:
    now = datetime.now(KST)
    payload: dict[str, Any] = {
        "recorded_at": now.isoformat(),
        "side": side,
        "symbol": symbol,
        "name": name,
        "price": price,
        "change_rate": change_rate,
        "volume": volume,
        "qty": qty,
        "reason": reason,
        "time_bucket": training_time_bucket(now),
        "market_session": training_market_session(now),
        "weekday": training_weekday(now),
        "minute_of_day": training_minute_of_day(now),
        "data_source": "real",
    }
    if pnl_percent is not None:
        payload["pnl_percent"] = pnl_percent
    if label_success is not None:
        payload["label_success"] = label_success
    if model_score is not None:
        payload["model_score"] = model_score
    if candidate_score is not None:
        payload["candidate_score"] = candidate_score
    if entry_price is not None:
        payload["entry_price"] = entry_price
    if holding_minutes is not None:
        payload["holding_minutes"] = holding_minutes
    if tracked_invested is not None:
        payload["tracked_invested"] = tracked_invested
    if remaining_budget is not None:
        payload["remaining_budget"] = remaining_budget
        total_budget = tracked_invested + remaining_budget if tracked_invested is not None else None
        if total_budget and total_budget > 0:
            payload["budget_utilization"] = tracked_invested / total_budget
    if tracked_position_count is not None:
        payload["tracked_position_count"] = tracked_position_count
    if stop_loss_percent is not None:
        payload["stop_loss_percent"] = stop_loss_percent
    if take_profit_percent is not None:
        payload["take_profit_percent"] = take_profit_percent
    if extra_features:
        payload.update(sanitize_training_payload(extra_features))
    append_training_row(payload)


def queue_training_labels(
    *,
    mode: str,
    account: str,
    symbol: str,
    name: str,
    entry_price: int,
    qty: int,
    change_rate: float,
    volume: int,
    reason: str,
    candidate_score: float | None,
    model_score: float | None,
    extra_features: dict[str, Any] | None,
) -> None:
    now = datetime.now(KST)
    items = read_pending_labels()
    for horizon in TRAINING_LABEL_HORIZONS_MINUTES:
        due_at = now + timedelta(minutes=horizon)
        items.append(
            {
                "mode": mode,
                "account": account,
                "symbol": symbol,
                "name": name,
                "entry_price": entry_price,
                "qty": qty,
                "change_rate": change_rate,
                "volume": volume,
                "reason": f"{reason}_{horizon}m_label",
                "candidate_score": candidate_score,
                "model_score": model_score,
                "extra_features": extra_features or {},
                "created_at": now.isoformat(),
                "due_at": due_at.isoformat(),
                "horizon_minutes": horizon,
            }
        )
    write_pending_labels(items)


def process_pending_training_labels(args: argparse.Namespace, client: KISClient) -> None:
    items = read_pending_labels()
    if not items:
        return

    now = datetime.now(KST)
    remaining: list[dict[str, Any]] = []
    processed = 0
    for item in items:
        try:
            due_at = datetime.fromisoformat(str(item.get("due_at", "")))
        except ValueError:
            continue
        if due_at > now:
            remaining.append(item)
            continue

        symbol = str(item.get("symbol", ""))
        entry_price = parse_int_field(item.get("entry_price"))
        qty = max(1, parse_int_field(item.get("qty"), 1))
        if not symbol or entry_price <= 0:
            continue

        try:
            quote = client.get_current_price(symbol)
        except (requests.RequestException, KISError) as exc:
            item["retry_count"] = int(item.get("retry_count", 0) or 0) + 1
            item["last_error"] = str(exc)
            if item["retry_count"] <= 3:
                remaining.append(item)
            continue

        current_price = parse_int_field(quote.get("stck_prpr"))
        pnl_percent = ((current_price - entry_price) / entry_price * 100.0) if entry_price else 0.0
        created_at_raw = str(item.get("created_at", "") or "")
        holding_minutes = parse_float_field(item.get("horizon_minutes"), 0.0)
        if created_at_raw:
            try:
                created_at = datetime.fromisoformat(created_at_raw)
                holding_minutes = max(0.0, (now - created_at).total_seconds() / 60.0)
            except ValueError:
                pass

        quote_features = build_quote_feature_payload(quote, reference_price=entry_price, qty=qty)
        extra_features = item.get("extra_features") if isinstance(item.get("extra_features"), dict) else {}
        extra_features.update(quote_features)
        extra_features["label_horizon_minutes"] = parse_int_field(item.get("horizon_minutes"))
        extra_features["label_current_price"] = current_price

        tracked_invested = invested_amount(mode=args.mode, account=args.account)
        record_trade_event(
            side="label",
            symbol=symbol,
            name=str(item.get("name") or quote.get("hts_kor_isnm", symbol)),
            price=current_price,
            change_rate=parse_float_field(quote.get("prdy_ctrt")),
            volume=parse_int_field(quote.get("acml_vol")),
            qty=qty,
            reason=str(item.get("reason", "timed_label")),
            pnl_percent=pnl_percent,
            label_success=1.0 if pnl_percent > 0 else 0.0,
            model_score=parse_float_field(item.get("model_score"), 0.0),
            candidate_score=parse_float_field(item.get("candidate_score"), 0.0),
            entry_price=entry_price,
            holding_minutes=holding_minutes,
            tracked_invested=tracked_invested,
            remaining_budget=max(0, args.budget - tracked_invested),
            tracked_position_count=len(list_positions(mode=args.mode, account=args.account)),
            stop_loss_percent=args.stop_loss_percent,
            take_profit_percent=args.take_profit_percent,
            extra_features=extra_features,
        )
        processed += 1

    write_pending_labels(remaining)
    if processed:
        print(f"학습 라벨 생성: {processed}개 처리, 대기 {len(remaining)}개", flush=True)
        auto_retrain_if_needed()


def mark_symbol_failure(symbol: str, cooldown_seconds: int, reason: str) -> None:
    failures = read_symbol_failures()
    failures[symbol] = {
        "expires_at": (datetime.now() + timedelta(seconds=cooldown_seconds)).isoformat(),
        "reason": reason,
    }
    write_symbol_failures(failures)


def is_symbol_blocked(symbol: str) -> tuple[bool, str]:
    failures = read_symbol_failures()
    info = failures.get(symbol)
    if not info:
        return False, ""
    try:
        expires_at = datetime.fromisoformat(info["expires_at"])
    except (KeyError, ValueError):
        return False, ""
    if expires_at <= datetime.now():
        failures.pop(symbol, None)
        write_symbol_failures(failures)
        return False, ""
    return True, info.get("reason", "")


def is_domestic_market_open(now: datetime | None = None) -> bool:
    current = now or datetime.now(KST)
    if current.weekday() >= 5:
        return False
    market_open = current.replace(hour=9, minute=0, second=0, microsecond=0)
    market_close = current.replace(hour=15, minute=20, second=0, microsecond=0)
    return market_open <= current <= market_close


def should_exclude_name(name: str) -> bool:
    cleaned = name.strip().upper()
    if any(token in cleaned for token in JUNK_NAME_TOKENS):
        return True
    return cleaned.endswith("우") or cleaned.endswith("우B") or cleaned.endswith("우C")


def parse_master_mst(content: bytes, market: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for raw_line in content.decode("cp949", errors="ignore").splitlines():
        if not raw_line.strip():
            continue
        body = raw_line[: len(raw_line) - 228] if len(raw_line) > 228 else raw_line
        short_code = body[0:9].strip()
        name = body[21:].strip()
        if not short_code or not short_code.isdigit():
            continue
        if should_exclude_name(name):
            continue
        rows.append({"symbol": short_code, "name": name, "market": market})
    return rows


def load_master_codes(refresh: bool = False) -> list[dict[str, str]]:
    if MASTER_CACHE_PATH.exists() and not refresh:
        try:
            payload = json.loads(MASTER_CACHE_PATH.read_text(encoding="utf-8"))
            expires_at = datetime.fromisoformat(payload["expires_at"])
            if expires_at > datetime.now():
                return payload["codes"]
        except (KeyError, ValueError, json.JSONDecodeError):
            pass

    codes: list[dict[str, str]] = []
    for market, url in MASTER_SOURCES.items():
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
            file_name = zf.namelist()[0]
            codes.extend(parse_master_mst(zf.read(file_name), market))

    payload = {
        "expires_at": (datetime.now() + timedelta(hours=12)).isoformat(),
        "codes": codes,
    }
    MASTER_CACHE_PATH.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return codes


def maybe_handle_position_exit(args: argparse.Namespace, client: KISClient, quote: dict[str, Any]) -> bool:
    saved = get_position(mode=args.mode, account=args.account, symbol=args.symbol)
    if not saved:
        return False

    current_price = int(quote["stck_prpr"])
    entry_price = int(saved["entry_price"])
    qty = int(saved["qty"])
    position_name = quote.get("hts_kor_isnm", args.symbol)
    stop_price = max(1, int(entry_price * (1 - args.stop_loss_percent / 100)))
    take_profit_enabled = args.take_profit_percent > 0
    take_profit_price = max(1, int(entry_price * (1 + args.take_profit_percent / 100))) if take_profit_enabled else None
    if take_profit_enabled:
        print(
            f"???: qty={qty} entry={entry_price} "
            f"stop_loss={args.stop_loss_percent}% ?????={stop_price} "
            f"take_profit={args.take_profit_percent}% ?????={take_profit_price}"
        )
    else:
        print(
            f"???: qty={qty} entry={entry_price} "
            f"stop_loss={args.stop_loss_percent}% ?????={stop_price} "
            "take_profit=disabled"
        )

    if current_price <= stop_price:
        exit_reason = f"?? ?? ??: ??? {current_price} <= ????? {stop_price}"
    elif take_profit_enabled and take_profit_price is not None and current_price >= take_profit_price:
        exit_reason = f"?? ?? ??: ??? {current_price} >= ????? {take_profit_price}"
    else:
        return False
        return False

    if not is_domestic_market_open():
        print("청산 조건 충족이지만 장중이 아니라 매도하지 않습니다.")
        return True

    print(exit_reason)
    if not args.execute:
        print("실제 청산 매도는 전송하지 않았습니다. 실제 실행하려면 --execute 를 추가하세요.")
        return True

    lock_key = make_lock_key(mode=args.mode, account=args.account, symbol=args.symbol, side="sell")
    if is_locked(lock_key, args.cooldown_seconds):
        print("청산 매도 중단: 최근 동일 매도 주문 기록이 있어 잠시 막아두었습니다.")
        return True

    sell_price = current_price if args.order_type == "00" else 0
    result = client.place_cash_order(
        side="sell",
        symbol=args.symbol,
        quantity=qty,
        price=sell_price,
        order_type=args.order_type,
    )
    lock_order(lock_key)
    pnl_percent = ((current_price - entry_price) / entry_price * 100) if entry_price else 0.0
    saved_at_raw = str(saved.get("saved_at", "") or "")
    holding_minutes: float | None = None
    if saved_at_raw:
        try:
            saved_at = datetime.fromisoformat(saved_at_raw)
            holding_minutes = max(0.0, (datetime.now(KST) - saved_at).total_seconds() / 60.0)
        except ValueError:
            holding_minutes = None
    tracked_invested = invested_amount(mode=args.mode, account=args.account)
    tracked_position_count = len(list_positions(mode=args.mode, account=args.account))
    current_model_score = score_with_model(
        load_model_weights(),
        current_price,
        float(quote.get("prdy_ctrt", "0") or "0"),
        int(quote.get("acml_vol", "0") or "0"),
    )
    record_trade_event(
        side="sell",
        symbol=args.symbol,
        name=position_name,
        price=current_price,
        change_rate=float(quote.get("prdy_ctrt", "0") or "0"),
        volume=int(quote.get("acml_vol", "0") or "0"),
        qty=qty,
        reason="take_profit" if take_profit_enabled and take_profit_price is not None and current_price >= take_profit_price else "stop_loss",
        pnl_percent=pnl_percent,
        label_success=1.0 if pnl_percent > 0 else 0.0,
        model_score=current_model_score,
        entry_price=entry_price,
        holding_minutes=holding_minutes,
        tracked_invested=tracked_invested,
        remaining_budget=max(0, args.budget - tracked_invested),
        tracked_position_count=tracked_position_count,
        stop_loss_percent=args.stop_loss_percent,
        take_profit_percent=args.take_profit_percent,
        extra_features=build_quote_feature_payload(quote, reference_price=entry_price, qty=qty),
    )
    clear_position(mode=args.mode, account=args.account, symbol=args.symbol)
    auto_retrain_if_needed()
    print("청산 매도 성공:", json.dumps(result, ensure_ascii=False))
    return True


def decide_signal(current_price: int, buy_below: int | None, sell_above: int | None) -> str:
    if buy_below is not None and current_price <= buy_below:
        return "buy"
    if sell_above is not None and current_price >= sell_above:
        return "sell"
    return "hold"


def print_price_summary(symbol: str, quote: dict[str, Any]) -> None:
    print(
        f"[{symbol}] {quote.get('hts_kor_isnm', '')} "
        f"현재가={quote.get('stck_prpr', '')} "
        f"전일대비={quote.get('prdy_vrss', '')} "
        f"등락률={quote.get('prdy_ctrt', '')}%"
    )


def parse_symbols(raw_symbols: str) -> list[dict[str, int | str | None]]:
    items = [item.strip() for item in raw_symbols.split(",") if item.strip()]
    if not items:
        raise ValueError("최소 1개 종목코드는 필요합니다.")
    result: list[dict[str, int | str | None]] = []
    for item in items:
        if ":" in item:
            symbol, raw_price = item.split(":", 1)
            if not raw_price.strip().isdigit():
                raise ValueError("종목별 설정은 137940:300 형식이어야 합니다.")
            result.append({"symbol": symbol.strip(), "buy_below": int(raw_price), "price": int(raw_price)})
        else:
            result.append({"symbol": item, "buy_below": None, "price": None})
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="한국투자 Open API 국내주식 자동매매")
    parser.add_argument("--symbol", help="종목코드 또는 종목코드:가격 목록. 예: 137940:300,001000:800")
    parser.add_argument("--auto-pick", action="store_true", help="국내 저가주 후보를 자동 선별해서 매수합니다.")
    parser.add_argument("--auto-pick-count", type=int, default=8, help="자동 후보 선별 개수")
    parser.add_argument("--candidate-scan-size", type=int, default=150, help="자동 선별 시 랜덤 조회 후보 수")
    parser.add_argument("--min-price", type=int, default=200, help="자동 선별 최소 가격")
    parser.add_argument("--max-price", type=int, default=5000, help="자동 선별 최대 가격")
    parser.add_argument("--min-change-rate", type=float, default=0.5, help="자동 선별 최소 등락률")
    parser.add_argument("--max-change-rate", type=float, default=25.0, help="자동 선별 최대 등락률")
    parser.add_argument("--min-volume", type=int, default=100000, help="자동 선별 최소 거래량")
    parser.add_argument("--relax-steps", type=int, default=3, help="후보가 없을 때 조건 자동 완화 단계 수")
    parser.add_argument("--qty", type=int, default=0, help="0이면 자동 수량, 양수면 해당 수량 이하로 제한")
    parser.add_argument("--buy-below", type=int, help="현재가가 이 값 이하이면 매수")
    parser.add_argument("--sell-above", type=int, help="현재가가 이 값 이상이면 매도")
    parser.add_argument("--mode", choices=["real", "demo"], default="real", help="실전(real) 또는 모의(demo)")
    parser.add_argument("--order-type", default="00", help="주문구분. 지정가=00, 시장가=01")
    parser.add_argument("--price", type=int, help="주문 단가. 지정가면 보통 입력, 시장가면 0")
    parser.add_argument("--credentials", type=Path, default=DEFAULT_CREDENTIALS_PATH, help="API 키 파일 경로")
    parser.add_argument(
        "--account",
        default=load_app_settings().get("kis_account", "00000000-00"),
        help="계좌번호. 예: 00000000-00",
    )
    parser.add_argument("--poll-seconds", type=int, default=0, help="0이면 1회, 5 이상이면 반복 감시")
    parser.add_argument("--execute", action="store_true", help="이 옵션이 있어야 실제 주문을 전송합니다.")
    parser.add_argument("--cooldown-seconds", type=int, default=300, help="같은 종목/같은 방향 주문 쿨다운")
    parser.add_argument("--stop-loss-percent", type=float, default=10.0, help="?? ? ?? ?? ???")
    parser.add_argument("--take-profit-percent", type=float, default=0.0, help="?? ? ?? ?? ???. 0 ??? ?? ?? ????")
    parser.add_argument("--budget", type=int, default=10000, help="총 투자 한도")
    parser.add_argument("--max-qty", type=int, default=10, help="종목당 최대 매수 수량")
    parser.add_argument("--failure-cooldown-seconds", type=int, default=1800, help="실패 종목 제외 시간(초)")
    parser.add_argument("--target-invest-rate", type=float, default=0.7, help="예산 대비 목표 투자율. 이보다 낮으면 더 적극 매수")
    parser.add_argument("--per-symbol-budget", type=int, default=10000, help="종목당 최대 투자금")
    return parser


def auto_pick_symbols(args: argparse.Namespace, client: KISClient) -> list[dict[str, int | str | None]]:
    master_codes = load_master_codes()
    positions = list_positions(mode=args.mode, account=args.account)
    blocked_symbols = set(positions.keys())
    tracked_invested = invested_amount(mode=args.mode, account=args.account)
    remaining_budget = max(0, args.budget - tracked_invested)
    target_invested = int(args.budget * args.target_invest_rate)
    model_weights = load_model_weights()
    print(
        f"자동 후보 탐색 시작: 전체종목={len(master_codes)} 보유제외={len(blocked_symbols)} "
        f"현재투자금={tracked_invested}원 남은예산={remaining_budget}원",
        flush=True,
    )

    scan_pool = [item for item in master_codes if item["symbol"] not in blocked_symbols]
    if not scan_pool:
        print("자동 후보 탐색 중단: 사용할 수 있는 스캔 풀이 없습니다.", flush=True)
        return []

    aggressive_mode = tracked_invested < target_invested
    desired_pick_count = args.auto_pick_count + (4 if aggressive_mode else 0)
    sample_size = min(len(scan_pool), max(args.candidate_scan_size, desired_pick_count * 20))
    print(
        f"자동 후보 스캔 설정: aggressive={'on' if aggressive_mode else 'off'} "
        f"샘플수={sample_size} 목표후보수={desired_pick_count}",
        flush=True,
    )
    quotes: list[Candidate] = []
    for item in random.sample(scan_pool, sample_size):
        symbol = item["symbol"]
        blocked, _ = is_symbol_blocked(symbol)
        if blocked:
            continue
        try:
            quote = client.get_current_price(symbol)
        except (requests.RequestException, KISError) as exc:
            mark_symbol_failure(symbol, args.failure_cooldown_seconds, f"시세조회 실패: {exc}")
            continue

        try:
            price = int(quote.get("stck_prpr", "0") or "0")
            change_rate = float(quote.get("prdy_ctrt", "0") or "0")
            volume = int(quote.get("acml_vol", "0") or "0")
        except ValueError:
            continue

        if price <= 0 or volume <= 0:
            continue

        # Higher volume and moderate positive momentum score better than extreme spikes.
        base_score = min(volume / 100000, 60) + max(0, 12 - abs(change_rate - 5)) - (price / 2500)
        ai_score = score_with_model(model_weights, price, change_rate, volume) * 18
        score = base_score + ai_score
        if aggressive_mode:
            score += 3
        quotes.append(
            Candidate(
                symbol=symbol,
                name=item["name"],
                price=price,
                change_rate=change_rate,
                volume=volume,
                score=score,
            )
        )

    candidates: list[Candidate] = []
    for step in range(max(1, args.relax_steps)):
        min_price = max(50, args.min_price - (step * 100))
        max_price = args.max_price + (step * 1000)
        min_change = max(-2.0, args.min_change_rate - step)
        max_change = args.max_change_rate + (step * 5)
        min_volume = max(10000, args.min_volume - (step * 50000))

        filtered = [
            quote
            for quote in quotes
            if min_price <= quote.price <= max_price
            and min_change <= quote.change_rate <= max_change
            and quote.volume >= min_volume
        ]
        filtered.sort(key=lambda item: item.score, reverse=True)
        if filtered:
            candidates = filtered[: desired_pick_count]
            if step > 0:
                print(
                    "자동 선별 조건 완화 적용:"
                    f" step={step} 가격={min_price}~{max_price}"
                    f" 등락률={min_change:.1f}~{max_change:.1f}"
                    f" 거래량>={min_volume}"
                )
            break

    if not candidates:
        print("자동 선별 후보를 찾지 못했습니다. 다음 주기에서 다시 시도합니다.")
        return []

    verify_count = min(len(candidates), max(args.auto_pick_count, 4))
    to_verify = candidates[:verify_count]
    verified_candidates: list[Candidate] = []
    tracked_invested = invested_amount(mode=args.mode, account=args.account)
    remaining_budget = max(0, args.budget - tracked_invested)
    print(f"자동 선별 1차 통과={len(candidates)}개, 주문 가능 검증 대상={len(to_verify)}개", flush=True)

    for candidate in to_verify:
        if candidate.price <= 0 or remaining_budget < candidate.price:
            continue
        try:
            buyable = client.get_buyable_cash(
                symbol=candidate.symbol,
                price=candidate.price,
                order_type=args.order_type,
            )
            max_buy_qty = int(buyable.get("max_buy_qty", "0") or "0")
            ord_psbl_cash = int(buyable.get("ord_psbl_cash", "0") or "0")
        except (requests.RequestException, KISError) as exc:
            mark_symbol_failure(candidate.symbol, args.failure_cooldown_seconds, f"매수가능조회 실패: {exc}")
            continue
        if max_buy_qty <= 0 or ord_psbl_cash < candidate.price:
            continue
        verified_candidates.append(candidate)

    candidates = verified_candidates
    if not candidates:
        print("자동 선별 후보는 있었지만 실제 매수가능조회까지 통과한 종목이 없었습니다.")
        return []

    print(
        f"자동 선별 모드: aggressive={'on' if aggressive_mode else 'off'} "
        f"추적투자금={tracked_invested}원 목표투자금={target_invested}원 남은예산={remaining_budget}원"
    )

    joined = ", ".join(
        f"{c.name}({c.symbol}:{c.price}, score={c.score:.1f})" for c in candidates
    )
    print(f"자동 선별 후보: {joined}", flush=True)
    print(f"AI model samples={model_weights.sample_count}", flush=True)
    return [
        {
            "symbol": candidate.symbol,
            "buy_below": candidate.price,
            "price": candidate.price,
            "name": candidate.name,
            "candidate_score": candidate.score,
        }
        for candidate in candidates
    ]


def run_once_for_symbol(
    args: argparse.Namespace,
    client: KISClient,
    symbol: str,
    symbol_buy_below: int | None,
    symbol_price: int | None,
    candidate_score: float | None = None,
) -> None:
    blocked, reason = is_symbol_blocked(symbol)
    if blocked:
        print(f"[{symbol}] 제외중: {reason}", flush=True)
        return

    quote = client.get_current_price(symbol)
    print_price_summary(symbol, quote)
    quote_name = quote.get("hts_kor_isnm", symbol)

    symbol_args = argparse.Namespace(**vars(args))
    symbol_args.symbol = symbol
    if symbol_buy_below is not None:
        symbol_args.buy_below = symbol_buy_below
    if symbol_price is not None:
        symbol_args.price = symbol_price

    if maybe_handle_position_exit(symbol_args, client, quote):
        return

    current_price = int(quote["stck_prpr"])
    signal = decide_signal(current_price, symbol_args.buy_below, symbol_args.sell_above)
    print(f"판단 결과: {signal}", flush=True)
    if signal == "hold":
        return

    if signal == "buy" and not is_domestic_market_open():
        print("주문 중단: 국내 장중(평일 09:00~15:20)일 때만 매수합니다.")
        return

    if signal == "buy" and get_position(mode=args.mode, account=args.account, symbol=symbol):
        print("주문 중단: 이미 이 종목 보유 기록이 있어서 재매수하지 않습니다.")
        return

    order_price = symbol_args.price or current_price if args.order_type == "00" else (symbol_args.price or 0)
    if signal == "buy":
        buyable = client.get_buyable_cash(symbol=symbol, price=order_price, order_type=args.order_type)
        buyable_cash = int(buyable.get("ord_psbl_cash", "0") or "0")
        max_buy_qty = int(buyable.get("max_buy_qty", "0") or "0")
        tracked_invested = invested_amount(mode=args.mode, account=args.account)
        remaining_budget = args.budget - tracked_invested
        budget_qty = remaining_budget // max(order_price, 1)
        per_symbol_qty = args.per_symbol_budget // max(order_price, 1)
        auto_qty = min(args.max_qty, max_buy_qty, budget_qty)
        if per_symbol_qty > 0:
            auto_qty = min(auto_qty, per_symbol_qty)
        order_qty = min(args.qty, auto_qty) if args.qty > 0 else auto_qty
        print(f"주문가능현금={buyable_cash}원 최대매수수량={max_buy_qty}주", flush=True)
        print(f"추적중 투자금={tracked_invested}원 남은예산={remaining_budget}원", flush=True)
        print(
            f"자동계산 주문수량={order_qty}주 종목당최대수량={args.max_qty}주 "
            f"종목당최대투자금={args.per_symbol_budget}원",
            flush=True
        )
        if buyable_cash <= 0 or max_buy_qty <= 0:
            print("주문 중단: 계좌의 주문가능현금 또는 가능수량이 부족합니다.", flush=True)
            return
        if order_qty <= 0:
            print("주문 중단: 설정한 총 투자 한도를 초과합니다.", flush=True)
            return
    else:
        order_qty = args.qty if args.qty > 0 else 1

    print(f"주문 예정: side={signal} qty={order_qty} price={order_price} mode={args.mode} execute={args.execute}", flush=True)
    if not args.execute:
        print("실제 주문은 전송하지 않았습니다. 실제 주문하려면 --execute 를 추가하세요.", flush=True)
        return

    lock_key = make_lock_key(mode=args.mode, account=args.account, symbol=symbol, side=signal)
    if is_locked(lock_key, args.cooldown_seconds):
        print("주문 중단: 같은 종목/같은 방향 주문이 최근에 실행되어 잠시 막아두었습니다.", flush=True)
        return

    result = client.place_cash_order(
        side=signal,
        symbol=symbol,
        quantity=order_qty,
        price=order_price,
        order_type=args.order_type,
    )
    lock_order(lock_key)
    if signal == "buy":
        save_position(
            mode=args.mode,
            account=args.account,
            symbol=symbol,
            qty=order_qty,
            entry_price=order_price if order_price > 0 else current_price,
        )
        entry_price_for_training = order_price if order_price > 0 else current_price
        change_rate_for_training = float(quote.get("prdy_ctrt", "0") or "0")
        volume_for_training = int(quote.get("acml_vol", "0") or "0")
        model_score_for_training = score_with_model(
            load_model_weights(),
            entry_price_for_training,
            change_rate_for_training,
            volume_for_training,
        )
        extra_features_for_training = build_quote_feature_payload(
            quote,
            reference_price=entry_price_for_training,
            qty=order_qty,
        )
        record_trade_event(
            side="buy",
            symbol=symbol,
            name=quote_name,
            price=entry_price_for_training,
            change_rate=change_rate_for_training,
            volume=volume_for_training,
            qty=order_qty,
            reason="signal_buy",
            model_score=model_score_for_training,
            candidate_score=candidate_score,
            entry_price=entry_price_for_training,
            tracked_invested=tracked_invested + (order_qty * entry_price_for_training),
            remaining_budget=max(
                0,
                args.budget - (tracked_invested + (order_qty * entry_price_for_training)),
            ),
            tracked_position_count=len(list_positions(mode=args.mode, account=args.account)) + 1,
            stop_loss_percent=args.stop_loss_percent,
            take_profit_percent=args.take_profit_percent,
            extra_features=extra_features_for_training,
        )
        queue_training_labels(
            mode=args.mode,
            account=args.account,
            symbol=symbol,
            name=quote_name,
            entry_price=entry_price_for_training,
            qty=order_qty,
            change_rate=change_rate_for_training,
            volume=volume_for_training,
            reason="signal_buy",
            candidate_score=candidate_score,
            model_score=model_score_for_training,
            extra_features=extra_features_for_training,
        )
    else:
        clear_position(mode=args.mode, account=args.account, symbol=symbol)
    print("주문 성공:", json.dumps(result, ensure_ascii=False), flush=True)


def build_symbol_specs(args: argparse.Namespace, client: KISClient) -> list[dict[str, int | str | None]]:
    if args.auto_pick:
        return auto_pick_symbols(args, client)
    if args.symbol:
        return parse_symbols(args.symbol)
    return []


def run_once(args: argparse.Namespace, client: KISClient) -> None:
    process_pending_training_labels(args, client)
    symbol_specs = build_symbol_specs(args, client)
    positions = list_positions(mode=args.mode, account=args.account)
    if positions:
        tracked = ", ".join(f"{symbol}(qty={info['qty']}, entry={info['entry_price']})" for symbol, info in positions.items())
        print(f"현재 추적 보유: {tracked}", flush=True)
    print(f"총 투자 한도={args.budget}원 현재 추적 투자금={invested_amount(mode=args.mode, account=args.account)}원", flush=True)

    if not symbol_specs:
        print("이번 주기에는 처리할 종목이 없습니다.", flush=True)
        return

    print(f"이번 주기 처리 종목 수={len(symbol_specs)}", flush=True)
    for spec in symbol_specs:
        symbol = str(spec["symbol"])
        try:
            run_once_for_symbol(
                args,
                client,
                symbol=symbol,
                symbol_buy_below=spec["buy_below"] if isinstance(spec["buy_below"], int) else args.buy_below,
                symbol_price=spec["price"] if isinstance(spec["price"], int) else args.price,
                candidate_score=float(spec["candidate_score"]) if isinstance(spec.get("candidate_score"), (int, float)) else None,
            )
        except (requests.RequestException, KISError) as exc:
            mark_symbol_failure(symbol, args.failure_cooldown_seconds, str(exc))
            print(f"[{symbol}] 건너뜀: {exc}", flush=True)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not args.symbol and not args.auto_pick:
        parser.error("--symbol 또는 --auto-pick 중 하나는 필요합니다.")
    if args.symbol:
        symbol_specs = parse_symbols(args.symbol)
        has_symbol_level_buy = any(isinstance(spec.get("buy_below"), int) for spec in symbol_specs)
        if args.buy_below is None and args.sell_above is None and not has_symbol_level_buy and not args.auto_pick:
            parser.error("--buy-below 또는 --sell-above 중 하나는 필요합니다.")
    if args.poll_seconds not in {0} and args.poll_seconds < 5:
        parser.error("--poll-seconds 는 0 또는 5 이상으로 설정하세요.")

    credentials = load_credentials(args.credentials)
    account = parse_account(args.account)
    client = KISClient(credentials=credentials, account=account, mode=args.mode)

    try:
        if args.poll_seconds == 0:
            run_once(args, client)
            return 0

        while True:
            print(f"\n[{datetime.now(KST).strftime('%Y-%m-%d %H:%M:%S')}] 감시 실행", flush=True)
            run_once(args, client)
            time.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        print("\n중단했습니다.", flush=True)
        return 0
    except (requests.RequestException, KISError, FileNotFoundError, ValueError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


