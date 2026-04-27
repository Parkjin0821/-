from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request

from kis_autotrade import DEFAULT_CREDENTIALS_PATH, KISClient, load_app_settings, load_credentials, parse_account


BASE_DIR = Path(__file__).resolve().parent
BOT_PATH = BASE_DIR / "kis_autotrade.py"
PYTHON_EXE = sys.executable
DEFAULT_ACCOUNT = load_app_settings().get("kis_account", "00000000-00")


@dataclass
class BotState:
    process: subprocess.Popen[str] | None = None
    logs: deque[str] = field(default_factory=lambda: deque(maxlen=600))
    command: list[str] = field(default_factory=list)
    last_error: str = ""
    lock: threading.Lock = field(default_factory=threading.Lock)

    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None


state = BotState()
app = Flask(__name__)
quote_cache: dict[str, dict[str, Any]] = {}
balance_cache: dict[str, Any] = {}
quote_client: KISClient | None = None


def append_log(line: str) -> None:
    cleaned = line.rstrip()
    if not cleaned:
        return
    with state.lock:
        state.logs.append(cleaned)


def to_int(value: Any, default: int = 0) -> int:
    try:
        if value in (None, ""):
            return default
        return int(float(str(value).replace(",", "")))
    except (TypeError, ValueError):
        return default


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return default


def get_quote_client(account_hint: str | None = None) -> KISClient | None:
    global quote_client
    if quote_client is not None:
        return quote_client
    try:
        credentials = load_credentials(DEFAULT_CREDENTIALS_PATH)
        account = parse_account(account_hint or DEFAULT_ACCOUNT)
        quote_client = KISClient(credentials=credentials, account=account, mode="real")
    except Exception:
        quote_client = None
    return quote_client


def get_cached_quote(symbol: str, account_hint: str | None = None) -> dict[str, Any] | None:
    cached = quote_cache.get(symbol)
    if cached and cached["expires_at"] > datetime.now():
        return cached["quote"]

    client = get_quote_client(account_hint)
    if client is None:
        return None

    try:
        quote = client.get_current_price(symbol)
    except Exception:
        return None

    quote_cache[symbol] = {
        "quote": quote,
        "expires_at": datetime.now() + timedelta(seconds=8),
    }
    return quote


def get_cached_balance(account_hint: str | None = None) -> dict[str, Any] | None:
    cached = balance_cache.get("balance")
    if cached and cached["expires_at"] > datetime.now():
        return cached["payload"]

    client = get_quote_client(account_hint)
    if client is None:
        return None

    try:
        payload = client.get_stock_balance()
    except Exception:
        return None

    balance_cache["balance"] = {
        "payload": payload,
        "expires_at": datetime.now() + timedelta(seconds=8),
    }
    return payload


def build_account_summary_from_live_balance(payload: dict[str, Any]) -> dict[str, Any]:
    output2 = payload.get("output2") or []
    if not output2:
        return {
            "cash": 0,
            "next_cash": 0,
            "today_buy_amount": 0,
            "today_sell_amount": 0,
            "daily_asset_change": 0,
            "daily_asset_change_rate": 0.0,
            "eval_profit": 0,
        }

    summary = output2[0]
    return {
        "cash": to_int(summary.get("dnca_tot_amt")),
        "next_cash": to_int(summary.get("nxdy_excc_amt")),
        "today_buy_amount": to_int(summary.get("thdt_buy_amt")),
        "today_sell_amount": to_int(summary.get("thdt_sll_amt")),
        "daily_asset_change": to_int(summary.get("asst_icdc_amt")),
        "daily_asset_change_rate": to_float(summary.get("asst_icdc_erng_rt")),
        "eval_profit": to_int(summary.get("evlu_pfls_smtl_amt")),
    }


def build_account_summary_from_local_cache(
    tracked_amount: int,
    total_eval_amount: int,
    total_pnl: int,
) -> dict[str, Any]:
    return {
        "cash": 0,
        "next_cash": 0,
        "today_buy_amount": 0,
        "today_sell_amount": 0,
        "daily_asset_change": total_pnl,
        "daily_asset_change_rate": ((total_pnl / tracked_amount) * 100) if tracked_amount else 0.0,
        "eval_profit": total_pnl,
    }


def build_positions_from_live_balance(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], int, int, int]:
    positions: list[dict[str, Any]] = []
    tracked_amount = 0
    total_eval_amount = 0
    total_pnl = 0

    for item in payload.get("output1", []):
        qty = to_int(item.get("hldg_qty"))
        if qty <= 0:
            continue

        entry_price = to_int(item.get("pchs_avg_pric"))
        current_price = to_int(item.get("prpr"))
        eval_amount = to_int(item.get("evlu_amt"), current_price * qty)
        pnl = to_int(item.get("evlu_pfls_amt"), eval_amount - entry_price * qty)
        pnl_rate = to_float(item.get("evlu_pfls_rt"))

        positions.append(
            {
                "symbol": item.get("pdno", ""),
                "name": item.get("prdt_name", ""),
                "qty": qty,
                "entry_price": entry_price,
                "invested": entry_price * qty,
                "current_price": current_price,
                "eval_amount": eval_amount,
                "pnl": pnl,
                "pnl_rate": pnl_rate,
            }
        )
        tracked_amount += entry_price * qty
        total_eval_amount += eval_amount
        total_pnl += pnl

    return positions, tracked_amount, total_eval_amount, total_pnl


def build_positions_from_local_cache() -> tuple[list[dict[str, Any]], int, int, int]:
    positions_path = BASE_DIR / ".cache" / "position_state.json"
    if not positions_path.exists():
        return [], 0, 0, 0

    try:
        payload = json.loads(positions_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return [], 0, 0, 0

    positions: list[dict[str, Any]] = []
    tracked_amount = 0
    total_eval_amount = 0
    total_pnl = 0

    for key, value in payload.items():
        symbol = key.split(":")[-1]
        qty = to_int(value.get("qty"))
        entry_price = to_int(value.get("entry_price"))
        invested = qty * entry_price

        current_price = None
        pnl = None
        pnl_rate = None
        eval_amount = invested

        quote = get_cached_quote(symbol, DEFAULT_ACCOUNT)
        if quote is not None:
            current_price = to_int(quote.get("stck_prpr"))
        if current_price is not None:
            eval_amount = current_price * qty
            pnl = eval_amount - invested
            pnl_rate = ((current_price - entry_price) / entry_price * 100) if entry_price else 0.0

        positions.append(
            {
                "symbol": symbol,
                "name": symbol,
                "qty": qty,
                "entry_price": entry_price,
                "invested": invested,
                "current_price": current_price,
                "eval_amount": eval_amount,
                "pnl": pnl,
                "pnl_rate": pnl_rate,
            }
        )
        tracked_amount += invested
        total_eval_amount += eval_amount
        total_pnl += pnl or 0

    return positions, tracked_amount, total_eval_amount, total_pnl


def pump_logs(process: subprocess.Popen[str]) -> None:
    try:
        assert process.stdout is not None
        for line in process.stdout:
            append_log(line)
    finally:
        return_code = process.poll()
        with state.lock:
            if state.process is process:
                state.process = None
        append_log(f"[dashboard] 자동매매 프로세스 종료 (code={return_code})")


def build_command() -> list[str]:
    return [
        PYTHON_EXE,
        "-X",
        "utf8",
        "-u",
        str(BOT_PATH),
        "--auto-pick",
        "--min-price",
        "100",
        "--max-price",
        "7000",
        "--min-change-rate",
        "-1.0",
        "--max-change-rate",
        "29.0",
        "--min-volume",
        "50000",
        "--auto-pick-count",
        "12",
        "--candidate-scan-size",
        "180",
        "--relax-steps",
        "5",
        "--budget",
        "50000",
        "--per-symbol-budget",
        "15000",
        "--max-qty",
        "20",
        "--poll-seconds",
        "10",
        "--stop-loss-percent",
        "10",
        "--take-profit-percent",
        "0",
        "--failure-cooldown-seconds",
        "600",
        "--target-invest-rate",
        "0.9",
        "--execute",
    ]


@app.get("/")
def index() -> str:
    return render_template("index.html")


@app.get("/api/status")
def api_status() -> Any:
    live_balance = get_cached_balance(DEFAULT_ACCOUNT)
    if live_balance:
        positions, tracked_amount, total_eval_amount, total_pnl = build_positions_from_live_balance(live_balance)
        account_summary = build_account_summary_from_live_balance(live_balance)
        data_source = "live_balance"
    else:
        positions, tracked_amount, total_eval_amount, total_pnl = build_positions_from_local_cache()
        account_summary = build_account_summary_from_local_cache(tracked_amount, total_eval_amount, total_pnl)
        data_source = "local_cache"

    with state.lock:
        running = state.is_running()
        return jsonify(
            {
                "running": running,
                "command": " ".join(shlex.quote(part) for part in state.command) if state.command else "",
                "last_error": state.last_error,
                "logs": list(state.logs),
                "positions": positions,
                "tracked_amount": tracked_amount,
                "total_eval_amount": total_eval_amount,
                "total_pnl": total_pnl,
                "total_pnl_rate": ((total_pnl / tracked_amount) * 100) if tracked_amount else 0,
                "account_summary": account_summary,
                "data_source": data_source,
            }
        )


@app.post("/api/start")
def api_start() -> Any:
    with state.lock:
        if state.is_running():
            return jsonify({"ok": False, "message": "이미 자동매매가 실행 중입니다."}), 400

    command = build_command()

    try:
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        process = subprocess.Popen(
            command,
            cwd=str(BASE_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )
    except OSError as exc:
        with state.lock:
            state.last_error = str(exc)
        return jsonify({"ok": False, "message": f"프로세스를 시작하지 못했습니다: {exc}"}), 500

    with state.lock:
        state.process = process
        state.command = command
        state.last_error = ""
        state.logs.clear()
        state.logs.append("[dashboard] 자동매매 프로세스를 시작했습니다.")
        state.logs.append(f"[dashboard] command: {' '.join(command)}")

    threading.Thread(target=pump_logs, args=(process,), daemon=True).start()
    return jsonify({"ok": True})


@app.post("/api/stop")
def api_stop() -> Any:
    with state.lock:
        process = state.process
    if process is None or process.poll() is not None:
        return jsonify({"ok": False, "message": "실행 중인 프로세스가 없습니다."}), 400

    process.terminate()
    append_log("[dashboard] 중지 요청을 보냈습니다.")
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=False)
