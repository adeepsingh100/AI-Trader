"""CoinDCX client: public market data (no auth) plus authenticated
account/order endpoints for RealExecutionAgent (build step 11)."""

from __future__ import annotations

import hashlib
import hmac
import json
import time

import requests

from src.config import COINDCX_API_KEY, COINDCX_API_SECRET
from src.resilience import retry_with_backoff

TICKER_URL = "https://api.coindcx.com/exchange/ticker"
MARKETS_DETAILS_URL = "https://api.coindcx.com/exchange/v1/markets_details"
ORDERBOOK_URL = "https://public.coindcx.com/market_data/orderbook"
CANDLES_URL = "https://public.coindcx.com/market_data/candles"
AUTH_BASE_URL = "https://api.coindcx.com"

# Retried below: every plain read (idempotent, safe to repeat on a
# transient network blip). create_order is deliberately NOT retried — a
# failed request whose response was lost but which actually succeeded
# server-side would place a second order on retry, a real double-submission
# risk unlike a re-read.


def get_ticker() -> list[dict]:
    def _fetch():
        resp = requests.get(TICKER_URL, timeout=10)
        resp.raise_for_status()
        return resp.json()

    return retry_with_backoff(_fetch, exceptions=(requests.RequestException,))


def get_markets_details() -> list[dict]:
    def _fetch():
        resp = requests.get(MARKETS_DETAILS_URL, timeout=10)
        resp.raise_for_status()
        return resp.json()

    return retry_with_backoff(_fetch, exceptions=(requests.RequestException,))


def top_inr_pairs_by_turnover(n: int = 10, ticker: list[dict] | None = None) -> list[dict]:
    """Top N INR markets ranked by 24h INR turnover (volume * last_price).

    Raw ticker 'volume' is base-currency volume, not comparable across
    symbols — turnover puts every market on the same INR scale.
    """
    ticker = ticker if ticker is not None else get_ticker()
    inr = [t for t in ticker if t["market"].endswith("INR")]
    for t in inr:
        t["turnover_inr"] = float(t["volume"]) * float(t["last_price"])
    inr.sort(key=lambda t: t["turnover_inr"], reverse=True)
    return inr[:n]


def symbol_to_pair(symbol: str, markets_details: list[dict] | None = None) -> str:
    """Map a ticker market symbol (e.g. 'BTCINR') to the pair id
    ('I-BTC_INR') the orderbook/candles endpoints expect."""
    markets_details = markets_details if markets_details is not None else get_markets_details()
    for m in markets_details:
        if m["symbol"] == symbol:
            return m["pair"]
    raise ValueError(f"unknown symbol: {symbol}")


def get_orderbook(pair: str) -> dict:
    def _fetch():
        resp = requests.get(ORDERBOOK_URL, params={"pair": pair}, timeout=10)
        resp.raise_for_status()
        return resp.json()

    return retry_with_backoff(_fetch, exceptions=(requests.RequestException,))


def get_candles(pair: str, interval: str = "1m", limit: int = 100) -> list[dict]:
    def _fetch():
        resp = requests.get(
            CANDLES_URL, params={"pair": pair, "interval": interval, "limit": limit}, timeout=10
        )
        resp.raise_for_status()
        return resp.json()

    return retry_with_backoff(_fetch, exceptions=(requests.RequestException,))


def _signed_post(path: str, body: dict) -> dict | list:
    body = {**body, "timestamp": int(time.time() * 1000)}
    payload = json.dumps(body, separators=(",", ":"))
    signature = hmac.new(
        COINDCX_API_SECRET.encode(), payload.encode(), hashlib.sha256
    ).hexdigest()
    headers = {
        "Content-Type": "application/json",
        "X-AUTH-APIKEY": COINDCX_API_KEY,
        "X-AUTH-SIGNATURE": signature,
    }
    resp = requests.post(AUTH_BASE_URL + path, data=payload, headers=headers, timeout=10)
    resp.raise_for_status()
    return resp.json()


def get_balances() -> list[dict]:
    return retry_with_backoff(
        lambda: _signed_post("/exchange/v1/users/balances", {}),
        exceptions=(requests.RequestException,),
    )


def create_order(
    market: str, side: str, total_quantity: float, order_type: str = "market_order"
) -> dict:
    # Deliberately not retried — see the module-level note above.
    return _signed_post(
        "/exchange/v1/orders/create",
        {
            "side": side,
            "order_type": order_type,
            "market": market,
            "total_quantity": total_quantity,
        },
    )


def get_order_status(order_id: str) -> dict:
    return retry_with_backoff(
        lambda: _signed_post("/exchange/v1/orders/status", {"id": order_id}),
        exceptions=(requests.RequestException,),
    )
