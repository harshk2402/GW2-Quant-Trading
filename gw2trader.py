import ssl
import certifi
import io
import json
import sys
import gzip
import urllib.request
import urllib.error
import ssl

HEADERS = {
    # Reasonable modern desktop UA
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Accept-Language": "en-US,en;q=0.9",
    # gzip only (avoid brotli complexity)
    "Accept-Encoding": "gzip, deflate",
    "Referer": (
        "https://gw2trader.gg/items?"
        "filters=buy-0-1500000_sell-0-5000000_volume24h-10000000-0"
    ),
    "Connection": "keep-alive",
}


class GW2Trader:
    """Reusable client for gw2trader.gg endpoints.

    Parameters
    ----------
    headers : dict | None
        Optional headers to merge/override the defaults.
    timeout : int | float
        Request timeout (seconds).
    ssl_context : ssl.SSLContext | None
        SSL context to use. Defaults to the module-level CTX.
    base_next_data : str
        The _next/data build key. Defaults to the one observed in this script.
    """

    def __init__(
        self,
        headers: dict | None = None,
        timeout: float = 30,
        ssl_context: ssl.SSLContext | None = None,
        base_next_data: str = "qCar717yjeoxwvFlOUH8t",
    ) -> None:
        try:
            CTX = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            # fallback: unverified
            CTX = ssl._create_unverified_context()
        self.timeout = timeout
        self.ctx = ssl_context or CTX
        # start from module headers and allow overrides
        self.headers = dict(HEADERS)
        if headers:
            self.headers.update(headers)
        self.base_next_data = base_next_data

    # ----- low-level fetch -----
    def fetch(self, url: str) -> bytes:
        req = urllib.request.Request(url, headers=self.headers)
        try:
            with urllib.request.urlopen(
                req, timeout=self.timeout, context=self.ctx
            ) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"HTTP {resp.status}")
                data = resp.read()
                enc = resp.headers.get("Content-Encoding", "").lower()
                if "gzip" in enc:
                    data = gzip.GzipFile(fileobj=io.BytesIO(data)).read()
                return data
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"HTTPError {e.code}: {e.reason}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"URLError: {e.reason}") from e

    # ----- helpers -----
    def _json(self, raw: bytes) -> dict:
        try:
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                text = raw.decode("latin-1", errors="replace")
            return json.loads(text)
        except json.JSONDecodeError as e:
            snippet = (
                raw[:200].decode("latin-1", errors="replace")
                if isinstance(raw, (bytes, bytearray))
                else str(raw)[:200]
            )
            raise RuntimeError(
                f"Failed to parse JSON: {e}. First 200 chars: {snippet!r}"
            ) from e

    # ----- public endpoints -----
    def get_flippable_items(
        self, buy_min=0, buy_max=0, sell_min=0, sell_max=0, vol_min=0, vol_max=0
    ) -> dict:
        url = (
            f"https://gw2trader.gg/_next/data/{self.base_next_data}/items.json?"
            f"sortBy=stat_profit%3Adesc&filters=buy-{buy_min}-{buy_max}_sell-{sell_min}-{sell_max}_volume24h-{vol_min}-{vol_max}"
        )
        try:
            raw = self.fetch(url)
            payload = self._json(raw)
            return payload["pageProps"]["data"]
        except Exception as e:
            sys.stderr.write(f"Error fetching flippable items: {e}\n")
            # Best-effort schema hints
            return {}

    def get_hourly_data(
        self,
        item_id: int,
        since_ms: int | None = 1753112177694,
        page_size: int = 800,
        page: int = 1,
        hour_interval: bool = True,
    ) -> dict:
        interval_str = "true" if hour_interval else "false"
        url = (
            "https://gw2trader.gg/backend/api/items/"
            f"{item_id}?populate[item_prices][fields][0]=item_id&"
            "populate[item_prices][fields][1]=buys_unit_price&"
            "populate[item_prices][fields][2]=buys_quantity&"
            "populate[item_prices][fields][3]=sells_unit_price&"
            "populate[item_prices][fields][4]=sells_quantity&"
            "populate[item_prices][fields][5]=timestamp&"
            "populate[item_prices][fields][6]=market_cap&"
            "populate[item_prices][fields][7]=volume_24h&"
            f"populate[item_prices][filters][timestamp][$gt]={since_ms or 0}&"
            f"populate[item_prices][filters][is_hour_interval]={interval_str}&"
            f"populate[item_prices][pagination][pageSize]={page_size}&"
            f"populate[item_prices][pagination][page]={page}"
        )
        try:
            raw = self.fetch(url)
            return self._json(raw)
        except Exception as e:
            sys.stderr.write(f"Error fetching hourly for {item_id}: {e}\n")
            return {}
