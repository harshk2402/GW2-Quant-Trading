# gw2_client.py
from __future__ import annotations

import time
import math
import json
from typing import Iterable, List, Optional, Union, Dict, Any
import requests
import os


class GW2:
    """
    Minimal Guild Wars 2 API client.

    Supports:
      - /v2/commerce/transactions/{current|history}/{buys|sells}  (auth required)
      - /v2/commerce/listings[/{id}] or ?ids=...                  (public)

    Docs:
      - Transactions: https://wiki.guildwars2.com/wiki/API:2/commerce/transactions
      - Listings:     https://wiki.guildwars2.com/wiki/API:2/commerce/listings
    """

    BASE = "https://api.guildwars2.com/v2"
    # GW2 API practical rate limit ~300 req/min. We'll be conservative.
    MAX_PER_MIN = 300

    def __init__(
        self,
        api_key: Optional[str] = os.getenv(
            "GW2_API_KEY", None
        ),  # or pass None for public endpoints only
        timeout: float = 20.0,
        session: Optional[requests.Session] = None,
        user_agent: str = "GW2-Client/1.0 (+https://github.com/you/yourrepo)",
        backoff_base: float = 0.5,
        max_retries: int = 3,
    ):
        self.api_key = api_key
        self.timeout = timeout
        self.backoff_base = backoff_base
        self.max_retries = max_retries

        self.s = session or requests.Session()
        self.s.headers.update(
            {
                "Accept": "application/json",
                "User-Agent": user_agent,
            }
        )
        if api_key:
            # official auth: Authorization: Bearer <API key>
            self.s.headers["Authorization"] = (
                f"Bearer {api_key}"  #  [oai_citation:0‡Guild Wars 2 Wiki](https://wiki.guildwars2.com/wiki/API%3AAPI_key?utm_source=chatgpt.com)
            )

        # rudimentary token bucket
        self._allowance = self.MAX_PER_MIN
        self._last_check = time.time()

    # ------------------- internals -------------------

    def _throttle(self):
        now = time.time()
        elapsed = now - self._last_check
        self._last_check = now
        # refill allowance
        self._allowance = min(
            self.MAX_PER_MIN, self._allowance + elapsed * (self.MAX_PER_MIN / 60.0)
        )
        if self._allowance < 1.0:
            sleep_for = (1.0 - self._allowance) * (60.0 / self.MAX_PER_MIN)
            time.sleep(max(0.0, sleep_for))
            self._allowance = 0
        else:
            self._allowance -= 1.0

    def _request(
        self, method: str, path: str, params: Optional[Dict[str, Any]] = None
    ) -> Any:
        url = f"{self.BASE}{path}"
        params = dict(params or {})
        # If no Authorization header, allow access_token=? as fallback
        if self.api_key and "Authorization" not in self.s.headers:
            params.setdefault("access_token", self.api_key)

        last_err = None
        for attempt in range(self.max_retries + 1):
            self._throttle()
            try:
                resp = self.s.request(method, url, params=params, timeout=self.timeout)
                if resp.status_code == 429:  # rate-limited
                    retry_after = float(resp.headers.get("Retry-After", "1"))
                    time.sleep(max(retry_after, self.backoff_base * (2**attempt)))
                    continue
                resp.raise_for_status()
                if not resp.content:
                    return None
                return resp.json()
            except requests.HTTPError as e:
                # For 5xx, retry with backoff; for 4xx (except 429 above), raise
                if 500 <= e.response.status_code < 600 and attempt < self.max_retries:
                    time.sleep(self.backoff_base * (2**attempt))
                    last_err = e
                    continue
                raise
            except (requests.ConnectionError, requests.Timeout) as e:
                if attempt < self.max_retries:
                    time.sleep(self.backoff_base * (2**attempt))
                    last_err = e
                    continue
                raise
        # if we somehow fall through
        raise RuntimeError(f"Request failed after retries: {last_err}")

    # ------------------- public: commerce/transactions -------------------
    # Structure per docs: /v2/commerce/transactions/{current|history}/{buys|sells}
    # Auth required. Cached for ~5 minutes.  [oai_citation:1‡Guild Wars 2 Wiki](https://wiki.guildwars2.com/wiki/API%3A2/commerce/transactions?utm_source=chatgpt.com)

    def transactions(
        self,
        scope: str,  # "current" or "history"
        side: str,  # "buys" or "sells"
        page: Optional[int] = None,
        page_size: Optional[int] = None,
    ) -> Any:
        """
        Fetch account transactions.

        Example:
            gw2.transactions("current", "buys")
            gw2.transactions("history", "sells", page=0, page_size=200)

        Notes:
          - Requires API key with Tradingpost scope.
          - Supports pagination on many responses (page, page_size up to ~200).   [oai_citation:2‡Guild Wars 2 Wiki](https://wiki.guildwars2.com/wiki/API%3ABest_practices?utm_source=chatgpt.com)
        """
        if scope not in {"current", "history"}:
            raise ValueError("scope must be 'current' or 'history'")
        if side not in {"buys", "sells"}:
            raise ValueError("side must be 'buys' or 'sells'")

        params: Dict[str, Any] = {}
        if page is not None:
            params["page"] = int(page)
        if page_size is not None:
            params["page_size"] = int(page_size)

        path = f"/commerce/transactions/{scope}/{side}"
        return self._request("GET", path, params=params)

    def transactions_current_buys(self, **kw) -> Any:
        return self.transactions("current", "buys", **kw)

    def transactions_current_sells(self, **kw) -> Any:
        return self.transactions("current", "sells", **kw)

    def transactions_history_buys(self, **kw) -> Any:
        return self.transactions("history", "buys", **kw)

    def transactions_history_sells(self, **kw) -> Any:
        return self.transactions("history", "sells", **kw)

    # ------------------- public: commerce/listings -------------------
    # Returns current order book for items.
    # Notes: /v2/commerce/listings/{id} returns object with buys/sells arrays.
    #        Bulk expansion via ?ids=1,2,... (max ~200 per request).   [oai_citation:3‡Guild Wars 2 Wiki](https://wiki.guildwars2.com/wiki/API%3A2/commerce/listings?utm_source=chatgpt.com)

    def listings_one(self, item_id: int) -> Dict[str, Any]:
        """
        Get full buy/sell listings for a single item id.
        """
        path = f"/commerce/listings/{int(item_id)}"
        return self._request("GET", path)

    def listings_many(self, ids: Iterable[int]) -> List[Dict[str, Any]]:
        """
        Get listings for up to 200 ids per request; splits automatically.
        """
        ids_list = [int(x) for x in ids]
        out: List[Dict[str, Any]] = []
        chunk_size = 200  # per API bulk best practices.  [oai_citation:4‡Guild Wars 2 Wiki](https://wiki.guildwars2.com/wiki/API%3ABest_practices?utm_source=chatgpt.com)
        for i in range(0, len(ids_list), chunk_size):
            chunk = ids_list[i : i + chunk_size]
            params = {"ids": ",".join(map(str, chunk))}
            out.extend(self._request("GET", "/commerce/listings", params=params))
        return out

    def listings_ids(self, page: int = 0, page_size: int = 200) -> List[int]:
        """
        Returns the list of item ids that have listings (paged).
        """
        params = {"page": int(page), "page_size": int(page_size)}
        data = self._request("GET", "/commerce/listings", params=params)
        # Per API conventions, root without id returns a list of ids (paged).   [oai_citation:5‡forum-en.gw2archive.eu](https://forum-en.gw2archive.eu/forum/community/api/API-v2-commerce-listings-and-paging?utm_source=chatgpt.com)
        return list(map(int, data or []))
