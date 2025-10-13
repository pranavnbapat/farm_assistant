import httpx
import logging
from typing import Dict, Any, List, Optional
from app.schemas import AskIn
from app.config import get_settings

logger = logging.getLogger("farm-assistant.search")
S = get_settings()

def build_search_payload(inp: AskIn) -> Dict[str, Any]:
    page = 1 if (inp.page is None or inp.page < 1) else inp.page
    default_k = 5

    payload: Dict[str, Any] = {
        "search_term": inp.question,
        "page": page,
        "dev": False,
        "model": "msmarco",
        "include_fulltext": True,
        "sort_by": "score_desc",
    }

    k_val = inp.k if inp.k is not None else default_k
    if isinstance(k_val, int) and k_val > 0:
        payload["k"] = k_val

    return payload

async def fetch_os_page(
    client: httpx.AsyncClient,
    base_payload: Dict[str, Any],
    page_no: int,
    headers: Dict[str, str],
    auth: Optional[httpx.Auth]
) -> Dict[str, Any]:
    payload = dict(base_payload)
    payload["page"] = page_no
    url = f"{S.OPENSEARCH_API_URL.rstrip('/')}{S.OS_API_PATH}"
    res = await client.post(url, json=payload, headers=headers, auth=auth)
    res.raise_for_status()
    return res.json()

async def collect_os_items(
    client: httpx.AsyncClient,
    base_payload: Dict[str, Any],
    pages: List[int],
    headers: Dict[str, str],
    auth: Optional[httpx.Auth]
) -> List[Dict[str, Any]]:
    all_items: List[Dict[str, Any]] = []
    seen_ids = set()
    for p in pages:
        os_json = await fetch_os_page(client, base_payload, p, headers, auth)
        items = (
            os_json.get("data")
            or os_json.get("results")
            or os_json.get("hits")
            or os_json.get("items")
            or []
        )
        for it in items:
            _id = (it.get("_id") if isinstance(it, dict) else None) or id(it)
            if _id in seen_ids:
                continue
            seen_ids.add(_id)
            all_items.append(it)
    logger.info(f"Collected total_items={len(all_items)} from pages={pages}")
    return all_items
