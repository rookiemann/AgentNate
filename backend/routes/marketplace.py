"""
Workflow Marketplace Routes

Fetches real n8n workflows from the official n8n template API (api.n8n.io).
Provides:
- Category listing with workflow counts
- Paginated workflow browsing with category filter
- Search functionality
- Individual workflow JSON (with full connections) for deployment
"""

import httpx
import logging
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException


def _sanitize_surrogates(obj):
    """Replace lone surrogate characters that break JSON/UTF-8 serialization."""
    if isinstance(obj, str):
        return obj.encode("utf-8", errors="replace").decode("utf-8")
    if isinstance(obj, dict):
        return {k: _sanitize_surrogates(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_surrogates(item) for item in obj]
    return obj

router = APIRouter()
logger = logging.getLogger("marketplace")

N8N_API_BASE = "https://api.n8n.io/api/templates"

CACHE_TTL_MINUTES = 30
_cache: Dict[str, Dict[str, Any]] = {}


def _get_cached(key: str) -> Optional[Any]:
    """Get cached data if not expired."""
    if key in _cache:
        entry = _cache[key]
        if datetime.now() < entry["expires"]:
            return entry["data"]
        else:
            del _cache[key]
    return None


def _set_cached(key: str, data: Any, ttl_minutes: int = CACHE_TTL_MINUTES):
    """Cache data with expiration."""
    _cache[key] = {
        "data": data,
        "expires": datetime.now() + timedelta(minutes=ttl_minutes),
    }


def _extract_node_type_short(full_type: str) -> str:
    """Extract short service name from n8n node type string."""
    # "@n8n/n8n-nodes-langchain.agent" -> "agent"
    # "n8n-nodes-base.httpRequest" -> "httpRequest"
    parts = full_type.rsplit(".", 1)
    return parts[-1] if len(parts) > 1 else full_type


def _detect_trigger_type(nodes: list) -> str:
    """Detect trigger type from node list."""
    for n in nodes:
        ntype = (n.get("type") or n.get("name") or "").lower()
        if "webhook" in ntype:
            return "webhook"
        if "cron" in ntype or "schedule" in ntype:
            return "scheduled"
        if "emailtrigger" in ntype or "emailreadimaptrigger" in ntype:
            return "triggered"
        if "chattrigger" in ntype:
            return "chat"
    return "manual"


def _estimate_complexity(node_count: int) -> str:
    """Estimate complexity from node count."""
    if node_count <= 5:
        return "Low"
    elif node_count <= 15:
        return "Medium"
    return "High"


def _transform_workflow_list_item(wf: dict) -> dict:
    """Transform n8n API workflow to our frontend format."""
    nodes = wf.get("nodes", [])
    node_names = [_extract_node_type_short(n.get("name", "")) for n in nodes]

    # Filter out generic types for integrations list
    skip = {"stickyNote", "noOp", "set", "code", "function", "if", "switch",
            "merge", "splitInBatches", "itemLists", "manualTrigger", "start",
            "httpRequest", "respondToWebhook"}
    integrations = [n for n in node_names if n and n not in skip][:6]

    # Detect trigger type from node types
    trigger_type = _detect_trigger_type(nodes)

    # Rough node count (search results only list key nodes, not all)
    node_count = len(nodes)
    complexity = _estimate_complexity(node_count)

    # Extract first category if available
    categories = wf.get("categories", [])
    category = categories[0]["name"] if categories else "Other"

    return {
        "id": str(wf.get("id", "")),
        "name": wf.get("name", "Untitled"),
        "description": wf.get("description", ""),
        "category": category,
        "complexity": complexity,
        "trigger_type": trigger_type,
        "node_count": node_count,
        "integrations": integrations,
        "tags": [],
        "totalViews": wf.get("totalViews", 0),
        "createdAt": wf.get("createdAt", ""),
        "user": wf.get("user", {}).get("username", ""),
    }


@router.get("/categories")
async def list_categories():
    """List all workflow categories with counts."""
    cached = _get_cached("categories")
    if cached:
        return cached

    async with httpx.AsyncClient(timeout=30) as client:
        try:
            resp = await client.get(
                f"{N8N_API_BASE}/search",
                params={"page": 1, "rows": 1, "search": "", "category": ""},
            )
            if resp.status_code != 200:
                raise HTTPException(status_code=502, detail=f"n8n API error: {resp.status_code}")

            data = resp.json()
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"n8n API error: {e}")

    total = data.get("totalWorkflows", 0)

    # Extract categories from filters
    categories = []
    for f in data.get("filters", []):
        counts = f.get("counts", [])
        if counts and any("AI" in c.get("value", "") for c in counts[:3]):
            for c in counts:
                categories.append({
                    "name": c.get("value", "Unknown"),
                    "count": c.get("count", 0),
                })
            break

    result = {
        "categories": categories,
        "count": len(categories),
        "total_workflows": total,
    }
    _set_cached("categories", result, ttl_minutes=60)
    return result


@router.get("/workflows")
async def list_workflows(
    category: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
):
    """List workflows with optional category filter."""
    page = (offset // limit) + 1 if limit > 0 else 1
    rows = min(limit, 100)

    cache_key = f"workflows_{category or 'all'}_{page}_{rows}"
    cached = _get_cached(cache_key)
    if cached:
        return cached

    params = {"page": page, "rows": rows, "search": ""}
    if category:
        params["category"] = category

    async with httpx.AsyncClient(timeout=30) as client:
        try:
            resp = await client.get(f"{N8N_API_BASE}/search", params=params)
            if resp.status_code != 200:
                raise HTTPException(status_code=502, detail=f"n8n API error: {resp.status_code}")
            data = _sanitize_surrogates(resp.json())
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"n8n API error: {e}")

    workflows = [_transform_workflow_list_item(w) for w in data.get("workflows", [])]
    total = data.get("totalWorkflows", 0)

    result = {
        "workflows": workflows,
        "count": len(workflows),
        "total": total,
        "offset": offset,
        "limit": limit,
    }
    _set_cached(cache_key, result, ttl_minutes=15)
    return result


@router.get("/workflows/{category}")
async def list_category_workflows(category: str, limit: int = 100, offset: int = 0):
    """List all workflows in a category."""
    return await list_workflows(category=category, limit=limit, offset=offset)


@router.get("/workflow/{workflow_id}")
async def get_workflow(workflow_id: str):
    """
    Get a single workflow by ID.

    Fetches full workflow JSON (with connections) from n8n API.
    """
    cache_key = f"workflow_json_{workflow_id}"
    cached = _get_cached(cache_key)
    if cached:
        return cached

    async with httpx.AsyncClient(timeout=30) as client:
        try:
            resp = await client.get(
                f"{N8N_API_BASE}/workflows/{workflow_id}",
            )
            if resp.status_code == 404:
                raise HTTPException(status_code=404, detail=f"Workflow not found: {workflow_id}")
            if resp.status_code != 200:
                raise HTTPException(status_code=502, detail=f"n8n API error: {resp.status_code}")

            data = resp.json()
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"n8n API error: {e}")

    wf_meta = _sanitize_surrogates(data.get("workflow", {}))
    workflow_json = wf_meta.get("workflow", {})

    if not workflow_json or not workflow_json.get("nodes"):
        raise HTTPException(status_code=502, detail="Workflow JSON empty or missing nodes")

    # Extract metadata
    nodes = workflow_json.get("nodes", [])
    categories = wf_meta.get("categories", [])
    category = categories[0]["name"] if categories else "Other"

    node_types = [_extract_node_type_short(n.get("type", "")) for n in nodes]
    skip = {"stickyNote", "noOp", "set", "code", "function", "if", "switch",
            "merge", "splitInBatches", "itemLists", "manualTrigger", "start"}
    integrations = list(dict.fromkeys(n for n in node_types if n and n not in skip))[:8]

    result = {
        "workflow": {
            "json": workflow_json,
            "metadata": {
                "id": str(wf_meta.get("id", workflow_id)),
                "name": wf_meta.get("name", "Untitled"),
                "description": wf_meta.get("description", ""),
                "category": category,
                "complexity": _estimate_complexity(len(nodes)),
                "trigger_type": _detect_trigger_type(nodes),
                "node_count": len(nodes),
                "integrations": integrations,
                "tags": [],
                "totalViews": wf_meta.get("totalViews", 0),
                "user": wf_meta.get("user", {}).get("username", ""),
            }
        }
    }
    _set_cached(cache_key, result, ttl_minutes=60)
    return result


@router.get("/search")
async def search_workflows(q: str, category: Optional[str] = None, limit: int = 50):
    """Search workflows by name or description."""
    if not q or len(q) < 2:
        raise HTTPException(status_code=400, detail="Search query must be at least 2 characters")

    rows = min(limit, 100)
    params = {"page": 1, "rows": rows, "search": q}
    if category:
        params["category"] = category

    cache_key = f"search_{q}_{category or ''}_{rows}"
    cached = _get_cached(cache_key)
    if cached:
        return cached

    async with httpx.AsyncClient(timeout=30) as client:
        try:
            resp = await client.get(f"{N8N_API_BASE}/search", params=params)
            if resp.status_code != 200:
                raise HTTPException(status_code=502, detail=f"n8n API error: {resp.status_code}")
            data = _sanitize_surrogates(resp.json())
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"n8n API error: {e}")

    results = [_transform_workflow_list_item(w) for w in data.get("workflows", [])]

    result = {
        "query": q,
        "results": results,
        "count": len(results),
    }
    _set_cached(cache_key, result, ttl_minutes=10)
    return result


@router.get("/stats")
async def get_marketplace_stats():
    """Get marketplace statistics."""
    cached = _get_cached("categories")
    if cached:
        return {
            "total_workflows": cached.get("total_workflows", 0),
            "categories": cached.get("count", 0),
            "source": "api.n8n.io",
            "cache_entries": len(_cache),
        }

    # Fetch fresh
    try:
        cats = await list_categories()
        return {
            "total_workflows": cats.get("total_workflows", 0),
            "categories": cats.get("count", 0),
            "source": "api.n8n.io",
            "cache_entries": len(_cache),
        }
    except Exception:
        return {
            "error": "Failed to fetch stats",
            "cache_entries": len(_cache),
        }


@router.delete("/cache")
async def clear_cache():
    """Clear the marketplace cache."""
    global _cache
    count = len(_cache)
    _cache = {}
    return {"success": True, "cleared": count}
