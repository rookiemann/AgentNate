"""
Data Tools - HTTP requests, JSON/HTML parsing, data conversion, database queries.
"""

from typing import Dict, Any, List, Optional, Union
import logging
import json
import re
import csv
import io
import sqlite3
from pathlib import Path

logger = logging.getLogger("tools.data")


TOOL_DEFINITIONS = [
    {
        "name": "http_request",
        "description": "Make an HTTP request to any API endpoint.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Request URL"
                },
                "method": {
                    "type": "string",
                    "description": "HTTP method",
                    "enum": ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"],
                    "default": "GET"
                },
                "headers": {
                    "type": "object",
                    "description": "Request headers"
                },
                "params": {
                    "type": "object",
                    "description": "URL query parameters"
                },
                "json_body": {
                    "type": "object",
                    "description": "JSON request body"
                },
                "body": {
                    "type": "string",
                    "description": "Raw request body (string)"
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (default 30)",
                    "default": 30
                },
                "follow_redirects": {
                    "type": "boolean",
                    "description": "Follow redirects (default true)",
                    "default": True
                }
            },
            "required": ["url"]
        }
    },
    {
        "name": "parse_json",
        "description": "Parse JSON and optionally extract data using JSONPath-like queries.",
        "parameters": {
            "type": "object",
            "properties": {
                "json_string": {
                    "type": "string",
                    "description": "JSON string to parse"
                },
                "query": {
                    "type": "string",
                    "description": "JSONPath-like query (e.g., 'data.items[0].name', 'results[*].id')"
                }
            },
            "required": ["json_string"]
        }
    },
    {
        "name": "parse_html",
        "description": "Parse HTML and extract data using CSS selectors.",
        "parameters": {
            "type": "object",
            "properties": {
                "html": {
                    "type": "string",
                    "description": "HTML string to parse"
                },
                "selector": {
                    "type": "string",
                    "description": "CSS selector to match elements"
                },
                "attribute": {
                    "type": "string",
                    "description": "Attribute to extract (default: text content)"
                },
                "multiple": {
                    "type": "boolean",
                    "description": "Return all matches (default true)",
                    "default": True
                }
            },
            "required": ["html", "selector"]
        }
    },
    {
        "name": "convert_data",
        "description": "Convert data between formats: JSON, CSV, YAML.",
        "parameters": {
            "type": "object",
            "properties": {
                "data": {
                    "type": "string",
                    "description": "Input data string"
                },
                "from_format": {
                    "type": "string",
                    "description": "Input format",
                    "enum": ["json", "csv", "yaml"]
                },
                "to_format": {
                    "type": "string",
                    "description": "Output format",
                    "enum": ["json", "csv", "yaml"]
                }
            },
            "required": ["data", "from_format", "to_format"]
        }
    },
    {
        "name": "database_query",
        "description": "Execute a SQL query on a SQLite database.",
        "parameters": {
            "type": "object",
            "properties": {
                "database": {
                    "type": "string",
                    "description": "Path to SQLite database file"
                },
                "query": {
                    "type": "string",
                    "description": "SQL query to execute"
                },
                "params": {
                    "type": "array",
                    "description": "Query parameters for parameterized queries",
                    "items": {}
                },
                "fetch": {
                    "type": "string",
                    "description": "Fetch mode: 'all', 'one', 'many' (for SELECT)",
                    "enum": ["all", "one", "many"],
                    "default": "all"
                },
                "limit": {
                    "type": "integer",
                    "description": "Limit rows for 'many' fetch (default 100)",
                    "default": 100
                }
            },
            "required": ["database", "query"]
        }
    },
]


class DataTools:
    """Tools for data manipulation and API calls."""

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize data tools.

        Args:
            config: Optional configuration dict
        """
        self.config = config or {}
        self.max_response_size = self.config.get("max_response_size", 100000)

    async def http_request(
        self,
        url: str,
        method: str = "GET",
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, str]] = None,
        json_body: Optional[Dict] = None,
        body: Optional[str] = None,
        timeout: int = 30,
        follow_redirects: bool = True
    ) -> Dict[str, Any]:
        """
        Make an HTTP request.

        Args:
            url: Request URL
            method: HTTP method
            headers: Request headers
            params: Query parameters
            json_body: JSON body
            body: Raw body string
            timeout: Timeout in seconds
            follow_redirects: Follow redirects

        Returns:
            Dict with response
        """
        try:
            import httpx
        except ImportError:
            return {"success": False, "error": "httpx not installed"}

        try:
            async with httpx.AsyncClient(
                timeout=timeout,
                follow_redirects=follow_redirects
            ) as client:
                kwargs = {
                    "method": method,
                    "url": url,
                    "headers": headers or {},
                    "params": params
                }

                if json_body is not None:
                    kwargs["json"] = json_body
                elif body is not None:
                    kwargs["content"] = body

                response = await client.request(**kwargs)

            # Parse response
            content_type = response.headers.get("content-type", "")

            # Try to parse as JSON
            try:
                response_data = response.json()
            except Exception:
                response_data = response.text[:self.max_response_size]
                if len(response.text) > self.max_response_size:
                    response_data += "\n\n[Response truncated...]"

            return {
                "success": response.status_code < 400,
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "content_type": content_type,
                "data": response_data
            }

        except httpx.TimeoutException:
            return {"success": False, "error": f"Request timed out after {timeout}s"}
        except Exception as e:
            logger.error(f"http_request error: {e}")
            return {"success": False, "error": str(e)}

    async def parse_json(
        self,
        json_string: str,
        query: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Parse JSON and optionally extract data with a query.

        Args:
            json_string: JSON string
            query: JSONPath-like query

        Returns:
            Dict with parsed data
        """
        try:
            data = json.loads(json_string)

            if not query:
                return {
                    "success": True,
                    "data": data
                }

            # Simple JSONPath-like extraction
            result = self._extract_json_path(data, query)

            return {
                "success": True,
                "query": query,
                "data": result
            }

        except json.JSONDecodeError as e:
            return {"success": False, "error": f"Invalid JSON: {e}"}
        except Exception as e:
            logger.error(f"parse_json error: {e}")
            return {"success": False, "error": str(e)}

    def _extract_json_path(self, data: Any, query: str) -> Any:
        """
        Simple JSONPath-like extraction.

        Supports:
        - data.field - Object property
        - data[0] - Array index
        - data[*] - All array items
        - data.items[*].name - Nested access
        """
        if not query:
            return data

        parts = re.split(r'\.|\[|\]', query)
        parts = [p for p in parts if p]  # Remove empty strings

        current = data

        for part in parts:
            if current is None:
                return None

            if part == '*':
                # Wildcard - return list
                if isinstance(current, list):
                    continue
                return None

            if part.isdigit():
                # Array index
                idx = int(part)
                if isinstance(current, list) and idx < len(current):
                    current = current[idx]
                else:
                    return None

            elif isinstance(current, list):
                # Apply to all items
                current = [
                    item.get(part) if isinstance(item, dict) else None
                    for item in current
                ]
                current = [c for c in current if c is not None]

            elif isinstance(current, dict):
                current = current.get(part)

            else:
                return None

        return current

    async def parse_html(
        self,
        html: str,
        selector: str,
        attribute: Optional[str] = None,
        multiple: bool = True
    ) -> Dict[str, Any]:
        """
        Parse HTML and extract data with CSS selectors.

        Args:
            html: HTML string
            selector: CSS selector
            attribute: Attribute to extract
            multiple: Return all matches

        Returns:
            Dict with extracted data
        """
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            return {"success": False, "error": "beautifulsoup4 not installed"}

        try:
            soup = BeautifulSoup(html, "html.parser")

            if multiple:
                elements = soup.select(selector)
            else:
                el = soup.select_one(selector)
                elements = [el] if el else []

            results = []
            for el in elements:
                if attribute:
                    value = el.get(attribute)
                else:
                    value = el.get_text(strip=True)
                results.append(value)

            return {
                "success": True,
                "selector": selector,
                "count": len(results),
                "data": results if multiple else (results[0] if results else None)
            }

        except Exception as e:
            logger.error(f"parse_html error: {e}")
            return {"success": False, "error": str(e)}

    async def convert_data(
        self,
        data: str,
        from_format: str,
        to_format: str
    ) -> Dict[str, Any]:
        """
        Convert data between formats.

        Args:
            data: Input data string
            from_format: Input format (json, csv, yaml)
            to_format: Output format (json, csv, yaml)

        Returns:
            Dict with converted data
        """
        try:
            # Parse input
            if from_format == "json":
                parsed = json.loads(data)

            elif from_format == "csv":
                reader = csv.DictReader(io.StringIO(data))
                parsed = list(reader)

            elif from_format == "yaml":
                try:
                    import yaml
                    parsed = yaml.safe_load(data)
                except ImportError:
                    return {"success": False, "error": "pyyaml not installed"}

            else:
                return {"success": False, "error": f"Unknown input format: {from_format}"}

            # Convert to output format
            if to_format == "json":
                output = json.dumps(parsed, indent=2)

            elif to_format == "csv":
                if not isinstance(parsed, list):
                    parsed = [parsed]

                if not parsed:
                    output = ""
                else:
                    # Get all keys
                    keys = set()
                    for item in parsed:
                        if isinstance(item, dict):
                            keys.update(item.keys())
                    keys = sorted(keys)

                    output_io = io.StringIO()
                    writer = csv.DictWriter(output_io, fieldnames=keys)
                    writer.writeheader()
                    for item in parsed:
                        if isinstance(item, dict):
                            writer.writerow(item)
                    output = output_io.getvalue()

            elif to_format == "yaml":
                try:
                    import yaml
                    output = yaml.dump(parsed, default_flow_style=False)
                except ImportError:
                    return {"success": False, "error": "pyyaml not installed"}

            else:
                return {"success": False, "error": f"Unknown output format: {to_format}"}

            return {
                "success": True,
                "from_format": from_format,
                "to_format": to_format,
                "data": output
            }

        except json.JSONDecodeError as e:
            return {"success": False, "error": f"Invalid JSON: {e}"}
        except Exception as e:
            logger.error(f"convert_data error: {e}")
            return {"success": False, "error": str(e)}

    async def database_query(
        self,
        database: str,
        query: str,
        params: Optional[List] = None,
        fetch: str = "all",
        limit: int = 100
    ) -> Dict[str, Any]:
        """
        Execute a SQL query on a SQLite database.

        Args:
            database: Path to SQLite database
            query: SQL query
            params: Query parameters
            fetch: Fetch mode
            limit: Row limit for 'many' fetch

        Returns:
            Dict with query results
        """
        try:
            db_path = Path(database)

            # Create database if it doesn't exist (for new databases)
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            try:
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)

                # Check if it's a SELECT query
                if query.strip().upper().startswith("SELECT"):
                    if fetch == "one":
                        row = cursor.fetchone()
                        data = dict(row) if row else None
                        count = 1 if row else 0
                    elif fetch == "many":
                        rows = cursor.fetchmany(limit)
                        data = [dict(row) for row in rows]
                        count = len(data)
                    else:  # all
                        rows = cursor.fetchall()
                        data = [dict(row) for row in rows]
                        count = len(data)

                    return {
                        "success": True,
                        "query_type": "SELECT",
                        "count": count,
                        "data": data
                    }

                else:
                    # For INSERT, UPDATE, DELETE
                    conn.commit()
                    return {
                        "success": True,
                        "query_type": query.strip().split()[0].upper(),
                        "rows_affected": cursor.rowcount,
                        "last_row_id": cursor.lastrowid
                    }

            finally:
                cursor.close()
                conn.close()

        except sqlite3.Error as e:
            return {"success": False, "error": f"SQLite error: {e}"}
        except Exception as e:
            logger.error(f"database_query error: {e}")
            return {"success": False, "error": str(e)}
