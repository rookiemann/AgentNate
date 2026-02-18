"""
Workflow Templates Library

Contains n8n workflow node builders with parameter schemas.
Each NODE_REGISTRY entry has: category, params (schema), builder (lambda).
"""

from typing import Dict, List, Any, Optional
import uuid


def node_id() -> str:
    """Generate a unique node ID."""
    return str(uuid.uuid4())


# ============================================================
# HELPER FUNCTIONS (used by builders)
# ============================================================

def _build_http_params(params: Dict) -> Dict:
    """Build HTTP request parameters."""
    result = {
        "method": params.get("method", "GET"),
        "url": params.get("url", "")
    }

    if params.get("headers"):
        result["sendHeaders"] = True
        result["headerParameters"] = {
            "parameters": [
                {"name": k, "value": v} for k, v in params.get("headers", {}).items()
            ]
        }

    if params.get("body"):
        result["sendBody"] = True
        result["specifyBody"] = "json"
        if isinstance(params.get("body"), str):
            result["jsonBody"] = params["body"]
        else:
            import json
            result["jsonBody"] = json.dumps(params["body"])

    if params.get("response_format") == "file":
        result["options"] = {
            "response": {
                "response": {
                    "responseFormat": "file"
                }
            }
        }

    return result


def _build_llm_body(params: Dict) -> str:
    """Build LLM request body using n8n expression syntax.

    n8n expressions: ={{ js_expression }} — no nested {{ }}.
    Use JSON.stringify() so $json references resolve at runtime.
    """
    prompt_field = params.get("prompt_field", "input")
    system_prompt = params.get("system_prompt", "")
    max_tokens = params.get("max_tokens", 1024)

    # Escape quotes in system prompt for JS string
    sys_escaped = system_prompt.replace("\\", "\\\\").replace('"', '\\"')

    if system_prompt:
        return (
            f'={{{{ JSON.stringify({{"messages": ['
            f'{{"role": "system", "content": "{sys_escaped}"}},'
            f'{{"role": "user", "content": $json.{prompt_field} || ""}}'
            f'], "max_tokens": {max_tokens}, "stream": false}}) }}}}'
        )
    else:
        return (
            f'={{{{ JSON.stringify({{"messages": ['
            f'{{"role": "user", "content": $json.{prompt_field} || ""}}'
            f'], "max_tokens": {max_tokens}, "stream": false}}) }}}}'
        )


def _build_summarize_body(params: Dict) -> str:
    """Build summarize request body."""
    input_field = params.get("input_field", "text")
    return (
        f'={{{{ JSON.stringify({{"messages": ['
        f'{{"role": "system", "content": "Summarize the following text concisely."}},'
        f'{{"role": "user", "content": $json.{input_field} || ""}}'
        f'], "max_tokens": 500}}) }}}}'
    )


def _build_classify_body(params: Dict) -> str:
    """Build classify request body."""
    input_field = params.get("input_field", "text")
    categories = params.get("categories", "positive, negative, neutral")
    cat_escaped = categories.replace("\\", "\\\\").replace('"', '\\"')
    return (
        f'={{{{ JSON.stringify({{"messages": ['
        f'{{"role": "system", "content": "Classify the text into one of these categories: {cat_escaped}. Reply with only the category name."}},'
        f'{{"role": "user", "content": $json.{input_field} || ""}}'
        f'], "max_tokens": 20}}) }}}}'
    )


def _build_if_node(params: Dict) -> Dict:
    """Build IF node with auto-detected value types (string/number/boolean)."""
    compare_value = params.get("compare_value", "")
    value_type = params.get("value_type", "string")

    # Auto-detect type from compare_value if not explicitly set
    if value_type == "string":
        cv_lower = str(compare_value).lower().strip()
        if cv_lower in ("true", "false"):
            value_type = "boolean"
        elif cv_lower.replace(".", "", 1).replace("-", "", 1).isdigit():
            value_type = "number"

    # Build operator based on type
    operation = params.get("operation", "equals")
    if value_type == "boolean":
        right_value = str(compare_value).lower().strip() == "true"
        operator = {"type": "boolean", "operation": operation}
    elif value_type == "number":
        try:
            right_value = float(compare_value)
            if right_value == int(right_value):
                right_value = int(right_value)
        except (ValueError, TypeError):
            right_value = 0
        operator = {"type": "number", "operation": operation}
    else:
        right_value = str(compare_value)
        operator = {"type": "string", "operation": operation}

    return {
        "id": node_id(),
        "name": params.get("name", "IF"),
        "type": "n8n-nodes-base.if",
        "typeVersion": 2,
        "parameters": {
            "conditions": {
                "options": {
                    "caseSensitive": True,
                    "leftValue": "",
                    "typeValidation": "loose"
                },
                "conditions": [
                    {
                        "leftValue": f"={{{{ $json.{params.get('field', 'value')} }}}}",
                        "rightValue": right_value,
                        "operator": operator
                    }
                ],
                "combinator": "and"
            }
        }
    }


# ============================================================
# NODE REGISTRY - Single source of truth for all node types
# Each entry has: category, params (schema), builder (lambda)
# ============================================================

NODE_REGISTRY = {
    # === TRIGGERS ===
    "manual_trigger": {
        "category": "trigger",
        "params": {
            "name": {"type": "string", "required": False, "default": "Manual Trigger", "description": "Node display name"},
        },
        "builder": lambda params: {
            "id": node_id(),
            "name": params.get("name", "Manual Trigger"),
            "type": "n8n-nodes-base.manualTrigger",
            "typeVersion": 1,
            "parameters": {}
        }
    },
    "webhook": {
        "category": "trigger",
        "params": {
            "path": {"type": "string", "required": False, "default": "webhook", "description": "URL path for the webhook"},
            "method": {"type": "string", "required": False, "default": "POST", "description": "HTTP method", "options": ["GET", "POST", "PUT", "DELETE", "PATCH"]},
            "response_mode": {"type": "string", "required": False, "default": "onReceived", "description": "When to respond", "options": ["onReceived", "lastNode", "responseNode"]},
        },
        "builder": lambda params: {
            "id": node_id(),
            "name": params.get("name", "Webhook"),
            "type": "n8n-nodes-base.webhook",
            "typeVersion": 1,
            "webhookId": node_id(),
            "parameters": {
                "path": params.get("path", "webhook"),
                "httpMethod": params.get("method", "POST"),
                "responseMode": params.get("response_mode", "onReceived"),
                "responseData": "allEntries"
            }
        }
    },
    "schedule": {
        "category": "trigger",
        "params": {
            "cron": {"type": "string", "required": False, "default": "0 9 * * *", "description": "Cron expression (e.g. '0 9 * * *' = daily 9am, '*/5 * * * *' = every 5min)"},
        },
        "builder": lambda params: {
            "id": node_id(),
            "name": params.get("name", "Schedule"),
            "type": "n8n-nodes-base.scheduleTrigger",
            "typeVersion": 1,
            "parameters": {
                "rule": {
                    "interval": [{"field": "cronExpression", "expression": params.get("cron", "0 9 * * *")}]
                }
            }
        }
    },
    "email_trigger": {
        "category": "trigger",
        "params": {
            "mailbox": {"type": "string", "required": False, "default": "INBOX", "description": "Mailbox to monitor"},
        },
        "builder": lambda params: {
            "id": node_id(),
            "name": params.get("name", "Email Trigger"),
            "type": "n8n-nodes-base.emailReadImap",
            "typeVersion": 2,
            "parameters": {
                "mailbox": params.get("mailbox", "INBOX"),
                "options": {}
            }
        }
    },
    "error_trigger": {
        "category": "trigger",
        "params": {},
        "builder": lambda params: {
            "id": node_id(),
            "name": params.get("name", "Error Trigger"),
            "type": "n8n-nodes-base.errorTrigger",
            "typeVersion": 1,
            "parameters": {}
        }
    },

    # === HTTP / API ===
    "http_request": {
        "category": "action",
        "params": {
            "url": {"type": "string", "required": True, "description": "URL to request"},
            "method": {"type": "string", "required": False, "default": "GET", "description": "HTTP method", "options": ["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD"]},
            "headers": {"type": "object", "required": False, "description": "Request headers as {name: value} dict"},
            "body": {"type": "string|object", "required": False, "description": "Request body (string or JSON object)"},
            "response_format": {"type": "string", "required": False, "description": "Response format", "options": ["json", "file"]},
        },
        "builder": lambda params: {
            "id": node_id(),
            "name": params.get("name", "HTTP Request"),
            "type": "n8n-nodes-base.httpRequest",
            "typeVersion": 4.2,
            "parameters": _build_http_params(params)
        }
    },
    "http_request_file": {
        "category": "action",
        "params": {
            "url": {"type": "string", "required": True, "description": "URL to download file from"},
            "method": {"type": "string", "required": False, "default": "GET", "description": "HTTP method"},
        },
        "builder": lambda params: {
            "id": node_id(),
            "name": params.get("name", "Fetch File"),
            "type": "n8n-nodes-base.httpRequest",
            "typeVersion": 4.2,
            "parameters": {
                "method": params.get("method", "GET"),
                "url": params.get("url", ""),
                "options": {
                    "response": {
                        "response": {
                            "responseFormat": "file"
                        }
                    }
                }
            }
        }
    },

    # === FILE OPERATIONS ===
    "write_file": {
        "category": "data",
        "params": {
            "file_path": {"type": "string", "required": True, "description": "Full path to write file to"},
            "binary_field": {"type": "string", "required": False, "default": "data", "description": "Binary property name containing data to write"},
        },
        "builder": lambda params: {
            "id": node_id(),
            "name": params.get("name", "Write File"),
            "type": "n8n-nodes-base.readWriteFile",
            "typeVersion": 1,
            "parameters": {
                "operation": "write",
                "fileName": params.get("file_path", params.get("path", "output.txt")),
                "dataPropertyName": params.get("binary_field", "data")
            }
        }
    },
    "read_file": {
        "category": "data",
        "params": {
            "file_path": {"type": "string", "required": True, "description": "Full path to read file from"},
        },
        "builder": lambda params: {
            "id": node_id(),
            "name": params.get("name", "Read File"),
            "type": "n8n-nodes-base.readWriteFile",
            "typeVersion": 1,
            "parameters": {
                "operation": "read",
                "fileSelector": params.get("file_path", params.get("path", "input.txt"))
            }
        }
    },

    # === LLM / AI ===
    "local_llm": {
        "category": "ai",
        "params": {
            "prompt_field": {"type": "string", "required": False, "default": "input", "description": "JSON field containing the user prompt (from previous node's output)"},
            "system_prompt": {"type": "string", "required": False, "default": "", "description": "System prompt to set LLM behavior"},
            "max_tokens": {"type": "integer", "required": False, "default": 1024, "description": "Max tokens in response"},
        },
        "builder": lambda params: {
            "id": node_id(),
            "name": params.get("name", "Local LLM"),
            "type": "n8n-nodes-base.httpRequest",
            "typeVersion": 4.2,
            "parameters": {
                "method": "POST",
                "url": "http://127.0.0.1:8000/v1/chat/completions",
                "sendHeaders": True,
                "headerParameters": {
                    "parameters": [{"name": "Content-Type", "value": "application/json"}]
                },
                "sendBody": True,
                "specifyBody": "json",
                "jsonBody": _build_llm_body(params)
            }
        }
    },
    "llm_summarize": {
        "category": "ai",
        "params": {
            "input_field": {"type": "string", "required": False, "default": "text", "description": "JSON field containing text to summarize"},
        },
        "builder": lambda params: {
            "id": node_id(),
            "name": params.get("name", "Summarize"),
            "type": "n8n-nodes-base.httpRequest",
            "typeVersion": 4.2,
            "parameters": {
                "method": "POST",
                "url": "http://127.0.0.1:8000/v1/chat/completions",
                "sendHeaders": True,
                "headerParameters": {
                    "parameters": [{"name": "Content-Type", "value": "application/json"}]
                },
                "sendBody": True,
                "specifyBody": "json",
                "jsonBody": _build_summarize_body(params)
            }
        }
    },
    "llm_classify": {
        "category": "ai",
        "params": {
            "input_field": {"type": "string", "required": False, "default": "text", "description": "JSON field containing text to classify"},
            "categories": {"type": "string", "required": False, "default": "positive, negative, neutral", "description": "Comma-separated list of categories"},
        },
        "builder": lambda params: {
            "id": node_id(),
            "name": params.get("name", "Classify"),
            "type": "n8n-nodes-base.httpRequest",
            "typeVersion": 4.2,
            "parameters": {
                "method": "POST",
                "url": "http://127.0.0.1:8000/v1/chat/completions",
                "sendHeaders": True,
                "headerParameters": {
                    "parameters": [{"name": "Content-Type", "value": "application/json"}]
                },
                "sendBody": True,
                "specifyBody": "json",
                "jsonBody": _build_classify_body(params)
            }
        }
    },

    # === DATA PROCESSING ===
    "set_field": {
        "category": "data",
        "params": {
            "field": {"type": "string", "required": True, "description": "Field name to set"},
            "value": {"type": "string", "required": True, "description": "Value to assign (can use n8n expressions like ={{ $json.x }})"},
        },
        "builder": lambda params: {
            "id": node_id(),
            "name": params.get("name", "Set Data"),
            "type": "n8n-nodes-base.set",
            "typeVersion": 3,
            "parameters": {
                "mode": "manual",
                "duplicateItem": False,
                "assignments": {
                    "assignments": [
                        {
                            "id": node_id(),
                            "name": params.get("field", "output"),
                            "value": params.get("value", ""),
                            "type": "string"
                        }
                    ]
                }
            }
        }
    },
    "code": {
        "category": "data",
        "params": {
            "code": {"type": "string", "required": True, "description": "JavaScript code. MUST return items array. No require() or Node.js modules (sandboxed)."},
            "mode": {"type": "string", "required": False, "default": "runOnceForAllItems", "description": "Execution mode", "options": ["runOnceForAllItems", "runOnceForEachItem"]},
        },
        "builder": lambda params: {
            "id": node_id(),
            "name": params.get("name", "Code"),
            "type": "n8n-nodes-base.code",
            "typeVersion": 2,
            "parameters": {
                "mode": params.get("mode", "runOnceForAllItems"),
                "jsCode": params.get("code", "return items;")
            }
        }
    },
    "if": {
        "category": "flow",
        "params": {
            "field": {"type": "string", "required": True, "description": "JSON field to evaluate (e.g. 'status')"},
            "operation": {"type": "string", "required": False, "default": "equals", "description": "Comparison operation", "options": ["equals", "notEquals", "contains", "notContains", "startsWith", "endsWith", "regex", "exists", "notExists"]},
            "compare_value": {"type": "string", "required": False, "default": "", "description": "Value to compare against (use 'true'/'false' for booleans, numbers for numeric comparisons)"},
            "value_type": {"type": "string", "required": False, "default": "string", "description": "Type of comparison", "options": ["string", "number", "boolean"]},
        },
        "note": "Has 2 outputs: output 0 = true branch, output 1 = false branch. Use custom connections to wire both.",
        "builder": lambda params: _build_if_node(params)
    },
    "parse_json": {
        "category": "data",
        "params": {
            "input_field": {"type": "string", "required": False, "default": "data", "description": "JSON field containing the JSON string to parse"},
        },
        "builder": lambda params: {
            "id": node_id(),
            "name": params.get("name", "Parse JSON"),
            "type": "n8n-nodes-base.code",
            "typeVersion": 2,
            "parameters": {
                "mode": "runOnceForEachItem",
                "jsCode": f"const parsed = JSON.parse($json.{params.get('input_field', 'data')});\nreturn {{ json: {{ ...parsed }} }};"
            }
        }
    },
    "html_extract": {
        "category": "data",
        "params": {
            "input_field": {"type": "string", "required": False, "default": "data", "description": "JSON field containing HTML"},
            "selector": {"type": "string", "required": False, "default": "body", "description": "CSS selector to extract"},
            "output_key": {"type": "string", "required": False, "default": "extracted", "description": "Key name for extracted content"},
            "return_type": {"type": "string", "required": False, "default": "text", "description": "What to return", "options": ["text", "html", "value"]},
        },
        "builder": lambda params: {
            "id": node_id(),
            "name": params.get("name", "HTML Extract"),
            "type": "n8n-nodes-base.html",
            "typeVersion": 1,
            "parameters": {
                "operation": "extractHtmlContent",
                "sourceData": "json",
                "jsonProperty": params.get("input_field", "data"),
                "extractionValues": {
                    "values": [
                        {
                            "key": params.get("output_key", "extracted"),
                            "cssSelector": params.get("selector", "body"),
                            "returnValue": params.get("return_type", "text")
                        }
                    ]
                },
                "options": {}
            }
        }
    },
    "xml": {
        "category": "data",
        "params": {
            "mode": {"type": "string", "required": False, "default": "xmlToJson", "description": "Conversion mode", "options": ["xmlToJson", "jsonToXml"]},
            "input_field": {"type": "string", "required": False, "default": "data", "description": "Field containing XML/JSON data"},
        },
        "builder": lambda params: {
            "id": node_id(),
            "name": params.get("name", "XML"),
            "type": "n8n-nodes-base.xml",
            "typeVersion": 1,
            "parameters": {
                "mode": params.get("mode", "xmlToJson"),
                "dataPropertyName": params.get("input_field", "data"),
                "options": {}
            }
        }
    },
    "spreadsheet": {
        "category": "data",
        "params": {
            "operation": {"type": "string", "required": False, "default": "fromFile", "description": "Operation", "options": ["fromFile", "toFile"]},
            "format": {"type": "string", "required": False, "default": "csv", "description": "File format", "options": ["csv", "xlsx", "ods"]},
        },
        "builder": lambda params: {
            "id": node_id(),
            "name": params.get("name", "Spreadsheet"),
            "type": "n8n-nodes-base.spreadsheetFile",
            "typeVersion": 2,
            "parameters": {
                "operation": params.get("operation", "fromFile"),
                "fileFormat": params.get("format", "csv"),
                "options": {}
            }
        }
    },
    "crypto": {
        "category": "data",
        "params": {
            "action": {"type": "string", "required": False, "default": "hash", "description": "Crypto action", "options": ["hash", "hmac", "sign"]},
            "algorithm": {"type": "string", "required": False, "default": "SHA256", "description": "Algorithm", "options": ["MD5", "SHA256", "SHA384", "SHA512"]},
            "input_field": {"type": "string", "required": False, "default": "data", "description": "Field to hash"},
            "encoding": {"type": "string", "required": False, "default": "hex", "description": "Output encoding", "options": ["hex", "base64"]},
        },
        "builder": lambda params: {
            "id": node_id(),
            "name": params.get("name", "Crypto"),
            "type": "n8n-nodes-base.crypto",
            "typeVersion": 1,
            "parameters": {
                "action": params.get("action", "hash"),
                "type": params.get("algorithm", "SHA256"),
                "value": f"={{{{ $json.{params.get('input_field', 'data')} }}}}",
                "encoding": params.get("encoding", "hex")
            }
        }
    },
    "date_time": {
        "category": "data",
        "params": {
            "operation": {"type": "string", "required": False, "default": "formatDate", "description": "Operation", "options": ["formatDate", "addToDate", "subtractFromDate"]},
            "input_field": {"type": "string", "required": False, "default": "date", "description": "Field containing the date"},
            "format": {"type": "string", "required": False, "default": "yyyy-MM-dd HH:mm:ss", "description": "Date format string"},
        },
        "builder": lambda params: {
            "id": node_id(),
            "name": params.get("name", "Date & Time"),
            "type": "n8n-nodes-base.dateTime",
            "typeVersion": 2,
            "parameters": {
                "operation": params.get("operation", "formatDate"),
                "date": f"={{{{ $json.{params.get('input_field', 'date')} }}}}",
                "format": params.get("format", "yyyy-MM-dd HH:mm:ss"),
                "options": {}
            }
        }
    },
    "rename_keys": {
        "category": "data",
        "params": {
            "old_key": {"type": "string", "required": True, "description": "Current key name to rename"},
            "new_key": {"type": "string", "required": True, "description": "New key name"},
        },
        "builder": lambda params: {
            "id": node_id(),
            "name": params.get("name", "Rename Keys"),
            "type": "n8n-nodes-base.renameKeys",
            "typeVersion": 1,
            "parameters": {
                "keys": {
                    "key": [
                        {
                            "currentKey": params.get("old_key", "oldName"),
                            "newKey": params.get("new_key", "newName")
                        }
                    ]
                }
            }
        }
    },
    "filter": {
        "category": "data",
        "params": {
            "field": {"type": "string", "required": True, "description": "Field to filter on"},
            "operation": {"type": "string", "required": False, "default": "equals", "description": "Comparison operation", "options": ["equals", "notEquals", "contains", "notContains", "startsWith", "endsWith", "regex", "exists", "notExists"]},
            "compare_value": {"type": "string", "required": False, "default": "", "description": "Value to compare against"},
        },
        "builder": lambda params: {
            "id": node_id(),
            "name": params.get("name", "Filter"),
            "type": "n8n-nodes-base.filter",
            "typeVersion": 2,
            "parameters": {
                "conditions": {
                    "options": {"caseSensitive": True, "leftValue": "", "typeValidation": "strict"},
                    "conditions": [
                        {
                            "leftValue": f"={{{{ $json.{params.get('field', 'value')} }}}}",
                            "rightValue": params.get("compare_value", ""),
                            "operator": {"type": "string", "operation": params.get("operation", "equals")}
                        }
                    ],
                    "combinator": "and"
                },
                "options": {}
            }
        }
    },
    "limit": {
        "category": "data",
        "params": {
            "max_items": {"type": "integer", "required": False, "default": 10, "description": "Maximum number of items"},
            "keep": {"type": "string", "required": False, "default": "firstItems", "description": "Which items to keep", "options": ["firstItems", "lastItems"]},
        },
        "builder": lambda params: {
            "id": node_id(),
            "name": params.get("name", "Limit"),
            "type": "n8n-nodes-base.limit",
            "typeVersion": 1,
            "parameters": {
                "maxItems": params.get("max_items", 10),
                "keep": params.get("keep", "firstItems")
            }
        }
    },
    "sort": {
        "category": "data",
        "params": {
            "field": {"type": "string", "required": True, "description": "Field to sort by"},
            "order": {"type": "string", "required": False, "default": "ascending", "description": "Sort order", "options": ["ascending", "descending"]},
        },
        "builder": lambda params: {
            "id": node_id(),
            "name": params.get("name", "Sort"),
            "type": "n8n-nodes-base.sort",
            "typeVersion": 1,
            "parameters": {
                "sortFieldsUi": {
                    "sortField": [
                        {
                            "fieldName": params.get("field", "name"),
                            "order": params.get("order", "ascending")
                        }
                    ]
                },
                "options": {}
            }
        }
    },
    "remove_duplicates": {
        "category": "data",
        "params": {
            "field": {"type": "string", "required": True, "description": "Field to check for duplicates"},
        },
        "builder": lambda params: {
            "id": node_id(),
            "name": params.get("name", "Remove Duplicates"),
            "type": "n8n-nodes-base.removeDuplicates",
            "typeVersion": 1,
            "parameters": {
                "compare": "selectedFields",
                "fieldsToCompare": {
                    "fields": [{"fieldName": params.get("field", "id")}]
                },
                "options": {}
            }
        }
    },
    "split_out": {
        "category": "data",
        "params": {
            "field": {"type": "string", "required": True, "description": "Array field to split into separate items"},
        },
        "builder": lambda params: {
            "id": node_id(),
            "name": params.get("name", "Split Out"),
            "type": "n8n-nodes-base.splitOut",
            "typeVersion": 1,
            "parameters": {
                "fieldToSplitOut": params.get("field", "items"),
                "options": {}
            }
        }
    },
    "aggregate": {
        "category": "data",
        "params": {},
        "builder": lambda params: {
            "id": node_id(),
            "name": params.get("name", "Aggregate"),
            "type": "n8n-nodes-base.aggregate",
            "typeVersion": 1,
            "parameters": {
                "aggregate": "aggregateAllItemData",
                "options": {}
            }
        }
    },
    "html_to_markdown": {
        "category": "data",
        "params": {
            "html_field": {"type": "string", "required": False, "default": "html", "description": "Field containing HTML to convert"},
        },
        "builder": lambda params: {
            "id": node_id(),
            "name": params.get("name", "HTML to Markdown"),
            "type": "n8n-nodes-base.markdown",
            "typeVersion": 1,
            "parameters": {
                "mode": "htmlToMarkdown",
                "html": f"={{{{ $json.{params.get('html_field', 'html')} }}}}"
            }
        }
    },
    "markdown_to_html": {
        "category": "data",
        "params": {
            "markdown_field": {"type": "string", "required": False, "default": "markdown", "description": "Field containing Markdown to convert"},
        },
        "builder": lambda params: {
            "id": node_id(),
            "name": params.get("name", "Markdown to HTML"),
            "type": "n8n-nodes-base.markdown",
            "typeVersion": 1,
            "parameters": {
                "mode": "markdownToHtml",
                "markdown": f"={{{{ $json.{params.get('markdown_field', 'markdown')} }}}}"
            }
        }
    },
    "compare_datasets": {
        "category": "data",
        "params": {
            "field1": {"type": "string", "required": False, "default": "id", "description": "Field from first input to match on"},
            "field2": {"type": "string", "required": False, "default": "id", "description": "Field from second input to match on"},
        },
        "builder": lambda params: {
            "id": node_id(),
            "name": params.get("name", "Compare Datasets"),
            "type": "n8n-nodes-base.compareDatasets",
            "typeVersion": 1,
            "parameters": {
                "mergeByFields": {
                    "values": [{"field1": params.get("field1", "id"), "field2": params.get("field2", "id")}]
                },
                "options": {}
            }
        }
    },
    "summarize": {
        "category": "data",
        "params": {
            "fields": {"type": "array", "required": False, "default": [{"field": "value", "aggregation": "sum"}], "description": "Fields to summarize with aggregation type (sum, average, min, max, count)"},
        },
        "builder": lambda params: {
            "id": node_id(),
            "name": params.get("name", "Summarize"),
            "type": "n8n-nodes-base.summarize",
            "typeVersion": 1,
            "parameters": {
                "fieldsToSummarize": {
                    "values": params.get("fields", [{"field": "value", "aggregation": "sum"}])
                },
                "options": {}
            }
        }
    },
    "item_lists": {
        "category": "data",
        "params": {
            "operation": {"type": "string", "required": False, "default": "concatenateItems", "description": "Operation", "options": ["concatenateItems", "limit", "removeDuplicates", "sort", "summarize"]},
        },
        "builder": lambda params: {
            "id": node_id(),
            "name": params.get("name", "Item Lists"),
            "type": "n8n-nodes-base.itemLists",
            "typeVersion": 3.1,
            "parameters": {
                "operation": params.get("operation", "concatenateItems"),
                "options": {}
            }
        }
    },
    "convert": {
        "category": "data",
        "params": {
            "operation": {"type": "string", "required": False, "default": "toJson", "description": "Convert operation", "options": ["toJson", "toCsv", "toHtml", "toXml"]},
        },
        "builder": lambda params: {
            "id": node_id(),
            "name": params.get("name", "Convert to File"),
            "type": "n8n-nodes-base.convertToFile",
            "typeVersion": 1.1,
            "parameters": {
                "operation": params.get("operation", "toJson"),
                "options": {}
            }
        }
    },

    # === MESSAGING ===
    "discord_webhook": {
        "category": "messaging",
        "params": {
            "webhook_url": {"type": "string", "required": False, "default": "={{ $env.DISCORD_WEBHOOK }}", "description": "Discord webhook URL (or use env var)"},
            "message_field": {"type": "string", "required": False, "default": "content", "description": "JSON field containing the message text"},
        },
        "builder": lambda params: {
            "id": node_id(),
            "name": params.get("name", "Discord"),
            "type": "n8n-nodes-base.httpRequest",
            "typeVersion": 4.2,
            "parameters": {
                "method": "POST",
                "url": params.get("webhook_url", "={{ $env.DISCORD_WEBHOOK }}"),
                "sendHeaders": True,
                "headerParameters": {
                    "parameters": [{"name": "Content-Type", "value": "application/json"}]
                },
                "sendBody": True,
                "specifyBody": "json",
                "jsonBody": f'={{"content": "{{{{ $json.{params.get("message_field", "content")} }}}}"}}'
            }
        }
    },
    "slack_webhook": {
        "category": "messaging",
        "params": {
            "webhook_url": {"type": "string", "required": False, "default": "={{ $env.SLACK_WEBHOOK }}", "description": "Slack webhook URL (or use env var)"},
            "message_field": {"type": "string", "required": False, "default": "content", "description": "JSON field containing the message text"},
        },
        "builder": lambda params: {
            "id": node_id(),
            "name": params.get("name", "Slack"),
            "type": "n8n-nodes-base.httpRequest",
            "typeVersion": 4.2,
            "parameters": {
                "method": "POST",
                "url": params.get("webhook_url", "={{ $env.SLACK_WEBHOOK }}"),
                "sendHeaders": True,
                "headerParameters": {
                    "parameters": [{"name": "Content-Type", "value": "application/json"}]
                },
                "sendBody": True,
                "specifyBody": "json",
                "jsonBody": f'={{"text": "{{{{ $json.{params.get("message_field", "content")} }}}}"}}'
            }
        }
    },
    "respond_webhook": {
        "category": "messaging",
        "params": {
            "respond_with": {"type": "string", "required": False, "default": "firstIncomingItem", "description": "Response mode", "options": ["firstIncomingItem", "allIncomingItems", "json", "text", "noData", "binary", "redirect"]},
            "response_body": {"type": "string", "required": False, "default": "", "description": "JSON expression for response body (only when respond_with='json'). Use n8n expression syntax like ={{ $json }}."},
        },
        "builder": lambda params: {
            "id": node_id(),
            "name": params.get("name", "Respond"),
            "type": "n8n-nodes-base.respondToWebhook",
            "typeVersion": 1,
            "parameters": {
                "respondWith": params.get("respond_with", "firstIncomingItem"),
                **({"responseBody": params.get("response_body")} if params.get("response_body") else {})
            }
        }
    },
    "telegram": {
        "category": "messaging",
        "params": {
            "message_field": {"type": "string", "required": False, "default": "content", "description": "JSON field containing the message text"},
        },
        "builder": lambda params: {
            "id": node_id(),
            "name": params.get("name", "Telegram"),
            "type": "n8n-nodes-base.httpRequest",
            "typeVersion": 4.2,
            "parameters": {
                "method": "POST",
                "url": f"https://api.telegram.org/bot{{{{ $env.TELEGRAM_BOT_TOKEN }}}}/sendMessage",
                "sendBody": True,
                "specifyBody": "json",
                "jsonBody": f'={{"chat_id": "{{{{ $env.TELEGRAM_CHAT_ID }}}}", "text": "{{{{ $json.{params.get("message_field", "content")} }}}}"}}'
            }
        }
    },
    "email_send": {
        "category": "messaging",
        "params": {
            "from": {"type": "string", "required": False, "default": "={{ $env.EMAIL_FROM }}", "description": "Sender email address"},
            "to": {"type": "string", "required": True, "description": "Recipient email address"},
            "subject": {"type": "string", "required": True, "description": "Email subject"},
            "body_field": {"type": "string", "required": False, "default": "content", "description": "JSON field containing email body text"},
            "credential_id": {"type": "string", "required": True, "description": "n8n credential ID for SMTP"},
            "credential_name": {"type": "string", "required": False, "default": "SMTP", "description": "Credential display name"},
        },
        "builder": lambda params: {
            "id": node_id(),
            "name": params.get("name", "Send Email"),
            "type": "n8n-nodes-base.emailSend",
            "typeVersion": 2.1,
            "parameters": {
                "fromEmail": params.get("from", "={{ $env.EMAIL_FROM }}"),
                "toEmail": params.get("to", ""),
                "subject": params.get("subject", ""),
                "emailType": "text",
                "message": f"={{{{ $json.{params.get('body_field', 'content')} }}}}"
            }
        }
    },
    "twilio": {
        "category": "messaging",
        "params": {
            "from": {"type": "string", "required": True, "description": "Twilio phone number (sender)"},
            "to": {"type": "string", "required": True, "description": "Recipient phone number"},
            "message": {"type": "string", "required": True, "description": "SMS message text"},
            "credential_id": {"type": "string", "required": True, "description": "n8n credential ID for Twilio"},
            "credential_name": {"type": "string", "required": False, "default": "Twilio", "description": "Credential display name"},
        },
        "builder": lambda params: {
            "id": node_id(),
            "name": params.get("name", "Twilio SMS"),
            "type": "n8n-nodes-base.twilio",
            "typeVersion": 1,
            "parameters": {
                "operation": "send",
                "from": params.get("from", ""),
                "to": params.get("to", ""),
                "message": params.get("message", "")
            },
            "credentials": {"twilioApi": {"id": params.get("credential_id", ""), "name": params.get("credential_name", "Twilio")}}
        }
    },
    "matrix": {
        "category": "messaging",
        "params": {
            "room_id": {"type": "string", "required": True, "description": "Matrix room ID"},
            "message": {"type": "string", "required": True, "description": "Message text"},
            "credential_id": {"type": "string", "required": True, "description": "n8n credential ID for Matrix"},
            "credential_name": {"type": "string", "required": False, "default": "Matrix", "description": "Credential display name"},
        },
        "builder": lambda params: {
            "id": node_id(),
            "name": params.get("name", "Matrix"),
            "type": "n8n-nodes-base.matrix",
            "typeVersion": 1,
            "parameters": {
                "resource": "message",
                "operation": "send",
                "roomId": params.get("room_id", ""),
                "text": params.get("message", "")
            },
            "credentials": {"matrixApi": {"id": params.get("credential_id", ""), "name": params.get("credential_name", "Matrix")}}
        }
    },

    # === FLOW CONTROL ===
    "switch": {
        "category": "flow",
        "params": {
            "field": {"type": "string", "required": True, "description": "JSON field to evaluate"},
            "operation": {"type": "string", "required": False, "default": "equals", "description": "Comparison operation"},
            "compare_value": {"type": "string", "required": False, "default": "", "description": "Value to compare against"},
            "output_key": {"type": "string", "required": False, "default": "output_0", "description": "Output key name"},
        },
        "note": "Has multiple outputs. Use custom connections to wire each output to different nodes.",
        "builder": lambda params: {
            "id": node_id(),
            "name": params.get("name", "Switch"),
            "type": "n8n-nodes-base.switch",
            "typeVersion": 3,
            "parameters": {
                "mode": "rules",
                "options": {},
                "rules": {
                    "rules": [
                        {
                            "outputKey": params.get("output_key", "output_0"),
                            "conditions": {
                                "conditions": [
                                    {
                                        "leftValue": f"={{{{ $json.{params.get('field', 'value')} }}}}",
                                        "rightValue": params.get("compare_value", ""),
                                        "operator": {"type": "string", "operation": params.get("operation", "equals")}
                                    }
                                ]
                            }
                        }
                    ]
                }
            }
        }
    },
    "merge": {
        "category": "flow",
        "params": {
            "mode": {"type": "string", "required": False, "default": "combine", "description": "Merge mode", "options": ["combine", "append", "chooseBranch"]},
            "field": {"type": "string", "required": False, "default": "id", "description": "Field to merge on (for combine mode)"},
        },
        "note": "Has 2 inputs. Connect two branches into this node using custom connections.",
        "builder": lambda params: {
            "id": node_id(),
            "name": params.get("name", "Merge"),
            "type": "n8n-nodes-base.merge",
            "typeVersion": 3,
            "parameters": {
                "mode": params.get("mode", "combine"),
                "mergeByFields": {
                    "values": [{"field1": params.get("field", "id"), "field2": params.get("field", "id")}]
                } if params.get("mode") == "combine" else {},
                "options": {}
            }
        }
    },
    "split_in_batches": {
        "category": "flow",
        "params": {
            "batch_size": {"type": "integer", "required": False, "default": 10, "description": "Items per batch"},
        },
        "builder": lambda params: {
            "id": node_id(),
            "name": params.get("name", "Split In Batches"),
            "type": "n8n-nodes-base.splitInBatches",
            "typeVersion": 3,
            "parameters": {
                "batchSize": params.get("batch_size", 10),
                "options": {}
            }
        }
    },
    "wait": {
        "category": "flow",
        "params": {
            "seconds": {"type": "integer", "required": False, "default": 1, "description": "Seconds to wait"},
        },
        "builder": lambda params: {
            "id": node_id(),
            "name": params.get("name", "Wait"),
            "type": "n8n-nodes-base.wait",
            "typeVersion": 1.1,
            "parameters": {
                "amount": params.get("seconds", 1),
                "unit": "seconds"
            }
        }
    },
    "no_op": {
        "category": "flow",
        "params": {},
        "builder": lambda params: {
            "id": node_id(),
            "name": params.get("name", "No Operation"),
            "type": "n8n-nodes-base.noOp",
            "typeVersion": 1,
            "parameters": {}
        }
    },
    "stop_and_error": {
        "category": "flow",
        "params": {
            "message": {"type": "string", "required": False, "default": "Workflow stopped due to error", "description": "Error message"},
        },
        "builder": lambda params: {
            "id": node_id(),
            "name": params.get("name", "Stop and Error"),
            "type": "n8n-nodes-base.stopAndError",
            "typeVersion": 1,
            "parameters": {
                "errorType": "errorMessage",
                "errorMessage": params.get("message", "Workflow stopped due to error")
            }
        }
    },
    "execute_workflow": {
        "category": "flow",
        "params": {
            "workflow_id": {"type": "string", "required": True, "description": "ID of workflow to execute"},
        },
        "builder": lambda params: {
            "id": node_id(),
            "name": params.get("name", "Execute Workflow"),
            "type": "n8n-nodes-base.executeWorkflow",
            "typeVersion": 1.1,
            "parameters": {
                "source": "database",
                "workflowId": params.get("workflow_id", "")
            }
        }
    },
    "loop": {
        "category": "flow",
        "params": {
            "batch_size": {"type": "integer", "required": False, "default": 1, "description": "Items per loop iteration"},
        },
        "builder": lambda params: {
            "id": node_id(),
            "name": params.get("name", "Loop Over Items"),
            "type": "n8n-nodes-base.splitInBatches",
            "typeVersion": 3,
            "parameters": {
                "batchSize": params.get("batch_size", 1),
                "options": {
                    "reset": params.get("reset", False)
                }
            }
        }
    },

    # === DATABASE ===
    "mysql": {
        "category": "database",
        "params": {
            "operation": {"type": "string", "required": False, "default": "select", "description": "DB operation", "options": ["select", "insert", "update", "delete", "executeQuery"]},
            "table": {"type": "string", "required": True, "description": "Table name"},
            "credential_id": {"type": "string", "required": True, "description": "n8n credential ID for MySQL"},
            "credential_name": {"type": "string", "required": False, "default": "MySQL", "description": "Credential display name"},
        },
        "builder": lambda params: {
            "id": node_id(),
            "name": params.get("name", "MySQL"),
            "type": "n8n-nodes-base.mySql",
            "typeVersion": 2.4,
            "parameters": {
                "operation": params.get("operation", "select"),
                "table": {"__rl": True, "mode": "raw", "value": params.get("table", "")},
                "options": {}
            },
            "credentials": {"mySql": {"id": params.get("credential_id", ""), "name": params.get("credential_name", "MySQL")}}
        }
    },
    "postgres": {
        "category": "database",
        "params": {
            "operation": {"type": "string", "required": False, "default": "select", "description": "DB operation", "options": ["select", "insert", "update", "delete", "executeQuery"]},
            "table": {"type": "string", "required": True, "description": "Table name"},
            "schema": {"type": "string", "required": False, "default": "public", "description": "Database schema"},
            "credential_id": {"type": "string", "required": True, "description": "n8n credential ID for Postgres"},
            "credential_name": {"type": "string", "required": False, "default": "Postgres", "description": "Credential display name"},
        },
        "builder": lambda params: {
            "id": node_id(),
            "name": params.get("name", "Postgres"),
            "type": "n8n-nodes-base.postgres",
            "typeVersion": 2.5,
            "parameters": {
                "operation": params.get("operation", "select"),
                "schema": {"__rl": True, "mode": "raw", "value": params.get("schema", "public")},
                "table": {"__rl": True, "mode": "raw", "value": params.get("table", "")},
                "options": {}
            },
            "credentials": {"postgres": {"id": params.get("credential_id", ""), "name": params.get("credential_name", "Postgres")}}
        }
    },
    "mongodb": {
        "category": "database",
        "params": {
            "operation": {"type": "string", "required": False, "default": "find", "description": "DB operation", "options": ["find", "insert", "update", "delete"]},
            "collection": {"type": "string", "required": True, "description": "Collection name"},
            "query": {"type": "string", "required": False, "default": "{}", "description": "MongoDB query as JSON string"},
            "credential_id": {"type": "string", "required": True, "description": "n8n credential ID for MongoDB"},
            "credential_name": {"type": "string", "required": False, "default": "MongoDB", "description": "Credential display name"},
        },
        "builder": lambda params: {
            "id": node_id(),
            "name": params.get("name", "MongoDB"),
            "type": "n8n-nodes-base.mongoDb",
            "typeVersion": 1.1,
            "parameters": {
                "operation": params.get("operation", "find"),
                "collection": params.get("collection", ""),
                "query": params.get("query", "{}")
            },
            "credentials": {"mongoDb": {"id": params.get("credential_id", ""), "name": params.get("credential_name", "MongoDB")}}
        }
    },
    "sqlite": {
        "category": "database",
        "params": {
            "operation": {"type": "string", "required": False, "default": "executeQuery", "description": "DB operation", "options": ["executeQuery", "insert"]},
            "query": {"type": "string", "required": True, "description": "SQL query to execute"},
        },
        "builder": lambda params: {
            "id": node_id(),
            "name": params.get("name", "SQLite"),
            "type": "n8n-nodes-base.sqlite",
            "typeVersion": 1,
            "parameters": {
                "operation": params.get("operation", "executeQuery"),
                "query": params.get("query", "SELECT * FROM table_name")
            }
        }
    },
    "redis": {
        "category": "database",
        "params": {
            "operation": {"type": "string", "required": False, "default": "get", "description": "Redis operation", "options": ["get", "set", "delete", "push", "pop", "info"]},
            "key": {"type": "string", "required": True, "description": "Redis key"},
            "credential_id": {"type": "string", "required": True, "description": "n8n credential ID for Redis"},
            "credential_name": {"type": "string", "required": False, "default": "Redis", "description": "Credential display name"},
        },
        "builder": lambda params: {
            "id": node_id(),
            "name": params.get("name", "Redis"),
            "type": "n8n-nodes-base.redis",
            "typeVersion": 1,
            "parameters": {
                "operation": params.get("operation", "get"),
                "key": params.get("key", "")
            },
            "credentials": {"redis": {"id": params.get("credential_id", ""), "name": params.get("credential_name", "Redis")}}
        }
    },

    # === CLOUD STORAGE ===
    "google_sheets": {
        "category": "cloud",
        "params": {
            "operation": {"type": "string", "required": False, "default": "read", "description": "Operation", "options": ["read", "append", "update", "delete"]},
            "spreadsheet_id": {"type": "string", "required": True, "description": "Google Sheets document ID"},
            "sheet_name": {"type": "string", "required": False, "default": "Sheet1", "description": "Sheet tab name"},
            "credential_id": {"type": "string", "required": True, "description": "n8n credential ID for Google Sheets"},
            "credential_name": {"type": "string", "required": False, "default": "Google Sheets", "description": "Credential display name"},
        },
        "builder": lambda params: {
            "id": node_id(),
            "name": params.get("name", "Google Sheets"),
            "type": "n8n-nodes-base.googleSheets",
            "typeVersion": 4.5,
            "parameters": {
                "operation": params.get("operation", "read"),
                "documentId": {"__rl": True, "mode": "id", "value": params.get("spreadsheet_id", "")},
                "sheetName": {"__rl": True, "mode": "name", "value": params.get("sheet_name", "Sheet1")}
            },
            "credentials": {"googleSheetsOAuth2Api": {"id": params.get("credential_id", ""), "name": params.get("credential_name", "Google Sheets")}}
        }
    },
    "google_drive": {
        "category": "cloud",
        "params": {
            "operation": {"type": "string", "required": False, "default": "download", "description": "Operation", "options": ["download", "upload", "delete", "list"]},
            "file_id": {"type": "string", "required": True, "description": "Google Drive file ID"},
            "credential_id": {"type": "string", "required": True, "description": "n8n credential ID for Google Drive"},
            "credential_name": {"type": "string", "required": False, "default": "Google Drive", "description": "Credential display name"},
        },
        "builder": lambda params: {
            "id": node_id(),
            "name": params.get("name", "Google Drive"),
            "type": "n8n-nodes-base.googleDrive",
            "typeVersion": 3,
            "parameters": {
                "operation": params.get("operation", "download"),
                "fileId": {"__rl": True, "mode": "id", "value": params.get("file_id", "")}
            },
            "credentials": {"googleDriveOAuth2Api": {"id": params.get("credential_id", ""), "name": params.get("credential_name", "Google Drive")}}
        }
    },
    "aws_s3": {
        "category": "cloud",
        "params": {
            "operation": {"type": "string", "required": False, "default": "download", "description": "Operation", "options": ["download", "upload", "delete", "list"]},
            "bucket": {"type": "string", "required": True, "description": "S3 bucket name"},
            "key": {"type": "string", "required": True, "description": "Object key (file path in bucket)"},
            "credential_id": {"type": "string", "required": True, "description": "n8n credential ID for AWS"},
            "credential_name": {"type": "string", "required": False, "default": "AWS", "description": "Credential display name"},
        },
        "builder": lambda params: {
            "id": node_id(),
            "name": params.get("name", "AWS S3"),
            "type": "n8n-nodes-base.awsS3",
            "typeVersion": 1,
            "parameters": {
                "operation": params.get("operation", "download"),
                "bucketName": params.get("bucket", ""),
                "fileKey": params.get("key", "")
            },
            "credentials": {"aws": {"id": params.get("credential_id", ""), "name": params.get("credential_name", "AWS")}}
        }
    },
    "dropbox": {
        "category": "cloud",
        "params": {
            "operation": {"type": "string", "required": False, "default": "download", "description": "Operation", "options": ["download", "upload", "delete", "list"]},
            "path": {"type": "string", "required": True, "description": "Dropbox file path"},
            "credential_id": {"type": "string", "required": True, "description": "n8n credential ID for Dropbox"},
            "credential_name": {"type": "string", "required": False, "default": "Dropbox", "description": "Credential display name"},
        },
        "builder": lambda params: {
            "id": node_id(),
            "name": params.get("name", "Dropbox"),
            "type": "n8n-nodes-base.dropbox",
            "typeVersion": 1,
            "parameters": {
                "operation": params.get("operation", "download"),
                "path": params.get("path", "")
            },
            "credentials": {"dropboxOAuth2Api": {"id": params.get("credential_id", ""), "name": params.get("credential_name", "Dropbox")}}
        }
    },
    "onedrive": {
        "category": "cloud",
        "params": {
            "operation": {"type": "string", "required": False, "default": "download", "description": "Operation", "options": ["download", "upload", "delete"]},
            "file_id": {"type": "string", "required": True, "description": "OneDrive file ID"},
            "credential_id": {"type": "string", "required": True, "description": "n8n credential ID for OneDrive"},
            "credential_name": {"type": "string", "required": False, "default": "OneDrive", "description": "Credential display name"},
        },
        "builder": lambda params: {
            "id": node_id(),
            "name": params.get("name", "OneDrive"),
            "type": "n8n-nodes-base.microsoftOneDrive",
            "typeVersion": 1,
            "parameters": {
                "operation": params.get("operation", "download"),
                "fileId": params.get("file_id", "")
            },
            "credentials": {"microsoftOneDriveOAuth2Api": {"id": params.get("credential_id", ""), "name": params.get("credential_name", "OneDrive")}}
        }
    },
    "notion": {
        "category": "cloud",
        "params": {
            "resource": {"type": "string", "required": False, "default": "page", "description": "Notion resource", "options": ["page", "database", "block"]},
            "operation": {"type": "string", "required": False, "default": "get", "description": "Operation", "options": ["get", "create", "update", "archive"]},
            "page_id": {"type": "string", "required": False, "description": "Notion page ID"},
            "credential_id": {"type": "string", "required": True, "description": "n8n credential ID for Notion"},
            "credential_name": {"type": "string", "required": False, "default": "Notion", "description": "Credential display name"},
        },
        "builder": lambda params: {
            "id": node_id(),
            "name": params.get("name", "Notion"),
            "type": "n8n-nodes-base.notion",
            "typeVersion": 2.2,
            "parameters": {
                "resource": params.get("resource", "page"),
                "operation": params.get("operation", "get"),
                "pageId": params.get("page_id", "")
            },
            "credentials": {"notionApi": {"id": params.get("credential_id", ""), "name": params.get("credential_name", "Notion")}}
        }
    },
    "airtable": {
        "category": "cloud",
        "params": {
            "operation": {"type": "string", "required": False, "default": "read", "description": "Operation", "options": ["read", "create", "update", "delete"]},
            "base_id": {"type": "string", "required": True, "description": "Airtable base ID"},
            "table_id": {"type": "string", "required": True, "description": "Airtable table ID"},
            "credential_id": {"type": "string", "required": True, "description": "n8n credential ID for Airtable"},
            "credential_name": {"type": "string", "required": False, "default": "Airtable", "description": "Credential display name"},
        },
        "builder": lambda params: {
            "id": node_id(),
            "name": params.get("name", "Airtable"),
            "type": "n8n-nodes-base.airtable",
            "typeVersion": 2.1,
            "parameters": {
                "operation": params.get("operation", "read"),
                "base": {"__rl": True, "mode": "id", "value": params.get("base_id", "")},
                "table": {"__rl": True, "mode": "id", "value": params.get("table_id", "")}
            },
            "credentials": {"airtableTokenApi": {"id": params.get("credential_id", ""), "name": params.get("credential_name", "Airtable")}}
        }
    },

    # === EXTERNAL AI ===
    "openai": {
        "category": "ai",
        "params": {
            "model": {"type": "string", "required": False, "default": "gpt-4o-mini", "description": "OpenAI model ID"},
            "prompt": {"type": "string", "required": True, "description": "Prompt text"},
            "credential_id": {"type": "string", "required": True, "description": "n8n credential ID for OpenAI"},
            "credential_name": {"type": "string", "required": False, "default": "OpenAI", "description": "Credential display name"},
        },
        "builder": lambda params: {
            "id": node_id(),
            "name": params.get("name", "OpenAI"),
            "type": "@n8n/n8n-nodes-langchain.openAi",
            "typeVersion": 1.8,
            "parameters": {
                "resource": params.get("resource", "chat"),
                "operation": params.get("operation", "message"),
                "modelId": {"__rl": True, "mode": "list", "value": params.get("model", "gpt-4o-mini")},
                "messages": {
                    "values": [{"content": params.get("prompt", "")}]
                },
                "options": {}
            },
            "credentials": {"openAiApi": {"id": params.get("credential_id", ""), "name": params.get("credential_name", "OpenAI")}}
        }
    },
    "anthropic": {
        "category": "ai",
        "params": {
            "model": {"type": "string", "required": False, "default": "claude-3-5-sonnet-20241022", "description": "Anthropic model ID"},
            "prompt": {"type": "string", "required": True, "description": "Prompt text"},
            "credential_id": {"type": "string", "required": True, "description": "n8n credential ID for Anthropic"},
            "credential_name": {"type": "string", "required": False, "default": "Anthropic", "description": "Credential display name"},
        },
        "builder": lambda params: {
            "id": node_id(),
            "name": params.get("name", "Anthropic"),
            "type": "@n8n/n8n-nodes-langchain.lmChatAnthropic",
            "typeVersion": 1.3,
            "parameters": {
                "modelId": {"__rl": True, "mode": "list", "value": params.get("model", "claude-3-5-sonnet-20241022")},
                "messages": {
                    "values": [{"content": params.get("prompt", "")}]
                },
                "options": {}
            },
            "credentials": {"anthropicApi": {"id": params.get("credential_id", ""), "name": params.get("credential_name", "Anthropic")}}
        }
    },

    # === DEV / CODE TOOLS ===
    "github": {
        "category": "dev",
        "params": {
            "resource": {"type": "string", "required": False, "default": "issue", "description": "GitHub resource", "options": ["issue", "repository", "release", "user"]},
            "operation": {"type": "string", "required": False, "default": "get", "description": "Operation", "options": ["get", "create", "update", "getAll"]},
            "owner": {"type": "string", "required": True, "description": "Repository owner/org"},
            "repo": {"type": "string", "required": True, "description": "Repository name"},
            "credential_id": {"type": "string", "required": True, "description": "n8n credential ID for GitHub"},
            "credential_name": {"type": "string", "required": False, "default": "GitHub", "description": "Credential display name"},
        },
        "builder": lambda params: {
            "id": node_id(),
            "name": params.get("name", "GitHub"),
            "type": "n8n-nodes-base.github",
            "typeVersion": 1,
            "parameters": {
                "resource": params.get("resource", "issue"),
                "operation": params.get("operation", "get"),
                "owner": params.get("owner", ""),
                "repository": params.get("repo", "")
            },
            "credentials": {"githubApi": {"id": params.get("credential_id", ""), "name": params.get("credential_name", "GitHub")}}
        }
    },
    "gitlab": {
        "category": "dev",
        "params": {
            "resource": {"type": "string", "required": False, "default": "issue", "description": "GitLab resource", "options": ["issue", "repository", "release"]},
            "operation": {"type": "string", "required": False, "default": "get", "description": "Operation", "options": ["get", "create", "update", "getAll"]},
            "project_id": {"type": "string", "required": True, "description": "GitLab project ID"},
            "credential_id": {"type": "string", "required": True, "description": "n8n credential ID for GitLab"},
            "credential_name": {"type": "string", "required": False, "default": "GitLab", "description": "Credential display name"},
        },
        "builder": lambda params: {
            "id": node_id(),
            "name": params.get("name", "GitLab"),
            "type": "n8n-nodes-base.gitlab",
            "typeVersion": 1,
            "parameters": {
                "resource": params.get("resource", "issue"),
                "operation": params.get("operation", "get"),
                "projectId": params.get("project_id", "")
            },
            "credentials": {"gitlabApi": {"id": params.get("credential_id", ""), "name": params.get("credential_name", "GitLab")}}
        }
    },
    "jira": {
        "category": "dev",
        "params": {
            "resource": {"type": "string", "required": False, "default": "issue", "description": "Jira resource", "options": ["issue", "project"]},
            "operation": {"type": "string", "required": False, "default": "get", "description": "Operation", "options": ["get", "create", "update", "getAll"]},
            "issue_key": {"type": "string", "required": False, "description": "Jira issue key (e.g. PROJ-123)"},
            "credential_id": {"type": "string", "required": True, "description": "n8n credential ID for Jira"},
            "credential_name": {"type": "string", "required": False, "default": "Jira", "description": "Credential display name"},
        },
        "builder": lambda params: {
            "id": node_id(),
            "name": params.get("name", "Jira"),
            "type": "n8n-nodes-base.jira",
            "typeVersion": 1,
            "parameters": {
                "resource": params.get("resource", "issue"),
                "operation": params.get("operation", "get"),
                "issueKey": params.get("issue_key", "")
            },
            "credentials": {"jiraSoftwareCloudApi": {"id": params.get("credential_id", ""), "name": params.get("credential_name", "Jira")}}
        }
    },

    # === UTILITY ===
    "debug": {
        "category": "utility",
        "params": {},
        "builder": lambda params: {
            "id": node_id(),
            "name": params.get("name", "Debug"),
            "type": "n8n-nodes-base.set",
            "typeVersion": 3,
            "parameters": {
                "mode": "raw",
                "jsonOutput": "={{ JSON.stringify($json, null, 2) }}",
                "options": {}
            }
        }
    },
    "rss_feed": {
        "category": "utility",
        "params": {
            "url": {"type": "string", "required": True, "description": "RSS feed URL"},
        },
        "builder": lambda params: {
            "id": node_id(),
            "name": params.get("name", "RSS Feed"),
            "type": "n8n-nodes-base.rssFeedRead",
            "typeVersion": 1,
            "parameters": {
                "url": params.get("url", "")
            }
        }
    },
    "ftp": {
        "category": "utility",
        "params": {
            "operation": {"type": "string", "required": False, "default": "download", "description": "FTP operation", "options": ["download", "upload", "list", "delete"]},
            "path": {"type": "string", "required": True, "description": "Remote file path"},
            "credential_id": {"type": "string", "required": True, "description": "n8n credential ID for FTP"},
            "credential_name": {"type": "string", "required": False, "default": "FTP", "description": "Credential display name"},
        },
        "builder": lambda params: {
            "id": node_id(),
            "name": params.get("name", "FTP"),
            "type": "n8n-nodes-base.ftp",
            "typeVersion": 1,
            "parameters": {
                "operation": params.get("operation", "download"),
                "path": params.get("path", "")
            },
            "credentials": {"ftp": {"id": params.get("credential_id", ""), "name": params.get("credential_name", "FTP")}}
        }
    },
    "ssh": {
        "category": "utility",
        "params": {
            "operation": {"type": "string", "required": False, "default": "executeCommand", "description": "SSH operation", "options": ["executeCommand", "download", "upload"]},
            "command": {"type": "string", "required": False, "description": "Shell command to execute (for executeCommand)"},
            "credential_id": {"type": "string", "required": True, "description": "n8n credential ID for SSH"},
            "credential_name": {"type": "string", "required": False, "default": "SSH", "description": "Credential display name"},
        },
        "builder": lambda params: {
            "id": node_id(),
            "name": params.get("name", "SSH"),
            "type": "n8n-nodes-base.ssh",
            "typeVersion": 1,
            "parameters": {
                "operation": params.get("operation", "executeCommand"),
                "command": params.get("command", "")
            },
            "credentials": {"sshPassword": {"id": params.get("credential_id", ""), "name": params.get("credential_name", "SSH")}}
        }
    },
    "compression": {
        "category": "utility",
        "params": {
            "operation": {"type": "string", "required": False, "default": "compress", "description": "Operation", "options": ["compress", "decompress"]},
            "binary_property": {"type": "string", "required": False, "default": "data", "description": "Binary property name"},
            "format": {"type": "string", "required": False, "default": "zip", "description": "Compression format", "options": ["zip", "gzip"]},
        },
        "builder": lambda params: {
            "id": node_id(),
            "name": params.get("name", "Compression"),
            "type": "n8n-nodes-base.compression",
            "typeVersion": 1,
            "parameters": {
                "operation": params.get("operation", "compress"),
                "binaryPropertyName": params.get("binary_property", "data"),
                "outputFormat": params.get("format", "zip")
            }
        }
    },
    "pdf": {
        "category": "utility",
        "params": {
            "binary_property": {"type": "string", "required": False, "default": "data", "description": "Binary property containing PDF"},
        },
        "builder": lambda params: {
            "id": node_id(),
            "name": params.get("name", "PDF Extract"),
            "type": "n8n-nodes-base.extractFromFile",
            "typeVersion": 1,
            "parameters": {
                "operation": "pdf",
                "binaryPropertyName": params.get("binary_property", "data")
            }
        }
    },
}


# ============================================================
# NODE BUILDER FUNCTIONS
# ============================================================

def build_node(node_type: str, params: Dict = None) -> Dict[str, Any]:
    """Build a single node from type and parameters.

    Handles both flat params and nested params:
    - Flat: {"type": "http_request", "url": "..."}
    - Nested: {"type": "http_request", "parameters": {"url": "..."}}
    """
    params = params or {}

    if node_type not in NODE_REGISTRY:
        # Reverse lookup: map full n8n type names (e.g. "n8n-nodes-base.httpRequest")
        # back to registry keys (e.g. "http_request") by calling each builder
        for key, entry in NODE_REGISTRY.items():
            builder = entry.get("builder")
            if builder:
                try:
                    built = builder({})
                    if built.get("type") == node_type:
                        node_type = key
                        break
                except Exception:
                    pass

    if node_type not in NODE_REGISTRY:
        # Handle "n8n-nodes-base." prefix stripping and camelCase to snake_case
        import re as _re
        stripped = node_type
        if stripped.startswith("n8n-nodes-base."):
            stripped = stripped[len("n8n-nodes-base."):]
        # camelCase to snake_case: "httpRequest" -> "http_request"
        stripped = _re.sub(r'([a-z])([A-Z])', r'\1_\2', stripped).lower()
        if stripped in NODE_REGISTRY:
            node_type = stripped

    if node_type not in NODE_REGISTRY:
        # Fuzzy match: normalize underscores/spaces, try substring, prefix match
        norm = node_type.lower().replace("_", " ").replace("-", " ")
        for key in NODE_REGISTRY:
            key_norm = key.lower().replace("_", " ").replace("-", " ")
            # Exact normalized match (respond_to_webhook -> respond webhook == respond webhook)
            if norm == key_norm:
                node_type = key
                break
            # One contains the other (respond to webhook contains respond webhook)
            if key_norm in norm or norm in key_norm:
                node_type = key
                break
            # Word overlap: if most words match
            norm_words = set(norm.split())
            key_words = set(key_norm.split())
            overlap = norm_words & key_words
            if len(overlap) >= max(1, len(key_words) - 1) and len(overlap) >= len(key_words) * 0.6:
                node_type = key
                break
        else:
            raise ValueError(f"Unknown node type: {node_type}. Available: {list(NODE_REGISTRY.keys())}")

    # Merge nested "parameters" or "settings" into top-level params for flexibility
    merged_params = dict(params)
    for nested_key in ("parameters", "settings"):
        if nested_key in params and isinstance(params[nested_key], dict):
            for k, v in params[nested_key].items():
                if k not in merged_params or k == nested_key:
                    merged_params[k] = v
            merged_params.pop(nested_key, None)

    # Handle common param name aliases (n8n names -> our names)
    param_aliases = {
        "jsCode": "code",
        "binaryPropertyName": "binary_field",
        "fileName": "file_path",
        "filePath": "file_path",
    }
    for n8n_name, our_name in param_aliases.items():
        if n8n_name in merged_params and our_name not in merged_params:
            merged_params[our_name] = merged_params[n8n_name]

    return NODE_REGISTRY[node_type]["builder"](merged_params)


def build_workflow(
    name: str,
    nodes: List[Dict[str, Any]],
    connections: Dict[str, Any]
) -> Dict[str, Any]:
    """Build a complete workflow from nodes and connections."""
    return {
        "name": name,
        "nodes": nodes,
        "connections": connections,
        "active": False,
        "settings": {
            "executionOrder": "v1",
            "saveManualExecutions": True,
            "callerPolicy": "workflowsFromSameOwner"
        },
        "versionId": node_id(),
        "id": node_id()
    }


def build_workflow_from_nodes(
    name: str,
    node_specs: List[Dict[str, Any]],
    connection_mode: str = "linear",
    custom_connections: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Build a complete workflow from simple node specifications.

    Args:
        name: Workflow name
        node_specs: List of node specs, each with "type" and optional params
                   Example: [{"type": "manual_trigger"}, {"type": "http_request", "url": "..."}]
        connection_mode: "linear" (default) connects nodes in sequence. Ignored if custom_connections provided.
        custom_connections: List of connection dicts for non-linear topologies.
                   Example: [{"from": "IF", "to": "Process", "output": 0}, ...]
                   Each dict: from (node name), to (node name), output (int, default 0), input (int, default 0)

    Returns:
        Complete n8n workflow JSON
    """
    nodes = []
    x_pos = 250
    y_pos = 300
    x_step = 200

    for spec in node_specs:
        node_type = spec.get("type")
        if not node_type:
            raise ValueError(f"Node spec missing 'type': {spec}")

        node = build_node(node_type, spec)
        node["position"] = [x_pos, y_pos]
        x_pos += x_step
        nodes.append(node)

    # Build connections
    connections = {}

    if custom_connections:
        # Custom connections for branching/merging workflows
        # Build lookups: by display name AND by original node type
        node_name_map = {n["name"]: n for n in nodes}
        # Also map original spec type → built node name (for type-based references)
        type_to_name = {}
        for i, spec in enumerate(node_specs):
            if i < len(nodes):
                type_to_name[spec.get("type", "")] = nodes[i]["name"]

        # Fuzzy match helper: try exact, type-based, case-insensitive, normalized, substring
        def _resolve_name(name, name_map):
            if name in name_map:
                return name
            # Type-based match (e.g. "respond_webhook" → "Respond")
            if name in type_to_name:
                return type_to_name[name]
            # Case-insensitive match
            lower_map = {k.lower(): k for k in name_map}
            if name.lower() in lower_map:
                return lower_map[name.lower()]
            # Underscore/space normalized match (http_request → HTTP Request)
            norm = name.lower().replace("_", " ")
            norm_map = {k.lower().replace("_", " "): k for k in name_map}
            if norm in norm_map:
                return norm_map[norm]
            # Type-based normalized match
            type_norm_map = {k.lower().replace("_", " "): v for k, v in type_to_name.items()}
            if norm in type_norm_map:
                return type_norm_map[norm]
            # Prefix match only (avoid false positives like "webhook" matching "respond webhook")
            best_match = None
            best_len = 0
            for real_name in name_map:
                real_norm = real_name.lower().replace("_", " ")
                if real_norm.startswith(norm) or norm.startswith(real_norm):
                    match_len = min(len(norm), len(real_norm))
                    if match_len > best_len:
                        best_len = match_len
                        best_match = real_name
            return best_match

        def _extract_name(val):
            """Extract string name from various connection formats.
            Handles: "NodeName", {"nodes": ["NodeName"]}, {"node": "NodeName"}, ["NodeName"]
            """
            if isinstance(val, str):
                return val
            if isinstance(val, dict):
                if "nodes" in val and isinstance(val["nodes"], list) and val["nodes"]:
                    return str(val["nodes"][0])
                if "node" in val:
                    return str(val["node"])
                return str(next(iter(val.values()), ""))
            if isinstance(val, list) and val:
                return str(val[0])
            return str(val)

        for conn in custom_connections:
            source_name = _extract_name(conn.get("from", ""))
            target_name = _extract_name(conn.get("to", ""))
            output_idx = conn.get("output", 0)
            input_idx = conn.get("input", 0)

            resolved_source = _resolve_name(source_name, node_name_map)
            resolved_target = _resolve_name(target_name, node_name_map)

            if not resolved_source:
                raise ValueError(f"Connection source '{source_name}' not found in nodes: {list(node_name_map.keys())}")
            if not resolved_target:
                raise ValueError(f"Connection target '{target_name}' not found in nodes: {list(node_name_map.keys())}")

            source_name = resolved_source
            target_name = resolved_target

            if source_name not in connections:
                connections[source_name] = {"main": []}

            # Ensure enough output slots
            while len(connections[source_name]["main"]) <= output_idx:
                connections[source_name]["main"].append([])

            connections[source_name]["main"][output_idx].append({
                "node": target_name,
                "type": "main",
                "index": input_idx
            })

    elif connection_mode == "linear" and len(nodes) > 1:
        # Auto-connect linearly
        for i in range(len(nodes) - 1):
            source_name = nodes[i]["name"]
            target_name = nodes[i + 1]["name"]
            connections[source_name] = {
                "main": [[{"node": target_name, "type": "main", "index": 0}]]
            }

    # Auto-fix: if workflow has a Respond to Webhook node, set webhook responseMode
    # to "responseNode" so it waits for the response instead of returning immediately.
    has_respond = any(n.get("type") == "n8n-nodes-base.respondToWebhook" for n in nodes)
    if has_respond:
        for n in nodes:
            if n.get("type") == "n8n-nodes-base.webhook":
                n["parameters"]["responseMode"] = "responseNode"

    return build_workflow(name, nodes, connections)


def get_node_types() -> Dict[str, str]:
    """Get available node types with descriptions (includes param hints)."""
    result = {}
    for type_name, entry in NODE_REGISTRY.items():
        params = entry.get("params", {})
        param_names = [f"{k}" for k, v in params.items() if k != "name" and v.get("required", False)]
        opt_params = [f"{k}" for k, v in params.items() if k != "name" and not v.get("required", False)]

        desc_parts = [entry.get("category", "")]
        if entry.get("note"):
            desc_parts.append(entry["note"])
        if param_names:
            desc_parts.append(f"required: {', '.join(param_names)}")
        if opt_params:
            desc_parts.append(f"optional: {', '.join(opt_params)}")

        result[type_name] = " | ".join(desc_parts)

    return result


def get_node_params(node_type: str) -> Optional[Dict]:
    """Get full parameter schema for a node type. Returns None if not found."""
    if node_type not in NODE_REGISTRY:
        return None
    entry = NODE_REGISTRY[node_type]
    return {
        "type": node_type,
        "category": entry.get("category", ""),
        "note": entry.get("note"),
        "params": entry.get("params", {}),
    }


def get_all_node_params() -> Dict[str, Dict]:
    """Get parameter schemas for all node types, grouped by category."""
    by_category = {}
    for type_name, entry in NODE_REGISTRY.items():
        cat = entry.get("category", "other")
        if cat not in by_category:
            by_category[cat] = {}
        by_category[cat][type_name] = {
            "params": entry.get("params", {}),
            "note": entry.get("note"),
        }
    return by_category


def get_short_to_n8n_type_map() -> Dict[str, str]:
    """Build mapping from short NODE_REGISTRY names to full n8n type strings.

    E.g. 'schedule' -> 'n8n-nodes-base.scheduleTrigger'
    """
    mapping = {}
    for short_name, entry in NODE_REGISTRY.items():
        builder = entry.get("builder")
        if builder:
            try:
                sample = builder({})
                n8n_type = sample.get("type", "")
                if n8n_type:
                    mapping[short_name] = n8n_type
            except Exception:
                pass
    return mapping
