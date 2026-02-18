"""
Utility Tools - DateTime, math, encoding, hashing, regex, etc.
"""

from typing import Dict, Any, List, Optional, Union
import logging
import uuid
import base64
import hashlib
import re
import math
import ast
import operator
from datetime import datetime, timezone
from urllib.parse import quote, unquote, urlencode, parse_qs

logger = logging.getLogger("tools.utility")


TOOL_DEFINITIONS = [
    {
        "name": "get_datetime",
        "description": "Get current date and time, optionally in a specific timezone.",
        "parameters": {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "Timezone name (e.g., 'US/Eastern', 'Europe/London', 'UTC'). Default is UTC."
                },
                "format": {
                    "type": "string",
                    "description": "Output format (strftime format or 'iso', 'unix', 'human')",
                    "default": "iso"
                }
            },
            "required": []
        }
    },
    {
        "name": "calculate",
        "description": "Evaluate a mathematical expression safely.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Math expression to evaluate (e.g., '2 + 2', 'sqrt(16)', '3.14 * 10**2')"
                }
            },
            "required": ["expression"]
        }
    },
    {
        "name": "generate_uuid",
        "description": "Generate a unique identifier.",
        "parameters": {
            "type": "object",
            "properties": {
                "version": {
                    "type": "integer",
                    "description": "UUID version (1 or 4, default 4)",
                    "enum": [1, 4],
                    "default": 4
                },
                "count": {
                    "type": "integer",
                    "description": "Number of UUIDs to generate (default 1, max 100)",
                    "default": 1
                }
            },
            "required": []
        }
    },
    {
        "name": "encode_decode",
        "description": "Encode or decode data using various methods.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Text to encode/decode"
                },
                "method": {
                    "type": "string",
                    "description": "Encoding method",
                    "enum": ["base64_encode", "base64_decode", "url_encode", "url_decode", "hex_encode", "hex_decode"]
                }
            },
            "required": ["text", "method"]
        }
    },
    {
        "name": "hash_text",
        "description": "Generate a hash of the input text.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Text to hash"
                },
                "algorithm": {
                    "type": "string",
                    "description": "Hash algorithm",
                    "enum": ["md5", "sha1", "sha256", "sha512"],
                    "default": "sha256"
                }
            },
            "required": ["text"]
        }
    },
    {
        "name": "regex_match",
        "description": "Test a regex pattern against text and extract matches.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regular expression pattern"
                },
                "text": {
                    "type": "string",
                    "description": "Text to match against"
                },
                "flags": {
                    "type": "string",
                    "description": "Regex flags: i=ignorecase, m=multiline, s=dotall",
                    "default": ""
                },
                "find_all": {
                    "type": "boolean",
                    "description": "Find all matches (default true)",
                    "default": True
                }
            },
            "required": ["pattern", "text"]
        }
    },
    {
        "name": "text_transform",
        "description": "Transform text: change case, trim, replace, split, etc.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Input text"
                },
                "operation": {
                    "type": "string",
                    "description": "Operation to perform",
                    "enum": [
                        "lowercase", "uppercase", "titlecase", "capitalize",
                        "trim", "strip_html", "remove_whitespace",
                        "reverse", "count_words", "count_chars"
                    ]
                }
            },
            "required": ["text", "operation"]
        }
    },
    {
        "name": "random_string",
        "description": "Generate a random string.",
        "parameters": {
            "type": "object",
            "properties": {
                "length": {
                    "type": "integer",
                    "description": "Length of string (default 16, max 1000)",
                    "default": 16
                },
                "charset": {
                    "type": "string",
                    "description": "Character set: 'alphanumeric', 'alpha', 'numeric', 'hex', 'custom'",
                    "enum": ["alphanumeric", "alpha", "numeric", "hex", "custom"],
                    "default": "alphanumeric"
                },
                "custom_chars": {
                    "type": "string",
                    "description": "Custom characters to use (only with charset='custom')"
                }
            },
            "required": []
        }
    },
]


class UtilityTools:
    """Utility tools for common operations."""

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize utility tools.

        Args:
            config: Optional configuration dict
        """
        self.config = config or {}

        # Safe math operators for calculate
        self._math_operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.FloorDiv: operator.floordiv,
            ast.Mod: operator.mod,
            ast.Pow: operator.pow,
            ast.USub: operator.neg,
            ast.UAdd: operator.pos,
        }

        # Safe math functions
        self._math_functions = {
            'abs': abs,
            'round': round,
            'min': min,
            'max': max,
            'sum': sum,
            'sqrt': math.sqrt,
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'log': math.log,
            'log10': math.log10,
            'log2': math.log2,
            'exp': math.exp,
            'pow': math.pow,
            'floor': math.floor,
            'ceil': math.ceil,
            'pi': math.pi,
            'e': math.e,
        }

    async def get_datetime(
        self,
        timezone: Optional[str] = None,
        format: str = "iso"
    ) -> Dict[str, Any]:
        """
        Get current date and time.

        Args:
            timezone: Timezone name
            format: Output format

        Returns:
            Dict with datetime info
        """
        try:
            now = datetime.now(tz=timezone_module.utc)

            if timezone:
                try:
                    import zoneinfo
                    tz = zoneinfo.ZoneInfo(timezone)
                    now = now.astimezone(tz)
                except ImportError:
                    # Try pytz as fallback
                    try:
                        import pytz
                        tz = pytz.timezone(timezone)
                        now = now.astimezone(tz)
                    except ImportError:
                        return {
                            "success": False,
                            "error": "Timezone support requires Python 3.9+ or pytz"
                        }
                except Exception as e:
                    return {"success": False, "error": f"Invalid timezone: {timezone}"}

            # Format output
            if format == "iso":
                formatted = now.isoformat()
            elif format == "unix":
                formatted = int(now.timestamp())
            elif format == "human":
                formatted = now.strftime("%A, %B %d, %Y at %I:%M %p %Z")
            else:
                # Custom strftime format
                formatted = now.strftime(format)

            return {
                "success": True,
                "datetime": formatted,
                "timezone": str(now.tzinfo) if now.tzinfo else "UTC",
                "components": {
                    "year": now.year,
                    "month": now.month,
                    "day": now.day,
                    "hour": now.hour,
                    "minute": now.minute,
                    "second": now.second,
                    "weekday": now.strftime("%A"),
                    "timestamp": int(now.timestamp())
                }
            }

        except Exception as e:
            logger.error(f"get_datetime error: {e}")
            return {"success": False, "error": str(e)}

    async def calculate(self, expression: str) -> Dict[str, Any]:
        """
        Safely evaluate a mathematical expression.

        Args:
            expression: Math expression

        Returns:
            Dict with result
        """
        try:
            # Replace function calls with markers we can recognize
            expr = expression

            # Handle math constants
            expr = expr.replace("pi", str(math.pi))
            expr = expr.replace("PI", str(math.pi))

            # Parse the expression
            tree = ast.parse(expr, mode='eval')

            result = self._eval_node(tree.body)

            return {
                "success": True,
                "expression": expression,
                "result": result
            }

        except ZeroDivisionError:
            return {"success": False, "error": "Division by zero"}
        except (ValueError, TypeError) as e:
            return {"success": False, "error": f"Math error: {e}"}
        except SyntaxError:
            return {"success": False, "error": "Invalid expression syntax"}
        except Exception as e:
            logger.error(f"calculate error: {e}")
            return {"success": False, "error": str(e)}

    def _eval_node(self, node):
        """Recursively evaluate AST node."""
        if isinstance(node, ast.Constant):
            return node.value
        elif hasattr(ast, 'Num') and isinstance(node, ast.Num):  # Python < 3.8 compat
            return node.n
        elif isinstance(node, ast.BinOp):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            op = self._math_operators.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
            return op(left, right)
        elif isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand)
            op = self._math_operators.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
            return op(operand)
        elif isinstance(node, ast.Call):
            func_name = node.func.id if isinstance(node.func, ast.Name) else None
            if func_name not in self._math_functions:
                raise ValueError(f"Unsupported function: {func_name}")
            args = [self._eval_node(arg) for arg in node.args]
            return self._math_functions[func_name](*args)
        elif isinstance(node, ast.Name):
            if node.id in self._math_functions:
                return self._math_functions[node.id]
            raise ValueError(f"Unknown name: {node.id}")
        else:
            raise ValueError(f"Unsupported expression type: {type(node).__name__}")

    async def generate_uuid(
        self,
        version: int = 4,
        count: int = 1
    ) -> Dict[str, Any]:
        """
        Generate UUID(s).

        Args:
            version: UUID version (1 or 4)
            count: Number of UUIDs

        Returns:
            Dict with UUID(s)
        """
        try:
            count = min(max(count, 1), 100)

            if version == 1:
                uuids = [str(uuid.uuid1()) for _ in range(count)]
            else:
                uuids = [str(uuid.uuid4()) for _ in range(count)]

            return {
                "success": True,
                "version": version,
                "count": count,
                "uuids": uuids if count > 1 else uuids[0]
            }

        except Exception as e:
            logger.error(f"generate_uuid error: {e}")
            return {"success": False, "error": str(e)}

    async def encode_decode(
        self,
        text: str,
        method: str
    ) -> Dict[str, Any]:
        """
        Encode or decode text.

        Args:
            text: Input text
            method: Encoding method

        Returns:
            Dict with result
        """
        try:
            if method == "base64_encode":
                result = base64.b64encode(text.encode()).decode()
            elif method == "base64_decode":
                result = base64.b64decode(text).decode()
            elif method == "url_encode":
                result = quote(text, safe='')
            elif method == "url_decode":
                result = unquote(text)
            elif method == "hex_encode":
                result = text.encode().hex()
            elif method == "hex_decode":
                result = bytes.fromhex(text).decode()
            else:
                return {"success": False, "error": f"Unknown method: {method}"}

            return {
                "success": True,
                "method": method,
                "input_length": len(text),
                "output_length": len(result),
                "result": result
            }

        except Exception as e:
            logger.error(f"encode_decode error: {e}")
            return {"success": False, "error": str(e)}

    async def hash_text(
        self,
        text: str,
        algorithm: str = "sha256"
    ) -> Dict[str, Any]:
        """
        Hash text.

        Args:
            text: Input text
            algorithm: Hash algorithm

        Returns:
            Dict with hash
        """
        try:
            if algorithm == "md5":
                h = hashlib.md5(text.encode()).hexdigest()
            elif algorithm == "sha1":
                h = hashlib.sha1(text.encode()).hexdigest()
            elif algorithm == "sha256":
                h = hashlib.sha256(text.encode()).hexdigest()
            elif algorithm == "sha512":
                h = hashlib.sha512(text.encode()).hexdigest()
            else:
                return {"success": False, "error": f"Unknown algorithm: {algorithm}"}

            return {
                "success": True,
                "algorithm": algorithm,
                "input_length": len(text),
                "hash": h
            }

        except Exception as e:
            logger.error(f"hash_text error: {e}")
            return {"success": False, "error": str(e)}

    async def regex_match(
        self,
        pattern: str,
        text: str,
        flags: str = "",
        find_all: bool = True
    ) -> Dict[str, Any]:
        """
        Test regex pattern against text.

        Args:
            pattern: Regex pattern
            text: Text to match
            flags: Regex flags
            find_all: Find all matches

        Returns:
            Dict with matches
        """
        try:
            # Parse flags
            re_flags = 0
            if 'i' in flags:
                re_flags |= re.IGNORECASE
            if 'm' in flags:
                re_flags |= re.MULTILINE
            if 's' in flags:
                re_flags |= re.DOTALL

            compiled = re.compile(pattern, re_flags)

            if find_all:
                matches = compiled.findall(text)
                # Handle groups
                if matches and isinstance(matches[0], tuple):
                    matches = [list(m) for m in matches]
            else:
                match = compiled.search(text)
                if match:
                    matches = [match.group(0)]
                    groups = match.groups()
                    if groups:
                        matches = [{"full": match.group(0), "groups": list(groups)}]
                else:
                    matches = []

            return {
                "success": True,
                "pattern": pattern,
                "match_count": len(matches),
                "matches": matches
            }

        except re.error as e:
            return {"success": False, "error": f"Invalid regex: {e}"}
        except Exception as e:
            logger.error(f"regex_match error: {e}")
            return {"success": False, "error": str(e)}

    async def text_transform(
        self,
        text: str,
        operation: str
    ) -> Dict[str, Any]:
        """
        Transform text.

        Args:
            text: Input text
            operation: Transformation to apply

        Returns:
            Dict with result
        """
        try:
            if operation == "lowercase":
                result = text.lower()
            elif operation == "uppercase":
                result = text.upper()
            elif operation == "titlecase":
                result = text.title()
            elif operation == "capitalize":
                result = text.capitalize()
            elif operation == "trim":
                result = text.strip()
            elif operation == "strip_html":
                result = re.sub(r'<[^>]+>', '', text)
            elif operation == "remove_whitespace":
                result = re.sub(r'\s+', ' ', text).strip()
            elif operation == "reverse":
                result = text[::-1]
            elif operation == "count_words":
                words = text.split()
                return {
                    "success": True,
                    "operation": operation,
                    "result": len(words)
                }
            elif operation == "count_chars":
                return {
                    "success": True,
                    "operation": operation,
                    "result": len(text),
                    "without_spaces": len(text.replace(" ", ""))
                }
            else:
                return {"success": False, "error": f"Unknown operation: {operation}"}

            return {
                "success": True,
                "operation": operation,
                "input_length": len(text),
                "output_length": len(result),
                "result": result
            }

        except Exception as e:
            logger.error(f"text_transform error: {e}")
            return {"success": False, "error": str(e)}

    async def random_string(
        self,
        length: int = 16,
        charset: str = "alphanumeric",
        custom_chars: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a random string.

        Args:
            length: String length
            charset: Character set to use
            custom_chars: Custom characters

        Returns:
            Dict with random string
        """
        import secrets
        import string

        try:
            length = min(max(length, 1), 1000)

            if charset == "alphanumeric":
                chars = string.ascii_letters + string.digits
            elif charset == "alpha":
                chars = string.ascii_letters
            elif charset == "numeric":
                chars = string.digits
            elif charset == "hex":
                chars = string.hexdigits[:16]
            elif charset == "custom" and custom_chars:
                chars = custom_chars
            else:
                return {"success": False, "error": "Invalid charset or missing custom_chars"}

            result = ''.join(secrets.choice(chars) for _ in range(length))

            return {
                "success": True,
                "length": length,
                "charset": charset,
                "result": result
            }

        except Exception as e:
            logger.error(f"random_string error: {e}")
            return {"success": False, "error": str(e)}


# Fix for get_datetime - import timezone properly
from datetime import timezone as timezone_module
