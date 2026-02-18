"""
GGUF file utilities for reading model metadata.

GGUF format stores model metadata including context length, architecture, etc.
"""

import struct
import os
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

# GGUF magic number and version
GGUF_MAGIC = 0x46554747  # "GGUF" in little-endian
GGUF_VERSION = 3

# GGUF value types
GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12


def read_gguf_string(f) -> str:
    """Read a GGUF string (length-prefixed)."""
    length = struct.unpack('<Q', f.read(8))[0]
    return f.read(length).decode('utf-8')


def read_gguf_value(f, value_type: int) -> Any:
    """Read a GGUF value based on its type."""
    if value_type == GGUF_TYPE_UINT8:
        return struct.unpack('<B', f.read(1))[0]
    elif value_type == GGUF_TYPE_INT8:
        return struct.unpack('<b', f.read(1))[0]
    elif value_type == GGUF_TYPE_UINT16:
        return struct.unpack('<H', f.read(2))[0]
    elif value_type == GGUF_TYPE_INT16:
        return struct.unpack('<h', f.read(2))[0]
    elif value_type == GGUF_TYPE_UINT32:
        return struct.unpack('<I', f.read(4))[0]
    elif value_type == GGUF_TYPE_INT32:
        return struct.unpack('<i', f.read(4))[0]
    elif value_type == GGUF_TYPE_FLOAT32:
        return struct.unpack('<f', f.read(4))[0]
    elif value_type == GGUF_TYPE_BOOL:
        return struct.unpack('<B', f.read(1))[0] != 0
    elif value_type == GGUF_TYPE_STRING:
        return read_gguf_string(f)
    elif value_type == GGUF_TYPE_UINT64:
        return struct.unpack('<Q', f.read(8))[0]
    elif value_type == GGUF_TYPE_INT64:
        return struct.unpack('<q', f.read(8))[0]
    elif value_type == GGUF_TYPE_FLOAT64:
        return struct.unpack('<d', f.read(8))[0]
    elif value_type == GGUF_TYPE_ARRAY:
        array_type = struct.unpack('<I', f.read(4))[0]
        array_len = struct.unpack('<Q', f.read(8))[0]
        return [read_gguf_value(f, array_type) for _ in range(array_len)]
    else:
        raise ValueError(f"Unknown GGUF value type: {value_type}")


def get_gguf_metadata(file_path: str) -> Dict[str, Any]:
    """
    Read metadata from a GGUF file.

    Returns dict with keys like:
    - context_length: Max context window size
    - architecture: Model architecture (llama, mistral, etc.)
    - name: Model name
    - parameters: Parameter count
    - quantization: Quantization type
    """
    metadata = {}

    try:
        with open(file_path, 'rb') as f:
            # Read header
            magic = struct.unpack('<I', f.read(4))[0]
            if magic != GGUF_MAGIC:
                logger.warning(f"Invalid GGUF magic: {hex(magic)}")
                return metadata

            version = struct.unpack('<I', f.read(4))[0]
            if version < 2:
                logger.warning(f"Unsupported GGUF version: {version}")
                return metadata

            tensor_count = struct.unpack('<Q', f.read(8))[0]
            metadata_kv_count = struct.unpack('<Q', f.read(8))[0]

            # Read metadata key-value pairs
            for _ in range(metadata_kv_count):
                key = read_gguf_string(f)
                value_type = struct.unpack('<I', f.read(4))[0]
                value = read_gguf_value(f, value_type)
                metadata[key] = value

    except FileNotFoundError:
        logger.error(f"GGUF file not found: {file_path}")
    except Exception as e:
        logger.error(f"Error reading GGUF metadata: {e}")

    return metadata


def get_model_context_length(file_path: str) -> Optional[int]:
    """
    Get the maximum context length for a GGUF model.

    Uses optimized reading that stops early once context_length is found.
    """
    # Common keys for context length (check these first)
    context_keys = {
        'llama.context_length',
        'mistral.context_length',
        'qwen2.context_length',
        'phi3.context_length',
        'gemma.context_length',
        'general.context_length',
        'general.architecture',  # Need this to construct arch-specific key
    }

    try:
        with open(file_path, 'rb') as f:
            # Read header
            magic = struct.unpack('<I', f.read(4))[0]
            if magic != GGUF_MAGIC:
                return None

            version = struct.unpack('<I', f.read(4))[0]
            if version < 2:
                return None

            struct.unpack('<Q', f.read(8))[0]  # tensor_count (skip)
            metadata_kv_count = struct.unpack('<Q', f.read(8))[0]

            found_values = {}

            # Read metadata key-value pairs, stopping early if possible
            for _ in range(metadata_kv_count):
                key = read_gguf_string(f)
                value_type = struct.unpack('<I', f.read(4))[0]

                # Only read value if it's a key we care about
                if key in context_keys or key.endswith('.context_length'):
                    value = read_gguf_value(f, value_type)
                    found_values[key] = value

                    # If we found a direct context_length, return immediately
                    if key.endswith('.context_length') and isinstance(value, int) and value > 0:
                        return value
                else:
                    # Skip this value - need to read but discard
                    read_gguf_value(f, value_type)

            # Check found values for context length
            for key in ['llama.context_length', 'mistral.context_length', 'qwen2.context_length',
                       'phi3.context_length', 'gemma.context_length', 'general.context_length']:
                if key in found_values:
                    value = found_values[key]
                    if isinstance(value, int) and value > 0:
                        return value

            # Try architecture-specific key
            arch = found_values.get('general.architecture', '')
            if arch:
                arch_key = f'{arch}.context_length'
                if arch_key in found_values:
                    return found_values[arch_key]

    except Exception as e:
        logger.debug(f"Error reading GGUF context length: {e}")

    return None


def get_model_info(file_path: str) -> Dict[str, Any]:
    """
    Get comprehensive model info from GGUF file.

    Returns:
        Dict with context_length, architecture, name, etc.
    """
    metadata = get_gguf_metadata(file_path)

    info = {
        'context_length': None,
        'architecture': metadata.get('general.architecture'),
        'name': metadata.get('general.name'),
        'quantization': metadata.get('general.quantization_version'),
        'file_type': metadata.get('general.file_type'),
    }

    # Get context length
    info['context_length'] = get_model_context_length(file_path)

    # If context_length still None, try to estimate from common patterns
    if info['context_length'] is None:
        file_name = os.path.basename(file_path).lower()
        # Common context length patterns in model names
        if '128k' in file_name:
            info['context_length'] = 131072
        elif '64k' in file_name:
            info['context_length'] = 65536
        elif '32k' in file_name:
            info['context_length'] = 32768
        elif '16k' in file_name:
            info['context_length'] = 16384
        elif '8k' in file_name:
            info['context_length'] = 8192
        else:
            # Default fallback based on architecture
            arch = info.get('architecture', '').lower()
            if 'llama' in arch:
                info['context_length'] = 8192  # Llama 2 default
            elif 'mistral' in arch:
                info['context_length'] = 32768  # Mistral default
            elif 'qwen' in arch:
                info['context_length'] = 32768  # Qwen default
            elif 'phi' in arch:
                info['context_length'] = 16384  # Phi default
            else:
                info['context_length'] = 4096  # Safe fallback

    return info
