"""
Post-refactor health checks — verifies no broken imports, orphaned references,
or mismatched tool registrations after the race executor rewrite.

Run: cd E:\AgentNate && python\python.exe tests\post_refactor_health.py
"""

import sys
import os
import importlib
import inspect
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS = 0
FAIL = 0

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} — {detail}")


print("=" * 70)
print("POST-REFACTOR HEALTH CHECKS")
print("=" * 70)

# ──────────────────────────────────────────────────────────────────────
# 1. Module import health — all core backend modules import cleanly
# ──────────────────────────────────────────────────────────────────────
print("\n--- 1. Module Import Health ---")

CORE_MODULES = [
    "backend.race_executor",
    "backend.agent_loop",
    "backend.tools.agent_tools",
    "backend.tools.tool_router",
    "backend.personas",
    "backend.tools.workflow_tools",
    "backend.tools.comfyui_tools",
    "backend.tools.marketplace_tools",
    "backend.routing_presets",
    "backend.conversation_store",
    "backend.agent_memory",
    "backend.agent_intelligence",
    "settings.settings_manager",
]

for mod_name in CORE_MODULES:
    try:
        m = importlib.import_module(mod_name)
        check(f"import {mod_name}", True)
    except Exception as e:
        check(f"import {mod_name}", False, f"{type(e).__name__}: {e}")

# ──────────────────────────────────────────────────────────────────────
# 2. Race executor internals
# ──────────────────────────────────────────────────────────────────────
print("\n--- 2. Race Executor Internals ---")

from backend.race_executor import (
    RACEABLE_TOOLS, RACE_VALIDATORS, _SAFE_ARG_OVERRIDES, _ARG_KEYS,
    run_tool_race, _extract_raw_args,
)

check("RACEABLE_TOOLS is a set", isinstance(RACEABLE_TOOLS, set))
check("RACEABLE_TOOLS has build_workflow", "build_workflow" in RACEABLE_TOOLS)
check("RACEABLE_TOOLS has comfyui_build_workflow", "comfyui_build_workflow" in RACEABLE_TOOLS)
check("RACEABLE_TOOLS size == 2", len(RACEABLE_TOOLS) == 2,
      f"got {len(RACEABLE_TOOLS)}: {RACEABLE_TOOLS}")

check("RACE_VALIDATORS has entries for all raceable tools",
      all(t in RACE_VALIDATORS for t in RACEABLE_TOOLS),
      f"missing: {RACEABLE_TOOLS - set(RACE_VALIDATORS.keys())}")

check("_SAFE_ARG_OVERRIDES has entries for all raceable tools",
      all(t in _SAFE_ARG_OVERRIDES for t in RACEABLE_TOOLS),
      f"missing: {RACEABLE_TOOLS - set(_SAFE_ARG_OVERRIDES.keys())}")

check("_ARG_KEYS has entries for all raceable tools",
      all(t in _ARG_KEYS for t in RACEABLE_TOOLS),
      f"missing: {RACEABLE_TOOLS - set(_ARG_KEYS.keys())}")

# Verify run_tool_race is async
import asyncio
check("run_tool_race is a coroutine function",
      inspect.iscoroutinefunction(run_tool_race))

# Verify no run_creative_race exists
check("run_creative_race removed",
      not hasattr(importlib.import_module("backend.race_executor"), "run_creative_race"))
check("_score_creative removed",
      not hasattr(importlib.import_module("backend.race_executor"), "_score_creative"))
check("CREATIVE_VARIANT_PROMPTS removed",
      not hasattr(importlib.import_module("backend.race_executor"), "CREATIVE_VARIANT_PROMPTS"))

# ──────────────────────────────────────────────────────────────────────
# 3. Agent tools — removed code is gone
# ──────────────────────────────────────────────────────────────────────
print("\n--- 3. Agent Tools Dead Code Removal ---")

from backend.tools.agent_tools import AgentTools, SubAgentState

check("AgentTools has spawn_agent", hasattr(AgentTools, "spawn_agent"))
check("AgentTools has batch_spawn_agents", hasattr(AgentTools, "batch_spawn_agents"))
check("AgentTools has check_agents", hasattr(AgentTools, "check_agents"))
check("AgentTools has abort_all", hasattr(AgentTools, "abort_all"))

# Removed methods
check("super_spawn removed", not hasattr(AgentTools, "super_spawn"))
check("_validate_candidate removed", not hasattr(AgentTools, "_validate_candidate"))
check("_summarize_candidate_failure removed", not hasattr(AgentTools, "_summarize_candidate_failure"))
check("_run_race removed", not hasattr(AgentTools, "_run_race"))

# RACE_CANDIDATE_FORBIDDEN_TOOLS should be gone
check("RACE_CANDIDATE_FORBIDDEN_TOOLS removed from module",
      not hasattr(importlib.import_module("backend.tools.agent_tools"),
                  "RACE_CANDIDATE_FORBIDDEN_TOOLS"))

# SubAgentState still has race_id attribute (harmless, used as None)
check("SubAgentState has abort method", hasattr(SubAgentState, "abort"))
check("SubAgentState has should_abort method", hasattr(SubAgentState, "should_abort"))

# ──────────────────────────────────────────────────────────────────────
# 4. Personas — no orphaned tool references
# ──────────────────────────────────────────────────────────────────────
print("\n--- 4. Persona Tool Group Validation ---")

from backend.personas import TOOL_GROUPS
from backend.tools.tool_router import ToolRouter

# Collect all tool names referenced in TOOL_GROUPS
all_tool_refs = set()
for group_name, tools in TOOL_GROUPS.items():
    if tools is None:
        continue
    for t in tools:
        all_tool_refs.add(t)

check("super_spawn not in any TOOL_GROUPS",
      "super_spawn" not in all_tool_refs,
      f"found in groups")

check("_run_race not in any TOOL_GROUPS",
      "_run_race" not in all_tool_refs)

# Verify spawn_agent IS still present
check("spawn_agent still in TOOL_GROUPS",
      "spawn_agent" in all_tool_refs)

# batch_spawn_agents is registered directly in tool_router, not via persona groups
check("batch_spawn_agents registered in ToolRouter",
      hasattr(ToolRouter, "_batch_spawn_wrapper"))

# ──────────────────────────────────────────────────────────────────────
# 5. Tool router — no orphaned wrappers
# ──────────────────────────────────────────────────────────────────────
print("\n--- 5. Tool Router Integrity ---")

check("ToolRouter has no _super_spawn_wrapper",
      not hasattr(ToolRouter, "_super_spawn_wrapper"))

# Check that essential wrappers still exist
check("ToolRouter has _spawn_agent_wrapper",
      hasattr(ToolRouter, "_spawn_agent_wrapper"))
check("ToolRouter has _batch_spawn_wrapper",
      hasattr(ToolRouter, "_batch_spawn_wrapper"))

# ──────────────────────────────────────────────────────────────────────
# 6. Settings — race settings exist and have valid defaults
# ──────────────────────────────────────────────────────────────────────
print("\n--- 6. Settings Integration ---")

from settings.settings_manager import SettingsManager

sm = SettingsManager()

race_enabled = sm.get("agent.tool_race_enabled")
check("agent.tool_race_enabled exists", race_enabled is not None,
      f"got {race_enabled}")
check("agent.tool_race_enabled is bool", isinstance(race_enabled, bool),
      f"got {type(race_enabled)}")

race_candidates = sm.get("agent.tool_race_candidates")
check("agent.tool_race_candidates exists", race_candidates is not None,
      f"got {race_candidates}")
check("agent.tool_race_candidates is int >= 2",
      isinstance(race_candidates, int) and race_candidates >= 2,
      f"got {race_candidates}")

# ──────────────────────────────────────────────────────────────────────
# 7. Agent loop — race interception code path exists
# ──────────────────────────────────────────────────────────────────────
print("\n--- 7. Agent Loop Race Integration ---")

import inspect
from backend.agent_loop import run_agent_loop

sig = inspect.signature(run_agent_loop)
params = list(sig.parameters.keys())

check("run_agent_loop has race_enabled param", "race_enabled" in params)
check("run_agent_loop has race_candidates param", "race_candidates" in params)

# Read source to verify RACEABLE_TOOLS import exists in the function
source = inspect.getsource(run_agent_loop)
check("agent_loop imports RACEABLE_TOOLS", "RACEABLE_TOOLS" in source)
check("agent_loop calls run_tool_race", "run_tool_race" in source)
check("agent_loop does NOT reference run_creative_race", "run_creative_race" not in source)

# ──────────────────────────────────────────────────────────────────────
# 8. _extract_raw_args handles edge cases
# ──────────────────────────────────────────────────────────────────────
print("\n--- 8. _extract_raw_args Edge Cases ---")

import json

# _extract_raw_args takes RAW TEXT (string), not parsed dicts

# Normal case: proper JSON with tool wrapper
normal = '{"tool": "build_workflow", "arguments": {"name": "test", "nodes": []}}'
result = _extract_raw_args(normal, "build_workflow")
check("Normal JSON extraction", result is not None and result.get("name") == "test")

# Direct args (no tool wrapper)
direct = '{"name": "test2", "nodes": [{"type": "webhook"}]}'
result = _extract_raw_args(direct, "build_workflow")
check("Direct args (no wrapper)", result is not None and result.get("name") == "test2")

# JSON with preamble text
preamble = 'Here is the workflow:\n{"tool": "build_workflow", "arguments": {"name": "test3", "nodes": []}}'
result = _extract_raw_args(preamble, "build_workflow")
check("JSON with preamble text", result is not None and result.get("name") == "test3")

# Empty/garbage
result = _extract_raw_args("no json here", "build_workflow")
check("Garbage text returns None", result is None)

# ──────────────────────────────────────────────────────────────────────
# 9. SYSTEM_CHANGING_TOOLS sync check
# ──────────────────────────────────────────────────────────────────────
print("\n--- 9. SYSTEM_CHANGING_TOOLS Sync ---")

# Both agent_loop.py and routes/tools.py define SYSTEM_CHANGING_TOOLS
# They must stay in sync
try:
    agent_loop_src = open("backend/agent_loop.py", "r", encoding="utf-8").read()
    routes_tools_src = open("backend/routes/tools.py", "r", encoding="utf-8").read()

    # Extract the sets (rough but functional)
    import re

    def extract_set(src, varname="SYSTEM_CHANGING_TOOLS"):
        pattern = rf'{varname}\s*=\s*\{{([^}}]+)\}}'
        m = re.search(pattern, src)
        if m:
            items = re.findall(r'"([^"]+)"', m.group(1))
            return set(items)
        return None

    al_set = extract_set(agent_loop_src)
    rt_set = extract_set(routes_tools_src)

    if al_set and rt_set:
        check("SYSTEM_CHANGING_TOOLS match between agent_loop and routes/tools",
              al_set == rt_set,
              f"agent_loop has {al_set - rt_set} extra, routes/tools has {rt_set - al_set} extra")
    else:
        check("SYSTEM_CHANGING_TOOLS found in both files",
              al_set is not None and rt_set is not None,
              f"agent_loop={'found' if al_set else 'MISSING'}, routes/tools={'found' if rt_set else 'MISSING'}")
except Exception as e:
    check("SYSTEM_CHANGING_TOOLS sync check", False, str(e))

# ──────────────────────────────────────────────────────────────────────
# 10. Stale test files
# ──────────────────────────────────────────────────────────────────────
print("\n--- 10. Stale Test Imports ---")

stale_imports = []
test_files = [
    "test_sub_agents.py",
    "test_agent_gaps.py",
    "tests/test_agent_isolation_and_router.py",
]

for tf in test_files:
    if not os.path.exists(tf):
        continue
    try:
        with open(tf, "r", encoding="utf-8") as f:
            content = f.read()
        if "RACE_CANDIDATE_FORBIDDEN_TOOLS" in content:
            stale_imports.append(f"{tf}: RACE_CANDIDATE_FORBIDDEN_TOOLS")
        if "super_spawn" in content:
            stale_imports.append(f"{tf}: super_spawn")
        if "_run_race" in content and "run_tool_race" not in content:
            stale_imports.append(f"{tf}: _run_race")
    except:
        pass

if stale_imports:
    check("No stale test imports", False, "; ".join(stale_imports))
else:
    check("No stale test imports", True)

# ──────────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"RESULTS: {PASS} passed, {FAIL} failed out of {PASS+FAIL}")
print(f"{'='*70}")
