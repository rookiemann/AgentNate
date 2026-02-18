"""
Tests for n8n Deep Integration â€” 12 New Workflow Tools

Tests tool definitions, routing, persona groups, and method existence.
Does NOT require a running n8n instance.

Run with: python/python.exe test_n8n_integration.py
"""
import sys
import os
import inspect

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

passed = 0
failed = 0
errors = []


def test(name, condition, detail=""):
    global passed, failed, errors
    if condition:
        passed += 1
        print(f"  PASS: {name}")
    else:
        failed += 1
        msg = f"  FAIL: {name}"
        if detail:
            msg += f" -- {detail}"
        print(msg)
        errors.append(name)


# The 12 new tools
NEW_TOOLS = [
    "describe_credential_types",
    "create_credential",
    "update_credential",
    "delete_credential",
    "list_executions",
    "get_execution_result",
    "update_workflow",
    "activate_workflow",
    "deactivate_workflow",
    "trigger_webhook",
    "set_variable",
    "list_variables",
]

# ============================================================================
# 1. Tool Definitions
# ============================================================================
print("\n=== Tool Definitions ===")

from backend.tools.workflow_tools import TOOL_DEFINITIONS, WorkflowTools

tool_names = [t["name"] for t in TOOL_DEFINITIONS]

# Check all 12 new tools exist in TOOL_DEFINITIONS
for name in NEW_TOOLS:
    test(f"{name} in TOOL_DEFINITIONS", name in tool_names)

# Existing 7 should still be there
existing = ["describe_node", "list_credentials", "build_workflow",
            "deploy_workflow", "list_workflows", "delete_workflow",
            "delete_all_workflows"]
for name in existing:
    test(f"{name} still in TOOL_DEFINITIONS", name in tool_names)

test("Workflow tool count covers baseline",
     len(TOOL_DEFINITIONS) >= len(existing) + len(NEW_TOOLS),
     f"got {len(TOOL_DEFINITIONS)}, expected at least {len(existing) + len(NEW_TOOLS)}")


# ============================================================================
# 2. AVAILABLE_TOOLS
# ============================================================================
print("\n=== AVAILABLE_TOOLS ===")

from backend.tools.tool_router import AVAILABLE_TOOLS

all_names = [t["name"] for t in AVAILABLE_TOOLS]
for name in NEW_TOOLS:
    test(f"{name} in AVAILABLE_TOOLS", name in all_names)

print(f"  (Total AVAILABLE_TOOLS: {len(all_names)})")


# ============================================================================
# 3. TOOL_GROUPS
# ============================================================================
print("\n=== TOOL_GROUPS ===")

from backend.personas import TOOL_GROUPS

workflow_group = TOOL_GROUPS.get("workflow", [])
for name in NEW_TOOLS:
    test(f"{name} in TOOL_GROUPS['workflow']", name in workflow_group)

test("Workflow group covers baseline",
     len(workflow_group) >= len(existing) + len(NEW_TOOLS),
     f"got {len(workflow_group)}, expected at least {len(existing) + len(NEW_TOOLS)}")


# ============================================================================
# 4. Route Wiring
# ============================================================================
print("\n=== Route Wiring ===")

# We can't instantiate ToolRouter without real deps, but we can check
# the route setup code references all tool names
router_source = inspect.getsource(
    __import__("backend.tools.tool_router", fromlist=["ToolRouter"]).ToolRouter.__init__
)

for name in NEW_TOOLS:
    test(f"{name} wired in ToolRouter", f'"{name}"' in router_source)


# ============================================================================
# 5. Parameter Validation
# ============================================================================
print("\n=== Parameter Validation ===")

tool_defs = {t["name"]: t for t in TOOL_DEFINITIONS}

# describe_credential_types
td = tool_defs["describe_credential_types"]
test("describe_credential_types has filter param",
     "filter" in td["parameters"]["properties"])
test("describe_credential_types has no required params",
     td["parameters"]["required"] == [])

# create_credential
td = tool_defs["create_credential"]
test("create_credential requires name, credential_type, data",
     set(td["parameters"]["required"]) == {"name", "credential_type", "data"})
test("create_credential has data as object type",
     td["parameters"]["properties"]["data"]["type"] == "object")

# update_credential
td = tool_defs["update_credential"]
test("update_credential requires credential_id",
     td["parameters"]["required"] == ["credential_id"])

# delete_credential
td = tool_defs["delete_credential"]
test("delete_credential requires credential_id",
     td["parameters"]["required"] == ["credential_id"])

# list_executions
td = tool_defs["list_executions"]
test("list_executions has status enum",
     "enum" in td["parameters"]["properties"]["status"])
test("list_executions status options include error",
     "error" in td["parameters"]["properties"]["status"]["enum"])

# get_execution_result
td = tool_defs["get_execution_result"]
test("get_execution_result requires execution_id",
     td["parameters"]["required"] == ["execution_id"])

# update_workflow
td = tool_defs["update_workflow"]
test("update_workflow requires workflow_id and workflow_json",
     set(td["parameters"]["required"]) == {"workflow_id", "workflow_json"})

# activate_workflow
td = tool_defs["activate_workflow"]
test("activate_workflow requires workflow_id",
     td["parameters"]["required"] == ["workflow_id"])

# deactivate_workflow
td = tool_defs["deactivate_workflow"]
test("deactivate_workflow requires workflow_id",
     td["parameters"]["required"] == ["workflow_id"])

# trigger_webhook
td = tool_defs["trigger_webhook"]
test("trigger_webhook requires webhook_path",
     td["parameters"]["required"] == ["webhook_path"])
test("trigger_webhook has method enum",
     "enum" in td["parameters"]["properties"]["method"])
test("trigger_webhook has test_mode param",
     "test_mode" in td["parameters"]["properties"])

# set_variable
td = tool_defs["set_variable"]
test("set_variable requires key and value",
     set(td["parameters"]["required"]) == {"key", "value"})

# list_variables
td = tool_defs["list_variables"]
test("list_variables has no required params",
     td["parameters"]["required"] == [])


# ============================================================================
# 6. Method Existence on WorkflowTools
# ============================================================================
print("\n=== WorkflowTools Methods ===")

for name in NEW_TOOLS:
    test(f"WorkflowTools.{name} exists",
         hasattr(WorkflowTools, name) and callable(getattr(WorkflowTools, name)))

# Check they're all async
for name in NEW_TOOLS:
    method = getattr(WorkflowTools, name)
    test(f"WorkflowTools.{name} is async",
         inspect.iscoroutinefunction(method))

# Check helper methods
test("WorkflowTools._n8n_request helper exists",
     hasattr(WorkflowTools, '_n8n_request'))
test("WorkflowTools._check_n8n helper exists",
     hasattr(WorkflowTools, '_check_n8n'))


# ============================================================================
# 7. _check_n8n returns error for missing instances
# ============================================================================
print("\n=== Instance Check ===")

class MockN8nManager:
    instances = {5678: "running"}

tools = WorkflowTools(orchestrator=None, n8n_manager=MockN8nManager())
result = tools._check_n8n(5678)
test("_check_n8n returns None for running instance", result is None)

result = tools._check_n8n(9999)
test("_check_n8n returns error for missing instance", result is not None)
test("_check_n8n error has success=False", result.get("success") == False)


# ============================================================================
# 8. Persona Prompts
# ============================================================================
print("\n=== Persona Prompts ===")

from backend.personas import PREDEFINED_PERSONAS

persona_map = {p.id: p for p in PREDEFINED_PERSONAS}

# workflow_builder should mention credentials and management
wb = persona_map.get("workflow_builder")
if wb:
    test("workflow_builder mentions describe_credential_types",
         "describe_credential_types" in wb.system_prompt)
    test("workflow_builder mentions create_credential",
         "create_credential" in wb.system_prompt)
    test("workflow_builder mentions trigger_webhook",
         "trigger_webhook" in wb.system_prompt)
    test("workflow_builder mentions list_executions",
         "list_executions" in wb.system_prompt)
    test("workflow_builder mentions set_variable",
         "set_variable" in wb.system_prompt)
else:
    test("workflow_builder persona found", False)

# system_agent should mention new tools
sa = persona_map.get("system_agent")
if sa:
    test("system_agent mentions describe_credential_types",
         "describe_credential_types" in sa.system_prompt)
    test("system_agent mentions trigger_webhook",
         "trigger_webhook" in sa.system_prompt)
    test("system_agent mentions set_variable",
         "set_variable" in sa.system_prompt)
else:
    test("system_agent persona found", False)

# power_agent should mention new tools
pa = persona_map.get("power_agent")
if pa:
    test("power_agent mentions create_credential",
         "create_credential" in pa.system_prompt)
    test("power_agent mentions trigger_webhook",
         "trigger_webhook" in pa.system_prompt)
else:
    test("power_agent persona found", False)

# automator should mention new flow
auto = persona_map.get("automator")
if auto:
    test("automator mentions describe_credential_types",
         "describe_credential_types" in auto.system_prompt)
    test("automator mentions activate_workflow",
         "activate_workflow" in auto.system_prompt)
    test("automator mentions list_executions",
         "list_executions" in auto.system_prompt)
else:
    test("automator persona found", False)


# ============================================================================
# Summary
# ============================================================================
print(f"\n{'='*60}")
print(f"Results: {passed} passed, {failed} failed")
if errors:
    print(f"Failed tests:")
    for e in errors:
        print(f"  - {e}")
print(f"{'='*60}")

sys.exit(0 if failed == 0 else 1)


