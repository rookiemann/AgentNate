"""
Tests for Workflow Creation System Overhaul
"""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.workflow_templates import (
    NODE_REGISTRY, build_node, build_workflow_from_nodes,
    get_node_types, get_node_params, get_all_node_params
)
from backend.workflow_generator import validate_workflow, fix_workflow, create_generation_prompt


def test_all_entries_have_params():
    """Every NODE_REGISTRY entry must have a 'params' dict."""
    missing = [k for k, v in NODE_REGISTRY.items() if "params" not in v]
    assert len(missing) == 0, f"Entries missing params: {missing}"
    print(f"  PASS: all {len(NODE_REGISTRY)} entries have params")


def test_all_entries_have_builder():
    """Every NODE_REGISTRY entry must have a 'builder' callable."""
    missing = [k for k, v in NODE_REGISTRY.items() if "builder" not in v or not callable(v["builder"])]
    assert len(missing) == 0, f"Entries missing builder: {missing}"
    print(f"  PASS: all {len(NODE_REGISTRY)} entries have builder")


def test_all_entries_have_category():
    """Every NODE_REGISTRY entry must have a 'category' string."""
    missing = [k for k, v in NODE_REGISTRY.items() if "category" not in v]
    assert len(missing) == 0, f"Entries missing category: {missing}"
    print(f"  PASS: all {len(NODE_REGISTRY)} entries have category")


def test_describe_http_request():
    """describe_node for http_request returns url, method, headers, body params."""
    info = get_node_params("http_request")
    assert info is not None
    assert info["category"] == "action"
    params = info["params"]
    assert "url" in params
    assert params["url"]["required"] == True
    assert "method" in params
    assert "headers" in params
    assert "body" in params
    print("  PASS: describe http_request returns correct params")


def test_describe_postgres():
    """describe_node for postgres returns operation, table, schema, credential_id."""
    info = get_node_params("postgres")
    assert info is not None
    assert info["category"] == "database"
    params = info["params"]
    assert "operation" in params
    assert "table" in params
    assert params["table"]["required"] == True
    assert "schema" in params
    assert params["schema"]["default"] == "public"
    assert "credential_id" in params
    assert params["credential_id"]["required"] == True
    print("  PASS: describe postgres returns correct params")


def test_describe_all():
    """get_all_node_params returns all types grouped by category."""
    cats = get_all_node_params()
    assert len(cats) >= 8, f"Expected 8+ categories, got {len(cats)}"
    total = sum(len(v) for v in cats.values())
    assert total == len(NODE_REGISTRY), f"Expected {len(NODE_REGISTRY)} types, got {total}"
    print(f"  PASS: get_all_node_params returns {total} types in {len(cats)} categories")


def test_describe_nonexistent():
    """get_node_params for unknown type returns None."""
    info = get_node_params("nonexistent_node")
    assert info is None
    print("  PASS: nonexistent node returns None")


def test_build_linear_workflow():
    """build_workflow_from_nodes creates valid linear workflow."""
    wf = build_workflow_from_nodes("Test Linear", [
        {"type": "manual_trigger"},
        {"type": "http_request", "url": "https://example.com"},
        {"type": "set_field", "field": "result", "value": "done"},
    ])
    assert wf["name"] == "Test Linear"
    assert len(wf["nodes"]) == 3
    assert len(wf["connections"]) == 2  # 2 linear connections
    is_valid, errors = validate_workflow(wf)
    assert is_valid, f"Workflow not valid: {errors}"
    print("  PASS: linear workflow builds and validates correctly")


def test_build_branching_workflow():
    """build_workflow_from_nodes with custom_connections creates branching workflow."""
    wf = build_workflow_from_nodes("Test Branch", [
        {"type": "manual_trigger"},
        {"type": "if", "field": "status", "compare_value": "ok"},
        {"type": "set_field", "name": "True Path", "field": "msg", "value": "yes"},
        {"type": "set_field", "name": "False Path", "field": "msg", "value": "no"},
        {"type": "merge", "name": "Merge"},
    ], custom_connections=[
        {"from": "Manual Trigger", "to": "IF"},
        {"from": "IF", "to": "True Path", "output": 0},
        {"from": "IF", "to": "False Path", "output": 1},
        {"from": "True Path", "to": "Merge"},
        {"from": "False Path", "to": "Merge", "input": 1},
    ])

    assert len(wf["nodes"]) == 5
    conns = wf["connections"]
    # IF should have 2 outputs
    assert len(conns["IF"]["main"]) == 2
    assert conns["IF"]["main"][0][0]["node"] == "True Path"
    assert conns["IF"]["main"][1][0]["node"] == "False Path"
    # Merge should receive from both
    assert conns["True Path"]["main"][0][0]["node"] == "Merge"
    assert conns["False Path"]["main"][0][0]["node"] == "Merge"
    assert conns["False Path"]["main"][0][0]["index"] == 1  # input 1

    is_valid, errors = validate_workflow(wf)
    assert is_valid, f"Branching workflow not valid: {errors}"
    print("  PASS: branching workflow builds with correct connections")


def test_build_invalid_node():
    """build_node with unknown type raises ValueError."""
    try:
        build_node("nonexistent_type_xyz")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Unknown node type" in str(e)
    print("  PASS: unknown node type raises ValueError")


def test_build_custom_connection_bad_name():
    """Custom connection with bad node name raises ValueError."""
    try:
        build_workflow_from_nodes("Bad", [
            {"type": "manual_trigger"},
        ], custom_connections=[
            {"from": "Manual Trigger", "to": "Nonexistent Node"},
        ])
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "not found" in str(e)
    print("  PASS: bad connection target raises ValueError")


def test_old_dicts_removed():
    """Old template dicts (TRIGGERS, AI_NODES, etc.) should not exist."""
    import backend.workflow_templates as wt
    assert not hasattr(wt, "TRIGGERS"), "TRIGGERS dict should be removed"
    assert not hasattr(wt, "AI_NODES"), "AI_NODES dict should be removed"
    assert not hasattr(wt, "ACTION_NODES"), "ACTION_NODES dict should be removed"
    assert not hasattr(wt, "DATA_NODES"), "DATA_NODES dict should be removed"
    assert not hasattr(wt, "get_template_descriptions"), "get_template_descriptions should be removed"
    print("  PASS: old template dicts removed")


def test_get_node_types_has_param_hints():
    """get_node_types returns descriptions with required/optional param hints."""
    types = get_node_types()
    pg = types.get("postgres", "")
    assert "required:" in pg, f"postgres description should show required params: {pg}"
    assert "table" in pg, f"postgres should mention 'table': {pg}"
    print("  PASS: get_node_types includes param hints")


def test_workflow_generator_import():
    """workflow_generator.py should import without old dicts."""
    # This would fail if it still tries to import TRIGGERS etc
    from backend.workflow_generator import create_generation_prompt
    system, user = create_generation_prompt("test workflow", "manual")
    assert "manual_trigger" in system, "System prompt should mention node types"
    print("  PASS: workflow_generator imports and works without old dicts")


def test_all_builders_callable():
    """Every builder should produce a valid node when called with empty params."""
    failures = []
    for type_name, entry in NODE_REGISTRY.items():
        try:
            node = entry["builder"]({})
            if not isinstance(node, dict):
                failures.append(f"{type_name}: builder returned {type(node)}, expected dict")
            elif "type" not in node:
                failures.append(f"{type_name}: builder output missing 'type' field")
            elif "name" not in node:
                failures.append(f"{type_name}: builder output missing 'name' field")
        except Exception as e:
            failures.append(f"{type_name}: builder raised {e}")

    assert len(failures) == 0, f"Builder failures:\n" + "\n".join(failures)
    print(f"  PASS: all {len(NODE_REGISTRY)} builders produce valid nodes with empty params")


def test_param_schema_structure():
    """Param schemas should have required fields."""
    issues = []
    for type_name, entry in NODE_REGISTRY.items():
        for pname, pschema in entry.get("params", {}).items():
            if "type" not in pschema:
                issues.append(f"{type_name}.{pname}: missing 'type'")
            if "description" not in pschema:
                issues.append(f"{type_name}.{pname}: missing 'description'")
            if "required" not in pschema and pname != "name":
                issues.append(f"{type_name}.{pname}: missing 'required'")

    assert len(issues) == 0, f"Schema issues:\n" + "\n".join(issues)
    print(f"  PASS: all param schemas have type, description, required")


def test_renamed_ai_nodes():
    """LLM summarize/classify should be llm_summarize/llm_classify (not conflicting with data nodes)."""
    assert "llm_summarize" in NODE_REGISTRY, "llm_summarize should exist"
    assert "llm_classify" in NODE_REGISTRY, "llm_classify should exist"
    # The data-level summarize should still exist
    assert "summarize" in NODE_REGISTRY, "data summarize should still exist"
    # They should be different categories
    assert NODE_REGISTRY["llm_summarize"]["category"] == "ai"
    assert NODE_REGISTRY["summarize"]["category"] == "data"
    print("  PASS: LLM and data summarize nodes properly separated")


if __name__ == "__main__":
    print("=== Workflow Creation Overhaul Tests ===\n")

    tests = [
        test_all_entries_have_params,
        test_all_entries_have_builder,
        test_all_entries_have_category,
        test_describe_http_request,
        test_describe_postgres,
        test_describe_all,
        test_describe_nonexistent,
        test_build_linear_workflow,
        test_build_branching_workflow,
        test_build_invalid_node,
        test_build_custom_connection_bad_name,
        test_old_dicts_removed,
        test_get_node_types_has_param_hints,
        test_workflow_generator_import,
        test_all_builders_callable,
        test_param_schema_structure,
        test_renamed_ai_nodes,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {test.__name__}: {e}")
            failed += 1

    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    if failed == 0:
        print("All tests passed!")
