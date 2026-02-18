"""
Workflow Bridge - Generate n8n workflow JSON from routing presets.

Four patterns for generating workflows that use the routing system:
  1. swarm     - Parallel personas processing same input
  2. pipeline  - Sequential stages, each persona's output feeds the next
  3. multi_coder - N parallel coders + reviewer
  4. image_pipeline - LLM generates prompt → ComfyUI generates image

Every generated workflow:
  - Has a Webhook trigger node
  - Starts with a Code node that resolves personas via /api/routing/resolve/{persona}
  - Uses resolved instance_ids for LLM HTTP Request calls
  - Has a Respond to Webhook node returning results
"""

import uuid
import json
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger("workflow_bridge")

AGENTNATE_BASE = "http://localhost:8000"


def _node_id() -> str:
    return str(uuid.uuid4())


def _webhook_id() -> str:
    return str(uuid.uuid4())


def _pos(col: int, row: int) -> list:
    """Position helper. col 0 = trigger, col 1+ = processing."""
    return [250 + col * 250, 200 + row * 150]


# ---------------------------------------------------------------------------
# Shared Code-Node JavaScript Generators
# ---------------------------------------------------------------------------

def _resolve_code_js(personas: List[str]) -> str:
    """
    JS for a Code node that calls /api/routing/resolve/{persona} for each
    persona. Returns: {resolved: {persona1: "instance-id", ...}, input: $json}
    """
    # Build fetch calls for each persona
    fetches = []
    for p in personas:
        fetches.append(
            f'    fetch("{AGENTNATE_BASE}/api/routing/resolve/{p}")'
            f'.then(r => r.json()).then(d => ["{p}", d.instance_id || null])'
        )
    fetches_str = ",\n".join(fetches)

    return f"""// Resolve persona → model instance via routing presets
const results = await Promise.all([
{fetches_str}
]);

const resolved = {{}};
for (const [persona, instanceId] of results) {{
  resolved[persona] = instanceId;
}}

return [{{ json: {{ resolved, input: $input.first().json }} }}];"""


def _llm_call_js(persona_key: str, system_prompt: str,
                 prompt_expr: str, max_tokens: int = 2048,
                 output_field: str = "") -> str:
    """
    JS for a Code node that POSTs to /api/chat/completions using a resolved
    instance_id. Returns the LLM response text in a named field.
    """
    field = output_field or f"{persona_key}_response"
    # Escape backticks and backslashes in system_prompt for JS template
    sys_escaped = system_prompt.replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")

    return f"""// Call LLM via resolved instance for persona: {persona_key}
const data = $input.first().json;
const instanceId = data.resolved["{persona_key}"];

if (!instanceId) {{
  return [{{ json: {{ ...data, {field}: "ERROR: No model routed for {persona_key}" }} }}];
}}

const resp = await fetch("{AGENTNATE_BASE}/api/chat/completions", {{
  method: "POST",
  headers: {{ "Content-Type": "application/json" }},
  body: JSON.stringify({{
    instance_id: instanceId,
    messages: [
      {{ role: "system", content: `{sys_escaped}` }},
      {{ role: "user", content: {prompt_expr} }}
    ],
    max_tokens: {max_tokens},
    stream: false
  }})
}});

const result = await resp.json();
const text = result.choices?.[0]?.message?.content || JSON.stringify(result);

return [{{ json: {{ ...data, {field}: text }} }}];"""


# ---------------------------------------------------------------------------
# Pattern 1: Swarm (Parallel Personas)
# ---------------------------------------------------------------------------

def generate_swarm_workflow(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parallel swarm: Webhook → Resolve → [Persona1, Persona2, ...] → Merge → Respond

    Config:
        name: str - workflow name
        personas: [{id, system_prompt?, max_tokens?}]
        task_field: str - JSON field from webhook containing the task (default "task")
        webhook_path: str - webhook URL path (default "swarm")
    """
    name = config.get("name", "Routing Swarm")
    personas = config.get("personas", [])
    if not personas:
        raise ValueError("swarm pattern requires at least one persona in config.personas")

    # Normalize personas: accept strings or dicts
    persona_list = []
    for p in personas:
        if isinstance(p, str):
            persona_list.append({"id": p, "system_prompt": f"You are a {p}.", "max_tokens": 2048})
        else:
            persona_list.append({
                "id": p.get("id", p.get("persona_id", "assistant")),
                "system_prompt": p.get("system_prompt", f"You are a {p.get('id', 'assistant')}."),
                "max_tokens": p.get("max_tokens", 2048),
            })

    task_field = config.get("task_field", "task")
    webhook_path = config.get("webhook_path", "swarm")
    persona_ids = [p["id"] for p in persona_list]

    nodes = []
    connections = {}

    # Node 1: Webhook trigger
    wh_name = "Webhook"
    nodes.append({
        "id": _node_id(),
        "name": wh_name,
        "type": "n8n-nodes-base.webhook",
        "typeVersion": 2,
        "position": _pos(0, len(persona_list) // 2),
        "webhookId": _webhook_id(),
        "parameters": {
            "path": webhook_path,
            "httpMethod": "POST",
            "responseMode": "responseNode",
        },
    })

    # Node 2: Resolve routes
    resolve_name = "Resolve Routes"
    nodes.append({
        "id": _node_id(),
        "name": resolve_name,
        "type": "n8n-nodes-base.code",
        "typeVersion": 2,
        "position": _pos(1, len(persona_list) // 2),
        "parameters": {
            "mode": "runOnceForAllItems",
            "jsCode": _resolve_code_js(persona_ids),
        },
    })
    connections[wh_name] = {"main": [[{"node": resolve_name, "type": "main", "index": 0}]]}

    # Nodes 3..N: One Code node per persona (all connected from Resolve)
    resolve_outputs = []
    persona_node_names = []
    for i, p in enumerate(persona_list):
        pname = f"LLM {p['id'].title()}"
        persona_node_names.append(pname)
        nodes.append({
            "id": _node_id(),
            "name": pname,
            "type": "n8n-nodes-base.code",
            "typeVersion": 2,
            "position": _pos(2, i),
            "parameters": {
                "mode": "runOnceForAllItems",
                "jsCode": _llm_call_js(
                    p["id"],
                    p["system_prompt"],
                    f'data.input.{task_field} || JSON.stringify(data.input)',
                    p["max_tokens"],
                ),
            },
        })
        resolve_outputs.append({"node": pname, "type": "main", "index": 0})

    # All personas branch from Resolve (parallel via separate connections)
    # n8n runs all outputs simultaneously
    connections[resolve_name] = {"main": [resolve_outputs]}

    # Merge node - combines all persona outputs
    merge_name = "Merge Results"
    if len(persona_list) > 1:
        nodes.append({
            "id": _node_id(),
            "name": merge_name,
            "type": "n8n-nodes-base.code",
            "typeVersion": 2,
            "position": _pos(3, len(persona_list) // 2),
            "parameters": {
                "mode": "runOnceForAllItems",
                "jsCode": _merge_results_js(persona_ids),
            },
        })
        for pname in persona_node_names:
            connections[pname] = {"main": [[{"node": merge_name, "type": "main", "index": 0}]]}
    else:
        # Single persona - no merge needed
        merge_name = persona_node_names[0]

    # Respond node
    respond_name = "Send Response"
    nodes.append({
        "id": _node_id(),
        "name": respond_name,
        "type": "n8n-nodes-base.respondToWebhook",
        "typeVersion": 1,
        "position": _pos(4, len(persona_list) // 2),
        "parameters": {
            "respondWith": "json",
            "responseBody": "={{ JSON.stringify($json) }}",
        },
    })
    if merge_name not in connections:
        connections[merge_name] = {"main": [[{"node": respond_name, "type": "main", "index": 0}]]}
    else:
        connections[merge_name]["main"][0].append({"node": respond_name, "type": "main", "index": 0})
        # Fix: merge should only go to respond, not to respond AND personas
    # Actually, let's rebuild the merge connection cleanly
    if len(persona_list) > 1:
        connections[merge_name] = {"main": [[{"node": respond_name, "type": "main", "index": 0}]]}

    return _wrap_workflow(name, nodes, connections, webhook_path)


def _merge_results_js(persona_ids: List[str]) -> str:
    """JS to collect all persona results into a single output."""
    fields = ", ".join(f'"{pid}_response": items[{i}]?.json?.{pid}_response || "no response"'
                       for i, pid in enumerate(persona_ids))
    return f"""// Merge results from all personas
const items = $input.all();
const merged = {{
  {fields}
}};
return [{{ json: merged }}];"""


# ---------------------------------------------------------------------------
# Pattern 2: Pipeline (Sequential Stages)
# ---------------------------------------------------------------------------

def generate_pipeline_workflow(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sequential pipeline: Webhook → Resolve → Stage1 → Stage2 → ... → Respond

    Config:
        name: str
        stages: [{persona_id, system_prompt, output_field, max_tokens?}]
        webhook_path: str (default "pipeline")
    """
    name = config.get("name", "Routing Pipeline")
    raw_stages = config.get("stages", [])
    if not raw_stages:
        raise ValueError("pipeline pattern requires at least one stage in config.stages")

    # Accept both string list ["coder","reviewer"] and dict list [{"persona_id":"coder",...}]
    stages = []
    for s in raw_stages:
        if isinstance(s, str):
            stages.append({"persona_id": s, "system_prompt": f"You are a {s}.", "output_field": f"{s}_output"})
        else:
            stages.append(s)

    webhook_path = config.get("webhook_path", "pipeline")
    persona_ids = list(set(s.get("persona_id", "assistant") for s in stages))

    nodes = []
    connections = {}

    # Webhook
    wh_name = "Webhook"
    nodes.append({
        "id": _node_id(),
        "name": wh_name,
        "type": "n8n-nodes-base.webhook",
        "typeVersion": 2,
        "position": _pos(0, 0),
        "webhookId": _webhook_id(),
        "parameters": {
            "path": webhook_path,
            "httpMethod": "POST",
            "responseMode": "responseNode",
        },
    })

    # Resolve
    resolve_name = "Resolve Routes"
    nodes.append({
        "id": _node_id(),
        "name": resolve_name,
        "type": "n8n-nodes-base.code",
        "typeVersion": 2,
        "position": _pos(1, 0),
        "parameters": {
            "mode": "runOnceForAllItems",
            "jsCode": _resolve_code_js(persona_ids),
        },
    })
    connections[wh_name] = {"main": [[{"node": resolve_name, "type": "main", "index": 0}]]}

    # Chain stages sequentially
    prev_name = resolve_name
    for i, stage in enumerate(stages):
        persona_id = stage.get("persona_id", "assistant")
        system_prompt = stage.get("system_prompt", f"You are a {persona_id}.")
        output_field = stage.get("output_field", f"stage{i+1}_output")
        max_tokens = stage.get("max_tokens", 2048)

        # Build prompt expression: first stage uses input, subsequent use previous output
        if i == 0:
            prompt_expr = 'data.input.task || JSON.stringify(data.input)'
        else:
            prev_field = stages[i-1].get("output_field", f"stage{i}_output")
            prompt_expr = f'data.{prev_field}'

        stage_name = f"Stage {i+1}: {persona_id.title()}"
        nodes.append({
            "id": _node_id(),
            "name": stage_name,
            "type": "n8n-nodes-base.code",
            "typeVersion": 2,
            "position": _pos(2 + i, 0),
            "parameters": {
                "mode": "runOnceForAllItems",
                "jsCode": _llm_call_js(
                    persona_id, system_prompt, prompt_expr, max_tokens, output_field
                ),
            },
        })
        connections[prev_name] = {"main": [[{"node": stage_name, "type": "main", "index": 0}]]}
        prev_name = stage_name

    # Respond
    respond_name = "Send Response"
    nodes.append({
        "id": _node_id(),
        "name": respond_name,
        "type": "n8n-nodes-base.respondToWebhook",
        "typeVersion": 1,
        "position": _pos(2 + len(stages), 0),
        "parameters": {
            "respondWith": "json",
            "responseBody": "={{ JSON.stringify($json) }}",
        },
    })
    connections[prev_name] = {"main": [[{"node": respond_name, "type": "main", "index": 0}]]}

    return _wrap_workflow(name, nodes, connections, webhook_path)


# ---------------------------------------------------------------------------
# Pattern 3: Multi-Coder (N Coders + Reviewer)
# ---------------------------------------------------------------------------

def generate_multi_coder_workflow(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Multi-coder: Webhook → Resolve → [Coder1, Coder2, ...] → Merge → Reviewer → Respond

    Config:
        name: str
        coder_count: int (default 3)
        coder_persona: str (default "coder")
        reviewer_persona: str (default "coder")
        coder_system_prompt: str
        reviewer_system_prompt: str
        webhook_path: str (default "multi-coder")
    """
    name = config.get("name", "Multi-Coder Review")
    coder_count = config.get("coder_count", 3)
    coder_persona = config.get("coder_persona", "coder")
    reviewer_persona = config.get("reviewer_persona", "coder")
    coder_prompt = config.get("coder_system_prompt",
                              "You are an expert programmer. Write clean, correct code for the given task. "
                              "Return ONLY the code with brief comments.")
    reviewer_prompt = config.get("reviewer_system_prompt",
                                 "You are a senior code reviewer. You will receive multiple code solutions. "
                                 "Analyze each, pick the best one or synthesize the best parts, "
                                 "and return the final polished solution with explanation.")
    webhook_path = config.get("webhook_path", "multi-coder")

    persona_ids = list(set([coder_persona, reviewer_persona]))

    nodes = []
    connections = {}

    # Webhook
    wh_name = "Webhook"
    nodes.append({
        "id": _node_id(),
        "name": wh_name,
        "type": "n8n-nodes-base.webhook",
        "typeVersion": 2,
        "position": _pos(0, coder_count // 2),
        "webhookId": _webhook_id(),
        "parameters": {
            "path": webhook_path,
            "httpMethod": "POST",
            "responseMode": "responseNode",
        },
    })

    # Resolve
    resolve_name = "Resolve Routes"
    nodes.append({
        "id": _node_id(),
        "name": resolve_name,
        "type": "n8n-nodes-base.code",
        "typeVersion": 2,
        "position": _pos(1, coder_count // 2),
        "parameters": {
            "mode": "runOnceForAllItems",
            "jsCode": _resolve_code_js(persona_ids),
        },
    })
    connections[wh_name] = {"main": [[{"node": resolve_name, "type": "main", "index": 0}]]}

    # Coder nodes (parallel)
    coder_names = []
    resolve_outputs = []
    for i in range(coder_count):
        cname = f"Coder {i+1}"
        coder_names.append(cname)
        variation = f" Approach the problem from a {'different angle' if i > 0 else 'straightforward approach'}. Variation {i+1} of {coder_count}."
        nodes.append({
            "id": _node_id(),
            "name": cname,
            "type": "n8n-nodes-base.code",
            "typeVersion": 2,
            "position": _pos(2, i),
            "parameters": {
                "mode": "runOnceForAllItems",
                "jsCode": _llm_call_js(
                    coder_persona,
                    coder_prompt + variation,
                    'data.input.task || JSON.stringify(data.input)',
                    4096,
                    f"coder{i+1}_solution",
                ),
            },
        })
        resolve_outputs.append({"node": cname, "type": "main", "index": 0})

    connections[resolve_name] = {"main": [resolve_outputs]}

    # Merge coder outputs
    merge_name = "Collect Solutions"
    merge_fields = ", ".join(
        f'"coder{i+1}": items[{i}]?.json?.coder{i+1}_solution || "no response"'
        for i in range(coder_count)
    )
    nodes.append({
        "id": _node_id(),
        "name": merge_name,
        "type": "n8n-nodes-base.code",
        "typeVersion": 2,
        "position": _pos(3, coder_count // 2),
        "parameters": {
            "mode": "runOnceForAllItems",
            "jsCode": f"""// Collect all coder solutions
const items = $input.all();
const data = items[0]?.json || {{}};
const solutions = {{ {merge_fields} }};
return [{{ json: {{ ...data, solutions }} }}];""",
        },
    })
    for cname in coder_names:
        connections[cname] = {"main": [[{"node": merge_name, "type": "main", "index": 0}]]}

    # Reviewer node
    reviewer_name = "Reviewer"
    review_prompt_expr = '`Review these ${Object.keys(data.solutions).length} code solutions:\\n\\n` + Object.entries(data.solutions).map(([k,v]) => `=== ${k} ===\\n${v}`).join("\\n\\n") + `\\n\\nOriginal task: ${data.input?.task || "unknown"}`'
    nodes.append({
        "id": _node_id(),
        "name": reviewer_name,
        "type": "n8n-nodes-base.code",
        "typeVersion": 2,
        "position": _pos(4, coder_count // 2),
        "parameters": {
            "mode": "runOnceForAllItems",
            "jsCode": _llm_call_js(
                reviewer_persona,
                reviewer_prompt,
                review_prompt_expr,
                4096,
                "final_review",
            ),
        },
    })
    connections[merge_name] = {"main": [[{"node": reviewer_name, "type": "main", "index": 0}]]}

    # Respond
    respond_name = "Send Response"
    nodes.append({
        "id": _node_id(),
        "name": respond_name,
        "type": "n8n-nodes-base.respondToWebhook",
        "typeVersion": 1,
        "position": _pos(5, coder_count // 2),
        "parameters": {
            "respondWith": "json",
            "responseBody": "={{ JSON.stringify({ review: $json.final_review, solutions: $json.solutions }) }}",
        },
    })
    connections[reviewer_name] = {"main": [[{"node": respond_name, "type": "main", "index": 0}]]}

    return _wrap_workflow(name, nodes, connections, webhook_path)


# ---------------------------------------------------------------------------
# Pattern 4: Image Pipeline (LLM → ComfyUI)
# ---------------------------------------------------------------------------

def generate_image_pipeline_workflow(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Image pipeline: Webhook → Resolve → LLM(prompt gen) → ComfyUI generate → Poll → Respond

    Config:
        name: str
        prompt_persona: str (default "coder") - persona to generate optimized SD prompts
        instance_id: str - ComfyUI instance ID (required)
        checkpoint: str - model checkpoint filename (optional, uses first available)
        webhook_path: str (default "image-pipeline")
    """
    name = config.get("name", "Image Pipeline")
    prompt_persona = config.get("prompt_persona", "coder")
    comfy_instance = config.get("instance_id", "")
    checkpoint = config.get("checkpoint", "")
    webhook_path = config.get("webhook_path", "image-pipeline")

    nodes = []
    connections = {}

    # Webhook
    wh_name = "Webhook"
    nodes.append({
        "id": _node_id(),
        "name": wh_name,
        "type": "n8n-nodes-base.webhook",
        "typeVersion": 2,
        "position": _pos(0, 0),
        "webhookId": _webhook_id(),
        "parameters": {
            "path": webhook_path,
            "httpMethod": "POST",
            "responseMode": "responseNode",
        },
    })

    # Resolve routes
    resolve_name = "Resolve Routes"
    nodes.append({
        "id": _node_id(),
        "name": resolve_name,
        "type": "n8n-nodes-base.code",
        "typeVersion": 2,
        "position": _pos(1, 0),
        "parameters": {
            "mode": "runOnceForAllItems",
            "jsCode": _resolve_code_js([prompt_persona]),
        },
    })
    connections[wh_name] = {"main": [[{"node": resolve_name, "type": "main", "index": 0}]]}

    # LLM: Generate optimized SD prompt
    llm_name = "Generate SD Prompt"
    sd_system = ("You are a Stable Diffusion prompt engineer. Given a user's image description, "
                 "create an optimized prompt for image generation. Include quality tags, style descriptors, "
                 "and composition details. Also create a negative prompt. "
                 "Return ONLY a JSON object: {\"prompt\": \"...\", \"negative_prompt\": \"...\"}")
    nodes.append({
        "id": _node_id(),
        "name": llm_name,
        "type": "n8n-nodes-base.code",
        "typeVersion": 2,
        "position": _pos(2, 0),
        "parameters": {
            "mode": "runOnceForAllItems",
            "jsCode": _llm_call_js(
                prompt_persona, sd_system,
                'data.input.description || data.input.task || JSON.stringify(data.input)',
                1024, "sd_prompts",
            ),
        },
    })
    connections[resolve_name] = {"main": [[{"node": llm_name, "type": "main", "index": 0}]]}

    # Code: Parse LLM output and prepare ComfyUI request
    prep_name = "Prepare ComfyUI Request"
    checkpoint_js = f'"{checkpoint}"' if checkpoint else 'data.input.checkpoint || ""'
    instance_js = f'"{comfy_instance}"' if comfy_instance else 'data.input.instance_id || ""'
    nodes.append({
        "id": _node_id(),
        "name": prep_name,
        "type": "n8n-nodes-base.code",
        "typeVersion": 2,
        "position": _pos(3, 0),
        "parameters": {
            "mode": "runOnceForAllItems",
            "jsCode": f"""// Parse LLM response and build ComfyUI generate request
const data = $input.first().json;
let sdPrompts;
try {{
  // Try to parse JSON from LLM response
  const raw = data.sd_prompts;
  const jsonMatch = raw.match(/\\{{[\\s\\S]*\\}}/);
  sdPrompts = jsonMatch ? JSON.parse(jsonMatch[0]) : {{ prompt: raw, negative_prompt: "" }};
}} catch (e) {{
  sdPrompts = {{ prompt: data.sd_prompts, negative_prompt: "" }};
}}

return [{{ json: {{
  ...data,
  comfyui_request: {{
    instance_id: {instance_js},
    prompt: sdPrompts.prompt,
    negative_prompt: sdPrompts.negative_prompt || "",
    checkpoint: {checkpoint_js},
    seed: Math.floor(Math.random() * 2147483647)
  }}
}} }}];""",
        },
    })
    connections[llm_name] = {"main": [[{"node": prep_name, "type": "main", "index": 0}]]}

    # HTTP: POST to ComfyUI generate endpoint
    generate_name = "Generate Image"
    nodes.append({
        "id": _node_id(),
        "name": generate_name,
        "type": "n8n-nodes-base.httpRequest",
        "typeVersion": 4,
        "position": _pos(4, 0),
        "parameters": {
            "method": "POST",
            "url": f"{AGENTNATE_BASE}/api/comfyui/generate",
            "sendHeaders": True,
            "headerParameters": {
                "parameters": [
                    {"name": "Content-Type", "value": "application/json"},
                ],
            },
            "sendBody": True,
            "specifyBody": "json",
            "jsonBody": "={{ JSON.stringify($json.comfyui_request) }}",
        },
    })
    connections[prep_name] = {"main": [[{"node": generate_name, "type": "main", "index": 0}]]}

    # Wait 5 seconds for generation
    wait_name = "Wait for Generation"
    nodes.append({
        "id": _node_id(),
        "name": wait_name,
        "type": "n8n-nodes-base.wait",
        "typeVersion": 1.1,
        "position": _pos(5, 0),
        "parameters": {
            "amount": 5,
            "unit": "seconds",
        },
    })
    connections[generate_name] = {"main": [[{"node": wait_name, "type": "main", "index": 0}]]}

    # Code: Poll for result
    poll_name = "Poll Result"
    nodes.append({
        "id": _node_id(),
        "name": poll_name,
        "type": "n8n-nodes-base.code",
        "typeVersion": 2,
        "position": _pos(6, 0),
        "parameters": {
            "mode": "runOnceForAllItems",
            "jsCode": f"""// Poll ComfyUI for generation result
const prevData = $('Generate Image').first().json;
const instanceId = prevData.instance_id || $('Prepare ComfyUI Request').first().json.comfyui_request.instance_id;
const promptId = prevData.prompt_id;

if (!promptId) {{
  return [{{ json: {{ success: false, error: "No prompt_id from generate", raw: prevData }} }}];
}}

// Poll up to 12 times (60 seconds total)
for (let i = 0; i < 12; i++) {{
  const resp = await fetch(`{AGENTNATE_BASE}/api/comfyui/result/${{instanceId}}/${{promptId}}`);
  const result = await resp.json();

  if (result.status === "completed") {{
    return [{{ json: {{ success: true, images: result.images, prompt_id: promptId, sd_prompt: $('Prepare ComfyUI Request').first().json.comfyui_request.prompt }} }}];
  }}
  if (result.status === "failed") {{
    return [{{ json: {{ success: false, error: "Generation failed", details: result }} }}];
  }}

  // Wait 5 seconds before retry
  await new Promise(r => setTimeout(r, 5000));
}}

return [{{ json: {{ success: false, error: "Generation timed out after 60s" }} }}];""",
        },
    })
    connections[wait_name] = {"main": [[{"node": poll_name, "type": "main", "index": 0}]]}

    # Respond
    respond_name = "Send Response"
    nodes.append({
        "id": _node_id(),
        "name": respond_name,
        "type": "n8n-nodes-base.respondToWebhook",
        "typeVersion": 1,
        "position": _pos(7, 0),
        "parameters": {
            "respondWith": "json",
            "responseBody": "={{ JSON.stringify($json) }}",
        },
    })
    connections[poll_name] = {"main": [[{"node": respond_name, "type": "main", "index": 0}]]}

    return _wrap_workflow(name, nodes, connections, webhook_path)


# ---------------------------------------------------------------------------
# Wrapper & Entry Point
# ---------------------------------------------------------------------------

def _wrap_workflow(name: str, nodes: list, connections: dict, webhook_path: str) -> dict:
    """Wrap nodes and connections into a full n8n workflow JSON."""
    return {
        "name": name,
        "active": False,
        "settings": {
            "executionOrder": "v1",
            "saveManualExecutions": True,
        },
        "nodes": nodes,
        "connections": connections,
        "meta": {
            "generator": "agentnate-workflow-bridge",
            "webhook_path": webhook_path,
        },
    }


PATTERNS = {
    "swarm": generate_swarm_workflow,
    "pipeline": generate_pipeline_workflow,
    "multi_coder": generate_multi_coder_workflow,
    "image_pipeline": generate_image_pipeline_workflow,
}


def generate_workflow(pattern: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate an n8n workflow JSON from a pattern name and config.

    Args:
        pattern: One of "swarm", "pipeline", "multi_coder", "image_pipeline"
        config: Pattern-specific configuration dict

    Returns:
        Complete n8n workflow JSON ready for deploy_workflow

    Raises:
        ValueError: If pattern is unknown
    """
    generator = PATTERNS.get(pattern)
    if not generator:
        raise ValueError(
            f"Unknown workflow pattern: '{pattern}'. "
            f"Available: {list(PATTERNS.keys())}"
        )
    return generator(config)
