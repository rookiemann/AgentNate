"""
n8n Management Routes

REST API for spawning, stopping, and managing n8n instances.
Includes:
- Legacy endpoints for backwards compatibility (N8nManager)
- New queue endpoints for isolated worker databases (N8nQueueManager)
- Reverse proxy for embedding n8n in iframes with auto-auth.
"""

from typing import Optional, List, Any
from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from pydantic import BaseModel
import os
import httpx
import websockets
import asyncio
import logging

router = APIRouter()
logger = logging.getLogger("n8n_routes")

# Store auth cookies for each n8n instance
_n8n_auth_cookies: dict = {}


# =============================================================================
# Request/Response Models
# =============================================================================

class SpawnRequest(BaseModel):
    port: Optional[int] = None


class SpawnWorkerRequest(BaseModel):
    workflow_id: str
    mode: str = "once"  # once, loop, standby
    loop_target: Optional[int] = None  # For loop mode


class DeployAndRunRequest(BaseModel):
    workflow_json: dict
    mode: str = "once"
    loop_target: Optional[int] = None


# =============================================================================
# Queue System Endpoints (N8nQueueManager)
# =============================================================================

@router.post("/main/start")
async def start_main_admin(request: Request):
    """Start the main admin n8n instance."""
    queue_manager = getattr(request.app.state, 'n8n_queue_manager', None)
    if not queue_manager:
        raise HTTPException(status_code=500, detail="Queue manager not initialized")

    try:
        instance = await queue_manager.start_main()
        return {
            "success": True,
            "instance": instance.to_dict(),
        }
    except Exception as e:
        logger.error(f"Failed to start main admin: {e}")
        return {
            "success": False,
            "error": str(e),
        }


@router.delete("/main/stop")
async def stop_main_admin(request: Request):
    """Stop the main admin n8n instance."""
    queue_manager = getattr(request.app.state, 'n8n_queue_manager', None)
    if not queue_manager:
        raise HTTPException(status_code=500, detail="Queue manager not initialized")

    success = await queue_manager.stop_main()
    return {"success": success}


@router.get("/main/status")
async def get_main_status(request: Request):
    """Get main admin instance status."""
    queue_manager = getattr(request.app.state, 'n8n_queue_manager', None)
    if not queue_manager:
        raise HTTPException(status_code=500, detail="Queue manager not initialized")

    return queue_manager.get_main_status()


@router.get("/workflows")
async def list_workflows(request: Request):
    """List all workflows from main database."""
    queue_manager = getattr(request.app.state, 'n8n_queue_manager', None)
    if not queue_manager:
        raise HTTPException(status_code=500, detail="Queue manager not initialized")

    workflows = await queue_manager.get_workflows()
    return {
        "workflows": workflows,
        "count": len(workflows),
    }


@router.get("/workflow/{workflow_id}")
async def get_workflow(request: Request, workflow_id: str):
    """Get a single workflow with full details including nodes."""
    queue_manager = getattr(request.app.state, 'n8n_queue_manager', None)
    if not queue_manager:
        raise HTTPException(status_code=500, detail="Queue manager not initialized")

    workflow = queue_manager.db_utils.get_workflow(workflow_id)
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")

    return {"workflow": workflow}


@router.put("/workflow/{workflow_id}")
async def update_workflow(request: Request, workflow_id: str):
    """Update a workflow's parameters."""
    queue_manager = getattr(request.app.state, 'n8n_queue_manager', None)
    if not queue_manager:
        raise HTTPException(status_code=500, detail="Queue manager not initialized")

    body = await request.json()
    workflow_data = body.get("workflow")

    if not workflow_data:
        raise HTTPException(status_code=400, detail="No workflow data provided")

    try:
        success = queue_manager.db_utils.update_workflow(workflow_id, workflow_data)
        if success:
            return {"success": True}
        else:
            return {"success": False, "error": "Failed to update workflow"}
    except Exception as e:
        logger.error(f"Failed to update workflow: {e}")
        return {"success": False, "error": str(e)}


async def _ensure_main_running(queue_manager):
    """Start main admin if not running and wait until ready."""
    main_port = queue_manager.main_port
    main_status = queue_manager.get_main_status()

    if not main_status.get("running"):
        logger.info("Auto-starting main admin")
        await queue_manager.start_main()
        for _ in range(30):
            await asyncio.sleep(1)
            try:
                async with httpx.AsyncClient(timeout=3) as probe:
                    r = await probe.get(f"http://127.0.0.1:{main_port}/healthz")
                    if r.status_code == 200:
                        return
            except Exception:
                pass


async def _delete_single_workflow(client, main_port, workflow_id, headers):
    """Deactivate, archive, and delete a single workflow. Returns (id, success, error)."""
    try:
        # Step 1: Deactivate
        await client.patch(
            f"http://127.0.0.1:{main_port}/rest/workflows/{workflow_id}",
            json={"active": False},
            headers=headers,
        )
        # Step 2: Archive
        await client.post(
            f"http://127.0.0.1:{main_port}/rest/workflows/{workflow_id}/archive",
            headers=headers,
        )
        # Step 3: Delete
        resp = await client.delete(
            f"http://127.0.0.1:{main_port}/rest/workflows/{workflow_id}",
            headers=headers,
        )
        if resp.status_code in (200, 204):
            return (workflow_id, True, None)
        else:
            return (workflow_id, False, f"n8n {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        return (workflow_id, False, str(e))


@router.delete("/workflow/{workflow_id}")
async def delete_workflow(request: Request, workflow_id: str):
    """Archive and delete a workflow from the main n8n admin instance."""
    queue_manager = getattr(request.app.state, 'n8n_queue_manager', None)
    if not queue_manager:
        raise HTTPException(status_code=500, detail="Queue manager not initialized")

    await _ensure_main_running(queue_manager)

    main_port = queue_manager.main_port
    auth_cookie = await _get_or_create_auth(main_port)
    headers = {"Cookie": f"n8n-auth={auth_cookie}"} if auth_cookie else {}

    async with httpx.AsyncClient(timeout=15) as client:
        wf_id, success, error = await _delete_single_workflow(client, main_port, workflow_id, headers)

        if success:
            logger.info(f"Deleted workflow {workflow_id} from main admin")
            return {"success": True}
        else:
            return {"success": False, "error": error}


@router.post("/workflows/batch-delete")
async def batch_delete_workflows(request: Request):
    """Archive and delete multiple workflows concurrently."""
    queue_manager = getattr(request.app.state, 'n8n_queue_manager', None)
    if not queue_manager:
        raise HTTPException(status_code=500, detail="Queue manager not initialized")

    body = await request.json()
    ids = body.get("ids", [])
    if not ids:
        return {"success": True, "deleted": 0, "errors": []}

    await _ensure_main_running(queue_manager)

    main_port = queue_manager.main_port
    auth_cookie = await _get_or_create_auth(main_port)
    headers = {"Cookie": f"n8n-auth={auth_cookie}"} if auth_cookie else {}

    async with httpx.AsyncClient(timeout=30) as client:
        results = await asyncio.gather(
            *[_delete_single_workflow(client, main_port, wf_id, headers) for wf_id in ids]
        )

    deleted = sum(1 for _, ok, _ in results if ok)
    errors = [{"id": wf_id, "error": err} for wf_id, ok, err in results if not ok]

    logger.info(f"Batch delete: {deleted}/{len(ids)} succeeded")
    return {"success": len(errors) == 0, "deleted": deleted, "errors": errors}


@router.post("/workers/spawn")
async def spawn_worker(request: Request, body: SpawnWorkerRequest):
    """
    Spawn a worker instance for a specific workflow.

    Creates an isolated n8n instance with its own database containing
    only the specified workflow and credentials.
    """
    queue_manager = getattr(request.app.state, 'n8n_queue_manager', None)
    if not queue_manager:
        raise HTTPException(status_code=500, detail="Queue manager not initialized")

    try:
        worker = await queue_manager.spawn_worker(
            workflow_id=body.workflow_id,
            mode=body.mode,
            loop_target=body.loop_target,
        )
        return {
            "success": True,
            "worker": worker.to_dict(),
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to spawn worker: {e}")
        return {
            "success": False,
            "error": str(e),
        }


@router.get("/workers")
async def list_workers(request: Request):
    """List all worker instances."""
    queue_manager = getattr(request.app.state, 'n8n_queue_manager', None)
    if not queue_manager:
        raise HTTPException(status_code=500, detail="Queue manager not initialized")

    workers = queue_manager.list_workers()
    return {
        "workers": workers,
        "count": len(workers),
    }


@router.get("/workers/{port}")
async def get_worker(request: Request, port: int):
    """Get details about a specific worker."""
    queue_manager = getattr(request.app.state, 'n8n_queue_manager', None)
    if not queue_manager:
        raise HTTPException(status_code=500, detail="Queue manager not initialized")

    worker = queue_manager.get_worker(port)
    if not worker:
        raise HTTPException(status_code=404, detail=f"No worker on port {port}")

    return worker.to_dict()


@router.delete("/workers/{port}")
async def stop_worker(request: Request, port: int, cleanup: bool = True):
    """Stop a worker instance."""
    queue_manager = getattr(request.app.state, 'n8n_queue_manager', None)
    if not queue_manager:
        raise HTTPException(status_code=500, detail="Queue manager not initialized")

    success = await queue_manager.stop_worker(port, cleanup=cleanup)
    if not success:
        raise HTTPException(status_code=404, detail=f"No worker on port {port}")

    return {"success": True, "port": port}


@router.delete("/workers/all")
async def stop_all_workers(request: Request):
    """Stop all worker instances."""
    queue_manager = getattr(request.app.state, 'n8n_queue_manager', None)
    if not queue_manager:
        raise HTTPException(status_code=500, detail="Queue manager not initialized")

    await queue_manager.stop_all_workers()
    return {"success": True, "message": "All workers stopped"}


@router.post("/workers/{port}/activate")
async def activate_worker(request: Request, port: int):
    """Activate the workflow on a worker (enable triggers)."""
    queue_manager = getattr(request.app.state, 'n8n_queue_manager', None)
    if not queue_manager:
        raise HTTPException(status_code=500, detail="Queue manager not initialized")

    result = await queue_manager.activate_worker(port)
    if not result.get("success"):
        return {"success": False, "port": port, "active": False, "error": result.get("error", "Unknown error")}

    return {"success": True, "port": port, "active": True}


@router.post("/workers/{port}/deactivate")
async def deactivate_worker(request: Request, port: int):
    """Deactivate the workflow on a worker (pause triggers, keep worker alive)."""
    queue_manager = getattr(request.app.state, 'n8n_queue_manager', None)
    if not queue_manager:
        raise HTTPException(status_code=500, detail="Queue manager not initialized")

    success = await queue_manager.deactivate_worker(port)
    if not success:
        raise HTTPException(status_code=404, detail=f"No running worker on port {port}")

    return {"success": True, "port": port, "active": False}


@router.post("/workers/{port}/execute")
async def execute_worker_workflow(request: Request, port: int):
    """Manually execute a workflow on a worker and wait for the result."""
    queue_manager = getattr(request.app.state, 'n8n_queue_manager', None)
    if not queue_manager:
        raise HTTPException(status_code=500, detail="Queue manager not initialized")

    worker = queue_manager.workers.get(port)
    if not worker or not worker.is_running:
        raise HTTPException(status_code=404, detail=f"No running worker on port {port}")

    # Prevent concurrent manual executions while queue processor is running
    if worker._processing:
        return {"success": False, "port": port, "error": "Queue is processing. Use pause to stop it."}

    try:
        auth_cookie = await _get_or_create_auth(port)
        headers = {"Cookie": f"n8n-auth={auth_cookie}"} if auth_cookie else {}

        async with httpx.AsyncClient(timeout=15) as client:
            # Step 1: Get the full workflow data from the worker's n8n
            wf_resp = await client.get(
                f"http://127.0.0.1:{port}/rest/workflows/{worker.workflow_id}",
                headers=headers,
            )
            if wf_resp.status_code != 200:
                return {"success": False, "port": port, "error": f"Failed to get workflow: {wf_resp.status_code}"}

            wf_data = wf_resp.json()
            workflow_json = wf_data.get("data", wf_data)

            # Find the start node (manual trigger preferred, else first node)
            nodes = workflow_json.get("nodes", [])
            start_node = None
            for node in nodes:
                ntype = node.get("type", "")
                if "manualTrigger" in ntype or "trigger" in ntype.lower():
                    start_node = node["name"]
                    break
            if not start_node and nodes:
                start_node = nodes[0]["name"]

            # Step 2: Execute via n8n internal run endpoint
            run_resp = await client.post(
                f"http://127.0.0.1:{port}/rest/workflows/{worker.workflow_id}/run",
                json={
                    "workflowData": workflow_json,
                    "triggerToStartFrom": {"name": start_node} if start_node else None,
                    "startNodes": [],
                    "pinData": {},
                },
                headers=headers,
            )

            if run_resp.status_code not in (200, 201):
                error_text = run_resp.text[:500]
                return {"success": False, "port": port, "error": f"n8n returned {run_resp.status_code}: {error_text}"}

            result = run_resp.json()
            exec_id = result.get("data", {}).get("executionId")

            if not exec_id:
                return {"success": False, "port": port, "error": "No executionId returned"}

            # Step 3: Poll for execution completion (5s timeout per poll request)
            exec_result = None
            async with httpx.AsyncClient(timeout=5) as poll_client:
                for _ in range(30):  # Up to 30 seconds
                    await asyncio.sleep(1)
                    try:
                        exec_resp = await poll_client.get(
                            f"http://127.0.0.1:{port}/rest/executions/{exec_id}",
                            headers=headers,
                        )
                        if exec_resp.status_code != 200:
                            continue

                        exec_data = exec_resp.json()
                        execution = exec_data.get("data", exec_data)
                        if not isinstance(execution, dict):
                            continue

                        finished = execution.get("finished")
                        status = execution.get("status", "")
                        stopped_at = execution.get("stoppedAt")

                        # Still running
                        if not finished and not stopped_at and status not in ("error", "crashed", "failed"):
                            continue

                        # Completed — check if success or error
                        if status == "success" or (finished and status not in ("error", "crashed", "failed")):
                            exec_result = {
                                "success": True,
                                "port": port,
                                "executionId": exec_id,
                                "status": "success",
                            }
                        else:
                            error_msg = _extract_execution_error(execution)
                            exec_result = {
                                "success": False,
                                "port": port,
                                "executionId": exec_id,
                                "status": status or "error",
                                "error": error_msg,
                            }
                        break
                    except Exception:
                        continue

            if exec_result:
                return exec_result

            # Timed out waiting — execution is still running
            return {
                "success": True,
                "port": port,
                "executionId": exec_id,
                "status": "running",
                "error": "Execution still running after 30s",
            }

    except Exception as e:
        return {"success": False, "port": port, "error": str(e)}


async def _fetch_workflow_data(port: int, workflow_id: str, headers: dict) -> Optional[dict]:
    """Fetch and cache workflow data + start node. Returns dict with keys 'json' and 'start_node', or None."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"http://127.0.0.1:{port}/rest/workflows/{workflow_id}",
                headers=headers,
            )
            if resp.status_code != 200:
                return None

            wf_data = resp.json()
            workflow_json = wf_data.get("data", wf_data)
            nodes = workflow_json.get("nodes", [])
            start_node = None
            for node in nodes:
                ntype = node.get("type", "")
                if "manualTrigger" in ntype or "trigger" in ntype.lower():
                    start_node = node["name"]
                    break
            if not start_node and nodes:
                start_node = nodes[0]["name"]

            return {"json": workflow_json, "start_node": start_node}
    except Exception:
        return None


async def _fire_single_execution(queue_manager, port: int, cached_wf: dict = None) -> Optional[str]:
    """Fire a single execution on a worker. Returns execution ID or None on error.

    Args:
        cached_wf: Pre-fetched workflow data dict with 'json' and 'start_node' keys.
                   If None, fetches fresh (slower).
    """
    from backend.middleware.debug_logger import debug_logger

    worker = queue_manager.workers.get(port)
    if not worker or not worker.is_running:
        return None

    try:
        auth = await _get_or_create_auth(port)
        headers = {"Cookie": f"n8n-auth={auth}"} if auth else {}

        # Use cached workflow data or fetch fresh
        wf = cached_wf
        if not wf:
            wf = await _fetch_workflow_data(port, worker.workflow_id, headers)
        if not wf:
            debug_logger.info(f"[QUEUE] Worker :{port} get workflow failed")
            return None

        async with httpx.AsyncClient(timeout=15) as client:
            run_resp = await client.post(
                f"http://127.0.0.1:{port}/rest/workflows/{worker.workflow_id}/run",
                json={
                    "workflowData": wf["json"],
                    "triggerToStartFrom": {"name": wf["start_node"]} if wf["start_node"] else None,
                    "startNodes": [],
                    "pinData": {},
                },
                headers=headers,
            )
            if run_resp.status_code not in (200, 201):
                debug_logger.info(f"[QUEUE] Worker :{port} run failed: {run_resp.status_code}")
                return None

            result = run_resp.json()
            exec_id = result.get("data", {}).get("executionId")
            if not exec_id:
                debug_logger.info(f"[QUEUE] Worker :{port} no executionId returned")
                return None

            return exec_id

    except (httpx.ConnectTimeout, httpx.ConnectError, httpx.ReadTimeout):
        debug_logger.info(f"[QUEUE] Worker :{port} unreachable")
        return None
    except Exception as e:
        debug_logger.info(f"[QUEUE] Worker :{port} fire error: {type(e).__name__}: {e}")
        return None


async def _wait_for_execution(port: int, exec_id: str, headers: dict, poll_client: httpx.AsyncClient = None, timeout: float = 60) -> Optional[str]:
    """Poll until an execution completes. Returns status string or None on timeout.

    Uses fast polling: 0.15s intervals for sub-second detection.
    """
    owns_client = poll_client is None
    if owns_client:
        poll_client = httpx.AsyncClient(timeout=5)

    try:
        max_polls = int(timeout / 0.15)
        for _ in range(max_polls):
            await asyncio.sleep(0.15)
            try:
                poll_resp = await poll_client.get(
                    f"http://127.0.0.1:{port}/rest/executions/{exec_id}",
                    headers=headers,
                )
                if poll_resp.status_code != 200:
                    continue
                exec_data = poll_resp.json()
                execution = exec_data.get("data", exec_data)
                if not isinstance(execution, dict):
                    continue
                status = execution.get("status", "")
                if execution.get("stoppedAt") or status in ("success", "error", "crashed", "failed"):
                    return status
            except Exception:
                continue
        return None
    finally:
        if owns_client:
            await poll_client.aclose()


async def _process_queue(queue_manager, port: int):
    """Background task: process the execution queue for a worker.

    Sequential: fire-wait-fire with minimal overhead.
    Parallel: pipeline model — keeps N executions in-flight, fires next as soon as one completes.
    """
    from backend.middleware.debug_logger import debug_logger

    worker = queue_manager.workers.get(port)
    if not worker:
        return

    worker._processing = True
    worker._paused = False
    local_completed = worker.execution_count
    debug_logger.info(f"[QUEUE] Starting processor for :{port} (total={worker._queued_total}, parallel={worker._queued_parallel}, initial_completed={local_completed})")

    # Pre-fetch workflow data once
    auth = await _get_or_create_auth(port)
    headers = {"Cookie": f"n8n-auth={auth}"} if auth else {}
    cached_wf = await _fetch_workflow_data(port, worker.workflow_id, headers)
    if not cached_wf:
        debug_logger.info(f"[QUEUE] Worker :{port} failed to fetch workflow data, aborting")
        worker._processing = False
        return

    # Shared HTTP client for polling (connection reuse)
    poll_client = httpx.AsyncClient(timeout=5)

    def _should_stop():
        w = queue_manager.workers.get(port)
        return not w or not w._processing or w._paused or not w.is_running

    def _target_reached():
        return worker._queued_total is not None and local_completed >= worker._queued_total

    try:
        if worker._queued_parallel:
            # === PIPELINE PARALLEL MODE ===
            MAX_INFLIGHT = 10
            local_fired = 0

            async def _run_one(fire_num: int):
                """Fire + wait for one execution."""
                nonlocal local_completed
                exec_id = await _fire_single_execution(queue_manager, port, cached_wf)
                if not exec_id:
                    return False
                debug_logger.info(f"[QUEUE] Worker :{port} parallel fired #{exec_id} (fire {fire_num})")
                status = await _wait_for_execution(port, exec_id, headers, poll_client, timeout=120)
                local_completed += 1
                worker.execution_count = local_completed
                debug_logger.info(f"[QUEUE] Worker :{port} parallel #{exec_id} -> {status or 'timeout'} (completed={local_completed}/{worker._queued_total})")
                return True

            pending = set()

            while not _should_stop():
                if _target_reached():
                    debug_logger.info(f"[QUEUE] Worker :{port} parallel target reached {local_completed}/{worker._queued_total}")
                    break

                # Calculate how many more to fire
                if worker._queued_total is None:
                    can_fire = MAX_INFLIGHT - len(pending)
                else:
                    remaining = worker._queued_total - local_completed - len(pending)
                    can_fire = min(remaining, MAX_INFLIGHT - len(pending))

                # Launch new tasks to fill pipeline
                for _ in range(max(0, can_fire)):
                    if _should_stop():
                        break
                    local_fired += 1
                    task = asyncio.create_task(_run_one(local_fired))
                    pending.add(task)
                    await asyncio.sleep(0.05)  # Tiny stagger to avoid HTTP thundering herd

                if not pending:
                    break

                # Wait for at least one to finish, then loop to refill pipeline
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)

            # Wait for remaining in-flight
            if pending:
                debug_logger.info(f"[QUEUE] Worker :{port} draining {len(pending)} in-flight tasks")
                await asyncio.wait(pending)

        else:
            # === SEQUENTIAL MODE ===
            while not _should_stop():
                if _target_reached():
                    debug_logger.info(f"[QUEUE] Worker :{port} sequential target reached {local_completed}/{worker._queued_total}")
                    break

                exec_id = await _fire_single_execution(queue_manager, port, cached_wf)
                if not exec_id:
                    debug_logger.info(f"[QUEUE] Worker :{port} sequential fire failed, stopping")
                    break

                status = await _wait_for_execution(port, exec_id, headers, poll_client, timeout=60)
                local_completed += 1
                worker.execution_count = local_completed
                debug_logger.info(f"[QUEUE] Worker :{port} sequential #{exec_id} -> {status or 'timeout'} (completed={local_completed}/{worker._queued_total})")

    except Exception as e:
        debug_logger.info(f"[QUEUE] Worker :{port} processor error: {type(e).__name__}: {e}")
    finally:
        await poll_client.aclose()
        worker = queue_manager.workers.get(port)
        if worker:
            worker._processing = False
            debug_logger.info(f"[QUEUE] Processor ended for :{port} (execution_count={worker.execution_count}, local_completed={local_completed})")


def _extract_execution_error(execution: dict) -> str:
    """Extract a human-readable error message from an n8n execution result."""
    try:
        # Check resultData for node-level errors
        data = execution.get("data")
        result_data = data.get("resultData", {}) if isinstance(data, dict) else {}

        # Check run data for node errors
        run_data = result_data.get("runData", {}) if isinstance(result_data, dict) else {}
        for node_name, node_runs in run_data.items():
            if not isinstance(node_runs, list):
                continue
            for run in node_runs:
                if not isinstance(run, dict):
                    continue
                error = run.get("error")
                if error:
                    msg = error.get("message", str(error)) if isinstance(error, dict) else str(error)
                    return f"Error in '{node_name}': {msg}"

        # Check top-level error
        error = result_data.get("error") if isinstance(result_data, dict) else None
        if error:
            return error.get("message", str(error)) if isinstance(error, dict) else str(error)

        # Check lastNodeExecuted
        last_node = result_data.get("lastNodeExecuted", "") if isinstance(result_data, dict) else ""
        if last_node:
            return f"Execution failed at node '{last_node}'"
    except Exception:
        pass

    return "Workflow execution failed"


class ChangeWorkerModeRequest(BaseModel):
    mode: str
    loop_target: Optional[int] = None


@router.patch("/workers/{port}/mode")
async def change_worker_mode(request: Request, port: int, body: ChangeWorkerModeRequest):
    """Change a worker's execution mode."""
    queue_manager = getattr(request.app.state, 'n8n_queue_manager', None)
    if not queue_manager:
        raise HTTPException(status_code=500, detail="Queue manager not initialized")

    if body.mode not in ("once", "loop", "standby"):
        raise HTTPException(status_code=400, detail=f"Invalid mode: {body.mode}")

    success = await queue_manager.change_worker_mode(port, body.mode, body.loop_target)
    if not success:
        raise HTTPException(status_code=404, detail=f"No running worker on port {port}")

    return {"success": True, "port": port, "mode": body.mode, "loop_target": body.loop_target}


@router.post("/workers/{port}/reset-counter")
async def reset_worker_counter(request: Request, port: int):
    """Reset a worker's execution counter."""
    queue_manager = getattr(request.app.state, 'n8n_queue_manager', None)
    if not queue_manager:
        raise HTTPException(status_code=500, detail="Queue manager not initialized")

    worker = queue_manager.workers.get(port)
    if not worker:
        raise HTTPException(status_code=404, detail=f"No worker on port {port}")

    # Clear queue + stop processing + reset counters
    from datetime import datetime
    worker._queued_total = 0
    worker._paused = False
    worker._processing = False  # Signal background task to stop
    worker.started_at = datetime.now()
    worker.loop_count = 0
    worker.execution_count = 0
    worker.last_execution = None
    worker.last_status = None

    return {"success": True, "port": port}


class EnqueueRequest(BaseModel):
    count: Optional[int] = None  # None = infinite
    parallel: bool = False


@router.post("/workers/{port}/enqueue")
async def enqueue_runs(request: Request, port: int, body: EnqueueRequest):
    """Enqueue N runs for a worker. Stacks on top of existing queue."""
    queue_manager = getattr(request.app.state, 'n8n_queue_manager', None)
    if not queue_manager:
        raise HTTPException(status_code=500, detail="Queue manager not initialized")

    worker = queue_manager.workers.get(port)
    if not worker or not worker.is_running:
        raise HTTPException(status_code=404, detail=f"No running worker on port {port}")

    # Stack runs
    if body.count is None:
        worker._queued_total = None  # Infinite
    elif worker._queued_total is None:
        pass  # Already infinite, ignore finite additions
    else:
        worker._queued_total += body.count

    worker._queued_parallel = body.parallel
    worker._paused = False  # Unpause if paused

    # Start background processor if not already running
    if not worker._processing:
        asyncio.create_task(_process_queue(queue_manager, port))

    return {"success": True, "queued_total": worker._queued_total}


@router.post("/workers/{port}/pause")
async def pause_worker(request: Request, port: int):
    """Pause queue processing on a worker. In-flight executions finish naturally."""
    queue_manager = getattr(request.app.state, 'n8n_queue_manager', None)
    if not queue_manager:
        raise HTTPException(status_code=500, detail="Queue manager not initialized")

    worker = queue_manager.workers.get(port)
    if not worker:
        raise HTTPException(status_code=404, detail=f"No worker on port {port}")

    worker._paused = True
    return {"success": True}


# =============================================================================
# Test Execution Log — receives POSTs from test workflows, appends to JSON file
# =============================================================================

_test_log_lock = asyncio.Lock()


def _get_test_log_path() -> str:
    base = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".n8n-instances")
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, "execution-log.json")


@router.post("/test-log")
async def append_test_log(request: Request):
    """Append an entry to the execution log file. Called by test workflows."""
    import json
    body = await request.json()
    log_path = _get_test_log_path()

    async with _test_log_lock:
        entries = []
        if os.path.exists(log_path):
            try:
                with open(log_path, 'r') as f:
                    entries = json.load(f)
            except (json.JSONDecodeError, Exception):
                entries = []

        entries.append(body)

        with open(log_path, 'w') as f:
            json.dump(entries, f, indent=2)

    return {"success": True, "total_entries": len(entries)}


@router.get("/test-log")
async def get_test_log(request: Request):
    """Read the execution log file."""
    import json
    log_path = _get_test_log_path()
    if not os.path.exists(log_path):
        return {"entries": [], "count": 0}

    try:
        with open(log_path, 'r') as f:
            entries = json.load(f)
        return {"entries": entries, "count": len(entries)}
    except Exception as e:
        return {"entries": [], "count": 0, "error": str(e)}


@router.delete("/test-log")
async def clear_test_log(request: Request):
    """Clear the execution log file."""
    log_path = _get_test_log_path()
    if os.path.exists(log_path):
        os.remove(log_path)
    return {"success": True}


@router.post("/credentials/sync")
async def sync_credentials(request: Request):
    """Manually trigger credential sync to all workers."""
    queue_manager = getattr(request.app.state, 'n8n_queue_manager', None)
    if not queue_manager:
        raise HTTPException(status_code=500, detail="Queue manager not initialized")

    await queue_manager.sync_credentials_now()
    return {"success": True, "message": "Credentials synced to all workers"}


@router.get("/history")
async def get_history(
    request: Request,
    workflow_id: Optional[str] = None,
    status: Optional[str] = None,
    since: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
):
    """Get aggregated execution history."""
    queue_manager = getattr(request.app.state, 'n8n_queue_manager', None)
    if not queue_manager:
        raise HTTPException(status_code=500, detail="Queue manager not initialized")

    history = queue_manager.get_execution_history(workflow_id, status, since, limit, offset)
    return {
        "history": history,
        "count": len(history),
    }


@router.get("/history/stats")
async def get_history_stats(
    request: Request,
    workflow_id: Optional[str] = None,
    since: Optional[str] = None
):
    """Get execution count stats grouped by status."""
    queue_manager = getattr(request.app.state, 'n8n_queue_manager', None)
    if not queue_manager:
        raise HTTPException(status_code=500, detail="Queue manager not initialized")

    return queue_manager.get_history_stats(workflow_id, since)


@router.get("/history/workflows")
async def get_history_workflows(request: Request):
    """Get distinct workflow names from history."""
    queue_manager = getattr(request.app.state, 'n8n_queue_manager', None)
    if not queue_manager:
        raise HTTPException(status_code=500, detail="Queue manager not initialized")

    return {"workflows": queue_manager.get_distinct_workflows()}


@router.delete("/history")
async def clear_history(request: Request):
    """Clear all execution history."""
    queue_manager = getattr(request.app.state, 'n8n_queue_manager', None)
    if not queue_manager:
        raise HTTPException(status_code=500, detail="Queue manager not initialized")

    success = queue_manager.clear_history()
    return {"success": success}


@router.get("/executions/live")
async def get_live_executions(request: Request):
    """Get live execution data from all running workers."""
    queue_manager = getattr(request.app.state, 'n8n_queue_manager', None)
    if not queue_manager:
        raise HTTPException(status_code=500, detail="Queue manager not initialized")

    return await queue_manager.get_live_executions()


@router.get("/executions")
async def get_executions(
    request: Request,
    workflow_id: Optional[str] = None,
    status: Optional[str] = None,
    since: Optional[str] = None,
    limit: int = 200
):
    """Combined endpoint: history + live + queued executions."""
    queue_manager = getattr(request.app.state, 'n8n_queue_manager', None)
    if not queue_manager:
        raise HTTPException(status_code=500, detail="Queue manager not initialized")

    # 1. Get history
    history = queue_manager.get_execution_history(workflow_id, status, since, limit)

    # 2. Get live data
    live_data = await queue_manager.get_live_executions()
    live_execs = live_data.get("executions", [])
    queued = live_data.get("queued", [])

    # 3. Apply filters to live executions
    if workflow_id:
        live_execs = [e for e in live_execs if e.get("workflow_id") == workflow_id]
        queued = [q for q in queued if q.get("workflow_id") == workflow_id]
    if status and status != "queued":
        live_execs = [e for e in live_execs if e.get("status") == status]
    if since:
        live_execs = [e for e in live_execs if (e.get("started_at") or "") >= since]

    # 4. Merge and deduplicate (live takes priority over history by ID)
    history_ids = {h["id"] for h in history}
    merged = list(history)
    for le in live_execs:
        if le["id"] not in history_ids:
            merged.append(le)

    # 5. Sort by started_at DESC
    merged.sort(key=lambda x: x.get("started_at") or "", reverse=True)

    # 6. Stats from history DB
    stats = queue_manager.get_history_stats(workflow_id, since)
    # Add live running count
    running_count = sum(1 for e in live_execs if e.get("status") in ("running", "waiting", "new"))
    stats["running"] = running_count
    # Add queued count
    total_queued = 0
    for q in queued:
        r = q.get("remaining")
        if r is None:
            total_queued = None
            break
        total_queued += r
    stats["queued"] = total_queued

    return {
        "executions": merged[:limit],
        "queued": queued,
        "stats": stats,
    }


@router.post("/history/aggregate")
async def aggregate_history(request: Request):
    """Manually trigger history aggregation from all workers."""
    queue_manager = getattr(request.app.state, 'n8n_queue_manager', None)
    if not queue_manager:
        raise HTTPException(status_code=500, detail="Queue manager not initialized")

    count = await queue_manager.aggregate_all_history()
    return {"success": True, "records_aggregated": count}


@router.post("/deploy-and-run")
async def deploy_and_run(request: Request, body: DeployAndRunRequest):
    """
    One-click deploy and run workflow.

    1. Auto-starts Main Admin if not running
    2. Deploys workflow JSON to Main Admin
    3. Spawns isolated worker for that workflow

    Returns worker info for the running workflow.
    """
    queue_manager = getattr(request.app.state, 'n8n_queue_manager', None)
    if not queue_manager:
        raise HTTPException(status_code=500, detail="Queue manager not initialized")

    try:
        # 1. Ensure main admin is running
        main_status = queue_manager.get_main_status()
        if not main_status.get("running"):
            logger.info("Auto-starting main admin for deploy-and-run")
            await queue_manager.start_main()
            # Wait a bit for main to be ready
            await asyncio.sleep(2)

        # 2. Deploy workflow to main admin via n8n API
        from backend.workflow_generator import fix_workflow
        fixed_wf = fix_workflow(body.workflow_json)

        main_port = queue_manager.main_port
        auth_cookie = await _get_or_create_auth(main_port)

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"http://127.0.0.1:{main_port}/rest/workflows",
                json=fixed_wf,
                headers={"Cookie": f"n8n-auth={auth_cookie}"} if auth_cookie else {},
            )

            if resp.status_code not in (200, 201):
                return {
                    "success": False,
                    "error": f"Failed to deploy workflow: {resp.status_code} - {resp.text}",
                }

            workflow_data = resp.json()
            workflow_id = workflow_data.get("data", {}).get("id") or workflow_data.get("id")

            if not workflow_id:
                return {
                    "success": False,
                    "error": "Workflow deployed but no ID returned",
                }

        # 3. Spawn isolated worker
        worker = await queue_manager.spawn_worker(
            workflow_id=workflow_id,
            mode=body.mode,
            loop_target=body.loop_target,
        )

        return {
            "success": True,
            "workflow_id": workflow_id,
            "main_status": queue_manager.get_main_status(),
            "worker": worker.to_dict(),
        }

    except Exception as e:
        logger.error(f"Deploy and run failed: {e}")
        return {
            "success": False,
            "error": str(e),
        }


# =============================================================================
# Legacy Endpoints (N8nManager - for backwards compatibility)
# =============================================================================


@router.get("/list")
async def list_instances(request: Request):
    """List all running n8n instances."""
    n8n_manager = request.app.state.n8n_manager
    return {
        "instances": n8n_manager.list(),
        "count": n8n_manager.count,
    }


@router.post("/spawn")
async def spawn_instance(request: Request, body: SpawnRequest = None):
    """
    Spawn a new n8n instance.

    Args:
        port: Optional specific port. If not provided, auto-assigns next available.

    Returns:
        Instance details including URL and port.
    """
    n8n_manager = request.app.state.n8n_manager

    try:
        port = body.port if body else None
        instance = await n8n_manager.spawn(port)

        return {
            "success": True,
            "instance": instance.to_dict(),
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


@router.delete("/all")
async def stop_all_instances(request: Request):
    """Stop all running n8n instances."""
    n8n_manager = request.app.state.n8n_manager

    await n8n_manager.stop_all()

    return {"success": True, "message": "All instances stopped"}


@router.delete("/{port}")
async def stop_instance(request: Request, port: int):
    """Stop an n8n instance by port."""
    n8n_manager = request.app.state.n8n_manager

    success = await n8n_manager.stop(port)

    if not success:
        raise HTTPException(status_code=404, detail=f"No n8n instance on port {port}")

    return {"success": True, "port": port}


@router.get("/{port}")
async def get_instance(request: Request, port: int):
    """Get details about a specific n8n instance."""
    n8n_manager = request.app.state.n8n_manager

    instance = n8n_manager.get(port)

    if not instance:
        raise HTTPException(status_code=404, detail=f"No n8n instance on port {port}")

    return instance.to_dict()


@router.get("/{port}/health")
async def check_instance_health(request: Request, port: int):
    """Check if an n8n instance is responding."""
    n8n_manager = request.app.state.n8n_manager
    instance = n8n_manager.get(port)

    if not instance:
        raise HTTPException(status_code=404, detail=f"No n8n instance on port {port}")

    ready = await n8n_manager.is_ready(port)

    return {
        "port": port,
        "healthy": instance.is_running,
        "ready": ready,
        "url": instance.url,
    }


@router.get("/{port}/ready")
async def check_instance_ready(request: Request, port: int):
    """Check if an n8n instance is ready to serve requests (for iframe polling)."""
    import httpx

    # Check queue manager first (main admin or workers) — takes priority
    queue_manager = getattr(request.app.state, 'n8n_queue_manager', None)
    if queue_manager:
        found = False

        # Check if it's the main admin
        if queue_manager.main and queue_manager.main.port == port:
            found = True

        # Check if it's a worker
        if not found and port in queue_manager.workers:
            found = True

        if found:
            # Do direct HTTP check
            url = f"http://127.0.0.1:{port}/"
            try:
                async with httpx.AsyncClient(timeout=3) as client:
                    resp = await client.get(url)
                    ready = resp.status_code in (200, 301, 302)
                    return {"ready": ready, "port": port, "url": url}
            except Exception:
                return {"ready": False, "port": port, "url": url}

    # Fallback: check legacy n8n_manager
    n8n_manager = getattr(request.app.state, 'n8n_manager', None)
    if not n8n_manager:
        return {"ready": False, "error": "Instance not found"}
    instance = n8n_manager.get(port)

    if instance and instance.is_running:
        ready = await n8n_manager.is_ready(port)
        return {"ready": ready, "port": port, "url": instance.url}

    return {"ready": False, "error": "Instance not found"}


@router.websocket("/{port}/proxy/rest/push")
async def n8n_websocket_proxy(websocket: WebSocket, port: int):
    """WebSocket proxy for n8n push notifications."""
    await websocket.accept()

    # Get query string from the request
    query = str(websocket.scope.get("query_string", b""), "utf-8")
    ws_url = f"ws://127.0.0.1:{port}/rest/push"
    if query:
        ws_url += f"?{query}"

    # Get auth cookie
    auth_cookie = await _get_or_create_auth(port)
    headers = {}
    if auth_cookie:
        headers["Cookie"] = f"n8n-auth={auth_cookie}"

    import logging
    logger = logging.getLogger("n8n_ws_proxy")
    logger.info(f"Connecting to n8n WebSocket: {ws_url}")

    try:
        async with websockets.connect(
            ws_url,
            additional_headers=headers
        ) as n8n_ws:
            logger.info(f"Connected to n8n WebSocket on port {port}")

            async def forward_to_client():
                try:
                    async for msg in n8n_ws:
                        logger.debug(f"n8n->client: {msg[:100] if len(msg) > 100 else msg}")
                        await websocket.send_text(msg)
                except Exception as e:
                    logger.error(f"forward_to_client error: {e}")

            async def forward_to_n8n():
                try:
                    while True:
                        data = await websocket.receive_text()
                        logger.debug(f"client->n8n: {data[:100] if len(data) > 100 else data}")
                        await n8n_ws.send(data)
                except WebSocketDisconnect:
                    logger.info("Client disconnected")
                except Exception as e:
                    logger.error(f"forward_to_n8n error: {e}")

            await asyncio.gather(forward_to_client(), forward_to_n8n())
    except Exception as e:
        logger.error(f"WebSocket proxy error: {e}")
        try:
            await websocket.close(code=1011, reason=str(e)[:120])
        except:
            pass


@router.api_route("/{port}/proxy/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
async def n8n_proxy(request: Request, port: int, path: str = ""):
    """
    Reverse proxy for n8n with auto-authentication.
    All requests go through here with auth cookie injected.
    """
    n8n_manager = request.app.state.n8n_manager
    instance = n8n_manager.get(port)

    if not instance:
        raise HTTPException(status_code=404, detail=f"No n8n instance on port {port}")

    # Get or create auth cookie
    auth_cookie = await _get_or_create_auth(port)

    # Build target URL
    target_url = f"http://127.0.0.1:{port}/{path}"
    if request.url.query:
        target_url += f"?{request.url.query}"

    # Forward request with auth - don't accept compressed responses
    headers = {k: v for k, v in request.headers.items()
               if k.lower() not in ("host", "cookie", "accept-encoding")}
    if auth_cookie:
        headers["cookie"] = f"n8n-auth={auth_cookie}"

    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=False) as client:
            body = await request.body()
            resp = await client.request(
                method=request.method,
                url=target_url,
                headers=headers,
                content=body if body else None,
            )

            content = resp.content
            content_type = resp.headers.get("content-type", "")
            proxy_base = f"/api/n8n/{port}/proxy"

            # Rewrite base-path.js to use proxy path
            if path == "static/base-path.js":
                content = f"window.BASE_PATH = '{proxy_base}/';\n".encode()

            # Rewrite URLs in HTML responses to go through proxy
            elif "text/html" in content_type:
                content = content.replace(b'href="/', f'href="{proxy_base}/'.encode())
                content = content.replace(b"href='/", f"href='{proxy_base}/".encode())
                content = content.replace(b'src="/', f'src="{proxy_base}/'.encode())
                content = content.replace(b"src='/", f"src='{proxy_base}/".encode())

            # Filter response headers
            skip_headers = {"content-encoding", "content-length", "transfer-encoding", "connection"}
            response_headers = {k: v for k, v in resp.headers.items()
                              if k.lower() not in skip_headers}

            return Response(
                content=content,
                status_code=resp.status_code,
                headers=response_headers,
            )
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


async def _get_or_create_auth(port: int) -> Optional[str]:
    """Get auth cookie, logging in if needed."""
    global _n8n_auth_cookies

    if port in _n8n_auth_cookies:
        return _n8n_auth_cookies[port]

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                f"http://127.0.0.1:{port}/rest/login",
                json={"emailOrLdapLoginId": "admin@example.com", "password": "password123"}
            )
            if resp.status_code == 200:
                cookie = resp.cookies.get("n8n-auth")
                if cookie:
                    _n8n_auth_cookies[port] = cookie
                    return cookie
    except:
        pass
    return None


@router.get("/{port}/auto-login")
async def auto_login_redirect(request: Request, port: int):
    """
    Auto-login to n8n and redirect. Opens in new tab (not for iframe).
    """
    n8n_manager = request.app.state.n8n_manager
    instance = n8n_manager.get(port)

    # Also check queue manager if legacy manager doesn't know about it
    if not instance:
        queue_manager = getattr(request.app.state, 'n8n_queue_manager', None)
        if queue_manager:
            if queue_manager.main and queue_manager.main.port == port:
                instance = queue_manager.main
            elif port in queue_manager.workers:
                instance = queue_manager.workers[port]

    if not instance:
        raise HTTPException(status_code=404, detail=f"No n8n instance on port {port}")

    # Ensure user exists and get auth
    await _ensure_n8n_user(port)
    auth_cookie = await _get_n8n_auth_cookie(port)

    if not auth_cookie:
        return HTMLResponse(content=f"""
            <html><body style="background:#1a1a2e;color:#fff;font-family:sans-serif;padding:20px;">
                <p>n8n on port {port} is starting up...</p>
                <script>setTimeout(()=>location.reload(), 3000)</script>
            </body></html>
        """)

    n8n_host = request.url.hostname or "localhost"

    # Return page that posts login form to n8n (sets cookie on n8n's domain)
    return HTMLResponse(content=f"""
        <!DOCTYPE html>
        <html>
        <head><title>Connecting to n8n...</title></head>
        <body style="background:#1a1a2e;color:#fff;font-family:sans-serif;padding:20px;">
            <p>Connecting to n8n on port {port}...</p>
            <form id="loginForm" method="POST" action="http://{n8n_host}:{port}/rest/login">
                <input type="hidden" name="emailOrLdapLoginId" value="admin@example.com">
                <input type="hidden" name="password" value="password123">
            </form>
            <script>
                // Direct redirect with cookie
                fetch('http://{n8n_host}:{port}/rest/login', {{
                    method: 'POST',
                    headers: {{'Content-Type': 'application/json'}},
                    body: JSON.stringify({{emailOrLdapLoginId: 'admin@example.com', password: 'password123'}}),
                    credentials: 'include'
                }}).then(() => {{
                    window.location.href = 'http://{n8n_host}:{port}/';
                }}).catch(() => {{
                    window.location.href = 'http://{n8n_host}:{port}/';
                }});
            </script>
        </body>
        </html>
    """)


async def _get_n8n_auth_cookie(port: int) -> Optional[str]:
    """Get auth cookie for n8n instance, logging in if needed."""
    global _n8n_auth_cookies

    # Return cached cookie if exists
    if port in _n8n_auth_cookies:
        return _n8n_auth_cookies[port]

    # Try to login
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                f"http://127.0.0.1:{port}/rest/login",
                json={
                    "emailOrLdapLoginId": "admin@example.com",
                    "password": "password123"
                }
            )
            if resp.status_code == 200:
                # Extract cookie from response
                cookie = resp.cookies.get("n8n-auth")
                if cookie:
                    _n8n_auth_cookies[port] = cookie
                    return cookie
    except Exception:
        pass

    return None


async def _ensure_n8n_user(port: int):
    """Ensure the n8n database has a pre-configured user."""
    import sqlite3
    import os

    # Find the database path
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    db_path = os.path.join(base_dir, ".n8n-instances", f"n8n-{port}", ".n8n", "database.sqlite")

    if not os.path.exists(db_path):
        return

    try:
        import bcrypt
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Check if user exists with email
        cursor.execute('SELECT email FROM user WHERE roleSlug = "global:owner"')
        row = cursor.fetchone()

        if row and row[0] != "admin@example.com":
            # Update user
            password = 'password123'.encode('utf-8')
            hashed = bcrypt.hashpw(password, bcrypt.gensalt(rounds=10)).decode('utf-8')

            cursor.execute('''
                UPDATE user SET
                    email = 'admin@example.com',
                    firstName = 'Admin',
                    lastName = 'User',
                    password = ?,
                    settings = '{"userActivated":true,"isOnboarded":true}'
                WHERE roleSlug = 'global:owner'
            ''', (hashed,))

            cursor.execute('''
                INSERT OR REPLACE INTO settings (key, value, loadOnStartup)
                VALUES ('userManagement.isInstanceOwnerSetUp', 'true', 1)
            ''')

            conn.commit()

        conn.close()
    except Exception as e:
        print(f"Error ensuring n8n user: {e}")
