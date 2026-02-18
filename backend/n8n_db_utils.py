"""
N8N Database Utilities

Utilities for copying workflows and credentials between n8n SQLite databases,
syncing credentials to workers, and aggregating execution history.
"""

import os
import sqlite3
import shutil
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime

logger = logging.getLogger("n8n_db_utils")


class N8nDatabaseUtils:
    """
    Utilities for n8n SQLite database operations.

    Handles:
    - Listing workflows from main database
    - Copying workflows and credentials to worker databases
    - Live credential syncing to running workers
    - Aggregating execution history
    """

    def __init__(self, main_db_path: str, history_db_path: str):
        """
        Initialize database utilities.

        Args:
            main_db_path: Path to main admin n8n database
            history_db_path: Path to aggregated history database
        """
        self.main_db_path = main_db_path
        self.history_db_path = history_db_path

        # Ensure history database exists with schema
        self._init_history_db()

    def _init_history_db(self):
        """Initialize history database with schema if it doesn't exist."""
        history_dir = os.path.dirname(self.history_db_path)
        if history_dir:
            os.makedirs(history_dir, exist_ok=True)

        conn = sqlite3.connect(self.history_db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS execution_history (
                id TEXT PRIMARY KEY,
                workflow_id TEXT NOT NULL,
                workflow_name TEXT,
                worker_port INTEGER,
                mode TEXT,
                status TEXT,
                started_at TEXT,
                finished_at TEXT,
                execution_time_ms INTEGER,
                data TEXT,
                error_message TEXT,
                aggregated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_workflow_id ON execution_history(workflow_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_started_at ON execution_history(started_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_status ON execution_history(status)")

        conn.commit()
        conn.close()
        logger.debug(f"History database initialized at {self.history_db_path}")

    def list_workflows(self) -> List[Dict[str, Any]]:
        """
        List all workflows from main database.

        Returns:
            List of workflow dictionaries with id, name, active, nodes count, etc.
        """
        if not os.path.exists(self.main_db_path):
            logger.warning(f"Main database not found: {self.main_db_path}")
            return []

        conn = sqlite3.connect(self.main_db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        try:
            cursor.execute("""
                SELECT id, name, active, nodes, createdAt, updatedAt, triggerCount
                FROM workflow_entity
                ORDER BY updatedAt DESC
            """)

            workflows = []
            for row in cursor.fetchall():
                # Count nodes from JSON
                nodes_json = row['nodes'] or '[]'
                try:
                    import json
                    nodes = json.loads(nodes_json)
                    node_count = len(nodes) if isinstance(nodes, list) else 0
                except:
                    node_count = 0

                workflows.append({
                    'id': row['id'],
                    'name': row['name'],
                    'active': bool(row['active']),
                    'node_count': node_count,
                    'trigger_count': row['triggerCount'] or 0,
                    'created_at': row['createdAt'],
                    'updated_at': row['updatedAt'],
                })

            return workflows

        except Exception as e:
            logger.error(f"Error listing workflows: {e}")
            return []
        finally:
            conn.close()

    def get_workflow(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a single workflow by ID with parsed JSON fields.

        Args:
            workflow_id: The workflow ID

        Returns:
            Workflow dictionary with parsed nodes/connections or None if not found
        """
        if not os.path.exists(self.main_db_path):
            return None

        conn = sqlite3.connect(self.main_db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        try:
            cursor.execute("SELECT * FROM workflow_entity WHERE id = ?", (workflow_id,))
            row = cursor.fetchone()

            if row:
                import json
                result = dict(row)
                # Parse JSON fields
                for field in ['nodes', 'connections', 'settings', 'staticData']:
                    if field in result and result[field]:
                        try:
                            result[field] = json.loads(result[field])
                        except (json.JSONDecodeError, TypeError):
                            pass
                return result
            return None

        except Exception as e:
            logger.error(f"Error getting workflow {workflow_id}: {e}")
            return None
        finally:
            conn.close()

    def update_workflow(self, workflow_id: str, workflow_data: Dict[str, Any]) -> bool:
        """
        Update a workflow's data (primarily nodes/parameters).

        Args:
            workflow_id: The workflow ID to update
            workflow_data: Dictionary containing the updated workflow data

        Returns:
            True if successful, False otherwise
        """
        if not os.path.exists(self.main_db_path):
            logger.error("Main database not found")
            return False

        conn = sqlite3.connect(self.main_db_path)
        cursor = conn.cursor()

        try:
            import json
            from datetime import datetime

            # Extract fields that can be updated
            nodes = workflow_data.get('nodes')
            connections = workflow_data.get('connections')
            settings = workflow_data.get('settings')

            updates = []
            params = []

            if nodes is not None:
                updates.append("nodes = ?")
                params.append(json.dumps(nodes) if isinstance(nodes, (list, dict)) else nodes)

            if connections is not None:
                updates.append("connections = ?")
                params.append(json.dumps(connections) if isinstance(connections, dict) else connections)

            if settings is not None:
                updates.append("settings = ?")
                params.append(json.dumps(settings) if isinstance(settings, dict) else settings)

            # Always update the timestamp
            updates.append("updatedAt = ?")
            params.append(datetime.utcnow().isoformat())

            if not updates:
                logger.warning("No fields to update")
                return False

            params.append(workflow_id)

            sql = f"UPDATE workflow_entity SET {', '.join(updates)} WHERE id = ?"
            cursor.execute(sql, params)
            conn.commit()

            if cursor.rowcount > 0:
                logger.info(f"Updated workflow {workflow_id}")
                return True
            else:
                logger.warning(f"Workflow {workflow_id} not found")
                return False

        except Exception as e:
            logger.error(f"Error updating workflow {workflow_id}: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()

    def copy_workflow_to_worker(
        self,
        workflow_id: str,
        worker_db_path: str,
        worker_name: str = "worker"
    ) -> bool:
        """
        Copy a single workflow and all credentials to a worker database.

        Creates a fresh n8n database with:
        - The specified workflow
        - ALL credentials (workflows may reference any)
        - Minimal settings and user configuration

        Args:
            workflow_id: The workflow ID to copy
            worker_db_path: Path to create the worker database
            worker_name: Name for logging

        Returns:
            True if successful, False otherwise
        """
        if not os.path.exists(self.main_db_path):
            logger.error(f"Main database not found: {self.main_db_path}")
            return False

        # Ensure worker directory exists
        worker_dir = os.path.dirname(worker_db_path)
        if worker_dir:
            os.makedirs(worker_dir, exist_ok=True)

        # Remove existing worker DB if present (retry on Windows file locks)
        if os.path.exists(worker_db_path):
            for attempt in range(5):
                try:
                    os.remove(worker_db_path)
                    break
                except PermissionError:
                    if attempt < 4:
                        import time
                        logger.debug(f"Worker DB locked, waiting... (attempt {attempt + 1})")
                        time.sleep(1)
                    else:
                        logger.warning(f"Could not remove locked worker DB: {worker_db_path}")
                        raise

        main_conn = sqlite3.connect(self.main_db_path)
        main_conn.row_factory = sqlite3.Row

        try:
            # Copy the main database as a starting point
            # This ensures we have the correct schema
            shutil.copy2(self.main_db_path, worker_db_path)
            logger.debug(f"Copied main DB to {worker_db_path}")

            # Now clean up the worker DB to only have what we need
            worker_conn = sqlite3.connect(worker_db_path)
            worker_cursor = worker_conn.cursor()

            # Delete all workflows except the one we want
            worker_cursor.execute(
                "DELETE FROM workflow_entity WHERE id != ?",
                (workflow_id,)
            )
            deleted_workflows = worker_cursor.rowcount
            logger.debug(f"Removed {deleted_workflows} other workflows from worker DB")

            # Keep workflow inactive initially — activation happens via API after spawn
            worker_cursor.execute(
                "UPDATE workflow_entity SET active = 0 WHERE id = ?",
                (workflow_id,)
            )

            # Clean up workflow-related tables
            worker_cursor.execute(
                "DELETE FROM shared_workflow WHERE workflowId != ?",
                (workflow_id,)
            )
            worker_cursor.execute(
                "DELETE FROM workflow_history WHERE workflowId != ?",
                (workflow_id,)
            )
            worker_cursor.execute(
                "DELETE FROM workflow_statistics WHERE workflowId != ?",
                (workflow_id,)
            )

            # Clear execution data (worker will create its own)
            worker_cursor.execute("DELETE FROM execution_entity")
            worker_cursor.execute("DELETE FROM execution_data")
            worker_cursor.execute("DELETE FROM execution_metadata")
            worker_cursor.execute("DELETE FROM execution_annotations")

            # Keep all credentials (we copy ALL to be safe)
            # Credentials are already in the copied DB

            # Keep user and settings
            # Already copied from main

            worker_conn.commit()
            worker_conn.close()

            logger.info(f"Created worker DB for workflow {workflow_id} at {worker_db_path}")
            return True

        except Exception as e:
            logger.error(f"Error copying workflow to worker: {e}")
            # Clean up on failure
            if os.path.exists(worker_db_path):
                try:
                    os.remove(worker_db_path)
                except:
                    pass
            return False
        finally:
            main_conn.close()

    def sync_credentials_to_worker(self, worker_db_path: str) -> bool:
        """
        Sync all credentials from main database to a worker database.

        This is used for live credential updates - when credentials are
        changed in main, they get synced to running workers.

        Args:
            worker_db_path: Path to worker database

        Returns:
            True if successful, False otherwise
        """
        if not os.path.exists(self.main_db_path):
            logger.error(f"Main database not found: {self.main_db_path}")
            return False

        if not os.path.exists(worker_db_path):
            logger.error(f"Worker database not found: {worker_db_path}")
            return False

        main_conn = sqlite3.connect(self.main_db_path)
        main_conn.row_factory = sqlite3.Row
        worker_conn = sqlite3.connect(worker_db_path)

        try:
            main_cursor = main_conn.cursor()
            worker_cursor = worker_conn.cursor()

            # Get all credentials from main
            main_cursor.execute("SELECT * FROM credentials_entity")
            credentials = main_cursor.fetchall()

            # Get column names
            main_cursor.execute("PRAGMA table_info(credentials_entity)")
            columns = [row[1] for row in main_cursor.fetchall()]

            # Upsert each credential
            synced = 0
            for cred in credentials:
                cred_dict = dict(cred)

                # Build upsert query
                placeholders = ','.join(['?' for _ in columns])
                update_parts = [f'{col}=excluded.{col}' for col in columns if col != 'id']

                query = f"""
                    INSERT INTO credentials_entity ({','.join(columns)})
                    VALUES ({placeholders})
                    ON CONFLICT(id) DO UPDATE SET {','.join(update_parts)}
                """

                values = [cred_dict[col] for col in columns]
                worker_cursor.execute(query, values)
                synced += 1

            # Also sync shared_credentials table
            main_cursor.execute("SELECT * FROM shared_credentials")
            shared_creds = main_cursor.fetchall()

            if shared_creds:
                main_cursor.execute("PRAGMA table_info(shared_credentials)")
                shared_columns = [row[1] for row in main_cursor.fetchall()]

                # Clear and re-insert (simpler than upsert for junction table)
                worker_cursor.execute("DELETE FROM shared_credentials")

                for shared in shared_creds:
                    shared_dict = dict(shared)
                    placeholders = ','.join(['?' for _ in shared_columns])
                    query = f"INSERT INTO shared_credentials ({','.join(shared_columns)}) VALUES ({placeholders})"
                    values = [shared_dict[col] for col in shared_columns]
                    worker_cursor.execute(query, values)

            worker_conn.commit()
            logger.debug(f"Synced {synced} credentials to worker")
            return True

        except Exception as e:
            logger.error(f"Error syncing credentials to worker: {e}")
            return False
        finally:
            main_conn.close()
            worker_conn.close()

    def aggregate_executions(
        self,
        worker_db_path: str,
        workflow_id: str,
        workflow_name: str,
        worker_port: int,
        mode: str
    ) -> int:
        """
        Copy execution records from worker database to history database.

        Args:
            worker_db_path: Path to worker database
            workflow_id: The workflow ID
            workflow_name: The workflow name (for display)
            worker_port: The worker port number
            mode: The execution mode (once/loop/standby)

        Returns:
            Number of records aggregated
        """
        if not os.path.exists(worker_db_path):
            logger.warning(f"Worker database not found: {worker_db_path}")
            return 0

        worker_conn = sqlite3.connect(worker_db_path)
        worker_conn.row_factory = sqlite3.Row
        history_conn = sqlite3.connect(self.history_db_path)

        try:
            worker_cursor = worker_conn.cursor()
            history_cursor = history_conn.cursor()

            # Get executions from worker
            worker_cursor.execute("""
                SELECT e.id, e.workflowId, e.status, e.startedAt, e.stoppedAt,
                       d.data
                FROM execution_entity e
                LEFT JOIN execution_data d ON e.id = d.executionId
                WHERE e.workflowId = ?
            """, (workflow_id,))

            count = 0
            for row in worker_cursor.fetchall():
                exec_id = row['id']
                status = row['status']
                started = row['startedAt']
                stopped = row['stoppedAt']
                data = row['data']

                # Calculate execution time
                exec_time_ms = None
                if started and stopped:
                    try:
                        start_dt = datetime.fromisoformat(started.replace('Z', '+00:00'))
                        stop_dt = datetime.fromisoformat(stopped.replace('Z', '+00:00'))
                        exec_time_ms = int((stop_dt - start_dt).total_seconds() * 1000)
                    except:
                        pass

                # Insert into history (ignore if already exists)
                history_cursor.execute("""
                    INSERT OR IGNORE INTO execution_history
                    (id, workflow_id, workflow_name, worker_port, mode, status,
                     started_at, finished_at, execution_time_ms, data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    f"{worker_port}-{exec_id}",  # Unique ID combining port and exec ID
                    workflow_id,
                    workflow_name,
                    worker_port,
                    mode,
                    status,
                    started,
                    stopped,
                    exec_time_ms,
                    data
                ))

                if history_cursor.rowcount > 0:
                    count += 1

            history_conn.commit()
            logger.info(f"Aggregated {count} execution records from worker {worker_port}")
            return count

        except Exception as e:
            logger.error(f"Error aggregating executions: {e}")
            return 0
        finally:
            worker_conn.close()
            history_conn.close()

    def get_execution_history(
        self,
        workflow_id: Optional[str] = None,
        status: Optional[str] = None,
        since: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Get execution history from aggregated history database.

        Args:
            workflow_id: Optional workflow ID to filter by
            status: Optional status to filter by (success/error/running)
            since: Optional ISO timestamp - only return executions after this time
            limit: Maximum records to return
            offset: Number of records to skip

        Returns:
            List of execution history dictionaries
        """
        if not os.path.exists(self.history_db_path):
            return []

        conn = sqlite3.connect(self.history_db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        try:
            conditions = []
            params = []

            if workflow_id:
                conditions.append("workflow_id = ?")
                params.append(workflow_id)
            if status:
                conditions.append("status = ?")
                params.append(status)
            if since:
                conditions.append("started_at >= ?")
                params.append(since)

            where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
            query = f"SELECT * FROM execution_history {where} ORDER BY started_at DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

        except Exception as e:
            logger.error(f"Error getting execution history: {e}")
            return []
        finally:
            conn.close()

    def get_history_stats(
        self,
        workflow_id: Optional[str] = None,
        since: Optional[str] = None,
    ) -> Dict[str, int]:
        """Get execution count stats grouped by status."""
        if not os.path.exists(self.history_db_path):
            return {"total": 0, "success": 0, "error": 0}

        conn = sqlite3.connect(self.history_db_path)
        cursor = conn.cursor()

        try:
            conditions = []
            params = []
            if workflow_id:
                conditions.append("workflow_id = ?")
                params.append(workflow_id)
            if since:
                conditions.append("started_at >= ?")
                params.append(since)

            where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

            cursor.execute(f"SELECT status, COUNT(*) as cnt FROM execution_history {where} GROUP BY status", params)
            rows = cursor.fetchall()

            stats = {"total": 0, "success": 0, "error": 0}
            for status_val, cnt in rows:
                stats["total"] += cnt
                if status_val in ("success", "finished"):
                    stats["success"] += cnt
                elif status_val in ("error", "failed", "crashed"):
                    stats["error"] += cnt
                else:
                    stats.setdefault(status_val or "unknown", 0)
                    stats[status_val or "unknown"] = cnt

            return stats

        except Exception as e:
            logger.error(f"Error getting history stats: {e}")
            return {"total": 0, "success": 0, "error": 0}
        finally:
            conn.close()

    def get_distinct_workflows(self) -> List[Dict[str, str]]:
        """Get distinct workflow IDs and names from history."""
        if not os.path.exists(self.history_db_path):
            return []

        conn = sqlite3.connect(self.history_db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                SELECT DISTINCT workflow_id, workflow_name
                FROM execution_history
                ORDER BY workflow_name
            """)
            return [{"id": row[0], "name": row[1]} for row in cursor.fetchall()]

        except Exception as e:
            logger.error(f"Error getting distinct workflows: {e}")
            return []
        finally:
            conn.close()

    def clear_history(self) -> bool:
        """Clear all execution history."""
        if not os.path.exists(self.history_db_path):
            return True

        conn = sqlite3.connect(self.history_db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("DELETE FROM execution_history")
            conn.commit()
            logger.info(f"Cleared execution history ({cursor.rowcount} records)")
            return True
        except Exception as e:
            logger.error(f"Error clearing history: {e}")
            return False
        finally:
            conn.close()

    def cleanup_worker_db(self, worker_db_path: str) -> bool:
        """
        Delete a worker database and its directory.

        Args:
            worker_db_path: Path to worker database

        Returns:
            True if successful, False otherwise
        """
        try:
            if os.path.exists(worker_db_path):
                for attempt in range(5):
                    try:
                        os.remove(worker_db_path)
                        logger.debug(f"Removed worker database: {worker_db_path}")
                        break
                    except PermissionError:
                        if attempt < 4:
                            import time
                            time.sleep(1)
                        else:
                            logger.warning(f"Could not remove locked worker DB: {worker_db_path}")
                            return False

            # Also try to remove the parent .n8n directory if empty
            n8n_dir = os.path.dirname(worker_db_path)
            if os.path.exists(n8n_dir) and not os.listdir(n8n_dir):
                os.rmdir(n8n_dir)

            # And the worker folder if empty
            worker_dir = os.path.dirname(n8n_dir)
            if os.path.exists(worker_dir) and not os.listdir(worker_dir):
                os.rmdir(worker_dir)

            return True

        except Exception as e:
            logger.error(f"Error cleaning up worker database: {e}")
            return False

    def get_encryption_key(self) -> Optional[str]:
        """
        Get the n8n encryption key from main database settings.

        n8n uses this key to encrypt credentials. Workers need the same key.

        Returns:
            Encryption key string or None if not found
        """
        if not os.path.exists(self.main_db_path):
            return None

        conn = sqlite3.connect(self.main_db_path)
        cursor = conn.cursor()

        try:
            cursor.execute(
                "SELECT value FROM settings WHERE key = 'encryptionKey'"
            )
            row = cursor.fetchone()

            if row:
                return row[0]
            return None

        except Exception as e:
            logger.error(f"Error getting encryption key: {e}")
            return None
        finally:
            conn.close()
