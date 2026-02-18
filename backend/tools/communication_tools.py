"""
Communication Tools - Send messages via Discord, Slack, email, etc.
"""

from typing import Dict, Any, List, Optional
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

logger = logging.getLogger("tools.communication")


TOOL_DEFINITIONS = [
    {
        "name": "send_discord",
        "description": "Send a message to a Discord channel via webhook.",
        "parameters": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "Message content to send"
                },
                "webhook_url": {
                    "type": "string",
                    "description": "Discord webhook URL (optional if configured globally)"
                },
                "username": {
                    "type": "string",
                    "description": "Override webhook username (optional)"
                },
                "embed": {
                    "type": "object",
                    "description": "Discord embed object (optional)",
                    "properties": {
                        "title": {"type": "string"},
                        "description": {"type": "string"},
                        "color": {"type": "integer"},
                        "url": {"type": "string"}
                    }
                }
            },
            "required": ["message"]
        }
    },
    {
        "name": "send_slack",
        "description": "Send a message to a Slack channel via webhook.",
        "parameters": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "Message text to send"
                },
                "webhook_url": {
                    "type": "string",
                    "description": "Slack webhook URL (optional if configured globally)"
                },
                "channel": {
                    "type": "string",
                    "description": "Override channel (e.g., '#general')"
                },
                "username": {
                    "type": "string",
                    "description": "Override username"
                },
                "icon_emoji": {
                    "type": "string",
                    "description": "Override icon emoji (e.g., ':robot_face:')"
                },
                "blocks": {
                    "type": "array",
                    "description": "Slack Block Kit blocks (optional)",
                    "items": {"type": "object"}
                }
            },
            "required": ["message"]
        }
    },
    {
        "name": "send_email",
        "description": "Send an email via SMTP.",
        "parameters": {
            "type": "object",
            "properties": {
                "to": {
                    "type": "string",
                    "description": "Recipient email address"
                },
                "subject": {
                    "type": "string",
                    "description": "Email subject"
                },
                "body": {
                    "type": "string",
                    "description": "Email body (plain text)"
                },
                "html_body": {
                    "type": "string",
                    "description": "HTML email body (optional)"
                },
                "cc": {
                    "type": "string",
                    "description": "CC recipients (comma-separated)"
                },
                "from_address": {
                    "type": "string",
                    "description": "From address (optional, uses config default)"
                }
            },
            "required": ["to", "subject", "body"]
        }
    },
    {
        "name": "send_telegram",
        "description": "Send a message via Telegram Bot API.",
        "parameters": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "Message text to send"
                },
                "chat_id": {
                    "type": "string",
                    "description": "Telegram chat ID (optional if configured)"
                },
                "bot_token": {
                    "type": "string",
                    "description": "Bot token (optional if configured)"
                },
                "parse_mode": {
                    "type": "string",
                    "description": "Parse mode: 'HTML', 'Markdown', 'MarkdownV2'",
                    "enum": ["HTML", "Markdown", "MarkdownV2"]
                },
                "disable_notification": {
                    "type": "boolean",
                    "description": "Send silently",
                    "default": False
                }
            },
            "required": ["message"]
        }
    },
    {
        "name": "send_webhook",
        "description": "Send data to any webhook URL (generic HTTP POST/PUT).",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Webhook URL"
                },
                "data": {
                    "type": "object",
                    "description": "JSON data to send"
                },
                "method": {
                    "type": "string",
                    "description": "HTTP method (default POST)",
                    "enum": ["POST", "PUT", "PATCH"],
                    "default": "POST"
                },
                "headers": {
                    "type": "object",
                    "description": "Additional headers"
                }
            },
            "required": ["url", "data"]
        }
    },
]


class CommunicationTools:
    """Tools for sending messages and notifications."""

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize communication tools.

        Args:
            config: Configuration dict with keys:
                - discord_webhook: str - Default Discord webhook URL
                - slack_webhook: str - Default Slack webhook URL
                - telegram_bot_token: str - Telegram bot token
                - telegram_chat_id: str - Default Telegram chat ID
                - smtp: dict - SMTP configuration:
                    - host: str
                    - port: int
                    - use_tls: bool
                    - username: str
                    - password: str
                    - from_address: str
        """
        self.config = config or {}

    async def send_discord(
        self,
        message: str,
        webhook_url: Optional[str] = None,
        username: Optional[str] = None,
        embed: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Send a message to Discord via webhook.

        Args:
            message: Message content
            webhook_url: Discord webhook URL
            username: Override webhook username
            embed: Discord embed object

        Returns:
            Dict with send result
        """
        try:
            import httpx
        except ImportError:
            return {"success": False, "error": "httpx not installed"}

        url = webhook_url or self.config.get("discord_webhook")
        if not url:
            return {
                "success": False,
                "error": "No Discord webhook URL provided or configured"
            }

        try:
            payload = {"content": message}
            if username:
                payload["username"] = username
            if embed:
                payload["embeds"] = [embed]

            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=payload, timeout=30)

            # Discord returns 204 on success
            if response.status_code in (200, 204):
                return {"success": True, "message": "Message sent to Discord"}
            else:
                return {
                    "success": False,
                    "error": f"Discord returned {response.status_code}: {response.text}"
                }

        except Exception as e:
            logger.error(f"send_discord error: {e}")
            return {"success": False, "error": str(e)}

    async def send_slack(
        self,
        message: str,
        webhook_url: Optional[str] = None,
        channel: Optional[str] = None,
        username: Optional[str] = None,
        icon_emoji: Optional[str] = None,
        blocks: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Send a message to Slack via webhook.

        Args:
            message: Message text
            webhook_url: Slack webhook URL
            channel: Override channel
            username: Override username
            icon_emoji: Override icon
            blocks: Slack Block Kit blocks

        Returns:
            Dict with send result
        """
        try:
            import httpx
        except ImportError:
            return {"success": False, "error": "httpx not installed"}

        url = webhook_url or self.config.get("slack_webhook")
        if not url:
            return {
                "success": False,
                "error": "No Slack webhook URL provided or configured"
            }

        try:
            payload = {"text": message}
            if channel:
                payload["channel"] = channel
            if username:
                payload["username"] = username
            if icon_emoji:
                payload["icon_emoji"] = icon_emoji
            if blocks:
                payload["blocks"] = blocks

            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=payload, timeout=30)

            if response.status_code == 200 and response.text == "ok":
                return {"success": True, "message": "Message sent to Slack"}
            else:
                return {
                    "success": False,
                    "error": f"Slack returned: {response.text}"
                }

        except Exception as e:
            logger.error(f"send_slack error: {e}")
            return {"success": False, "error": str(e)}

    async def send_email(
        self,
        to: str,
        subject: str,
        body: str,
        html_body: Optional[str] = None,
        cc: Optional[str] = None,
        from_address: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Send an email via SMTP.

        Args:
            to: Recipient email
            subject: Email subject
            body: Plain text body
            html_body: HTML body (optional)
            cc: CC recipients
            from_address: From address

        Returns:
            Dict with send result
        """
        smtp_config = self.config.get("smtp", {})
        if not smtp_config:
            return {
                "success": False,
                "error": "SMTP not configured. Set smtp settings in config."
            }

        try:
            # Build message
            if html_body:
                msg = MIMEMultipart("alternative")
                msg.attach(MIMEText(body, "plain"))
                msg.attach(MIMEText(html_body, "html"))
            else:
                msg = MIMEText(body)

            sender = from_address or smtp_config.get("from_address")
            if not sender:
                return {"success": False, "error": "No from_address configured"}

            msg["Subject"] = subject
            msg["From"] = sender
            msg["To"] = to
            if cc:
                msg["Cc"] = cc

            # Send
            host = smtp_config.get("host", "localhost")
            port = smtp_config.get("port", 587)

            with smtplib.SMTP(host, port, timeout=30) as server:
                if smtp_config.get("use_tls", True):
                    server.starttls()

                username = smtp_config.get("username")
                password = smtp_config.get("password")
                if username and password:
                    server.login(username, password)

                recipients = [to]
                if cc:
                    recipients.extend([addr.strip() for addr in cc.split(",")])

                server.sendmail(sender, recipients, msg.as_string())

            return {
                "success": True,
                "message": f"Email sent to {to}",
                "subject": subject
            }

        except smtplib.SMTPAuthenticationError:
            return {"success": False, "error": "SMTP authentication failed"}
        except smtplib.SMTPException as e:
            return {"success": False, "error": f"SMTP error: {e}"}
        except Exception as e:
            logger.error(f"send_email error: {e}")
            return {"success": False, "error": str(e)}

    async def send_telegram(
        self,
        message: str,
        chat_id: Optional[str] = None,
        bot_token: Optional[str] = None,
        parse_mode: Optional[str] = None,
        disable_notification: bool = False
    ) -> Dict[str, Any]:
        """
        Send a message via Telegram Bot API.

        Args:
            message: Message text
            chat_id: Telegram chat ID
            bot_token: Bot token
            parse_mode: Parse mode
            disable_notification: Send silently

        Returns:
            Dict with send result
        """
        try:
            import httpx
        except ImportError:
            return {"success": False, "error": "httpx not installed"}

        token = bot_token or self.config.get("telegram_bot_token")
        cid = chat_id or self.config.get("telegram_chat_id")

        if not token:
            return {"success": False, "error": "No Telegram bot token configured"}
        if not cid:
            return {"success": False, "error": "No Telegram chat_id provided or configured"}

        try:
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            payload = {
                "chat_id": cid,
                "text": message,
                "disable_notification": disable_notification
            }
            if parse_mode:
                payload["parse_mode"] = parse_mode

            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=payload, timeout=30)
                data = response.json()

            if data.get("ok"):
                return {
                    "success": True,
                    "message": "Message sent to Telegram",
                    "message_id": data.get("result", {}).get("message_id")
                }
            else:
                return {
                    "success": False,
                    "error": data.get("description", "Unknown Telegram error")
                }

        except Exception as e:
            logger.error(f"send_telegram error: {e}")
            return {"success": False, "error": str(e)}

    async def send_webhook(
        self,
        url: str,
        data: Dict[str, Any],
        method: str = "POST",
        headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Send data to any webhook URL.

        Args:
            url: Webhook URL
            data: JSON data to send
            method: HTTP method
            headers: Additional headers

        Returns:
            Dict with send result
        """
        try:
            import httpx
        except ImportError:
            return {"success": False, "error": "httpx not installed"}

        try:
            request_headers = {"Content-Type": "application/json"}
            if headers:
                request_headers.update(headers)

            async with httpx.AsyncClient() as client:
                response = await client.request(
                    method,
                    url,
                    json=data,
                    headers=request_headers,
                    timeout=30
                )

            success = response.status_code < 400

            # Try to parse response as JSON
            try:
                response_data = response.json()
            except Exception:
                response_data = response.text[:1000]

            return {
                "success": success,
                "status_code": response.status_code,
                "response": response_data
            }

        except Exception as e:
            logger.error(f"send_webhook error: {e}")
            return {"success": False, "error": str(e)}
