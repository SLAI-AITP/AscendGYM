#!/usr/bin/env python3
"""Local protocol bridge and run-scoped provider usage observation for native agents.

One process serves one protocol against an OpenAI-compatible Chat Completions
upstream:

  * ``--protocol anthropic``: POST /v1/messages   -> /chat/completions
  * ``--protocol responses``: POST /v1/responses  -> /chat/completions

The Responses path also handles Codex namespace/custom tools; the Anthropic
path translates Claude Code's text/tool_use/tool_result blocks.  Both paths
stream SSE, forward usage where available, and never log prompts or credentials.
Each POST is submitted exactly once.
"""

from __future__ import annotations

import argparse
import hmac
from contextlib import contextmanager
from collections import OrderedDict
import hashlib
import json
import os
import queue
import re
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, parse_qs

import requests


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval" / "native_harness"))
from native_bench.usage import TransportMeter, provider_usage


def usage_endpoint(handler):
    route = urlsplit(handler.path)
    if route.path != "/__native_usage":
        return False
    send = getattr(handler, "send_json", None) or handler._json
    token = os.environ.get("NATIVE_BENCH_USAGE_TOKEN", "")
    if not token or not hmac.compare_digest(handler.headers.get("Authorization", ""), "Bearer " + token):
        send(403, {"error": "invalid usage observer credential"})
        return True
    try:
        if handler.command == "POST":
            size = int(handler.headers.get("Content-Length", "0"))
            if size > 8192 or size < 0:
                raise ValueError("invalid reset body")
            handler.rfile.read(size)
            payload = handler.server.usage_meter.reset()
        else:
            value = parse_qs(route.query).get("at_ns", [None])[0]
            payload = handler.server.usage_meter.snapshot(int(value) if value is not None else None)
        send(200, payload)
    except (ValueError, KeyError, TypeError):
        send(409, {"error": "invalid usage snapshot or active requests"})
    return True


class ObservedResponse:
    def __init__(self, response, meter, record):
        self.response, self.meter, self.record = response, meter, record
        self.raw_usage = {}
        self.kind = "openai"
        self.final = False

    def __getattr__(self, name):
        return getattr(self.response, name)

    def observe(self, payload):
        if not isinstance(payload, dict):
            return
        typ = payload.get("type", "")
        response_payload = payload.get("response")
        response_payload = response_payload if isinstance(response_payload, dict) else {}
        if typ == "message_start":
            self.kind = "anthropic"
            self.raw_usage.update(payload.get("message", {}).get("usage") or {})
        elif typ == "message_delta":
            self.kind = "anthropic"
            self.raw_usage.update(payload.get("usage") or {})
            delta = payload.get("delta")
            stop_reason = delta.get("stop_reason") if isinstance(delta, dict) else None
            if isinstance(stop_reason, str) and stop_reason and isinstance(payload.get("usage"), dict):
                self.final = provider_usage(self.raw_usage, self.kind) is not None
        elif typ == "message_stop":
            self.final = True
        elif typ == "message" and isinstance(payload.get("usage"), dict):
            self.kind = "anthropic"
            self.raw_usage.update(payload["usage"])
            self.final = True
        elif isinstance(response_payload.get("usage"), dict):
            self.raw_usage.update(response_payload["usage"])
            self.final = typ in ("response.completed", "response.incomplete", "response.failed")
        elif isinstance(payload.get("usage"), dict):
            self.raw_usage.update(payload["usage"])
            self.final = bool("choices" in payload or payload.get("object") == "response"
                              or payload.get("object") == "response.compaction" or bool(payload.get("error")))
        if self.final and self.record["finished_ns"] is None:
            self.meter.finish(self.record, provider_usage(self.raw_usage, self.kind))

    @property
    def content(self):
        raw = self.response.content
        try:
            self.observe(json.loads(raw))
        except (ValueError, TypeError):
            pass
        return raw

    def json(self, **kwargs):
        payload = self.response.json(**kwargs)
        self.observe(payload)
        return payload

    def iter_lines(self, **kwargs):
        pending = []
        for line in self.response.iter_lines(**kwargs):
            raw = line.decode("utf-8", "replace") if isinstance(line, bytes) else line
            if raw.startswith("data:"):
                value = raw[5:].strip()
                if value != "[DONE]":
                    pending.append(value)
                    try:
                        payload = json.loads("\n".join(pending))
                    except ValueError:
                        pass
                    else:
                        self.observe(payload)
                        pending = []
            elif not raw.strip():
                pending = []
            yield line


@contextmanager
def metered_post(server, session, *args, **kwargs):
    record = server.usage_meter.begin()
    try:
        with session.post(*args, **kwargs) as response:
            yield ObservedResponse(response, server.usage_meter, record)
    finally:
        if record["finished_ns"] is None:
            server.usage_meter.finish(record, None)


class PassthroughHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def send_json(self, status, payload):
        raw = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if not usage_endpoint(self):
            self.forward()

    def do_POST(self):
        if not usage_endpoint(self):
            self.forward()

    def forward(self):
        route = urlsplit(self.path)
        if route.scheme or route.netloc or route.path.startswith("//"):
            self.send_json(400, {"error": "invalid upstream path"})
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if size < 0 or size > 100 * 1024 * 1024:
                raise ValueError("invalid request size")
            raw = self.rfile.read(size) if size else b""
            body = json.loads(raw) if raw else {}
            stream = bool(body.get("stream"))
            if stream and route.path.rstrip("/").endswith("/chat/completions"):
                body.setdefault("stream_options", {})["include_usage"] = True
                raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
            headers = {k: v for k, v in self.headers.items() if k.lower() not in
                {"host", "content-length", "connection", "transfer-encoding", "proxy-authorization"}}
            cfg = self.server.config
            if not any(k.lower() in {"authorization", "x-api-key"} for k in headers) and cfg["api_key"]:
                headers["Authorization"] = "Bearer " + cfg["api_key"]
            base_path = urlsplit(cfg["upstream"]).path.rstrip("/")
            suffix = self.path
            if base_path and (route.path == base_path or route.path.startswith(base_path + "/")):
                suffix = route.path[len(base_path):] + ("?" + route.query if route.query else "")
            url = cfg["upstream"].rstrip("/") + suffix
            generation = self.command == "POST" and route.path.rstrip("/").endswith(
                ("/responses", "/responses/compact", "/messages", "/chat/completions"))
            with requests.Session() as session:
                kwargs = dict(data=raw or None, headers=headers, stream=stream,
                    timeout=(15, cfg["timeout"]), allow_redirects=False)
                context = metered_post(self.server, session, url, **kwargs) if generation else session.request(self.command, url, **kwargs)
                with context as response:
                    self.send_response(response.status_code)
                    for key, value in response.headers.items():
                        if key.lower() not in {"content-length", "content-encoding", "transfer-encoding", "connection"}:
                            self.send_header(key, value)
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.close_connection = True
                    if stream:
                        for line in response.iter_lines(chunk_size=1, decode_unicode=False):
                            self.wfile.write(line + b"\n")
                            self.wfile.flush()
                    else:
                        if generation:
                            try:
                                response.json()
                            except ValueError:
                                pass
                        self.wfile.write(response.content)
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        except (ValueError, TypeError, requests.RequestException):
            self.close_connection = True


def passthrough_main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--name", default="")
    parser.add_argument("--timeout", type=float, default=3600)
    args = parser.parse_args(argv)
    parsed = urlsplit(args.upstream)
    if parsed.scheme not in ("http", "https") or not parsed.netloc or parsed.query or parsed.fragment:
        parser.error("invalid upstream base URL")
    server = ThreadingHTTPServer((args.host, args.port), PassthroughHandler)
    server.usage_meter = TransportMeter()
    server.config = {"upstream": args.upstream, "api_key": os.environ.get("PROXY_API_KEY", ""), "timeout": args.timeout}
    start_parent_watch()
    try:
        server.serve_forever()
    finally:
        server.server_close()


def new_id(prefix: str) -> str:
    return prefix + uuid.uuid4().hex


def start_parent_watch() -> None:
    """Exit if the launching bridge process disappears (including SIGKILL)."""
    parent = int(os.environ.get("PROXY_PARENT_PID", "0") or 0)
    if parent <= 0:
        return

    def watch() -> None:
        while True:
            time.sleep(2)
            try:
                os.kill(parent, 0)
            except OSError:
                os._exit(0)

    threading.Thread(target=watch, name="parent-watch", daemon=True).start()


class BadRequest(ValueError):
    pass


class ToolTypes(dict):
    """Kinds plus reversible Responses namespace / Chat name mappings."""

    def __init__(self):
        super().__init__()
        self.names = {}
        self.aliases = {}

    def wire_name(self, name: str, namespace: str | None = None) -> str:
        qualified = namespace + "." + name if namespace else name
        if qualified in self.aliases:
            return self.aliases[qualified]
        if name in self:
            return name
        matches = [alias for alias, original in self.names.items() if original["name"] == name]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise BadRequest("Ambiguous unqualified historical tool name.")
        return safe_tool_name(qualified.replace(".", "__"))


def safe_tool_name(name: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
    if len(safe) > 64:
        safe = safe[:51] + "_" + hashlib.sha256(name.encode()).hexdigest()[:12]
    return safe


def convert_tools(definitions: list) -> tuple[list, ToolTypes]:
    kinds = ToolTypes()
    translated = []

    def add(tool: dict, namespace: str | None = None, namespace_description: str = "") -> None:
        kind = tool.get("type")
        if kind == "namespace":
            if namespace:
                raise BadRequest("Nested namespace tools are not supported.")
            if not tool.get("name") or not isinstance(tool.get("tools"), list):
                raise BadRequest("Namespace tools require a name and a tools list.")
            for child in tool["tools"]:
                add(child, tool["name"], tool.get("description", ""))
            return
        if kind not in ("function", "custom"):
            raise BadRequest("Unsupported Responses tool type: " + str(kind))
        name = tool.get("name")
        if not isinstance(name, str) or not name:
            raise BadRequest("Tools require a name.")
        qualified = namespace + "." + name if namespace else name
        alias = safe_tool_name((namespace + "__" + name) if namespace else name)
        if alias in kinds or qualified in kinds.aliases:
            raise BadRequest("Tool names collide after namespace translation.")
        kinds[alias] = kind
        kinds.names[alias] = {"name": name, "namespace": namespace}
        kinds.aliases[qualified] = alias
        description = tool.get("description", "")
        if namespace_description:
            description = namespace_description + "\n" + description
        fn = {"name": alias, "description": description}
        if kind == "custom":
            if tool.get("format"):
                fn["description"] += "\nInput format: " + json.dumps(tool["format"], ensure_ascii=False)
            fn["parameters"] = {"type": "object", "properties": {"input": {"type": "string", "description": "The complete raw input for this tool."}}, "required": ["input"], "additionalProperties": False}
        else:
            fn["parameters"] = tool.get("parameters") or {"type": "object", "properties": {}}
            if "strict" in tool:
                fn["strict"] = tool["strict"]
        translated.append({"type": "function", "function": fn})

    for definition in definitions:
        add(definition)
    return translated, kinds


def text_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def message_content(content: Any) -> Any:
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if not isinstance(content, list):
        raise BadRequest("Message content must be a string or a list.")
    result = []
    for part in content:
        kind = part.get("type")
        if kind in ("input_text", "output_text", "text"):
            result.append({"type": "text", "text": part.get("text", "")})
        elif kind == "input_image":
            image = part.get("image_url")
            if not isinstance(image, str):
                raise BadRequest("Only image_url input images are supported.")
            image_data = {"url": image}
            if part.get("detail"):
                image_data["detail"] = part["detail"]
            result.append({"type": "image_url", "image_url": image_data})
        elif kind == "refusal":
            result.append({"type": "text", "text": part.get("refusal", "")})
        else:
            raise BadRequest("Unsupported message content type: " + str(kind))
    if all(part["type"] == "text" for part in result):
        return "".join(part["text"] for part in result)
    return result


def reasoning_text(item: dict) -> str:
    parts = item.get("content") or item.get("summary") or []
    return "".join(part.get("text", "") for part in parts if isinstance(part, dict) and part.get("type") in ("summary_text", "reasoning_text", "text"))


def convert_request(body: dict, model_name: str, reasoning_lookup=None) -> tuple[dict, dict]:
    if body.get("previous_response_id"):
        raise BadRequest("previous_response_id is not supported; send the complete input history with store=false.")
    if body.get("background"):
        raise BadRequest("Background Responses requests are not supported.")
    tools, tool_types = convert_tools(body.get("tools") or [])
    messages = []
    if body.get("instructions"):
        messages.append({"role": "system", "content": body["instructions"]})
    inputs = body.get("input", [])
    if isinstance(inputs, str):
        inputs = [{"role": "user", "content": inputs}]
    if not isinstance(inputs, list):
        raise BadRequest("input must be a string or a list.")
    pending_reasoning = ""
    for item in inputs:
        if not isinstance(item, dict):
            raise BadRequest("Every input item must be an object.")
        kind = item.get("type", "message")
        if kind == "message":
            role = item.get("role", "user")
            if role == "developer":
                role = "system"
            if role not in ("system", "user", "assistant", "tool"):
                raise BadRequest("Unsupported message role: " + str(role))
            message = {"role": role, "content": message_content(item.get("content", ""))}
            if role == "assistant" and pending_reasoning:
                message["reasoning_content"] = pending_reasoning
                pending_reasoning = ""
            if role == "tool":
                message["tool_call_id"] = item.get("tool_call_id")
            messages.append(message)
        elif kind in ("function_call", "custom_tool_call"):
            call_id = item.get("call_id")
            if not call_id or not item.get("name"):
                raise BadRequest("Tool calls require call_id and name.")
            arguments = item.get("arguments", "") if kind == "function_call" else json.dumps({"input": item.get("input", "")}, ensure_ascii=False)
            tool_call = {"id": call_id, "type": "function", "function": {"name": tool_types.wire_name(item["name"], item.get("namespace")), "arguments": text_content(arguments)}}
            if messages and messages[-1]["role"] == "assistant":
                messages[-1].setdefault("tool_calls", []).append(tool_call)
            else:
                messages.append({"role": "assistant", "content": None, "tool_calls": [tool_call]})
            prior_reasoning = pending_reasoning or (reasoning_lookup(call_id) if reasoning_lookup else "")
            if prior_reasoning and not messages[-1].get("reasoning_content"):
                messages[-1]["reasoning_content"] = prior_reasoning
            pending_reasoning = ""
        elif kind in ("function_call_output", "custom_tool_call_output"):
            if not item.get("call_id"):
                raise BadRequest("Tool outputs require call_id.")
            output = item.get("output", "")
            if isinstance(output, list):
                output = message_content(output)
            messages.append({"role": "tool", "tool_call_id": item["call_id"], "content": text_content(output)})
        elif kind == "reasoning":
            restored = reasoning_text(item)
            if item.get("encrypted_content") and not restored:
                raise BadRequest("Opaque encrypted reasoning cannot be translated; replay this bridge's reasoning summary instead.")
            pending_reasoning += restored
        elif kind == "compaction":
            raise BadRequest("Opaque compaction items cannot be translated; send complete text/tool history.")
        else:
            raise BadRequest("Unsupported input item type: " + str(kind))
    if not messages:
        raise BadRequest("The request contains no supported input messages.")

    upstream = {"model": model_name, "messages": messages, "stream": bool(body.get("stream", False))}
    if tools:
        upstream["tools"] = tools
    choice = body.get("tool_choice")
    if isinstance(choice, dict):
        if choice.get("type") not in ("function", "custom") or not choice.get("name"):
            raise BadRequest("Only named function/custom tool_choice is supported.")
        upstream["tool_choice"] = {"type": "function", "function": {"name": tool_types.wire_name(choice["name"], choice.get("namespace"))}}
    elif choice is not None:
        upstream["tool_choice"] = choice
    for field in ("temperature", "top_p", "parallel_tool_calls", "user"):
        if field in body and body[field] is not None:
            upstream[field] = body[field]
    if body.get("max_output_tokens") is not None:
        upstream["max_tokens"] = body["max_output_tokens"]
    if isinstance(body.get("reasoning"), dict) and body["reasoning"].get("effort"):
        upstream["reasoning_effort"] = body["reasoning"]["effort"]
    if upstream["stream"]:
        upstream["stream_options"] = {"include_usage": True}
    return upstream, tool_types


def convert_usage(usage: Any) -> dict | None:
    if not isinstance(usage, dict):
        return None
    converted = {}
    for original, target in (("prompt_tokens", "input_tokens"), ("completion_tokens", "output_tokens"), ("total_tokens", "total_tokens")):
        if original in usage:
            converted[target] = usage[original]
    prompt_details = usage.get("prompt_tokens_details") or {}
    completion_details = usage.get("completion_tokens_details") or {}
    if "cached_tokens" in prompt_details:
        converted["input_tokens_details"] = {"cached_tokens": prompt_details["cached_tokens"]}
    if "reasoning_tokens" in completion_details:
        converted["output_tokens_details"] = {"reasoning_tokens": completion_details["reasoning_tokens"]}
    return converted or None


def response_object(response_id: str, model: str, created: int, status: str, output: list, usage: dict | None = None, error: dict | None = None, incomplete: dict | None = None) -> dict:
    return {"id": response_id, "object": "response", "created_at": created, "status": status, "error": error, "incomplete_details": incomplete, "model": model, "output": output, "usage": usage, "store": False, "parallel_tool_calls": True}


def tool_item(call: dict, tool_types: dict, status: str) -> dict:
    name = call.get("name", "")
    if not name:
        raise ValueError("Upstream returned an unnamed tool call.")
    custom = tool_types.get(name) == "custom"
    item = {"id": call.get("item_id") or new_id("ctc_" if custom else "fc_"), "type": "custom_tool_call" if custom else "function_call", "call_id": call.get("call_id") or new_id("call_"), "name": name, "status": status}
    if isinstance(tool_types, ToolTypes) and name in tool_types.names:
        original = tool_types.names[name]
        item["name"] = original["name"]
        if original["namespace"]:
            item["namespace"] = original["namespace"]
    arguments = call.get("arguments", "")
    if custom:
        try:
            decoded = json.loads(arguments)
            if not isinstance(decoded, dict) or not isinstance(decoded.get("input"), str):
                raise ValueError("Custom tool arguments require a string input field.")
            item["input"] = decoded["input"]
        except (ValueError, TypeError):
            if status == "incomplete":
                item["input"] = ""
            else:
                raise ValueError("Upstream returned invalid custom tool arguments.") from None
    else:
        item["arguments"] = arguments
    return item


class BridgeServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple, config: dict):
        self.config = config
        self.usage_meter = TransportMeter()
        self.reasoning_cache = OrderedDict()
        self.reasoning_cache_bytes = 0
        self.reasoning_lock = threading.Lock()
        super().__init__(address, ResponsesHandler)

    def remember_reasoning(self, call_id: str, reasoning: str) -> None:
        if not call_id or not reasoning:
            return
        with self.reasoning_lock:
            previous = self.reasoning_cache.pop(call_id, "")
            self.reasoning_cache_bytes -= len(previous.encode("utf-8"))
            self.reasoning_cache[call_id] = reasoning
            self.reasoning_cache_bytes += len(reasoning.encode("utf-8"))
            while len(self.reasoning_cache) > 128 or self.reasoning_cache_bytes > 32 * 1024 * 1024:
                _, removed = self.reasoning_cache.popitem(last=False)
                self.reasoning_cache_bytes -= len(removed.encode("utf-8"))

    def recall_reasoning(self, call_id: str) -> str:
        with self.reasoning_lock:
            return self.reasoning_cache.get(call_id, "")


class ResponsesHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "ResponsesChatBridge/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        # Deliberately exclude arbitrary URLs, request bodies, and credentials.
        sys.stderr.write("%s %s %s\n" % (time.strftime("%Y-%m-%dT%H:%M:%S%z"), self.command, urlsplit(self.path).path))

    def send_json(self, status: int, payload: dict) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def error_json(self, status: int, message: str, code: str = "invalid_request_error") -> None:
        self.send_json(status, {"error": {"type": code, "code": code, "message": message}})

    def do_GET(self) -> None:
        if usage_endpoint(self):
            return
        route = urlsplit(self.path).path.rstrip("/")
        if route in ("/health", "/v1/health"):
            self.send_json(200, {"status": "ok", "model": self.server.config["name"], "stateless": True})
        elif route in ("/models", "/v1/models"):
            self.send_json(200, {"object": "list", "data": [{"id": self.server.config["name"], "object": "model", "owned_by": "catalog"}]})
        else:
            self.error_json(404, "Unknown route.")

    def do_POST(self) -> None:
        if usage_endpoint(self):
            return
        if urlsplit(self.path).path.rstrip("/") not in ("/responses", "/v1/responses"):
            self.error_json(404, "Only /v1/responses is supported.")
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if size <= 0 or size > 100 * 1024 * 1024:
                raise BadRequest("A JSON request body of at most 100 MiB is required.")
            body = json.loads(self.rfile.read(size))
            if not isinstance(body, dict):
                raise BadRequest("The request must be a JSON object.")
            upstream, tool_types = convert_request(body, self.server.config["name"], self.server.recall_reasoning)
            if "max_tokens" not in upstream and self.server.config.get("max_output") is not None:
                upstream["max_tokens"] = self.server.config["max_output"]
        except (ValueError, TypeError, KeyError, IndexError):
            # BadRequest messages are controlled strings, never body excerpts.
            exc = sys.exc_info()[1]
            self.error_json(400, str(exc) if isinstance(exc, BadRequest) else "Invalid JSON request.")
            return
        try:
            with requests.Session() as session:
                if self.server.config.get("proxy") == "never":
                    session.trust_env = False
                headers = {"Content-Type": "application/json", "Accept": "text/event-stream" if upstream["stream"] else "application/json"}
                if self.server.config.get("api_key"):
                    headers["Authorization"] = "Bearer " + self.server.config["api_key"]
                # No retry adapter and no redirects: a generation POST is never replayed.
                with metered_post(self.server, session, self.server.config["url"], json=upstream, headers=headers, stream=upstream["stream"], timeout=(15, self.server.config["timeout"]), allow_redirects=False) as result:
                    if result.status_code != 200:
                        raw = result.content
                        self.send_response(result.status_code)
                        self.send_header("Content-Type", result.headers.get("Content-Type", "application/json"))
                        self.send_header("Content-Length", str(len(raw)))
                        self.end_headers()
                        self.wfile.write(raw)
                        return
                    if upstream["stream"]:
                        self.stream_result(result, tool_types)
                    else:
                        self.nonstream_result(result.json(), tool_types)
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        except requests.RequestException:
            self.error_json(502, "Upstream connection failed or timed out; the POST was not retried.", "upstream_connection_error")
        except (ValueError, TypeError, KeyError):
            self.error_json(502, "Upstream returned an invalid Chat Completions response.", "upstream_protocol_error")

    def nonstream_result(self, result: dict, tool_types: dict) -> None:
        choice = result["choices"][0]
        finish = choice.get("finish_reason")
        if finish not in ("stop", "tool_calls", "length", "content_filter"):
            raise ValueError("Missing or unsupported finish reason.")
        status = "incomplete" if finish in ("length", "content_filter") else "completed"
        message = choice["message"]
        output = []
        reasoning = message.get("reasoning_content") or ""
        if reasoning:
            output.append({"id": new_id("rs_"), "type": "reasoning", "summary": [{"type": "summary_text", "text": reasoning}], "content": [{"type": "reasoning_text", "text": reasoning}], "status": status})
        if message.get("content"):
            output.append({"id": new_id("msg_"), "type": "message", "status": status, "role": "assistant", "content": [{"type": "output_text", "text": message["content"], "annotations": []}]})
        if message.get("refusal"):
            output.append({"id": new_id("msg_"), "type": "message", "status": status, "role": "assistant", "content": [{"type": "refusal", "refusal": message["refusal"]}]})
        for call in message.get("tool_calls") or []:
            self.server.remember_reasoning(call.get("id"), reasoning)
            output.append(tool_item({"call_id": call.get("id"), "name": call["function"].get("name"), "arguments": call["function"].get("arguments", "")}, tool_types, status))
        incomplete = {"reason": "max_output_tokens" if finish == "length" else "content_filter"} if status == "incomplete" else None
        self.send_json(200, response_object(new_id("resp_"), self.server.config["name"], int(time.time()), status, output, convert_usage(result.get("usage")), incomplete=incomplete))

    def stream_result(self, upstream: requests.Response, tool_types: dict) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        self.close_connection = True
        sequence = 0
        response_id = new_id("resp_")
        created = int(time.time())
        output = []
        message = None
        reasoning_item = None
        message_index = None
        reasoning_index = None
        usage = None
        finish = None
        calls = {}
        events = queue.Queue(maxsize=256)
        stop = threading.Event()

        def emit(kind: str, **payload: Any) -> None:
            nonlocal sequence
            payload = {"type": kind, "sequence_number": sequence, **payload}
            sequence += 1
            raw = "event: " + kind + "\ndata: " + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n\n"
            self.wfile.write(raw.encode("utf-8"))
            self.wfile.flush()

        def put_event(event: tuple) -> None:
            while not stop.is_set():
                try:
                    events.put(event, timeout=0.5)
                    return
                except queue.Full:
                    pass

        def read_stream() -> None:
            try:
                # chunk_size=1 avoids requests buffering a quiet, small SSE stream.
                data_lines = []
                for raw in upstream.iter_lines(chunk_size=1, decode_unicode=False):
                    if stop.is_set():
                        return
                    line = raw.decode("utf-8")
                    if not line:
                        if data_lines:
                            data = "\n".join(data_lines)
                            data_lines = []
                            if data == "[DONE]":
                                put_event(("end", None))
                                return
                            put_event(("data", json.loads(data)))
                    elif line.startswith("data:"):
                        data_lines.append(line[5:].lstrip(" "))
                if data_lines:
                    data = "\n".join(data_lines)
                    if data == "[DONE]":
                        put_event(("end", None))
                        return
                    put_event(("data", json.loads(data)))
                put_event(("end_of_file", None))
            except Exception:
                put_event(("error", None))

        try:
            initial = response_object(response_id, self.server.config["name"], created, "in_progress", [])
            emit("response.created", response=initial)
            emit("response.in_progress", response=initial)
            threading.Thread(target=read_stream, name="upstream-sse-reader", daemon=True).start()
            while True:
                try:
                    kind, payload = events.get(timeout=10)
                except queue.Empty:
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
                    continue
                if kind in ("error", "end_of_file"):
                    raise ValueError("Upstream stream failed or was interrupted.")
                if kind == "end":
                    break
                if payload.get("error"):
                    raise ValueError("Upstream sent an error inside the stream.")
                if payload.get("usage"):
                    usage = convert_usage(payload["usage"])
                choices = payload.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                if choice.get("finish_reason") is not None:
                    finish = choice["finish_reason"]
                delta = choice.get("delta") or {}
                if delta.get("refusal"):
                    raise ValueError("Upstream refused this generation.")
                if delta.get("reasoning_content"):
                    if not isinstance(delta["reasoning_content"], str):
                        raise ValueError("Unexpected upstream reasoning type.")
                    if reasoning_item is None:
                        reasoning_index = len(output)
                        reasoning_item = {"id": new_id("rs_"), "type": "reasoning", "summary": [], "status": "in_progress"}
                        output.append(reasoning_item)
                        emit("response.output_item.added", output_index=reasoning_index, item=dict(reasoning_item))
                        part = {"type": "summary_text", "text": ""}
                        reasoning_item["summary"].append(part)
                        emit("response.reasoning_summary_part.added", item_id=reasoning_item["id"], output_index=reasoning_index, summary_index=0, part=dict(part))
                    reasoning_item["summary"][0]["text"] += delta["reasoning_content"]
                    emit("response.reasoning_summary_text.delta", item_id=reasoning_item["id"], output_index=reasoning_index, summary_index=0, delta=delta["reasoning_content"])
                if delta.get("content"):
                    if not isinstance(delta["content"], str):
                        raise ValueError("Unexpected upstream content type.")
                    if message is None:
                        message_index = len(output)
                        message = {"id": new_id("msg_"), "type": "message", "status": "in_progress", "role": "assistant", "content": []}
                        output.append(message)
                        emit("response.output_item.added", output_index=message_index, item=dict(message))
                        part = {"type": "output_text", "text": "", "annotations": []}
                        message["content"].append(part)
                        emit("response.content_part.added", item_id=message["id"], output_index=message_index, content_index=0, part=dict(part))
                    message["content"][0]["text"] += delta["content"]
                    emit("response.output_text.delta", item_id=message["id"], output_index=message_index, content_index=0, delta=delta["content"], logprobs=[])
                for part in delta.get("tool_calls") or []:
                    index = part.get("index", 0)
                    call = calls.setdefault(index, {"call_id": "", "name": "", "arguments": ""})
                    if part.get("id"):
                        call["call_id"] = part["id"]
                    fn = part.get("function") or {}
                    call["name"] += fn.get("name") or ""
                    call["arguments"] += fn.get("arguments") or ""
            if finish not in ("stop", "tool_calls", "length", "content_filter"):
                raise ValueError("Upstream stream ended without a supported finish reason.")
            status = "incomplete" if finish in ("length", "content_filter") else "completed"
            # Validate all complete custom wrappers before emitting tool completion.
            converted_calls = [tool_item(calls[index], tool_types, status) for index in sorted(calls)]
            if reasoning_item is not None:
                reasoning_item["status"] = status
                part = reasoning_item["summary"][0]
                # Preserve the provider's raw reasoning for full-history replay;
                # the summary events also make it visible to existing clients.
                reasoning_item["content"] = [{"type": "reasoning_text", "text": part["text"]}]
                emit("response.reasoning_summary_text.done", item_id=reasoning_item["id"], output_index=reasoning_index, summary_index=0, text=part["text"])
                emit("response.reasoning_summary_part.done", item_id=reasoning_item["id"], output_index=reasoning_index, summary_index=0, part=part)
                emit("response.output_item.done", output_index=reasoning_index, item=reasoning_item)
                for item in converted_calls:
                    self.server.remember_reasoning(item["call_id"], part["text"])
            if message is not None:
                message["status"] = status
                part = message["content"][0]
                emit("response.output_text.done", item_id=message["id"], output_index=message_index, content_index=0, text=part["text"], logprobs=[])
                emit("response.content_part.done", item_id=message["id"], output_index=message_index, content_index=0, part=part)
                emit("response.output_item.done", output_index=message_index, item=message)
            for item in converted_calls:
                index = len(output)
                output.append(item)
                custom = item["type"] == "custom_tool_call"
                field = "input" if custom else "arguments"
                added = dict(item, status="in_progress", **{field: ""})
                emit("response.output_item.added", output_index=index, item=added)
                event = "response.custom_tool_call_input" if custom else "response.function_call_arguments"
                emit(event + ".delta", item_id=item["id"], output_index=index, delta=item[field])
                emit(event + ".done", item_id=item["id"], output_index=index, **{field: item[field]})
                emit("response.output_item.done", output_index=index, item=item)
            incomplete = {"reason": "max_output_tokens" if finish == "length" else "content_filter"} if status == "incomplete" else None
            final = response_object(response_id, self.server.config["name"], created, status, output, usage, incomplete=incomplete)
            emit("response.incomplete" if status == "incomplete" else "response.completed", response=final)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception:
            error = {"code": "upstream_protocol_error", "message": "Upstream stream failed, was interrupted, or contained invalid tool arguments; the POST was not retried."}
            try:
                emit("response.failed", response=response_object(response_id, self.server.config["name"], created, "failed", output, usage, error=error))
            except (BrokenPipeError, ConnectionResetError):
                pass
        finally:
            stop.set()
            upstream.close()


def responses_main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path,
                        default=Path(__file__).resolve().parents[1] / "configs" / "op_catalog.json")
    parser.add_argument("--model", default="glm-5.3-bigmodel", help="Upstream catalog model key.")
    parser.add_argument("--upstream", default=None,
                        help="OpenAI-compatible base URL; when set, --catalog/--model are not read.")
    parser.add_argument("--name", default=None, help="Model name returned to the client (direct mode).")
    parser.add_argument("--api-key", default=os.environ.get("PROXY_API_KEY") or os.environ.get("MODEL_PROXY_API_KEY", ""))
    parser.add_argument("--max-output", type=int, default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18008)
    parser.add_argument("--timeout", type=float, default=900, help="Upstream read timeout in seconds.")
    args = parser.parse_args(argv)
    try:
        if args.upstream:
            base = args.upstream.rstrip("/")
            config = {"name": args.name or args.model, "api_key": args.api_key, "proxy": "never",
                      "url": base + "/chat/completions", "timeout": args.timeout,
                      "max_output": args.max_output}
        else:
            catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
            selected = catalog["models"][args.model]
            base = selected["openai_base"].rstrip("/")
            config = {"name": selected["name"], "api_key": selected.get("api_key", ""), "proxy": selected.get("proxy"), "url": base + "/chat/completions", "timeout": args.timeout, "max_output": selected.get("max_output")}
        if urlsplit(base).scheme not in ("http", "https"):
            raise ValueError("Invalid upstream URL scheme.")
    except (OSError, ValueError, KeyError, TypeError):
        parser.error("Cannot load the requested model configuration from the catalog.")
    server = BridgeServer((args.host, args.port), config)
    start_parent_watch()
    print("Responses bridge listening on %s:%d; model=%s; stateless full-history requests only" % (args.host, args.port, config["name"]), file=sys.stderr, flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _id(prefix: str) -> str:
    return prefix + uuid.uuid4().hex


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif isinstance(block, dict) and block.get("type") == "tool_result":
                parts.append(_text_of(block.get("content")))
        return "\n".join(p for p in parts if p)
    return json.dumps(content, ensure_ascii=False)


def _image_part(block: dict) -> dict | None:
    source = block.get("source") or {}
    if source.get("type") == "base64":
        media = source.get("media_type") or "image/png"
        data = source.get("data") or ""
        return {"type": "image_url", "image_url": {"url": f"data:{media};base64,{data}"}}
    if source.get("type") == "url" and source.get("url"):
        return {"type": "image_url", "image_url": {"url": source["url"]}}
    return None


def _user_message(content: Any) -> list[dict]:
    """Convert one Anthropic user message into OpenAI messages.

    tool_result blocks become role=tool messages; text/image blocks become a
    following role=user message.  Order matters: OpenAI requires tool results
    immediately after the assistant tool_calls message.
    """
    if isinstance(content, str):
        return [{"role": "user", "content": content}]
    if not isinstance(content, list):
        return [{"role": "user", "content": _text_of(content)}]

    tool_messages: list[dict] = []
    user_parts: list[dict] = []
    for block in content:
        if not isinstance(block, dict):
            user_parts.append({"type": "text", "text": str(block)})
            continue
        kind = block.get("type")
        if kind == "tool_result":
            text = _text_of(block.get("content"))
            if block.get("is_error"):
                text = "Error: " + text
            tool_messages.append({
                "role": "tool",
                "tool_call_id": block.get("tool_use_id") or "",
                "content": text,
            })
        elif kind == "text":
            user_parts.append({"type": "text", "text": str(block.get("text", ""))})
        elif kind == "image":
            part = _image_part(block)
            if part:
                user_parts.append(part)
        # thinking / redacted_thinking and other beta blocks are ignored.
    messages = tool_messages
    if user_parts:
        if len(user_parts) == 1 and user_parts[0].get("type") == "text":
            messages.append({"role": "user", "content": user_parts[0]["text"]})
        else:
            messages.append({"role": "user", "content": user_parts})
    return messages


def _assistant_message(content: Any) -> dict:
    if isinstance(content, str):
        return {"role": "assistant", "content": content}
    if not isinstance(content, list):
        return {"role": "assistant", "content": _text_of(content)}

    texts: list[str] = []
    tool_calls: list[dict] = []
    for block in content:
        if not isinstance(block, dict):
            texts.append(str(block))
        elif block.get("type") == "text":
            texts.append(str(block.get("text", "")))
        elif block.get("type") == "tool_use":
            tool_calls.append({
                "id": block.get("id") or _id("call_"),
                "type": "function",
                "function": {
                    "name": block.get("name") or "",
                    "arguments": json.dumps(block.get("input") or {}, ensure_ascii=False),
                },
            })
    message: dict[str, Any] = {"role": "assistant"}
    text = "\n".join(t for t in texts if t)
    if text:
        message["content"] = text
    else:
        message["content"] = None
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


def anthropic_convert_request(body: dict, max_tokens_cap: int | None) -> dict:
    messages: list[dict] = []
    system = body.get("system")
    if system:
        messages.append({"role": "system", "content": _text_of(system)})

    for item in body.get("messages") or []:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if role == "assistant":
            messages.append(_assistant_message(content))
        elif role in ("user", "system"):
            messages.extend(_user_message(content) if role == "user" else
                            [{"role": "system", "content": _text_of(content)}])
        else:
            messages.append({"role": role or "user", "content": _text_of(content)})

    upstream: dict[str, Any] = {
        "model": body.get("model"),
        "messages": messages,
        "stream": bool(body.get("stream")),
    }
    max_tokens = body.get("max_tokens")
    if max_tokens_cap:
        max_tokens = min(int(max_tokens or max_tokens_cap), max_tokens_cap)
    if max_tokens:
        upstream["max_tokens"] = max_tokens
    for key in ("temperature", "top_p", "stop", "seed"):
        if body.get(key) is not None:
            upstream[key] = body[key]

    tools = []
    for tool in body.get("tools") or []:
        if not isinstance(tool, dict):
            continue
        name = tool.get("name")
        if not name:
            continue
        tools.append({
            "type": "function",
            "function": {
                "name": name,
                "description": tool.get("description") or "",
                "parameters": tool.get("input_schema") or {"type": "object", "properties": {}},
            },
        })
    if tools:
        upstream["tools"] = tools
        choice = body.get("tool_choice")
        if isinstance(choice, dict):
            kind = choice.get("type")
            if kind == "tool" and choice.get("name"):
                upstream["tool_choice"] = {
                    "type": "function",
                    "function": {"name": choice["name"]},
                }
            elif kind == "any":
                upstream["tool_choice"] = "required"
            elif kind == "none":
                upstream["tool_choice"] = "none"
            else:
                upstream["tool_choice"] = "auto"
        elif choice:
            upstream["tool_choice"] = choice

    if upstream["stream"]:
        upstream["stream_options"] = {"include_usage": True}
    return upstream


def _usage(usage: dict | None, fallback_output: int = 0) -> dict:
    usage = usage or {}
    prompt = int(usage.get("prompt_tokens") or 0)
    completion = int(usage.get("completion_tokens") or fallback_output)
    return {"input_tokens": prompt, "output_tokens": completion}


def _stop_reason(finish: str | None, has_tool: bool) -> str:
    if has_tool or finish == "tool_calls":
        return "tool_use"
    if finish == "length":
        return "max_tokens"
    if finish == "content_filter":
        return "end_turn"
    return "end_turn"


def anthropic_convert_response(result: dict, model: str) -> dict:
    choice = (result.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content: list[dict] = []
    text = message.get("content")
    if text:
        content.append({"type": "text", "text": text})
    for call in message.get("tool_calls") or []:
        raw = (call.get("function") or {}).get("arguments") or "{}"
        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError):
            parsed = {"raw": raw}
        content.append({
            "type": "tool_use",
            "id": call.get("id") or _id("toolu_"),
            "name": (call.get("function") or {}).get("name") or "",
            "input": parsed if isinstance(parsed, dict) else {"value": parsed},
        })
    if not content:
        content.append({"type": "text", "text": ""})
    return {
        "id": _id("msg_"),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": _stop_reason(choice.get("finish_reason"), bool(message.get("tool_calls"))),
        "stop_sequence": None,
        "usage": _usage(result.get("usage")),
    }


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "anthropic-proxy/1.0"

    @property
    def cfg(self) -> argparse.Namespace:
        return self.server.cfg  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("[%s] %s\n" % (time.strftime("%H:%M:%S"), fmt % args))

    def _json(self, status: int, payload: dict) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _error(self, status: int, kind: str, message: str) -> None:
        self._json(status, {"type": "error", "error": {"type": kind, "message": message}})

    def do_GET(self) -> None:
        if usage_endpoint(self):
            return
        route = urlsplit(self.path).path.rstrip("/")
        if route in ("/health", "/v1/health"):
            self._json(200, {"status": "ok", "upstream": self.cfg.upstream})
        elif route in ("/models", "/v1/models"):
            self._json(200, {
                "object": "list",
                "data": [{"id": self.cfg.model or "model", "object": "model", "owned_by": "proxy"}],
            })
        else:
            self._error(404, "not_found_error", "Unknown route.")

    def do_POST(self) -> None:
        if usage_endpoint(self):
            return
        if urlsplit(self.path).path.rstrip("/") not in ("/v1/messages", "/messages"):
            self._error(404, "not_found_error", "Only /v1/messages is supported.")
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if size <= 0 or size > 100 * 1024 * 1024:
                raise ValueError("A JSON request body of at most 100 MiB is required.")
            body = json.loads(self.rfile.read(size))
            if not isinstance(body, dict):
                raise ValueError("The request must be a JSON object.")
            upstream_body = anthropic_convert_request(body, self.cfg.max_tokens)
        except (ValueError, TypeError, KeyError) as exc:
            self._error(400, "invalid_request_error", str(exc) or "Invalid request.")
            return

        url = self.cfg.upstream.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.cfg.api_key:
            headers["Authorization"] = "Bearer " + self.cfg.api_key
        try:
            with requests.Session() as session:
                session.trust_env = False
                with metered_post(self.server, session,
                    url, json=upstream_body, headers=headers,
                    stream=upstream_body["stream"],
                    timeout=(15, self.cfg.timeout), allow_redirects=False,
                ) as result:
                    if result.status_code != 200:
                        raw = result.content[:2000]
                        try:
                            detail = json.loads(raw).get("error", raw.decode("utf-8", "replace"))
                        except Exception:
                            detail = raw.decode("utf-8", "replace")
                        self._error(502, "upstream_error", str(detail))
                        return
                    if upstream_body["stream"]:
                        self._stream(result, body.get("model") or "")
                    else:
                        self._json(200, anthropic_convert_response(result.json(), body.get("model") or ""))
        except requests.RequestException as exc:
            self._error(502, "upstream_error", f"Upstream connection failed: {exc.__class__.__name__}")

    def _stream(self, result: requests.Response, model: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        self.close_connection = True

        def event(kind: str, payload: dict) -> None:
            data = json.dumps(payload, ensure_ascii=False)
            self.wfile.write(f"event: {kind}\ndata: {data}\n\n".encode())
            self.wfile.flush()

        started = False
        text_started = False
        text_index = 0
        next_index = 1
        calls: dict[int, dict] = {}
        finish: str | None = None
        usage: dict | None = None

        def start_message(prompt_tokens: int = 0) -> None:
            nonlocal started
            if started:
                return
            started = True
            event("message_start", {
                "type": "message_start",
                "message": {
                    "id": _id("msg_"),
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": prompt_tokens, "output_tokens": 0},
                },
            })

        try:
            for line in result.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                if chunk.get("usage"):
                    usage = chunk["usage"]
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                delta = choice.get("delta") or {}
                if not started:
                    start_message(int((usage or {}).get("prompt_tokens") or 0))
                text = delta.get("content")
                if text:
                    if not text_started:
                        text_started = True
                        event("content_block_start", {
                            "type": "content_block_start",
                            "index": text_index,
                            "content_block": {"type": "text", "text": ""},
                        })
                    event("content_block_delta", {
                        "type": "content_block_delta",
                        "index": text_index,
                        "delta": {"type": "text_delta", "text": text},
                    })
                for call in delta.get("tool_calls") or []:
                    index = int(call.get("index") or 0)
                    slot = calls.setdefault(index, {"id": "", "name": "", "arguments": ""})
                    if call.get("id"):
                        slot["id"] = call["id"]
                    function = call.get("function") or {}
                    if function.get("name"):
                        slot["name"] = function["name"]
                    if function.get("arguments"):
                        slot["arguments"] += function["arguments"]
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]

            if not started:
                start_message(int((usage or {}).get("prompt_tokens") or 0))
            if text_started:
                event("content_block_stop", {"type": "content_block_stop", "index": text_index})
            for _, slot in sorted(calls.items()):
                event("content_block_start", {
                    "type": "content_block_start",
                    "index": next_index,
                    "content_block": {
                        "type": "tool_use",
                        "id": slot["id"] or _id("toolu_"),
                        "name": slot["name"],
                        "input": {},
                    },
                })
                event("content_block_delta", {
                    "type": "content_block_delta",
                    "index": next_index,
                    "delta": {"type": "input_json_delta", "partial_json": slot["arguments"] or "{}"},
                })
                event("content_block_stop", {"type": "content_block_stop", "index": next_index})
                next_index += 1
            event("message_delta", {
                "type": "message_delta",
                "delta": {"stop_reason": _stop_reason(finish, bool(calls)), "stop_sequence": None},
                "usage": _usage(usage),
            })
            event("message_stop", {"type": "message_stop"})
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        except Exception as exc:  # noqa: BLE001 - report a protocol error to the client
            try:
                event("error", {
                    "type": "error",
                    "error": {"type": "api_error", "message": f"Proxy stream error: {exc.__class__.__name__}"},
                })
            except (BrokenPipeError, ConnectionResetError):
                self.close_connection = True


def anthropic_main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listen", "--host", dest="listen", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18008)
    parser.add_argument("--upstream", required=True, help="OpenAI-compatible base, e.g. http://host:8008/v1")
    parser.add_argument("--api-key", default=os.environ.get("PROXY_API_KEY") or os.environ.get("ANTHROPIC_PROXY_API_KEY", ""))
    parser.add_argument("--max-tokens", type=int, default=16384,
                        help="Clamp Anthropic max_tokens to this value (0 disables).")
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--model", default="", help="Optional model id returned by GET /v1/models.")
    args = parser.parse_args(argv)

    server = ThreadingHTTPServer((args.listen, args.port), Handler)
    server.cfg = args  # type: ignore[attr-defined]
    server.usage_meter = TransportMeter()
    start_parent_watch()
    print(f"anthropic proxy listening on http://{args.listen}:{args.port} -> {args.upstream}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

def main() -> None:
    """Select the protocol implementation and start one local server."""
    if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help"):
        print("usage: proxy.py --protocol {anthropic,responses,passthrough} [options]")
        print("  --protocol anthropic   expose POST /v1/messages")
        print("  --protocol responses   expose POST /v1/responses")
        print("  --protocol passthrough preserve native HTTP protocol and observe provider usage")
        print("Run with --protocol <name> --help for protocol-specific options.")
        return
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--protocol", choices=("anthropic", "responses", "passthrough"), required=True)
    parser.add_argument("--parent-pid", type=int, default=0)
    args, rest = parser.parse_known_args()
    if args.parent_pid > 0:
        os.environ["PROXY_PARENT_PID"] = str(args.parent_pid)
    if args.protocol == "passthrough":
        passthrough_main(rest)
    elif args.protocol == "anthropic":
        anthropic_main(rest)
    else:
        responses_main(rest)


if __name__ == "__main__":
    main()
