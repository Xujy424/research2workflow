"""作者：xujiayi | 创建日期：2026-11-11
本地单因子研究网页：标准库HTTP服务、后台串行研究队列、ECharts前端。
启动：python -m v2.SingleFactorTest.web --root Z:/ --port 8765
"""
from __future__ import annotations
import argparse
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import socket
import json
import mimetypes
from pathlib import Path
import threading
from time import time
from urllib.parse import urlparse, parse_qs
from uuid import uuid4

if not __package__:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "v2.SingleFactorTest"

from .run import ReportRunner, list_factors, run

STATIC = Path(__file__).with_name("static")
RUN_KEYS = {"factor_name", "construction", "normalization", "transform", "direction",
            "neutralize_industry", "neutralize_size", "long_group", "short_group",
            "start", "end", "description", "sections", "section_options"}


class CancelledReport(Exception):
    """用户切换因子后，在下一模块边界终止旧任务。"""


class ResearchService:
    """管理计算任务；Params: root/可注入runner；Return: 服务状态对象。"""
    def __init__(self, root=None, runner=None):
        self.runner = runner or ReportRunner(root)
        self.root = self.runner.root
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="single-factor")
        self.lock = threading.Lock()
        self.jobs = OrderedDict()

    def submit(self, params):
        """提交报告；Params: 显式入口参数dict；Return: 任务ID。"""
        if not isinstance(params, dict) or set(params) - RUN_KEYS:
            raise ValueError("请求参数包含未知字段")
        if not isinstance(params.get("factor_name"), str):
            raise ValueError("必须选择因子")
        for name in ("neutralize_industry", "neutralize_size"):
            if name in params and not isinstance(params[name], bool):
                raise ValueError(f"{name}须为布尔值")
        if params.get("direction", 1) not in (-1, 1):
            raise ValueError("因子方向须为±1")
        job_id = uuid4().hex
        with self.lock:
            active = sum(j["status"] in ("queued", "running") and not j["cancel"] for j in self.jobs.values())
            if active >= 8:
                raise ValueError("已有多个计算任务，请等待完成后再更新")
            # 只清理完成任务，避免仍被浏览器轮询的运行任务失踪。
            for key in list(self.jobs):
                if len(self.jobs) < 24:
                    break
                if self.jobs[key]["status"] not in ("queued", "running"):
                    del self.jobs[key]
            self.jobs[job_id] = {"status": "queued", "message": "等待计算", "events": [],
                                 "revision": 0, "cancel": False, "started": time(), "errors": {}}
        self.pool.submit(self._work, job_id, params)
        return job_id

    def _work(self, job_id, params):
        """后台执行run.py；Params: ID/入口参数；Return: None，结果写入任务事件。"""
        def progress(stage, message, payload):
            with self.lock:
                job = self.jobs[job_id]
                if job["cancel"]:
                    raise CancelledReport()
                job["status"], job["message"] = "running", message
                if payload is not None:
                    job["revision"] += 1
                    job["events"].append({"revision": job["revision"], "stage": stage, "payload": payload})
        try:
            progress("start", "开始计算", None)
            report = run(**params, runner=self.runner, progress=progress)
            with self.lock:
                job = self.jobs[job_id]
                job.update(status="partial" if report["errors"] else "complete",
                           message="部分模块需检查数据或参数" if report["errors"] else "全部计算完成",
                           errors=report["errors"])
        except CancelledReport:
            with self.lock:
                self.jobs[job_id].update(status="cancelled", message="已切换因子，旧任务停止")
        except Exception as error:
            import traceback
            traceback.print_exc()
            with self.lock:
                self.jobs[job_id].update(status="failed", message=str(error))

    def snapshot(self, job_id, since=0):
        """增量状态；Params: ID/已接收版本；Return: JSON任务状态。"""
        with self.lock:
            job = self.jobs[job_id]
            return {"status": job["status"], "message": job["message"], "revision": job["revision"],
                    "elapsed": round(time() - job["started"], 1),
                    "events": [e for e in job["events"] if e["revision"] > since], "errors": job["errors"]}


def create_server(root=None, host="127.0.0.1", port=8765, runner=None):
    """创建本地HTTP服务；localhost同时监听IPv4/IPv6回环地址。

    Params: 数据根/监听地址/端口/测试runner；Return: 可serve_forever/shutdown的本地服务。
    """
    service = ResearchService(root, runner)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            """仅记录非轮询请求；Params: 日志参数；Return: None。"""
            if "/api/jobs/" not in str(args):
                super().log_message(format, *args)

        def send(self, status, body, content_type="application/json; charset=utf-8"):
            """写响应；Params: 状态/内容/类型；Return: None。"""
            if not isinstance(body, bytes):
                body = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            """读取页面、因子目录或任务；Params: HTTP请求；Return: HTTP响应。"""
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/api/health":
                    return self.send(200, {"status": "ok"})
                if parsed.path == "/api/factors":
                    return self.send(200, {"factors": list_factors(service.root)})
                if parsed.path.startswith("/api/jobs/"):
                    job_id = parsed.path.rsplit("/", 1)[-1]
                    since = int(parse_qs(parsed.query).get("since", [0])[0])
                    return self.send(200, service.snapshot(job_id, since))
                files = {"/": "index.html", "/app.js": "app.js", "/style.css": "style.css",
                         "/echarts.min.js": "vendor/echarts.min.js"}
                if parsed.path not in files:
                    return self.send(404, {"error": "页面不存在"})
                path = STATIC / files[parsed.path]
                content_type = mimetypes.guess_type(path)[0] or "application/octet-stream"
                return self.send(200, path.read_bytes(), content_type + "; charset=utf-8")
            except KeyError:
                self.send(404, {"error": "任务已过期，请重新运行"})
            except (OSError, ValueError) as error:
                self.send(400, {"error": str(error)})

        def do_POST(self):
            """提交/取消任务；Params: JSON HTTP请求；Return: 任务ID或错误。"""
            try:
                # 本地页面同源访问，不允许其他网页跨站提交本地计算任务。
                origin = self.headers.get("Origin")
                if origin and origin != f"http://{self.headers.get('Host')}":
                    return self.send(403, {"error": "仅允许当前本地页面提交请求"})
                if "application/json" not in self.headers.get("Content-Type", ""):
                    return self.send(415, {"error": "需要JSON请求"})
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length <= 65536:
                    return self.send(400, {"error": "请求大小无效"})
                params = json.loads(self.rfile.read(length))
                if self.path == "/api/run":
                    return self.send(202, {"job_id": service.submit(params)})
                if self.path == "/api/cancel":
                    with service.lock:
                        job = service.jobs.get(params.get("job_id"))
                        if job:
                            job["cancel"] = True
                    return self.send(200, {"cancelled": True})
                self.send(404, {"error": "接口不存在"})
            except (ValueError, TypeError) as error:
                self.send(400, {"error": str(error)})

    class ThreadingHTTPServerV6(ThreadingHTTPServer):
        address_family = socket.AF_INET6

        def server_bind(self):
            self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
            super().server_bind()

    if host in ("localhost", "loopback"):
        servers = [ThreadingHTTPServer(("127.0.0.1", port), Handler)]
        actual_port = servers[0].server_port
        try:
            servers.append(ThreadingHTTPServerV6(("::1", actual_port), Handler))
        except OSError:
            # IPv6在操作系统中关闭时，保留标准IPv4本地访问。
            pass

        class LocalServerPair:
            """同端口本机双栈服务。"""
            def __init__(self):
                self.server_port = actual_port
                self.service = service
                self.daemon_threads = True
                self.instances = servers

            def serve_forever(self):
                for secondary in self.instances[1:]:
                    threading.Thread(target=secondary.serve_forever, daemon=True).start()
                self.instances[0].serve_forever()

            def shutdown(self):
                for instance in self.instances:
                    instance.shutdown()

            def server_close(self):
                for instance in self.instances:
                    instance.server_close()

        return LocalServerPair()
    server = ThreadingHTTPServer((host, port), Handler)
    server.service = service
    server.daemon_threads = True
    return server


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="单因子研究本地网页")
    parser.add_argument("--root", default=None)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = create_server(**vars(args))
    print(f"SingleFactorTest: http://{args.host}:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        server.service.pool.shutdown(wait=False, cancel_futures=True)
