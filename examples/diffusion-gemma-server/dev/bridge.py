#!/usr/bin/env python3
import ctypes
import io
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import uuid
from ctypes import wintypes
from pathlib import Path

WSLC = (
    os.environ.get("DG_WSLC")
    or shutil.which("wslc")
    or r"C:\Program Files\WSL\wslc.exe"
)
IMAGE = os.environ.get("DG_CONTAINER_IMAGE", "diffusion-context-runner:local")
LOG = logging.getLogger("diffusion-container-bridge")


def main() -> int:
    parent_pid = int(sys.argv[1])
    argument = sys.argv[2]
    if argument == "--capabilities":
        return subprocess.run(
            [
                WSLC,
                "run",
                "--rm",
                "--gpus",
                "all",
                "--network",
                "none",
                IMAGE,
                argument,
            ],
            stdin=subprocess.DEVNULL,
            check=False,
        ).returncode

    model = Path(argument).resolve(strict=True)
    if not model.is_file():
        raise ValueError(f"Not a model file: {model}")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    parent = kernel.OpenProcess(0x00100000, False, parent_pid)
    if not parent:
        raise ctypes.WinError(ctypes.get_last_error())
    name = f"diffusion-context-{uuid.uuid4().hex}"
    with tempfile.TemporaryDirectory(prefix="dg_container_") as temporary:
        requests = Path(temporary)
        command = [
            WSLC,
            "run",
            "--interactive",
            "--gpus",
            "all",
            "--network",
            "none",
            "--name",
            name,
            "--mount",
            f"type=bind,source={model.parent},target=/model,readonly",
            "--mount",
            f"type=bind,source={requests},target=/requests,readonly",
        ]
        for key in ("MAXTOK", "NGL", "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "FA"):
            if key in os.environ:
                command.extend(["--env", f"{key}={os.environ[key]}"])
        command.extend([IMAGE, f"/model/{model.name}"])
        removed = threading.Event()
        remove_lock = threading.Lock()

        def remove_container() -> None:
            with remove_lock:
                if removed.is_set():
                    return
                result = subprocess.run(
                    [WSLC, "remove", "--force", name],
                    capture_output=True,
                    text=True,
                    timeout=15,
                    check=False,
                )
                if result.returncode == 0:
                    removed.set()
                else:
                    LOG.warning("Container cleanup: %s", result.stderr.strip())

        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            input_pipe = process.stdin
            output_pipe = process.stdout
            assert input_pipe is not None
            assert isinstance(output_pipe, io.BufferedReader)

            def forward_requests() -> None:
                try:
                    for line in sys.stdin:
                        source = line.rstrip("\r\n")
                        if source == "QUIT":
                            input_pipe.write(b"QUIT\n")
                            input_pipe.flush()
                            return
                        payload = Path(source).read_bytes()
                        target = requests / "request.json"
                        pending = requests / "request.tmp"
                        pending.write_bytes(payload)
                        pending.replace(target)
                        input_pipe.write(b"/requests/request.json\n")
                        input_pipe.flush()
                    input_pipe.close()
                except (OSError, ValueError):
                    LOG.exception("Request forwarding failed")
                    remove_container()

            def watch_parent() -> None:
                while process.poll() is None:
                    state = kernel.WaitForSingleObject(parent, 250)
                    if state == 0:
                        try:
                            remove_container()
                        except (OSError, subprocess.TimeoutExpired):
                            LOG.exception("Parent-exit container cleanup failed")
                        removed.wait(0.25)
                    elif state != 258:
                        LOG.error("Parent process wait failed: %s", state)
                        remove_container()
                        return

            threading.Thread(target=forward_requests, daemon=True).start()
            watcher = threading.Thread(target=watch_parent, daemon=True)
            watcher.start()
            try:
                while data := output_pipe.read1(65536):
                    sys.stdout.buffer.write(data)
                    sys.stdout.buffer.flush()
                return process.wait()
            finally:
                remove_container()
                if process.poll() is None:
                    process.wait(timeout=15)
                watcher.join(timeout=2)
        finally:
            kernel.CloseHandle(parent)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    try:
        raise SystemExit(main())
    except (OSError, ValueError, subprocess.SubprocessError):
        LOG.exception("Diffusion container bridge failed")
        raise SystemExit(1) from None
