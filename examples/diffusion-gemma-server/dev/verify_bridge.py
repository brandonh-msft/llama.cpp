#!/usr/bin/env python3
import importlib.util
import os
import struct
import subprocess
import time
from importlib.machinery import SourceFileLoader
from pathlib import Path

from bridge import WSLC

ROOT = Path(__file__).parent
ZOO = Path(os.environ.get("DG_ZOO_SOURCE", str(ROOT.parents[3] / "unsloth-zoo")))
ENGINE = ZOO / "unsloth_zoo" / "diffusion_studio" / "visual_engine.py"


def containers() -> set[str]:
    result = subprocess.run(
        [WSLC, "ps", "--all", "--quiet", "--filter", "name=diffusion-context-"],
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
    )
    return set(result.stdout.split())


def wait_for_cleanup(before: set[str]) -> None:
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if containers() == before:
            return
        time.sleep(0.25)
    raise AssertionError("Native model container survived launcher exit")


def main() -> None:
    source = ROOT / "fixtures" / "diffusion-gemma-synthetic.gguf"
    payload = source.read_bytes()
    key = b"diffusion-gemma.context_length"
    original = struct.pack("<Q", len(key)) + key + struct.pack("<II", 4, 256)
    replacement = struct.pack("<Q", len(key)) + key + struct.pack("<II", 4, 65536)
    assert payload.count(original) == 1, "Unexpected fixture context metadata"
    model = source.with_name("diffusion bridge context \u6a21\u578b.gguf")
    model.write_bytes(payload.replace(original, replacement))
    loader = SourceFileLoader("visual_engine", str(ENGINE))
    specification = importlib.util.spec_from_loader(loader.name, loader)
    assert specification is not None
    engine = importlib.util.module_from_spec(specification)
    loader.exec_module(engine)
    binary = str(ROOT / "diffusion-context-runner.exe")
    start = time.monotonic()
    capabilities = subprocess.run(
        [binary, "--capabilities"],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    assert "exact_context_v1" in capabilities.stdout.split()
    print(f"Capability handshake: {time.monotonic() - start:.2f}s", flush=True)
    baseline = containers()
    messages = [{"role": "user", "content": "hello"}]
    large_messages = [{"role": "user", "content": "abcde" * 49700}]
    for context in (4096, 32768, 65535):
        server = engine.VisualServer(
            str(model),
            gpu="0",
            maxtok=context,
            server_bin=binary,
            ngl=99,
        )
        try:
            assert server.maxtok == context
            result = engine.generate_visual(server, messages, max_blocks=1)
            assert result, "No synthetic GPU output"
            print(
                f"GPU READY/generation: requested={context} resolved={server.maxtok}",
                flush=True,
            )
            if context == 32768:
                try:
                    engine.generate_visual(server, large_messages, max_blocks=1)
                except engine.ContextOverflow as error:
                    assert 32768 < error.needed < 65535 and error.budget == context
                    print(
                        f"Expected overflow: needed={error.needed} budget={error.budget}",
                        flush=True,
                    )
                else:
                    raise AssertionError("Large prompt incorrectly accepted at 32768")
            elif context == 65535:
                assert engine.generate_visual(server, large_messages, max_blocks=1)
                assert engine.generate_visual(server, messages, max_blocks=1)
                print(
                    "Large prompt GPU generation and short-prompt reuse passed",
                    flush=True,
                )
        finally:
            server.close()
        wait_for_cleanup(baseline)
    server = engine.VisualServer(
        str(model),
        gpu="0",
        maxtok=4096,
        server_bin=binary,
        ngl=99,
    )
    server.p.kill()
    server.p.wait(timeout=10)
    wait_for_cleanup(baseline)
    server.p.stdin.close()
    server.p.stdout.close()
    server.close()
    print("Forced launcher termination releases the GPU container", flush=True)


if __name__ == "__main__":
    main()
