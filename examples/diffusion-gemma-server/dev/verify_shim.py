#!/usr/bin/env python3
import json
import os
from pathlib import Path

from fastapi.testclient import TestClient


def main() -> None:
    # Match the environment Studio sets for its shim child.
    os.environ["UNSLOTH_IS_PRESENT"] = "1"
    from unsloth_zoo.diffusion_studio import shim

    model = (
        Path(__file__).parent
        / "fixtures"
        / "diffusion bridge context \u6a21\u578b.gguf"
    )
    server = shim.V.VisualServer(str(model), gpu="0", maxtok=65535, ngl=99)
    process = server.p
    assert process is not None
    try:
        shim._STATE["server"] = server
        with TestClient(shim.app, base_url="http://127.0.0.1") as client:
            response = client.get("/health")
            response.raise_for_status()
            health = response.json()
            assert health["context_length"] == 65535
            assert health["requested_context_length"] == 65535
            print(json.dumps(health))
    finally:
        shim._STATE.pop("server", None)
        server.close()
        if process.stdin is not None:
            process.stdin.close()
        if process.stdout is not None:
            process.stdout.close()


if __name__ == "__main__":
    main()
