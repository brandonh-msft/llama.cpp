ARG BASE_IMAGE=python:3.14-bookworm
FROM ${BASE_IMAGE} AS tools

RUN --mount=type=secret,id=pip_config,target=/etc/pip.conf,required=true \
    unset PIP_INDEX_URL PIP_EXTRA_INDEX_URL && \
    PIP_CONFIG_FILE=/etc/pip.conf python3 -m pip install --no-cache-dir --use-deprecated=legacy-resolver \
      cmake ninja ziglang==0.14.1 \
      nvidia-cuda-nvcc==13.4.92 nvidia-cuda-runtime==13.4.92 \
      nvidia-cublas==13.8.2.12 nvidia-cuda-cccl==13.3.3.3.1 \
      nvidia-nvvm==13.4.92 nvidia-cuda-crt==13.4.92 nvidia-cuda-nvrtc==13.4.92

RUN CUDA_ROOT="$(dirname "$(dirname "$(find /usr/local/lib -path '*/nvidia/*/bin/nvcc' -print -quit)")")" && \
    test -f "$CUDA_ROOT/bin/nvcc" && \
    ln -s "$CUDA_ROOT" /opt/cuda && \
    ln -sf libcudart.so.13 "$CUDA_ROOT/lib/libcudart.so" && \
    ln -sf libcublas.so.13 "$CUDA_ROOT/lib/libcublas.so" && \
    ln -sf libcublasLt.so.13 "$CUDA_ROOT/lib/libcublasLt.so"

FROM tools AS runner
ARG CUDA_ARCHITECTURES=121a-real
ARG BUILD_JOBS=4
COPY . /src
RUN --mount=type=secret,id=cuda_driver,target=/opt/cuda-driver/libcuda.so,required=true \
    --mount=type=cache,id=diffusion-context-cuda-build,target=/cuda-build \
    CUDA_ROOT="$(readlink -f /opt/cuda)" && \
    ln -sf libcuda.so /opt/cuda-driver/libcuda.so.1 && \
    if [ -f /src/.git ]; then rm /src/.git; fi && \
    cmake -S /src -B /cuda-build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      "-DCMAKE_EXE_LINKER_FLAGS=-Wl,-rpath-link,$CUDA_ROOT/lib -Wl,-rpath-link,/opt/cuda-driver" \
      -DCUDAToolkit_ROOT="$CUDA_ROOT" \
      -DCMAKE_CUDA_COMPILER="$CUDA_ROOT/bin/nvcc" \
      -DCUDA_cuda_driver_LIBRARY=/opt/cuda-driver/libcuda.so \
      -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHITECTURES" \
      -DGGML_CUDA=ON -DGGML_CUDA_NCCL=OFF \
      -DGGML_NATIVE=OFF -DGGML_OPENMP=ON -DGGML_CCACHE=OFF \
      -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_UI=OFF \
      -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_APP=OFF \
      -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF \
      -DLLAMA_BUILD_EXAMPLES=ON && \
    cmake --build /cuda-build --target \
      llama-diffusion-gemma-visual-server test-diffusion-context-sizing test-llama-archs \
      --parallel "$BUILD_JOBS" && \
    mkdir /build && cp -a /cuda-build/bin /build/bin

ENV LD_LIBRARY_PATH=/build/bin:/opt/cuda/lib:/usr/lib/wsl/lib
ENTRYPOINT ["/build/bin/llama-diffusion-gemma-visual-server"]
