# Demo container for serve_llama_mlp_pipeline.py: CustomMLPProbe attached to
# a live Llama-3.1-8B model via vLLM, served over a bare HTTP /generate
# endpoint. See that script (examples/openai_server/serve_llama_mlp_pipeline.py)
# for the request/response shape and the non-containerized dev-install path
# this Dockerfile mirrors.
#
# Build (from the REPO ROOT, not this directory -- the build context needs
# src/ and examples/, which a build context rooted here wouldn't see):
#
#   docker build -f examples/content_safety/Dockerfile -t wrynx-content-safety-demo .
#
# Run (needs nvidia-container-toolkit on the host, an HF token with access
# to meta-llama/Llama-3.1-8B -- a GATED repo: accept its license on the
# Hugging Face model page, then pass a token as HF_TOKEN, the in-container
# equivalent of `huggingface-cli login` -- and a `CustomMLPProbe`
# checkpoint -- see examples/content_safety/content_safety_demo/custom_mlp.py's
# docstring for the expected format, and note that --activation/
# --final-activation MUST match how that checkpoint was actually trained;
# getting them wrong loads without error but silently produces wrong
# scores):
#
#   docker run --gpus all -p 8000:8000 \
#     -e HF_TOKEN=<your token> \
#     -v /path/to/checkpoint.pt:/model/checkpoint.pt:ro \
#     -e PROBE_MODEL_PATH=/model/checkpoint.pt \
#     wrynx-content-safety-demo --activation gelu --final-activation
#
# The image pins the exact vllm/torch/CUDA stack this pipeline was built and
# tested against (vllm==0.28.0) by starting from vLLM's own published image
# rather than a bare CUDA base -- reproducing that CUDA/NCCL/driver-matched
# toolchain by hand here would be far more fragile than reusing vLLM's own.
# If your target GPU/driver needs a different vllm version, change this tag
# to match (and re-verify against this repo's own test suite -- vLLM's
# internal APIs this platform hooks into, e.g. worker_extension_cls and the
# decoder-layer module layout the vLLM adapter walks, are not a stable
# public contract across versions).
# Confirm `vllm/vllm-openai:v0.28.0` resolves on Docker Hub (or swap in
# whatever tag matches your installed vllm==0.28.0) before relying on this
# build.
FROM vllm/vllm-openai:v0.28.0

WORKDIR /app

# The root undercurrent project (src/) plus examples/. The demo probes and
# the server script are not installed: the script puts
# examples/content_safety/ on sys.path itself.
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY src ./src
COPY examples ./examples

# --no-deps: the base image already carries a GPU-driver-matched build of
# vllm/torch and everything else undercurrent needs -- but letting pip
# resolve them anyway still recomputes torch's OWN transitive pins.
# Confirmed by an earlier build of this exact Dockerfile: pip downgraded the
# base image's nvidia-nccl-cu13 from 2.30.7 to 2.29.7 (a fresh 206MB
# download) to match torch's wheel metadata's exact pin -- silently
# replacing a GPU/driver-matched CUDA library with a different one pip has
# no way to verify still works on this hardware. --no-deps sidesteps that
# resolution entirely; if the project ever grows a dependency NOT already in
# the base image, the server will fail loudly (ModuleNotFoundError at
# import, not a silent gap) and it needs installing explicitly here.
RUN pip install --no-cache-dir --no-deps .

EXPOSE 8000

# No PROBE_MODEL_PATH (or HF_TOKEN) default baked in: the checkpoint is
# private, per-user trained weights, and the HF token is a credential --
# neither belongs in an image layer. Both are supplied at `docker run` time
# (see the usage comment above); the script itself already fails loudly
# with a clear message if PROBE_MODEL_PATH is unset or doesn't exist.
# --host 0.0.0.0: the server defaults to 127.0.0.1, which `docker run -p`
# can't reach from outside the container.
ENTRYPOINT ["python3", "examples/openai_server/serve_llama_mlp_pipeline.py", "--host", "0.0.0.0"]
