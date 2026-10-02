# Security Policy

Undercurrent is Wrynx's activation-probing platform. It runs inside model-serving processes and
handles prompt-derived data, so we take security reports seriously. Thank you for helping keep
Undercurrent and its users safe.

## Supported versions

Undercurrent is pre-1.0. Security fixes go into the **latest minor release** only (for example,
if `0.3.x` is current, fixes ship as a new `0.3.x` patch and not as backports to `0.2.x`). Before
1.0, a fix may need a breaking change. When that happens, the release notes say so.

| Version               | Supported          |
| --------------------- | ------------------ |
| Latest minor release  | :white_check_mark: |
| Older releases        | :x:                |

## Reporting a vulnerability

**Please do not report security vulnerabilities through public GitHub issues, discussions or pull
requests.**

Report them privately through either channel:

- **Email:** [security@wrynx.com](mailto:security@wrynx.com)
- **GitHub private vulnerability reporting:** open the
  [Security tab of wrynx/undercurrent](https://github.com/wrynx/undercurrent/security) and click
  **"Report a vulnerability"**.

A useful report includes:

- A description of the issue and its impact: what an attacker can do, and under which configuration.
- The affected Undercurrent version (`pip show undercurrent`), plus the Python, PyTorch,
  Transformers and (if relevant) vLLM versions.
- Steps to reproduce: a minimal script, spec file or checkpoint, or a proof of concept.
- Any known mitigations or workarounds.
- Whether and how you would like to be credited.

If you believe the issue is in an upstream project (PyTorch, Transformers, vLLM, PyYAML, …),
please report it to that project as well. We are still glad to hear how it affects Undercurrent.

## Response timeline

- **Acknowledgement:** within **3 business days** of your report.
- **Assessment:** we will confirm whether we consider the report a vulnerability, and keep you
  updated as we work on it.
- **Fix or disclosure:** we aim to release a fix, or publicly disclose the issue with mitigations,
  within **90 days** of the report. For complex issues, we may agree on a different timeline with
  you.

## Scope and security considerations

These are the parts of Undercurrent where security matters most, with guidance for anyone
deploying it. Vulnerabilities in these areas are in scope. So is anything else in the library.

### Loading probe checkpoints

Some probes (such as the MLP safety classifier in the content-safety example) load their weights
from a checkpoint file you supply. Probe weights are loaded weights-only (`torch.load(...,
weights_only=True)`), which refuses arbitrary pickled objects. Even so, **only load checkpoints you
trust**, from sources you control. A malicious or corrupted checkpoint can still produce wrong
probe verdicts. Undercurrent v0.1 does not download probes from the Hugging Face Hub or any other
remote source.

### Webhook sink

`WebhookLogSink` (`src/undercurrent/sinks/webhook_sink.py`) POSTs probe results and
signals as JSON to a URL you configure. It sends the request from a background thread and retries
failed requests. Records that still fail after the retries are written to a local dead-letter file
(NDJSON) if you configured one. These records can contain data derived from user prompts and model
outputs, depending on what your probes put in their verdicts and metadata.

- Use an **HTTPS** endpoint you control. The sink sends no authentication headers of its own and
  does not restrict the URL scheme.
- Keep sensitive content out of probe verdicts and metadata, or redact it, before it leaves the
  process.
- Treat the dead-letter file like any other log that holds user data: restrict its filesystem
  permissions and retention.

### vLLM plugin (auto-loaded)

The vLLM adapter registers `undercurrent.adapters.vllm.plugin:register` under vLLM's
`vllm.general_plugins` entry-point group (see the root `pyproject.toml`). vLLM loads it
**automatically in every vLLM process** (API server, engine core, and each worker) of any
environment where the package is installed, even if you never import Undercurrent yourself. The
hook only logs a message and sets two environment variables if they aren't already set
(`VLLM_USE_V2_MODEL_RUNNER=0` and `VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1`). It does not enable
activation capture on its own; that still needs an explicit `--worker-extension-cls` flag.

- Install Undercurrent only in the vLLM environments or images where you intend to use it.
- Treat it like any dependency that runs code inside your serving process: pin its version and
  review upgrades.

### YAML specs

Extraction-point specs are parsed by `undercurrent.spec` (`src/undercurrent/spec/parser.py`) with
PyYAML's `yaml.safe_load`. Pydantic then validates them, so a spec file cannot construct arbitrary
Python objects. A spec still controls which layers get hooked and which probes run. **Only load
specs from trusted sources**, and treat them as configuration, not user input.

### Example HTTP server

The reference HTTP server shipped as example code (`examples/openai_server/serve_llama_mlp_pipeline.py`) is a **demo only**. It has **no authentication, authorization, TLS or
rate limiting**, and by default it binds to `0.0.0.0`. Do not expose it to the public internet or
to untrusted networks. Undercurrent v0.1 does not ship a production server.

## Disclosure policy

We follow **coordinated disclosure**:

1. You report the issue privately. We acknowledge it and investigate.
2. We develop and test a fix, and agree on a disclosure date with you.
3. We release the fix and publish a security advisory (a GitHub Security Advisory, and a CVE where
   appropriate).
4. We credit reporters in the release notes and advisory, unless they ask to stay anonymous.

Please give us a reasonable chance to fix the issue before you disclose it publicly, and do not
access or change other people's data while researching.
