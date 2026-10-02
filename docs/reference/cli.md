# CLI

Installing Undercurrent adds an `undercurrent` command (also runnable as
`python -m undercurrent`). It has three subcommands:

| Command | What it does |
|---------|--------------|
| [`undercurrent inspect-model`](#undercurrent-inspect-model) | List a model's hookable layers and tensor types |
| [`undercurrent validate`](#undercurrent-validate) | Check that probe-spec files are valid |
| [`undercurrent schema`](#undercurrent-schema) | Print the probe-spec JSON Schema |

!!! note
    This page mirrors the output of `undercurrent --help` and
    `undercurrent COMMAND --help`. If they ever disagree, `--help` is right.

The command line (its commands, options and exit codes) is part of the
public API; see [API stability](../api-stability.md).

## Global options

```text
undercurrent [-h] [--version] [--debug] COMMAND ...
```

| Option | Description |
|--------|-------------|
| `-h`, `--help` | Show the help message and exit. |
| `--version` | Show the version (`undercurrent X.Y.Z`) and exit. |
| `--debug` | Show full tracebacks for errors. May also be given after the command. |

Without `--debug`, an error caused by the input (an invalid spec, a model
that can't be inspected, a missing file) is printed as a one-line
`error: ...` message, without a traceback.

## Exit codes

| Code | Meaning |
|------|---------|
| `0` | Success. |
| `1` | A failure the command reports: an invalid spec, a model that can't be inspected. |
| `2` | A usage error: unknown command or option, a bad argument, or no command given. |
| `130` | Interrupted with Ctrl-C. |

## `undercurrent inspect-model`

```text
undercurrent inspect-model [-h] [--backend {hf,vllm,all}] [--format {text,json}]
                           [--emit-spec FILE] [--trust-remote-code] [--revision REV]
                           MODEL
```

List a model's decoder layers and which tensor types each backend can
capture. Only `config.json` is fetched; the model is built on the meta
device, so no weights are downloaded or allocated.

| Argument / option | Description |
|-------------------|-------------|
| `MODEL` | Hugging Face model id or local model directory. |
| `--backend {hf,vllm,all}` | Which adapter(s) to report on (default: `all`). |
| `--format {text,json}` | Output format (default: `text`). |
| `--emit-spec FILE` | Also write a commented starter spec for this model to `FILE`. |
| `--trust-remote-code` | Allow custom model code from the Hub (only for repos you trust). |
| `--revision REV` | Model revision (branch, tag or commit) to read. |

```bash
undercurrent inspect-model openai-community/gpt2
undercurrent inspect-model openai-community/gpt2 --backend hf --emit-spec probes.yaml
```

## `undercurrent validate`

```text
undercurrent validate [-h] [--strict] [--check-probes] [--import MODULE]
                      [--format {text,json}]
                      SPEC [SPEC ...]
```

Check that probe-spec YAML files are valid. Exits `0` if all are, `1` if
any isn't, `2` on a usage error, so it works as a CI check.

| Argument / option | Description |
|-------------------|-------------|
| `SPEC` | Spec file(s) to check. |
| `--strict` | Treat deprecation warnings (e.g. old key names) as errors. |
| `--check-probes` | Also check that every `probe_type` is registered: built-in, installed plugins in the `undercurrent.probes` entry-point group, and modules given with `--import`. |
| `--import MODULE` | Import `MODULE` first, so the probes it registers count for `--check-probes`. Repeatable. |
| `--format {text,json}` | Output format (default: `text`). |

```bash
undercurrent validate probes.yaml
undercurrent validate --check-probes --import my_package.probes specs/*.yaml
```

## `undercurrent schema`

```text
undercurrent schema [-h] [-o FILE]
```

Print the JSON Schema (Draft 2020-12) for probe-spec YAML files, for editor
autocompletion and for validating specs with other tools. The same schema
is available in Python as
[`json_schema()`][undercurrent.spec.json_schema.json_schema].

| Option | Description |
|--------|-------------|
| `-o FILE`, `--output FILE` | Write the schema to `FILE` instead of stdout. |

```bash
undercurrent schema -o probe-spec.schema.json
```

## Python entry point

::: undercurrent.cli.main
    options:
      heading_level: 3
      show_root_heading: true
      merge_init_into_class: true
      show_root_full_path: false
