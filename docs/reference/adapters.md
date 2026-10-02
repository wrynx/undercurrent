# `undercurrent.adapters`

!!! info "Advanced API"
    [`ProbedModel`][undercurrent.model.ProbedModel] picks an adapter for
    you with `backend="hf"` or `backend="vllm"`. Use an adapter directly when
    you drive generation yourself, or implement
    [`EngineAdapter`][undercurrent.adapters.EngineAdapter] to support another
    engine.

Importing `undercurrent.adapters` never loads torch, transformers or vLLM.
`EngineAdapter` and `MissingDependencyError` are also exported from
`undercurrent`; the concrete adapters are imported from their subpackages.

## The adapter interface

::: undercurrent.adapters.EngineAdapter
    options:
      heading_level: 3
      show_root_heading: true
      merge_init_into_class: true
      show_root_full_path: false

::: undercurrent.adapters.MissingDependencyError
    options:
      heading_level: 3
      show_root_heading: true
      merge_init_into_class: true
      show_root_full_path: false

## Hugging Face transformers

```python
from undercurrent.adapters.hf import HFEngineAdapter
```

::: undercurrent.adapters.hf.HFEngineAdapter
    options:
      heading_level: 3
      show_root_heading: true
      merge_init_into_class: true
      show_root_full_path: false

::: undercurrent.adapters.hf.HFAdapterLimitationError
    options:
      heading_level: 3
      show_root_heading: true
      merge_init_into_class: true
      show_root_full_path: false

## vLLM

```python
from undercurrent.adapters.vllm import VLLMEngineAdapter
```

`VLLMEngineAdapter` is public, but the modules behind it read undocumented vLLM internals and are [experimental](../api-stability.md#experimental-parts). vLLM is never installed by default; see [Deploy with vLLM](../guides/vllm-deployment.md).

::: undercurrent.adapters.vllm.VLLMEngineAdapter
    options:
      heading_level: 3
      show_root_heading: true
      merge_init_into_class: true
      show_root_full_path: false

::: undercurrent.adapters.vllm.VLLMAdapterLimitationError
    options:
      heading_level: 3
      show_root_heading: true
      merge_init_into_class: true
      show_root_full_path: false
