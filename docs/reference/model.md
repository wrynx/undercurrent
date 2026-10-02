# `undercurrent.model`

The front door. [`ProbedModel`][undercurrent.model.ProbedModel] loads a model,
attaches the probes your spec asks for, and returns
[`GenerationOutput`][undercurrent.model.GenerationOutput]s with each
extraction point's result. `ProbedModel` and `GenerationOutput` are also
exported from `undercurrent`.

```python
from undercurrent import ProbedModel
```

## Loading and generating

::: undercurrent.model.ProbedModel
    options:
      heading_level: 3
      show_root_heading: true
      merge_init_into_class: true
      show_root_full_path: false

::: undercurrent.model.GenerationOutput
    options:
      heading_level: 3
      show_root_heading: true
      merge_init_into_class: true
      show_root_full_path: false

## Errors

::: undercurrent.model.ProbedModelConfigError
    options:
      heading_level: 3
      show_root_heading: true
      merge_init_into_class: true
      show_root_full_path: false

## Defaults

::: undercurrent.model.DEFAULT_MAX_NEW_TOKENS
    options:
      heading_level: 3
      show_root_heading: true
      merge_init_into_class: true
      show_root_full_path: false

::: undercurrent.model.DEFAULT_VLLM_MAX_CONCURRENCY
    options:
      heading_level: 3
      show_root_heading: true
      merge_init_into_class: true
      show_root_full_path: false
