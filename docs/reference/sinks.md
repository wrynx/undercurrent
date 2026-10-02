# `undercurrent.sinks`

!!! info "Advanced API"
    For a simple results hook, pass `on_result=` to
    [`ProbedModel`][undercurrent.model.ProbedModel]. Sinks are for shipping
    observations out of process from a [`Router`][undercurrent.router.Router].

Observation sinks receive the signals and final results of async
(observe-mode) extraction points. The sinks and redaction helpers are also
exported from `undercurrent`.

## Sinks

::: undercurrent.sinks.LogSink
    options:
      heading_level: 3
      show_root_heading: true
      merge_init_into_class: true
      show_root_full_path: false

::: undercurrent.sinks.FileLogSink
    options:
      heading_level: 3
      show_root_heading: true
      merge_init_into_class: true
      show_root_full_path: false

::: undercurrent.sinks.WebhookLogSink
    options:
      heading_level: 3
      show_root_heading: true
      merge_init_into_class: true
      show_root_full_path: false

::: undercurrent.sinks.wire_router
    options:
      heading_level: 3
      show_root_heading: true
      merge_init_into_class: true
      show_root_full_path: false

## Redaction

::: undercurrent.sinks.redact_keys
    options:
      heading_level: 3
      show_root_heading: true
      merge_init_into_class: true
      show_root_full_path: false

::: undercurrent.sinks.drop_keys
    options:
      heading_level: 3
      show_root_heading: true
      merge_init_into_class: true
      show_root_full_path: false

::: undercurrent.sinks.chain
    options:
      heading_level: 3
      show_root_heading: true
      merge_init_into_class: true
      show_root_full_path: false

::: undercurrent.sinks.RedactFn
    options:
      heading_level: 3
      show_root_heading: true
      merge_init_into_class: true
      show_root_full_path: false

::: undercurrent.sinks.DEFAULT_PROMPT_TEXT_KEYS
    options:
      heading_level: 3
      show_root_heading: true
      merge_init_into_class: true
      show_root_full_path: false

## Writing a custom sink

::: undercurrent.sinks.to_jsonable
    options:
      heading_level: 3
      show_root_heading: true
      merge_init_into_class: true
      show_root_full_path: false
