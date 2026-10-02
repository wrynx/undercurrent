# `undercurrent.router`

!!! info "Advanced API"
    Most users want [`ProbedModel`][undercurrent.model.ProbedModel], which
    drives a router for you. Use the router directly when you embed
    Undercurrent in your own serving stack.

The router dispatches activation records to per-request probe instances,
runs inline probes on the generation thread and async probes on bounded
worker pools, and records metrics. `Router`, `RequestHandle`, `RouterError`,
`OverflowPolicy` and the metrics types are also exported from `undercurrent`.

## Router

::: undercurrent.router.Router
    options:
      heading_level: 3
      show_root_heading: true
      merge_init_into_class: true
      show_root_full_path: false

::: undercurrent.router.RequestHandle
    options:
      heading_level: 3
      show_root_heading: true
      merge_init_into_class: true
      show_root_full_path: false

::: undercurrent.router.RequestEndListener
    options:
      heading_level: 3
      show_root_heading: true
      merge_init_into_class: true
      show_root_full_path: false

::: undercurrent.router.RouterError
    options:
      heading_level: 3
      show_root_heading: true
      merge_init_into_class: true
      show_root_full_path: false

## Async execution

::: undercurrent.router.OverflowPolicy
    options:
      heading_level: 3
      show_root_heading: true
      merge_init_into_class: true
      show_root_full_path: false
      show_if_no_docstring: true

::: undercurrent.router.default_worker_pool_size
    options:
      heading_level: 3
      show_root_heading: true
      merge_init_into_class: true
      show_root_full_path: false

::: undercurrent.router.DEFAULT_QUEUE_DEPTH
    options:
      heading_level: 3
      show_root_heading: true
      merge_init_into_class: true
      show_root_full_path: false

::: undercurrent.router.DEFAULT_DRAIN_TIMEOUT
    options:
      heading_level: 3
      show_root_heading: true
      merge_init_into_class: true
      show_root_full_path: false

::: undercurrent.router.DEFAULT_CIRCUIT_BREAKER_THRESHOLD
    options:
      heading_level: 3
      show_root_heading: true
      merge_init_into_class: true
      show_root_full_path: false

## Metrics

::: undercurrent.router.MetricsSink
    options:
      heading_level: 3
      show_root_heading: true
      merge_init_into_class: true
      show_root_full_path: false

::: undercurrent.router.InMemoryMetricsRegistry
    options:
      heading_level: 3
      show_root_heading: true
      merge_init_into_class: true
      show_root_full_path: false

::: undercurrent.router.MetricsSnapshot
    options:
      heading_level: 3
      show_root_heading: true
      merge_init_into_class: true
      show_root_full_path: false
