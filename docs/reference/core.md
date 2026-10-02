# `undercurrent.core`

How a probe plugs in: the probe lifecycle, function probes, the probe
registry, and the signal and result types probes produce. The front-door
names (`probe`, `register_probe`, `Probe`) and the data types are also
exported from `undercurrent`.

## Writing probes

Decorate a plain function with `@probe` for a stateless single-shot probe, or subclass [`Probe`][undercurrent.core.Probe] for stateful and trajectory probes. `@register_probe` makes a class findable by the `probe_type` name used in specs.

::: undercurrent.core.function_probe.probe
    options:
      heading_level: 3
      show_root_heading: true
      merge_init_into_class: true
      show_root_full_path: false

::: undercurrent.core.register_probe
    options:
      heading_level: 3
      show_root_heading: true
      merge_init_into_class: true
      show_root_full_path: false

::: undercurrent.core.FunctionProbe
    options:
      heading_level: 3
      show_root_heading: true
      merge_init_into_class: true
      show_root_full_path: false

## Probe lifecycle

::: undercurrent.core.Probe
    options:
      heading_level: 3
      show_root_heading: true
      merge_init_into_class: true
      show_root_full_path: false

::: undercurrent.core.ProbeFactory
    options:
      heading_level: 3
      show_root_heading: true
      merge_init_into_class: true
      show_root_full_path: false

::: undercurrent.core.RequestContext
    options:
      heading_level: 3
      show_root_heading: true
      merge_init_into_class: true
      show_root_full_path: false

## Signals & results

::: undercurrent.core.ProbeSignal
    options:
      heading_level: 3
      show_root_heading: true
      merge_init_into_class: true
      show_root_full_path: false

::: undercurrent.core.ProbeAction
    options:
      heading_level: 3
      show_root_heading: true
      merge_init_into_class: true
      show_root_full_path: false
      show_if_no_docstring: true

::: undercurrent.core.ProbeResult
    options:
      heading_level: 3
      show_root_heading: true
      merge_init_into_class: true
      show_root_full_path: false

## Registry

::: undercurrent.core.get_probe_factory
    options:
      heading_level: 3
      show_root_heading: true
      merge_init_into_class: true
      show_root_full_path: false

::: undercurrent.core.list_probes
    options:
      heading_level: 3
      show_root_heading: true
      merge_init_into_class: true
      show_root_full_path: false

::: undercurrent.core.unregister_probe
    options:
      heading_level: 3
      show_root_heading: true
      merge_init_into_class: true
      show_root_full_path: false

::: undercurrent.core.ProbeRegistry
    options:
      heading_level: 3
      show_root_heading: true
      merge_init_into_class: true
      show_root_full_path: false

::: undercurrent.core.default_registry
    options:
      heading_level: 3
      show_root_heading: true
      merge_init_into_class: true
      show_root_full_path: false

::: undercurrent.core.ProbeNotFoundError
    options:
      heading_level: 3
      show_root_heading: true
      merge_init_into_class: true
      show_root_full_path: false

## Example probes

`undercurrent.core.examples` holds two reference probes with deterministic stub scoring. They show the two probe shapes and are used in tests and demos; they are not production probes.

::: undercurrent.core.examples.MLPClassifierProbe
    options:
      heading_level: 3
      show_root_heading: true
      merge_init_into_class: true
      show_root_full_path: false

::: undercurrent.core.examples.TrajectoryScoreProbe
    options:
      heading_level: 3
      show_root_heading: true
      merge_init_into_class: true
      show_root_full_path: false
