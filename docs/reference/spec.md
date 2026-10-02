# `undercurrent.spec`

What to capture during generation, expressed as data: extraction points,
position selectors and the YAML spec format. The most common names
(`ExtractionPoint`, `load_spec`, `ProbeSpec`, the enums) are also exported
from `undercurrent`.

## Loading specs

::: undercurrent.spec.load_spec
    options:
      heading_level: 3
      show_root_heading: true
      merge_init_into_class: true
      show_root_full_path: false

::: undercurrent.spec.load_yaml_file
    options:
      heading_level: 3
      show_root_heading: true
      merge_init_into_class: true
      show_root_full_path: false

::: undercurrent.spec.parse_yaml
    options:
      heading_level: 3
      show_root_heading: true
      merge_init_into_class: true
      show_root_full_path: false

::: undercurrent.spec.parse_dict
    options:
      heading_level: 3
      show_root_heading: true
      merge_init_into_class: true
      show_root_full_path: false

## Spec types

::: undercurrent.spec.ProbeSpec
    options:
      heading_level: 3
      show_root_heading: true
      merge_init_into_class: true
      show_root_full_path: false

::: undercurrent.spec.ExtractionPoint
    options:
      heading_level: 3
      show_root_heading: true
      merge_init_into_class: true
      show_root_full_path: false

::: undercurrent.spec.InterventionPolicy
    options:
      heading_level: 3
      show_root_heading: true
      merge_init_into_class: true
      show_root_full_path: false

::: undercurrent.spec.FrozenArgs
    options:
      heading_level: 3
      show_root_heading: true
      merge_init_into_class: true
      show_root_full_path: false

## Positions

::: undercurrent.spec.parse_position
    options:
      heading_level: 3
      show_root_heading: true
      merge_init_into_class: true
      show_root_full_path: false

::: undercurrent.spec.PositionSelector
    options:
      heading_level: 3
      show_root_heading: true
      merge_init_into_class: true
      show_root_full_path: false

::: undercurrent.spec.PositionKind
    options:
      heading_level: 3
      show_root_heading: true
      merge_init_into_class: true
      show_root_full_path: false
      show_if_no_docstring: true

## Enums

::: undercurrent.spec.TensorType
    options:
      heading_level: 3
      show_root_heading: true
      merge_init_into_class: true
      show_root_full_path: false
      show_if_no_docstring: true

::: undercurrent.spec.ProbeKind
    options:
      heading_level: 3
      show_root_heading: true
      merge_init_into_class: true
      show_root_full_path: false
      show_if_no_docstring: true

::: undercurrent.spec.ExecutionMode
    options:
      heading_level: 3
      show_root_heading: true
      merge_init_into_class: true
      show_root_full_path: false
      show_if_no_docstring: true

::: undercurrent.spec.InterventionMode
    options:
      heading_level: 3
      show_root_heading: true
      merge_init_into_class: true
      show_root_full_path: false
      show_if_no_docstring: true

::: undercurrent.spec.TimeoutAction
    options:
      heading_level: 3
      show_root_heading: true
      merge_init_into_class: true
      show_root_full_path: false
      show_if_no_docstring: true

::: undercurrent.spec.UntilKind
    options:
      heading_level: 3
      show_root_heading: true
      merge_init_into_class: true
      show_root_full_path: false
      show_if_no_docstring: true

## Activation records

::: undercurrent.spec.ActivationRecord
    options:
      heading_level: 3
      show_root_heading: true
      merge_init_into_class: true
      show_root_full_path: false

## Serialization and JSON Schema

::: undercurrent.spec.to_yaml
    options:
      heading_level: 3
      show_root_heading: true
      merge_init_into_class: true
      show_root_full_path: false

::: undercurrent.spec.probe_spec_to_dict
    options:
      heading_level: 3
      show_root_heading: true
      merge_init_into_class: true
      show_root_full_path: false

::: undercurrent.spec.extraction_point_to_dict
    options:
      heading_level: 3
      show_root_heading: true
      merge_init_into_class: true
      show_root_full_path: false

::: undercurrent.spec.json_schema.json_schema
    options:
      heading_level: 3
      show_root_heading: true
      merge_init_into_class: true
      show_root_full_path: false

## Errors

::: undercurrent.spec.ProbingSpecError
    options:
      heading_level: 3
      show_root_heading: true
      merge_init_into_class: true
      show_root_full_path: false

::: undercurrent.spec.SpecValidationError
    options:
      heading_level: 3
      show_root_heading: true
      merge_init_into_class: true
      show_root_full_path: false

::: undercurrent.spec.PositionSyntaxError
    options:
      heading_level: 3
      show_root_heading: true
      merge_init_into_class: true
      show_root_full_path: false
