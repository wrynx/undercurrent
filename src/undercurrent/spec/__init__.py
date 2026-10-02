"""undercurrent.spec: the extraction-point contract layer for the activation-probing platform.

This package defines *what* to capture during LLM inference and *how* that
gets expressed as data -- nothing more. It has no dependency on any specific
inference engine, and no adapter/router/probe execution logic lives here.
Those packages depend on this one, not the other way around.

Typical usage::

    from undercurrent.spec import load_yaml_file

    spec = load_yaml_file("probes.yaml")
    for point in spec:
        print(point.name, point.tensor_type, point.position)

Public surface:
    - Parsing: `load_spec()` (any of the forms below), `load_yaml_file()`,
      `parse_yaml()`, `parse_dict()`
    - Resolved types: `ProbeSpec`, `ExtractionPoint`,
      `InterventionPolicy`, `FrozenArgs`, `PositionSelector`
      (built by `parse_position()`)
    - Runtime data type: `ActivationRecord`
    - Serialization: `to_yaml()`, `probe_spec_to_dict()`,
      `extraction_point_to_dict()`
    - Enums: `TensorType`, `UntilKind`, `ProbeKind`,
      `ExecutionMode`, `InterventionMode`, `TimeoutAction`,
      `PositionKind`
    - JSON Schema for spec files: `json_schema()`
    - Errors: `ProbingSpecError`, `SpecValidationError`,
      `PositionSyntaxError`

The pydantic models in `undercurrent.spec.schema` (``ProbeSpecFile``,
``ExtractionPointSpec``, ``InterventionPolicySpec``) describe the YAML file
format and are internal; use `json_schema()` for the file format and the
resolved types above in Python.
"""

from .activation_record import ActivationRecord
from .errors import PositionSyntaxError, ProbingSpecError, SpecValidationError
from .json_schema import json_schema
from .parser import load_spec, load_yaml_file, parse_dict, parse_yaml
from .position import PositionKind, PositionSelector, parse_position
from .resolved import ExtractionPoint, FrozenArgs, InterventionPolicy, ProbeSpec
from .schema import (
    ExecutionMode,
    InterventionMode,
    ProbeKind,
    TensorType,
    TimeoutAction,
    UntilKind,
)
from .serialize import extraction_point_to_dict, probe_spec_to_dict, to_yaml

__all__ = [
    "ActivationRecord",
    "ExecutionMode",
    "ExtractionPoint",
    "FrozenArgs",
    "InterventionMode",
    "InterventionPolicy",
    "PositionKind",
    "PositionSelector",
    "PositionSyntaxError",
    "ProbeKind",
    "ProbeSpec",
    "ProbingSpecError",
    "SpecValidationError",
    "TensorType",
    "TimeoutAction",
    "UntilKind",
    "extraction_point_to_dict",
    "json_schema",
    "load_spec",
    "load_yaml_file",
    "parse_dict",
    "parse_position",
    "parse_yaml",
    "probe_spec_to_dict",
    "to_yaml",
]
