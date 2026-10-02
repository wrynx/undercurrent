import pytest

from undercurrent.spec import parse_dict, parse_yaml, probe_spec_to_dict, to_yaml

SPECS = [
    {
        "version": "1",
        "extraction_points": [
            {
                "name": "classic_probe",
                "layers": 12,
                "tensor_type": "residual_stream",
                "position": "prompt[-1]",
                "probe_type": "linear_probe",
                "probe_kind": "single_shot",
                "execution_mode": "inline",
            }
        ],
    },
    {
        "version": "1",
        "extraction_points": [
            {
                "name": "trajectory_probe",
                "layers": [4, 8, 12],
                "tensor_type": "mlp_out",
                "position": "generated[*]",
                "stride": 2,
                "until": "generation_end",
                "probe_type": "trajectory_probe",
                "probe_kind": "trajectory",
                "execution_mode": "async",
                "queue_depth": 8,
            }
        ],
    },
    {
        "version": "1",
        "extraction_points": [
            {
                "name": "offset_dual_position",
                "layers": 6,
                "tensor_type": "attn_out",
                "position": "prompt[-1]+1",
                "probe_type": "dual_position_probe",
                "probe_kind": "single_shot",
                "execution_mode": "inline",
            }
        ],
    },
    {
        "version": "1",
        "extraction_points": [
            {
                "name": "sliced_range",
                "layers": 20,
                "tensor_type": "kv",
                "position": "generated[5:20]",
                "probe_type": "trajectory_probe",
                "probe_kind": "trajectory",
                "execution_mode": "inline",
            }
        ],
    },
]


@pytest.mark.parametrize("original", SPECS)
def test_dict_round_trip(original):
    spec = parse_dict(original)
    round_tripped_dict = probe_spec_to_dict(spec)
    respec = parse_dict(round_tripped_dict)

    assert spec == respec


@pytest.mark.parametrize("original", SPECS)
def test_yaml_round_trip(original):
    spec = parse_dict(original)
    yaml_text = to_yaml(spec)
    respec = parse_yaml(yaml_text)

    assert spec == respec


def test_round_trip_preserves_all_fields():
    original = SPECS[1]
    spec = parse_dict(original)
    round_tripped_dict = probe_spec_to_dict(spec)
    point = round_tripped_dict["extraction_points"][0]

    assert point["name"] == "trajectory_probe"
    assert point["layers"] == [4, 8, 12]
    assert point["tensor_type"] == "mlp_out"
    assert point["position"] == "generated[*]"
    assert point["stride"] == 2
    assert point["until"] == "generation_end"
    assert point["probe_kind"] == "trajectory"
    assert point["execution_mode"] == "async"
    assert point["queue_depth"] == 8
