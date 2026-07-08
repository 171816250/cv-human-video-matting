from cv_human_video_matting.utils.config import (
    config_bool,
    config_float,
    config_int,
    config_value,
)


def test_config_value_reads_nested_path():
    config = {"matting": {"model": "dummy"}}

    assert config_value(config, "matting.model") == "dummy"


def test_config_value_returns_default_for_missing_path():
    config = {"matting": {"model": "dummy"}}

    assert config_value(config, "depth.enabled", False) is False


def test_config_int_falls_back_on_non_numeric_string():
    config = {"postprocess": {"edge_feather": "auto"}}

    # A non-numeric string must not raise; it returns the default.
    assert config_int(config, "postprocess.edge_feather", 3) == 3


def test_config_int_coerces_numeric_string():
    config = {"postprocess": {"edge_feather": "5"}}

    assert config_int(config, "postprocess.edge_feather", 0) == 5


def test_config_float_falls_back_on_non_numeric_string():
    config = {"postprocess": {"temporal_alpha": "none"}}

    assert config_float(config, "postprocess.temporal_alpha", 0.75) == 0.75


def test_config_bool_treats_quoted_false_as_false():
    config = {"postprocess": {"enable_temporal_smooth": "false"}}

    # bool("false") would be True; config_bool must interpret the string.
    assert config_bool(config, "postprocess.enable_temporal_smooth", True) is False


def test_config_bool_accepts_real_yaml_boolean():
    config = {"depth": {"enabled": True}}

    assert config_bool(config, "depth.enabled", False) is True


def test_config_bool_falls_back_on_unrecognized_value():
    config = {"depth": {"enabled": "maybe"}}

    assert config_bool(config, "depth.enabled", False) is False
