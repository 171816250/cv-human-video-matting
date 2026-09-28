"""UI offline mode cannot silently initiate another model download."""

import numpy as np
import pytest

from cv_human_video_matting.ui import gradio_app as ui


@pytest.fixture
def offline_config(monkeypatch):
    config = {"matting": {"model": "rvm_mobilenetv3", "bundle": "verified-mobile-bundle"}}
    monkeypatch.setattr(ui, "_ACTIVE_CONFIG", config)
    monkeypatch.setattr(ui, "create_matting_runner", lambda **kwargs: pytest.fail("model loaded"))
    return config


@pytest.mark.parametrize("model,depth", [("rvm_resnet50", False), ("birefnet", False),
                                        ("rvm_mobilenetv3", True)])
def test_batch_offline_rejects_model_or_depth_switch_before_loading(offline_config, model, depth):
    with pytest.raises(ValueError, match="离线"):
        ui._run_batch_demo("unused.mp4", None, model, "green", 0, "cpu", enable_depth=depth)


@pytest.mark.parametrize("model,depth", [("rvm_resnet50", False), ("birefnet", False),
                                        ("rvm_mobilenetv3", True)])
def test_live_offline_rejects_model_or_depth_switch_before_loading(offline_config, model, depth):
    with pytest.raises(ValueError, match="离线"):
        ui.render_live_frame(np.zeros((24, 32, 3), np.uint8), None, model, "green", "cpu", 32,
                             enable_depth=depth, cache=ui.LiveRunnerCache())


@pytest.mark.parametrize("alias", ["rvm", "mobilenetv3", "rvm_mobilenetv3"])
def test_aliases_keep_matching_bundle(offline_config, alias):
    assert ui._model_assets(alias) == (None, "verified-mobile-bundle")


@pytest.mark.parametrize("model", ["rvm_mobilenetv3", "rvm_resnet50"])
def test_offline_controls_only_offer_bundled_model_and_disable_depth(offline_config, model):
    offline_config["matting"]["model"] = model
    demo = ui.build_demo(offline_config)
    try:
        assert demo.analytics_enabled is False
        controls = [item["props"] for item in demo.config["components"]]
        models = [item for item in controls if item.get("label") in {"抠像模型", "Model"}]
        assert len(models) == 2
        for item in models:
            assert item["value"] == model
            assert [choice[1] for choice in item["choices"]] == [model]
        depth = [item for item in controls if item.get("label") in {
            "启用相对远近标注", "Relative Distance (Near / Mid / Far)"}]
        assert len(depth) == 2
        assert all(item["interactive"] is False and item["value"] is False for item in depth)
    finally:
        demo.close()
