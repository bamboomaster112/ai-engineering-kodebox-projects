import pytest

from aidesk.playground import PlaygroundError, available_models, compare


def test_compare_runs_each_model():
    models = available_models()[:3]
    out = compare("Explain CI in one sentence", models, temperature=0.5, runs=2)
    assert len(out["results"]) == 6
    assert {r["model"] for r in out["results"]} == set(models)
    assert out["summary"]["errors"] == 0 and out["summary"]["fastest"] in models


def test_compare_rejects_models_outside_the_list():
    with pytest.raises(PlaygroundError):
        compare("hi", ["some-unlisted-model"])


def test_compare_validates_settings():
    with pytest.raises(PlaygroundError):
        compare("hi", available_models()[:1], temperature=3)
    with pytest.raises(PlaygroundError):
        compare("", available_models()[:1])
