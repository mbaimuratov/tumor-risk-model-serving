from types import SimpleNamespace

import pytest

import training.train as train


def test_promote_candidate_updates_champion_after_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aliases = {
        "candidate": SimpleNamespace(version="3"),
        "champion": SimpleNamespace(version="2"),
    }
    promoted: list[tuple[str, str]] = []

    monkeypatch.setattr(train, "model_version_by_alias", aliases.__getitem__)
    monkeypatch.setattr(train, "validate_model_artifacts", lambda version: None)
    monkeypatch.setattr(
        train,
        "validate_candidate_metrics",
        lambda candidate, champion: None,
    )
    monkeypatch.setattr(
        train,
        "run_candidate_predict_smoke_test",
        lambda version: None,
    )
    monkeypatch.setattr(
        train,
        "set_model_alias",
        lambda alias, version: promoted.append((alias, version)),
    )

    train.promote_candidate_to_champion()

    assert promoted == [("champion", "3")]


def test_promote_candidate_keeps_champion_when_validation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aliases = {
        "candidate": SimpleNamespace(version="3"),
        "champion": SimpleNamespace(version="2"),
    }
    promoted: list[tuple[str, str]] = []

    monkeypatch.setattr(train, "model_version_by_alias", aliases.__getitem__)
    monkeypatch.setattr(train, "validate_model_artifacts", lambda version: None)
    monkeypatch.setattr(
        train,
        "validate_candidate_metrics",
        lambda candidate, champion: None,
    )

    def fail_smoke_test(version: str) -> None:
        raise RuntimeError("smoke test failed")

    monkeypatch.setattr(train, "run_candidate_predict_smoke_test", fail_smoke_test)
    monkeypatch.setattr(
        train,
        "set_model_alias",
        lambda alias, version: promoted.append((alias, version)),
    )

    with pytest.raises(RuntimeError, match="smoke test failed"):
        train.promote_candidate_to_champion()

    assert promoted == []


def test_validate_candidate_metrics_requires_non_regression(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = SimpleNamespace(version="3")
    champion = SimpleNamespace(version="2")
    metrics_by_version = {
        "3": {"recall": 0.97, "false_negative_rate": 0.03},
        "2": {"recall": 0.98, "false_negative_rate": 0.02},
    }

    monkeypatch.setattr(
        train,
        "model_version_run_metrics",
        lambda model_version: metrics_by_version[model_version.version],
    )

    with pytest.raises(RuntimeError, match="Candidate recall"):
        train.validate_candidate_metrics(candidate, champion)
