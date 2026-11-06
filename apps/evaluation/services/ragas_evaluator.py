"""Ragas adapter. Stores faithfulness, answer relevancy and context recall/precision with framework="ragas"."""

from apps.evaluation.services import framework_bridge


def score_run(run, definitions) -> None:
    framework_bridge.score_run("ragas", run, definitions)
