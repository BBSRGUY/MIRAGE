from mirage.experiments.report import score_decision


def test_decision_scoring():
    assert score_decision({"a": {"passed": True}, "b": {"passed": True}}) == "PASS"
    assert score_decision({"a": {"passed": True}, "b": {"passed": False}}) == "PARTIAL"
    assert score_decision({"a": {"passed": False}}) == "FAIL"
