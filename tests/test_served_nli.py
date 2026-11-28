"""서빙 NLI 는 호출이 실패하면 통과로 세지 않는다(감사 지적, 2026-11-28)."""
from compliance_gateway.serving.openai_client import ServedNLI


class _Boom:
    def generate(self, *a, **k):
        raise RuntimeError("down")


class _Ok:
    def generate(self, *a, **k):
        return "SUPPORT"


def test_failure_is_fail_closed_and_not_cached():
    nli = ServedNLI("http://localhost:1", "m")
    nli.client = _Boom()
    assert nli("evidence", "claim") == 0.0
    assert nli.failures == 1
    assert ("evidence", "claim") not in nli.cache
    nli.client = _Ok()
    assert nli("evidence", "claim") == 1.0
