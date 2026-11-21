import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from compliance_gateway.agent.wf_a import DRAFT_PROMPT, GAP_PROMPT, REGEN_PROMPT  # noqa: E402
from run_wf_a_local import stage_of  # noqa: E402


def test_stage_of_matches_the_three_prompts():
    assert stage_of(GAP_PROMPT.format(topic="t", sources="s")) == "gap"
    assert stage_of(DRAFT_PROMPT.format(topic="t", sources="s", gap="g")) == "draft"
    assert stage_of(REGEN_PROMPT.format(vcr=0.5, threshold=0.6, problems="p", sources="s", draft="d")) == "regen"
