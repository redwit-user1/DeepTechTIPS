import argparse
from pathlib import Path

from compliance_gateway.ontology.checks import alcoa_coverage, coverage, to_turtle

ap = argparse.ArgumentParser(prog="python -m compliance_gateway.ontology")
ap.add_argument("--ttl", type=Path, help="Turtle 로 내보낼 경로")
a = ap.parse_args()

for reg, m in coverage().items():
    print(f"{reg}\n  조항 {m['clauses']} · 모델링 {m['modeled']:.0%} · 자동점검 대상 {m['automatable']}"
          f" · 레포 구현 {m['automated']:.1%} · 플랫폼 포함 {m['with_platform']:.1%}")
    if m["uncovered"]:
        print(f"  미구현: {', '.join(m['uncovered'])}")
print("\nALCOA+ 속성별 점검 연결 조항 / 전체")
for k, (c, t) in alcoa_coverage().items():
    print(f"  {k:16s} {c:3d} / {t:3d}")
if a.ttl:
    a.ttl.write_text(to_turtle(), encoding="utf-8")
    print(f"\n→ {a.ttl}")
