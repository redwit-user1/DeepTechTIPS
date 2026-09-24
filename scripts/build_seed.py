"""bioRxiv 시드 생성기.

출처: bioRxiv MCP (pharmacology and toxicology, 2024) 로 수집한 실제 preprint.
네트워크 정책상 repo 코드의 api.biorxiv.org 직접 접근이 차단(403)되므로,
수집 결과를 시드로 고정해 파이프라인이 오프라인에서도 재현되도록 한다.

갱신: bioRxiv MCP `search_preprints` + `get_preprint` 로 재수집 후 본 파일의
RECORDS 를 교체하면 된다.

실행: python scripts/build_seed.py
"""

from __future__ import annotations

import json
from pathlib import Path

RECORDS = [
    {
        "doi": "10.1101/2024.01.08.574589",
        "title": "In vitro 5-LOX inhibitory potential and antioxidant activity of new isoxazole derivatives",
        "authors": "Alam, W.; Khan, H.; Jan, M. S.; Daglia, M.",
        "date": "2024-01-08",
        "category": "pharmacology and toxicology",
        "license": "cc_by",
        "abstract": (
            "5-Lipoxygenase (5-LOX) is a key enzyme involved in the biosynthesis of "
            "pro-inflammatory leukotrienes, leading to asthma. Developing potent 5-LOX "
            "inhibitors are highly attractive. In this research the previously synthesized "
            "isoxazole derivatives has been investigated against 5-Lox inhibitory and "
            "antioxidant in vitro assay. The most potent compound C3 showed an IC50 of "
            "8.47 micro-M against 5-LOX. The compound C5 exhibited an IC50 of 10.48 micro-M. "
            "The most potent antioxidant activity was reported for C3 with an IC50 value of "
            "10.96 micro-M in the DPPH assay. Compound C6 also showed a potent dose dependent "
            "antioxidant effect with an IC50 value of 18.87 micro-M. Among the tested "
            "compounds, C6 was found most potent and reported the minimum IC50 value comparable "
            "to the reference drug."
        ),
    },
    {
        "doi": "10.1101/2024.01.10.575040",
        "title": "Propofol directly binds and inhibits skeletal muscle ryanodine receptor 1 (RyR1)",
        "authors": "Joseph, T. T.; Bu, W.; Haji-Ghassemi, O.; Chen, Y. S.; Woll, K.; Allen, P. D.; Brannigan, G.; van Petegem, F.; Eckenhoff, R. G.",
        "date": "2024-01-12",
        "category": "pharmacology and toxicology",
        "license": "cc_no",
        "abstract": (
            "As the primary Ca2+ release channel in skeletal muscle sarcoplasmic reticulum (SR), "
            "mutations in the type 1 ryanodine receptor (RyR1) underlie a constellation of muscle "
            "disorders, including malignant hyperthermia (MH). When anesthetizing patients with "
            "known MH mutations, the non-triggering intravenous general anesthetic propofol is "
            "commonly substituted for triggering anesthetics. Here, we show that propofol "
            "decreases RyR1 opening in heavy SR vesicles and planar lipid bilayers, and that it "
            "inhibits activator-induced Ca2+ release from SR in human skeletal muscle. "
            "Photoaffinity labeling using m-azipropofol revealed several putative propofol binding "
            "sites on RyR1. These findings invite the hypothesis that propofol may be protective "
            "against MH by inhibiting induced Ca2+ flux through RyR1."
        ),
    },
    {
        "doi": "10.1101/2024.01.12.575425",
        "title": "Antibiotics affect the pharmacokinetics of n-butylphthalide in vivo by altering the intestinal microbiota",
        "authors": "Li, X.; Guo, X.; Liu, Y.; Ren, F.; Li, S.; Yang, X.; Liu, J.; Zhang, Z.",
        "date": "2024-01-14",
        "category": "pharmacology and toxicology",
        "license": "cc_by",
        "abstract": (
            "N-butylphthalide (NBP) is a monomeric compound extracted from natural plant celery "
            "seeds. This study investigated the effect of intestinal microbiota alteration on the "
            "pharmacokinetics of NBP in SD rats. Compared to the control group, the values of "
            "Cmax and AUC0-8 in the antibiotic group increased by 56.1% and 56.4%, respectively. "
            "In contrast, the CL value decreased by 57.1%. CYP3A1 protein expression in the small "
            "intestine of the antibiotic group was 66.1% of that of the control group. Antibiotic "
            "treatment could affect the intestinal microbiota, decrease CYP3A1 expression and "
            "increase NBP exposure in vivo by inhibiting pathways related to NBP metabolism."
        ),
    },
    {
        "doi": "10.1101/2024.01.03.574006",
        "title": "Synergistic modulation of macrophages by methotrexate and RELA siRNA folate-liposome in collagen-induced arthritic rats",
        "authors": "Nasra, S.; Bhatia, D.; Kumar, A.",
        "date": "2024-01-04",
        "category": "pharmacology and toxicology",
        "license": "cc_no",
        "abstract": (
            "Rheumatoid arthritis (RA) is a chronic autoimmune disorder characterized by "
            "inflammation and joint destruction. This study explores a synergistic approach to RA "
            "therapy using folate-liposomal co-delivery of methotrexate (MTX) and RELA siRNA, "
            "aimed at RAW264.7 macrophage repolarization through inhibition of the NF-kappaB "
            "pathway. In a collagen-induced arthritis rat model, we observed a reduction in "
            "synovial inflammation and improved mobility following treatment. The combined MTX and "
            "RELA siRNA approach inhibits inflammatory cytokines and biochemical parameters such "
            "as C-reactive protein, potentially modulating the M1 to M2 macrophage polarization."
        ),
    },
]


def main() -> None:
    out = Path("compliance_gateway/data/seed/biorxiv_pharma.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(RECORDS, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[*] wrote {len(RECORDS)} preprints → {out}")


if __name__ == "__main__":
    main()
