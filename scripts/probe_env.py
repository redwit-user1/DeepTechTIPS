#!/usr/bin/env python3
"""실행 환경 진단 — 무엇을 할 수 있는지 자동 판정.

KT Cloud AI Nexus 는 AI Train(학습 컨테이너)과 AI Serv(추론 서빙)를 합친 플랫폼이라,
프로비저닝에 따라 가능한 작업이 다르다. 이 스크립트가 실제로 확인해준다.

  python scripts/probe_env.py
  python scripts/probe_env.py --endpoint http://<서빙주소>/v1   # 서빙 엔드포인트도 점검
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys

OK, NO, WARN = "✅", "❌", "⚠️ "


def _mod(name: str) -> tuple[bool, str]:
    try:
        m = importlib.import_module(name)
        return True, getattr(m, "__version__", "?")
    except Exception:
        return False, ""


def probe_gpu() -> dict:
    info: dict = {"available": False, "count": 0, "devices": [], "fp8": False}
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=20,
            ).stdout.strip()
            for line in filter(None, out.splitlines()):
                parts = [p.strip() for p in line.split(",")]
                name = parts[0]
                mem = parts[1] if len(parts) > 1 else "?"
                cap = parts[2] if len(parts) > 2 else "0.0"
                info["devices"].append({"name": name, "memory": mem, "compute_cap": cap})
                # FP8 은 Hopper(9.0)+ 에서 지원
                try:
                    if float(cap) >= 9.0:
                        info["fp8"] = True
                except ValueError:
                    pass
            info["count"] = len(info["devices"])
            info["available"] = info["count"] > 0
        except Exception as e:
            info["error"] = str(e)
    return info


def probe_endpoint(url: str) -> dict:
    """OpenAI 호환 서빙 엔드포인트 점검 (/models)."""
    import urllib.request

    base = url.rstrip("/")
    target = base + ("/models" if base.endswith("/v1") else "/v1/models")
    headers = {}
    key = os.getenv("OPENAI_API_KEY") or os.getenv("KT_API_KEY")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    try:
        req = urllib.request.Request(target, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read().decode("utf-8"))
        models = [m.get("id") for m in data.get("data", [])] or [str(data)[:120]]
        return {"reachable": True, "url": target, "models": models}
    except Exception as e:
        return {"reachable": False, "url": target, "error": f"{type(e).__name__}: {e}"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default=os.getenv("GOONO_SERVING_URL"),
                    help="OpenAI 호환 서빙 엔드포인트 (예: http://host:8000/v1)")
    a = ap.parse_args()

    print("=" * 62)
    print(" GOONO AI 실행 환경 진단")
    print("=" * 62)

    print(f"\npython {sys.version.split()[0]}")

    gpu = probe_gpu()
    print("\n[GPU]")
    if gpu["available"]:
        for d in gpu["devices"]:
            print(f"  {OK} {d['name']}  mem={d['memory']}  compute={d['compute_cap']}")
        print(f"  개수: {gpu['count']}   FP8(Hopper+): {'지원' if gpu['fp8'] else '미지원'}")
    else:
        print(f"  {NO} GPU 없음 (nvidia-smi 미검출) — 학습 불가, 평가/데이터 작업만 가능")

    print("\n[학습 스택]")
    deps = ["torch", "transformers", "trl", "peft", "datasets", "accelerate", "unsloth", "vllm"]
    have = {}
    for name in deps:
        ok, ver = _mod(name)
        have[name] = ok
        print(f"  {OK if ok else NO} {name}{(' ' + ver) if ok else ''}")

    print("\n[데이터]")
    import pathlib
    checks = {
        "SciFact(외부 EN 평가)": pathlib.Path("data/raw/data/corpus.jsonl"),
        "bioRxiv 시드": pathlib.Path("compliance_gateway/data/seed/biorxiv_pharma.json"),
        "국내 실데이터 시드": pathlib.Path("compliance_gateway/data/korean/seed/kr_protocols.json"),
    }
    for label, p in checks.items():
        print(f"  {OK if p.exists() else NO} {label}")

    if a.endpoint:
        print("\n[서빙 엔드포인트]")
        r = probe_endpoint(a.endpoint)
        if r["reachable"]:
            print(f"  {OK} 연결됨: {r['url']}")
            print(f"     모델: {', '.join(str(m) for m in r['models'][:5])}")
        else:
            print(f"  {NO} 연결 실패: {r['url']}")
            print(f"     {r['error']}")

    # ---- 판정 ----
    print("\n" + "=" * 62)
    print(" 판정")
    print("=" * 62)
    can_train = gpu["available"] and have.get("torch") and have.get("transformers")
    if can_train:
        n = gpu["count"]
        print(f"  {OK} 학습 가능 (GPU {n}장)")
        if gpu["fp8"]:
            print(f"     H100/Hopper 감지 → FP8 RL 사용 가능(Unsloth GRPO)")
        if n >= 2:
            print("     권장: GRPO 는 vLLM server 모드(GPU0=생성 / GPU1=학습)")
            print("       CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model <id>")
            print("       CUDA_VISIBLE_DEVICES=1 python -m compliance_gateway.train.grpo --vllm-mode server")
        else:
            print("     권장: GRPO 는 colocate 모드 (--vllm-mode colocate)")
        print("\n  다음 실행: bash scripts/run_m1_h100.sh")
    elif gpu["available"]:
        print(f"  {WARN} GPU 는 있으나 학습 스택 미설치")
        print('     실행: pip install -e ".[train]"')
    else:
        print(f"  {NO} 학습 불가 (GPU 없음)")
        print("     이 환경에서 가능한 것: 데이터 구축, KPI 평가(통계 NLI), 서빙 엔드포인트 연동")
        print("     서빙만 있는 경우 → Gateway 를 엔드포인트에 연결해 평가:")
        print("       python -m compliance_gateway.eval.external --split dev \\")
        print("           --nli-endpoint http://<서빙>/v1 --nli-model <모델명>")


if __name__ == "__main__":
    main()
