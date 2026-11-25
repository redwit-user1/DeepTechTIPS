"""(GPU 서버의 repo/ 에서 실행, 결과는 ../data/prop_search.json) 제안서 주제 검색 다시: Crossref 폴백(초록 없음)으로 채워진 주제만 OpenAlex 로 천천히 다시 찾는다."""
import dataclasses
import json
import sys
import time

sys.path.insert(0, ".")
from compliance_gateway.verify.backends import OpenAlexBackend

out = "../data/prop_search.json"
cache = json.load(open(out, encoding="utf-8"))
topics = json.load(open("../data/prop_topics.json", encoding="utf-8"))
oa = OpenAlexBackend()
todo = [t for t in topics if sum(1 for r in cache.get(t[1], []) if r.get("abstract")) < 3]
print("다시 찾을 주제", len(todo), flush=True)
fails = 0
for n, (ko, en, _) in enumerate(todo):
    q = " ".join(en.split()[:8])
    for attempt in range(4):
        try:
            recs = oa.search(q, limit=5)
        except Exception as e:
            recs = []
            print("err", e, flush=True)
        if recs:
            break
        time.sleep(10 * (attempt + 1))
    if recs:
        streak = 0
        cache[en] = [dataclasses.asdict(r) for r in recs]
    else:
        fails += 1
        streak = locals().get("streak", 0) + 1
        if streak >= 5:
            print("한도 소진으로 멈춤", n, flush=True)
            break
    time.sleep(1.0)
    if n % 50 == 0:
        json.dump(cache, open(out, "w", encoding="utf-8"), ensure_ascii=False)
        ok = sum(1 for v in cache.values() if sum(1 for r in v if r.get("abstract")) >= 3)
        print(n, time.strftime("%H:%M"), "ok topics", ok, "fails", fails, flush=True)
json.dump(cache, open(out, "w", encoding="utf-8"), ensure_ascii=False)
ok = sum(1 for v in cache.values() if sum(1 for r in v if r.get("abstract")) >= 3)
print("done", "ok topics", ok, "fails", fails)
