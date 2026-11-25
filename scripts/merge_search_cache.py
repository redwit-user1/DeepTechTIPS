#!/usr/bin/env python3
"""검색 캐시 합치기: 뒤 파일의 검색어가 앞 파일을 덮는다(기존 캐시를 고정하려면 기존 파일을 뒤에 둔다).

    python scripts/merge_search_cache.py out.json new.json old.json
"""
import json
import sys

out, *ins = sys.argv[1:]
merged: dict = {}
for p in ins:
    merged.update(json.load(open(p, encoding="utf-8")))
json.dump(merged, open(out, "w", encoding="utf-8"), ensure_ascii=False)
print(out, len(merged))
