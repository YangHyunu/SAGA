# 헌터스 건초더미 provenance — hunters-drm-r0 (2026-08-11)

고정 건초더미 벤치([스펙 §9](../specs/2026-08-10-fixed-haystack-bench-design.md) 데이터 위생)의
세 번째 재료. 소연 2종(fix-drm-r0/fix-van-r0)의 out-of-domain 편향 통제용.

## 산출물 (전부 gitignore, 로컬)

- 런 기록: `dreaming_data/eval/v2-hunters-drm-r0-run0.json` (대화 100턴 + ledger + 프로브 + 채점)
- 기억 저장소: `dreaming_data/hunters-drm-r0/` (episodes/facts/commits/raw)
- 카드: `dreaming_data/eval/card-hunters-v1.json` (동결 — 이후 수정 금지)
- 로그: `dreaming_data/eval/hunters-run.log`, `hunters-drm.log`, `proxy-8791.log`

## 원본과 변환 체인

- 원본: `~/Downloads/Alternate Hunters.charx` (헌터×게이트 시뮬, 영/한 이중언어, NPC ~40)
- 변환: `benchmarks.eval.charx2card` (keyed 63 / 상시 lore 12 / depth0→post 18)
- 전처리: `scripts/hunters_card_prep.py` (멱등) —
  ① lang 게이트 24곳 → 한국어(lang=1) 고정 해석 (유지 12/제거 12)
  ② `roll::500` 확률 이벤트 지시문 17개 제거 — 정적 조립은 턴별 롤 에뮬 불가,
     미해석 잔존 시 나레이터가 전부 무조건 지시로 오독
  ③ 이미지 지침 예시 자리표시자 `{{Character Image Command}}` → `CharacterName_expression`
  ④ 페르소나 주입: user_name "렌", [측정] 스킬 E랭크 각성자 (소연 런과 동일 구조)

## 생성 조건 (`scripts/hunters_run.zsh`)

- 파일럿(pilot-rules-r0) 검증 조합: `DREAMING_IDLE_SECONDS=100` + `--ttl-wait`(10턴마다 305초 정지)
- 나레이터 `deepseek/deepseek-v4-flash-0731` (reasoning 4000, max_tokens 30000, trim 100K)
- 꿈 `google/gemini-3-flash-preview`, 프록시 포트 8791
- Dreamer 프롬프트: 규칙 2개 추가판 (ae346d1 — 배경 진술·부수 묘사 추출)
- 스모크 6턴(격리 0) 통과 후 본런. 총 75분, $0.77 (나레이터 $0.49 + Lucid $0.22 + judge $0.06)

## 결과 요약

| 항목 | 값 | 비고 |
|---|---|---|
| 턴 | 100 | 완주, aborted 없음 |
| 격리 | 0 | night2-drm 병리 재발 없음 |
| 에피소드 | 24, **다턴 24/24 (100%)** | 스팬 2~7, 중앙값 4 — 파일럿(92%)보다 개선 |
| fact | 56 (0.56/턴) | 소연 3.5/턴의 1/6 — 파일럿과 같은 밀도대 |
| key_excerpts | 45 | |
| 프로브 | 유효 8/9 | G3: 1개 정답 누출로 드랍 (동결 시 excluded 라벨) |
| 라이브 judge | 2/8 | G9 유보 (judge-사람 일치율 미검증) — 건초더미 용도엔 비관건 |
| 평균 응답 | 1,030자 | 소연 fix-drm 1,929자의 절반 — roll 제거 영향 가능성 |

## 사용 제한 (라벨)

1. **조건부 주장만 가능**: dreaming 저장소는 "10턴 배치 후 5분+ 유휴" 리듬의 산물 —
   상한 조건 통제이지 실사용 재현이 아니다. 무조건부 성능 주장 금지.
   (트리거의 유저 리듬 종속성은 별도 설계 트랙: 유휴 AND 백로그≥k)
2. **소연 건초더미와 절대 평균 금지** (스펙 §2.3) — 응답 길이 분포 자체가 다르다.
3. 누적 히스토리가 가벼워(T99 예상 ~90K) vanilla가 128K 창에 들어감 —
   소연 fix-drm에선 불가능했던 순정 완주 비교가 여기선 가능.
4. 스킬 타임라인: [측정]은 페르소나 슬롯에만 있고 원장 미기록(드리머는 대화 신정보만 추출),
   "공간 도약"은 스토리 중 획득으로 원장에 숫자 스펙 포함 기록(쿨 30초·사거리 10m) —
   update/false 프로브 재료. 원장 수준 모순 없음.
