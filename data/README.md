# Data

The raw data is **not included** in this repository. It belongs to KAMP (Korea AI Manufacturing Platform) and is distributed through their portal.

## How to get it

1. Sign in to [KAMP](https://www.kamp-ai.kr) and open **제조AI데이터셋 → 사출성형기 AI 데이터셋** (Injection Molding Machine AI Dataset).
2. Download the dataset and place these four files in this folder:

```
data/
├── moldset_labeled_cn7.csv      1,211 rows · PassOrFail label
├── moldset_labeled_rg3.csv      1,182 rows · PassOrFail label
├── moldset_unlabeled_cn7.csv   35,239 rows · no label
└── moldset_unlabeled_rg3.csv   35,941 rows · no label
```

The competition release used the same file names. All process variables are already standardized (no physical units).

---

# 데이터

원본 데이터는 **저장소에 포함하지 않았다.** KAMP(인공지능 제조 플랫폼) 소유이며 KAMP 포털에서 받을 수 있다.

1. [KAMP](https://www.kamp-ai.kr) 로그인 → **제조AI데이터셋 → 사출성형기 AI 데이터셋**
2. 위 네 개 CSV를 이 폴더에 넣는다. 변수는 이미 표준화돼 있다(단위 없음).
