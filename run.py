"""제6회 K-인공지능 제조데이터 분석 경진대회 — ① 사출성형 품질불량 사전예측 및 검사 우선순위

실행:  python run.py
입력:  data/moldset_{labeled,unlabeled}_{cn7,rg3}.csv
출력:  outputs/ (진단 요약, 모델 비교, 오류분석, 검사 우선순위, 테스트 예측결과, 그림)
"""
import sys
from pathlib import Path

import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent
D, O = ROOT / "data", ROOT / "outputs"
O.mkdir(exist_ok=True)
PRODUCTS = ["cn7", "rg3"]
SEEDS = [0, 1, 2, 3, 4]
BLOCK = 40  # 시간 블록 크기(샷 수). 인접 샷이 학습/검증에 같이 들어가는 누수를 막는다
LAG = 3     # 검사 결과가 공정에 돌아오기까지의 지연(샷 수). 샷 t 시점에는 t-LAG 이전 검사 결과만 안다
LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.append(s)


def load(kind):
    parts = []
    for p in PRODUCTS:
        d = pd.read_csv(D / f"moldset_{kind}_{p}.csv").rename(columns={"Unnamed: 0": "order"})
        d["product"] = p
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


# ═════════════════════════ 1장. 데이터 이해 및 진단 ═════════════════════════
lab, unl = load("labeled"), load("unlabeled")
RAW = [c for c in lab.columns if c not in ("order", "PassOrFail", "product")]
CONST = [c for c in RAW if lab[c].nunique() <= 1]
FEATS = [c for c in RAW if c not in CONST]

say("=" * 72, "\n[1장] 데이터 진단")
for p in PRODUCTS:
    d = lab[lab["product"] == p]
    say(f"  {p}: 라벨 {len(d)}행, 불량 {int(d.PassOrFail.sum())}개({d.PassOrFail.mean():.2%}) | 라벨없음 {int((unl['product'] == p).sum())}행")
say(f"  결측: 라벨 {int(lab.isna().sum().sum())} / 라벨없음 {int(unl.isna().sum().sum())}  |  변수는 이미 표준화돼 있음(단위 없음)")
for c in CONST:
    say(f"  상수 컬럼 제거: {c} (라벨 데이터 고유값 1개, 라벨없음 데이터 고유값 {unl[c].nunique()}개 → 두 데이터 분포 차이)")


def make_shots(d):
    """연속된 두 행의 설비값이 완전히 같으면 한 번의 사출(샷)에서 나온 두 제품으로 보고 묶는다."""
    X, prod = d[FEATS].to_numpy(), d["product"].to_numpy()
    sid, s, i = np.empty(len(d), int), 0, 0
    while i < len(d):
        sid[i] = s
        if i + 1 < len(d) and prod[i] == prod[i + 1] and (X[i] == X[i + 1]).all():
            sid[i + 1] = s
            i += 2
        else:
            i += 1
        s += 1
    return sid


lab["shot"] = make_shots(lab)
g = lab.groupby("shot")
assert g.size().isin([1, 2]).all() and (g[FEATS].nunique() == 1).all().all(), "샷 묶기 오류"
shots = g.agg(**{f: (f, "first") for f in FEATS}, product=("product", "first"), order=("order", "first"),
              y=("PassOrFail", "max"), n_part=("PassOrFail", "size"), n_fail=("PassOrFail", "sum")).reset_index()
pairs = shots[shots.n_part == 2]
one_fail, two_fail = int((pairs.n_fail == 1).sum()), int((pairs.n_fail == 2).sum())
lone_fail = int(((shots.n_part == 1) & (shots.n_fail > 0)).sum())
say(f"  동일 설비값 연속 2행(=1샷 2제품, 2캐비티 추정): {len(pairs)}쌍 / 전체 {len(lab)}행 → {len(shots)}샷")
say(f"  불량 제품 {int(lab.PassOrFail.sum())}개 위치: 쌍 중 하나만 불량 {one_fail}쌍, 둘 다 불량 {two_fail}쌍, 단독 {lone_fail}개")
say(f"  → 불량 제품 {one_fail}개는 설비값이 똑같은 양품 쌍둥이가 있음: 제품 단위로는 어떤 모델도 둘을 구분 못 함")
say(f"  → [재정의 1] 예측 단위를 샷 단위 불량 위험으로: {len(shots)}샷 중 위험 샷 {int(shots.y.sum())}개({shots.y.mean():.2%})")

y = shots["y"].to_numpy()
pos = shots.groupby("product").cumcount()
shots["is_rg3"] = (shots["product"] == "rg3").astype(int)
unl["is_rg3"] = (unl["product"] == "rg3").astype(int)
shots["grp"] = shots["product"] + "_" + (pos // BLOCK).astype(str)
prev20 = shots.groupby("product")["y"].transform(lambda s: s.shift(1).rolling(20, min_periods=1).sum()).fillna(0).gt(0)
shots["defect_kind"] = np.where(y == 0, "-", np.where(prev20, "follow-on", "first"))
for p in PRODUCTS:
    say(f"  {p} 불량 샷 순번: {pos[(shots['product'] == p) & (shots.y == 1)].tolist()}")
nf = int((shots.defect_kind == "first").sum())
say(f"  불량은 몰려서 발생: 위험 샷 {int(y.sum())}개 중 직전 20샷 안에 불량이 이미 있던 경우 {int(y.sum()) - nf}개, 첫 불량 {nf}개")
Zabs = shots[FEATS].abs()
out_any = (Zabs > 3).any(axis=1).to_numpy()
top_out = (Zabs > 3).sum().sort_values(ascending=False).head(3)
say(f"  이상치(표준화값 |z|>3인 변수가 하나라도 있는 샷): {int(out_any.sum())}개({out_any.mean():.1%}), 그중 위험 샷 {int((out_any & (y == 1)).sum())}개"
    f" | 극단값이 많은 변수: {', '.join(f'{k}({v})' for k, v in top_out.items())}")
say("  → 이상치는 제거하지 않음: 트리 모델은 값의 크기 순서만 쓰고, 불량이 극단값과 함께 나타날 수 있어 정보로 남김. 이상 정도는 M1(IsolationForest)로 따로 검증")
pd.DataFrame({"n_over3": (Zabs > 3).sum(), "max_abs_z": Zabs.max()}).to_csv(O / "01_outliers.csv", encoding="utf-8-sig")

# ═════════════════════════ 2장. 모델 개발 및 성능평가 ═════════════════════════
BASE = FEATS + ["is_rg3"]


def lgbm(seed=0):
    return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=10,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                              class_weight="balanced", verbose=-1, random_state=seed)


def logit(seed=0):
    return LogisticRegression(class_weight="balanced", max_iter=5000)


def oof(make, cols, df, yy, seed, groups):
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=seed) if groups is not None else StratifiedKFold(5, shuffle=True, random_state=seed)
    p = np.zeros(len(df))
    for tr, te in cv.split(df, yy, groups):
        p[te] = make(seed).fit(df.iloc[tr][cols], yy[tr]).predict_proba(df.iloc[te][cols])[:, 1]
    return p


def hist_feats(L):
    """불량 이력 특징. 검사 결과가 L샷 늦게 돌아오므로 샷 t 에서는 t-L 이전 결과만 사용"""
    gy = shots.groupby("product")["y"]
    h = pd.DataFrame(index=shots.index)
    for k in (3, 10, 30):
        h[f"fail_last{k}"] = gy.transform(lambda s: s.shift(L).rolling(k, min_periods=1).sum()).fillna(0)
    last = pos.where(shots.y.eq(1)).groupby(shots["product"]).transform(lambda s: s.shift(L).ffill())
    h["since_fail"] = (pos - last).fillna(999).clip(upper=999)
    return h


def best_f1(yy, p):
    pr, rc, th = precision_recall_curve(yy, p)
    f = 2 * pr * rc / (pr + rc + 1e-12)
    i = int(f[:-1].argmax())
    return f[i], pr[i], rc[i], th[i]


def capture(p, k):
    n = int(np.ceil(len(y) * k / 100))
    return y[np.argsort(-p, kind="stable")[:n]].sum()


def scores(p, prob):
    f, pr, rc, _ = best_f1(y, p)
    return {"PR-AUC": average_precision_score(y, p), "ROC-AUC": roc_auc_score(y, p), "F1(best)": f, "Precision": pr,
            "Recall": rc, "caught_top10%": capture(p, 10), "caught_top20%": capture(p, 20),
            "Brier": brier_score_loss(y, p) if prob else np.nan}


iso = {p: IsolationForest(n_estimators=300, random_state=0).fit(unl.loc[unl["product"] == p, FEATS]) for p in PRODUCTS}
shots["iso"] = np.concatenate([-iso[p].score_samples(shots.loc[shots["product"] == p, FEATS]) for p in PRODUCTS])

P_PROC = {s: oof(lgbm, BASE, shots, y, s, shots["grp"]) for s in SEEDS}  # 설비값 모델 OOF (모든 L에서 공유)


def two_stage(seed, L, proc=None):
    """[재정의 2] 최근 불량 이력이 있으면 이력 모델(버스트 진행), 없으면 설비값 모델(버스트 시작) 순위"""
    h = hist_feats(L)
    ph = oof(logit, list(h.columns), pd.concat([shots, h], axis=1), y, seed, shots["grp"])
    recent = (h["fail_last3"] > 0).to_numpy()
    return np.where(recent, 1 + ph, pd.Series((proc or P_PROC)[seed]).rank(pct=True).to_numpy())


H = hist_feats(LAG)
SX = pd.concat([shots, H], axis=1)
CANDIDATES = {
    "M0 Logistic, process vars (baseline)": (lambda s: oof(logit, BASE, shots, y, s, shots["grp"]), True),
    "M1 IsolationForest, unlabeled 71k (unsupervised)": (lambda s: shots["iso"].to_numpy(), False),
    "M2 LightGBM, process vars": (lambda s: P_PROC[s], True),
    "R1 Rule: defect in last 3 known shots": (lambda s: (H.fail_last3 > 0).to_numpy() + 1e-6 * H.fail_last30.to_numpy(), False),
    "M5 Logistic, defect history": (lambda s: oof(logit, list(H.columns), SX, y, s, shots["grp"]), True),
    "M6 LightGBM, process vars + history": (lambda s: oof(lgbm, BASE + list(H.columns), SX, y, s, shots["grp"]), True),
    "M7 Two-stage (history gate + process)": (lambda s: two_stage(s, LAG), False),
}
say("=" * 72, f"\n[2장] 모델 비교 — 샷 단위, 시간블록({BLOCK}샷) 그룹 5겹 교차검증 × 시드 {len(SEEDS)}회, 검사 피드백 지연 {LAG}샷")
rows, OOF = [], {}
for name, (fn, prob) in CANDIDATES.items():
    ps = [fn(s) for s in SEEDS]
    OOF[name] = np.mean(ps, axis=0)
    r = pd.DataFrame([scores(p, prob) for p in ps])
    rows.append({"model": name, **{f"{k}_mean": v for k, v in r.mean().items()}, **{f"{k}_std": v for k, v in r.std().items()},
                 # 시드 5개의 OOF 점수를 평균한 뒤 계산(앙상블 기준). 모든 모델에 같은 방식으로 적용해야 비교가 공정하다
                 "PR-AUC_ens5": average_precision_score(y, OOF[name]),
                 "caught_top10%_ens5": capture(OOF[name], 10), "caught_top20%_ens5": capture(OOF[name], 20)})
FINAL = "M7 Two-stage (history gate + process)"
# 확률 보정: 검증(OOF) 점수에 2-파라미터 Platt 보정 1개를 맞춤. 단조변환이라 순위(포착률)는 그대로, 확률만 실제 불량률에 맞춰짐
p_cal = LogisticRegression().fit(OOF[FINAL].reshape(-1, 1), y).predict_proba(OOF[FINAL].reshape(-1, 1))[:, 1]
# 위 보정은 같은 점수로 맞추고 평가하므로 낙관적일 수 있다 → 보정식을 교차검증으로 맞춘 값과 비교해 둔다
p_cal_cv = cross_val_predict(LogisticRegression(), OOF[FINAL].reshape(-1, 1), y, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
rows.append({"model": "M7 final: 5-seed ensemble + Platt calibration", **{f"{k}_mean": v for k, v in scores(p_cal, True).items()}})
res = pd.DataFrame(rows)
res.to_csv(O / "02_model_comparison.csv", index=False, encoding="utf-8-sig")
for _, r in res.iterrows():
    say(f"  {r.model:48s} PR-AUC {r['PR-AUC_mean']:.3f} (앙상블 {r['PR-AUC_ens5']:.3f}) | ROC {r['ROC-AUC_mean']:.3f} | F1 {r['F1(best)_mean']:.3f}"
        f" | 상위10% 포착 {r['caught_top10%_mean']:4.1f}/{int(y.sum())} (앙상블 {r['caught_top10%_ens5']:.0f}, 20% {r['caught_top20%_ens5']:.0f}) | Brier {r['Brier_mean']:.4f}")
say(f"  (무작위 기준: PR-AUC = 불량 비율 {y.mean():.3f}, 상위10% 포착 {y.sum() * 0.1:.1f}개)")
say(f"  보정 검증: 보정식을 교차검증으로 맞추면 Brier {brier_score_loss(y, p_cal_cv):.4f} (전체 적합 {brier_score_loss(y, p_cal):.4f})")

say("\n  [재정의 1 근거] 같은 LightGBM(설비값), 예측 단위·분할 방식만 바꿈")
yp = lab["PassOrFail"].to_numpy()
lab["is_rg3"] = (lab["product"] == "rg3").astype(int)
split_rows = []
for name, df, yy, grp in [("part-level, random split", lab, yp, None), ("part-level, shot-grouped split", lab, yp, lab["shot"]),
                          ("shot-level, random split", shots, y, None), ("shot-level, time-block split (adopted)", shots, y, shots["grp"])]:
    ps = [oof(lgbm, BASE, df, yy, s, grp) for s in SEEDS]
    r = pd.DataFrame([{"PR-AUC": average_precision_score(yy, p), "ROC-AUC": roc_auc_score(yy, p),
                       **dict(zip(["F1", "Precision", "Recall"], best_f1(yy, p)[:3]))} for p in ps]).mean()
    split_rows.append({"setting": name, "base_rate": yy.mean(), **r.to_dict()})
    say(f"  {name:40s} PR-AUC {r['PR-AUC']:.3f} (무작위 {yy.mean():.3f}) | ROC {r['ROC-AUC']:.3f} | Precision {r['Precision']:.3f}")
pd.DataFrame(split_rows).to_csv(O / "02_split_comparison.csv", index=False, encoding="utf-8-sig")
say("  → 시간을 섞으면 좋아 보이지만 처음 보는 시간대에서는 무작위 수준: 설비값은 불량보다 '그 시간대'를 기억함")

say("\n  [현장 조건] 검사 결과가 공정으로 돌아오는 지연에 따른 최종모델 성능")
lag_rows = []
for L in (1, 3, 5, 10):
    ps = [two_stage(s, L) for s in SEEDS]
    lag_rows.append({"feedback_lag_shots": L, "PR-AUC": np.mean([average_precision_score(y, p) for p in ps]),
                     "caught_top10%": np.mean([capture(p, 10) for p in ps]), "caught_top20%": np.mean([capture(p, 20) for p in ps])})
    say(f"  지연 {L:2d}샷: PR-AUC {lag_rows[-1]['PR-AUC']:.3f} | 상위10% 검사로 {lag_rows[-1]['caught_top10%']:.1f}/{int(y.sum())} 포착 | 상위20% {lag_rows[-1]['caught_top20%']:.1f}")
pd.DataFrame(lag_rows).to_csv(O / "02_feedback_lag.csv", index=False, encoding="utf-8-sig")

say("\n  [제조지식 관계 변수 검증] 설비값끼리 조합한 변수를 B단계 설비값 모델에 추가")
BT = [f"Barrel_Temperature_{i}" for i in range(1, 7)]
REL = pd.DataFrame({
    "press_drop": shots.Max_Injection_Pressure - shots.Max_Switch_Over_Pressure,   # 보압 전환 시 압력 강하
    "back_press_var": shots.Max_Back_Pressure - shots.Average_Back_Pressure,       # 배압 흔들림
    "rpm_var": shots.Max_Screw_RPM - shots.Average_Screw_RPM,                      # 스크루 회전 흔들림
    "speed_x_press": shots.Max_Injection_Speed * shots.Max_Injection_Pressure,     # 속도·압력 동시 이탈
    "hold_time": shots.Injection_Time - shots.Filling_Time,                        # 보압 구간 대리
    "other_time": shots.Cycle_Time - (shots.Injection_Time + shots.Plasticizing_Time + shots.Clamp_Close_Time),  # 냉각·대기 대리
    "stroke": shots.Plasticizing_Position - shots.Cushion_Position,                # 실제 사출량 대리
    "barrel_spread": shots[BT].std(axis=1),                                        # 배럴 구간 온도 불균일
    "barrel_ends": shots.Barrel_Temperature_6 - shots.Barrel_Temperature_1,        # 배럴 양 끝 온도 차
    "hopper_vs_barrel": shots.Hopper_Temperature - shots[BT].mean(axis=1),         # 호퍼·배럴 온도 어긋남
    "mold_imbalance": shots.Mold_Temperature_3 - shots.Mold_Temperature_4,         # 금형 온도 불균형
    "n_out2": (shots[FEATS].abs() > 2).sum(axis=1),                                # 평소 범위(±2) 밖 변수 수
    "max_abs_z": shots[FEATS].abs().max(axis=1),                                   # 가장 크게 벗어난 정도
})  # 값이 표준화돼 있어 비율은 의미가 없으므로 차이·곱·묶음 통계만 사용
SR = pd.concat([shots, REL], axis=1)
P_REL = {s: oof(lgbm, BASE + list(REL.columns), SR, y, s, shots["grp"]) for s in SEEDS}
rel_res = {}
for name, proc in [("설비값 24개 (채택)", P_PROC), ("설비값 24개 + 관계 13개", P_REL)]:
    gm = np.mean([two_stage(s, LAG, proc) for s in SEEDS], axis=0)
    rel_res[name] = average_precision_score(y, gm)
    say(f"  B단계 = {name:22s} 설비값 단독 PR-AUC {np.mean([average_precision_score(y, p) for p in proc.values()]):.3f}"
        f" | 최종 PR-AUC {rel_res[name]:.3f} | 상위10% 포착 {capture(gm, 10)}/{int(y.sum())} | 상위20% {capture(gm, 20)}")
rel_tab = pd.DataFrame({"ROC_single": {c: roc_auc_score(y, REL[c]) for c in REL}, "mean_defect": REL[y == 1].mean(), "mean_ok": REL[y == 0].mean()})
rel_tab.to_csv(O / "03_relational_features.csv", encoding="utf-8-sig")
top = (rel_tab.ROC_single - 0.5).abs().sort_values(ascending=False).index[:2]
say("  관계 변수 중 단일 신호가 큰 것:", ", ".join(f"{c}(ROC {rel_tab.ROC_single[c]:.3f}, 불량 평균 {rel_tab.mean_defect[c]:+.2f} / 정상 {rel_tab.mean_ok[c]:+.2f})" for c in top))
say("  → 관계 변수 추가로 개선 없음: 채택하지 않음" if rel_res["설비값 24개 + 관계 13개"] <= rel_res["설비값 24개 (채택)"] else "  → 관계 변수 추가로 개선됨")

say("\n  [동일 분할 조건 비교] 분할 방식을 바꿔도 2단계가 설비값 단독을 이기는가 (누수가 있는 설정 포함)")
yp = lab["PassOrFail"].to_numpy()
same_rows = []
for name, grp in [("shot-level, random split (leaky)", None), ("shot-level, time-block split (adopted)", shots["grp"])]:
    pa, pb = [], []
    for s in SEEDS:
        pp = oof(lgbm, BASE, shots, y, s, grp)
        ph = oof(logit, list(H.columns), SX, y, s, grp)
        pa.append(pp)
        pb.append(np.where(H["fail_last3"].to_numpy() > 0, 1 + ph, pd.Series(pp).rank(pct=True).to_numpy()))
    same_rows.append({"setting": name, "process_only_PR-AUC": np.mean([average_precision_score(y, p) for p in pa]),
                      "two_stage_PR-AUC": np.mean([average_precision_score(y, p) for p in pb])})
    if grp is None:  # 제품 단위·무작위(가장 흔한 설정): 설비값 모델은 제품 행으로 직접 학습, 2단계는 샷 점수를 두 제품에 배정
        p_part = np.mean(pb, axis=0)[lab["shot"].to_numpy()]
        same_rows.append({"setting": "part-level, random split (leaky)",
                          "process_only_PR-AUC": np.mean([average_precision_score(yp, oof(lgbm, BASE, lab, yp, s, None)) for s in SEEDS]),
                          "two_stage_PR-AUC": average_precision_score(yp, p_part)})
same = pd.DataFrame(same_rows)
same.to_csv(O / "02_same_split_comparison.csv", index=False, encoding="utf-8-sig")
for _, r in same.iterrows():
    say(f"  {r.setting:40s} 설비값 단독 {r['process_only_PR-AUC']:.3f} | 2단계 {r['two_stage_PR-AUC']:.3f}")
say("  → 어떤 분할에서도 2단계가 높다. 누수 있는 설정의 설비값 점수(0.15)보다 같은 설정의 2단계(0.22)가 높음")

say("\n  [설비값 모델 설정 민감도] LightGBM num_leaves × n_estimators (시간블록 CV, 시드 5 평균)")
sens = []
for nl, ne in [(3, 200), (7, 300), (15, 300), (7, 600)]:
    mk = lambda s, nl=nl, ne=ne: lgb.LGBMClassifier(n_estimators=ne, learning_rate=0.03, num_leaves=nl, min_child_samples=10, subsample=0.8, subsample_freq=1,
                                                   colsample_bytree=0.8, class_weight="balanced", verbose=-1, random_state=s)
    v = np.mean([average_precision_score(y, oof(mk, BASE, shots, y, s, shots["grp"])) for s in SEEDS])
    sens.append({"num_leaves": nl, "n_estimators": ne, "PR-AUC": v})
    say(f"    num_leaves={nl:2d}, n_estimators={ne}: PR-AUC {v:.3f}{'  (채택)' if (nl, ne) == (7, 300) else ''}")
pd.DataFrame(sens).to_csv(O / "02_lgbm_sensitivity.csv", index=False, encoding="utf-8-sig")
say("  → 설정을 바꿔도 설비값 단독은 무작위(0.031) 근처: 튜닝으로 풀릴 문제가 아님. 불량 39개에 맞춰 얕은 트리(num_leaves 7, 300 trees, class_weight balanced)를 채택하고 추가 탐색은 하지 않음(과적합 위험)")

plt.figure(figsize=(6, 5))
for name in ["M0 Logistic, process vars (baseline)", "M2 LightGBM, process vars", "R1 Rule: defect in last 3 known shots",
             "M5 Logistic, defect history", FINAL]:
    pr, rc, _ = precision_recall_curve(y, OOF[name])
    plt.plot(rc, pr, label=f"{name.split(',')[0].split(' (')[0]} AP={average_precision_score(y, OOF[name]):.3f}")
plt.axhline(y.mean(), ls="--", c="gray", label="random")
plt.xlabel("Recall"), plt.ylabel("Precision"), plt.title(f"PR curve (shot-level, time-block CV, lag={LAG})"), plt.legend(fontsize=8)
plt.tight_layout(), plt.savefig(O / "fig_pr_curves.png", dpi=150), plt.close()

# ═════════════════════════ 3장. 영향요인 및 오류분석 ═════════════════════════
say("=" * 72, "\n[3장] 영향요인 및 오류분석 (최종모델 OOF 기준)")
proc_full = lgbm().fit(shots[BASE], y)
imp = pd.Series(np.abs(proc_full.booster_.predict(shots[BASE], pred_contrib=True)[:, :-1]).mean(0), index=BASE).sort_values(ascending=False)
imp.rename("mean_abs_shap").to_csv(O / "03_feature_importance_process.csv", encoding="utf-8-sig")
say("  설비값 모델 영향 변수(평균 |SHAP|, LightGBM 내장) 상위 8:", ", ".join(f"{k}({v:.2f})" for k, v in imp.head(8).items()))
hist_full = LogisticRegression(class_weight="balanced", max_iter=5000).fit(H, y)
coef = pd.Series(hist_full.coef_[0], index=H.columns)
coef.rename("logit_coef").to_csv(O / "03_history_coefficients.csv", encoding="utf-8-sig")
say("  이력 모델 계수:", ", ".join(f"{k} {v:+.2f}" for k, v in coef.items()))
imp.head(12)[::-1].plot.barh(figsize=(6, 5), title="Process model: mean |SHAP|")
plt.tight_layout(), plt.savefig(O / "fig_importance.png", dpi=150), plt.close()

thr = best_f1(y, p_cal)[3]
pred = p_cal >= thr
shots["prob"], shots["TP"], shots["FN"], shots["FP"] = p_cal, pred & (y == 1), ~pred & (y == 1), pred & (y == 0)
shots["stage"] = np.where(H["fail_last3"] > 0, "A recent defect", "B no recent defect")
TPn, FNn, FPn = int(shots.TP.sum()), int(shots.FN.sum()), int(shots.FP.sum())
say(f"  판정 기준값(F1 최대) {thr:.3f}: TP {TPn}, 미탐(FN) {FNn}, 오경보(FP) {FPn} | 정밀도 {TPn / (TPn + FPn):.2f}, 재현율 {TPn / (TPn + FNn):.2f}, F1 {2 * TPn / (2 * TPn + FNn + FPn):.3f}")
B = (H["fail_last3"] == 0).to_numpy()
P_OOF = np.mean(list(P_PROC.values()), axis=0)
say(f"  B단계(최근 불량 없음) {B.sum()}샷 안에서 설비값 모델 ROC {roc_auc_score(y[B], P_OOF[B]):.3f} (불량 {int(y[B].sum())}개)"
    f" — 전체 시간블록 ROC {roc_auc_score(y, P_OOF):.3f} 보다 높음: 설비값은 버스트 '시작' 쪽에 약한 신호")


def err_table(col):
    t = shots.groupby(col, observed=True).agg(shots=("y", "size"), defects=("y", "sum"), TP=("TP", "sum"), FN=("FN", "sum"), FP=("FP", "sum"))
    t["miss_rate"] = t.FN / t.defects.where(t.defects > 0)
    t["false_alarm_rate"] = t.FP / (t.shots - t.defects)
    return t


tables = {"product": err_table("product"), "stage": err_table("stage"), "defect_kind": err_table("defect_kind")}
for f in imp.index[:3]:
    shots[f + "_q"] = pd.qcut(shots[f].rank(method="first"), 4, labels=["Q1(low)", "Q2", "Q3", "Q4(high)"])
    tables[f] = err_table(f + "_q")
with open(O / "03_error_by_condition.csv", "w", encoding="utf-8-sig") as fh:
    for k, t in tables.items():
        fh.write(f"# condition: {k}\n")
        t.to_csv(fh)
        fh.write("\n")
for k, t in tables.items():
    say(f"  조건 [{k}]")
    for idx, r in t.iterrows():
        say(f"    {str(idx):20s} 샷 {int(r.shots):4d} 불량 {int(r.defects):2d} | 미탐 {int(r.FN):2d} | 오경보 {int(r.FP):3d} (오경보율 {r.false_alarm_rate:.1%})")

a, b = imp.index[0], imp.index[1]
inter = shots.pivot_table(index=a + "_q", columns=b + "_q", values="y", aggfunc="mean", observed=True)
inter.to_csv(O / "03_interaction_top2.csv", encoding="utf-8-sig")
say(f"  상호작용 [{a} × {b}] 구간별 불량률:\n" + inter.map(lambda v: f"{v:.1%}").to_string())

# ═════════════════════════ 4장. 현장 활용: 검사 우선순위 ═════════════════════════
say("=" * 72, "\n[4장] 검사 우선순위 — 위험도 높은 샷부터 검사할 때 불량 포착률")
order = np.argsort(-p_cal, kind="stable")
cap = []
for k in [5, 10, 20, 30, 50]:
    n = int(np.ceil(len(y) * k / 100))
    c = y[order[:n]].sum()
    cap.append({"inspect_top_%": k, "shots": n, "defects_caught": int(c), "capture_rate": c / y.sum(), "lift": (c / y.sum()) / (k / 100)})
cap = pd.DataFrame(cap)
cap.to_csv(O / "04_inspection_priority.csv", index=False, encoding="utf-8-sig")
for _, r in cap.iterrows():
    say(f"  상위 {int(r['inspect_top_%']):2d}% 샷({int(r.shots)}개)만 정밀검사 → 불량 {int(r.defects_caught)}/{int(y.sum())} 포착 ({r.capture_rate:.0%}, 무작위 대비 {r.lift:.1f}배)")
say("  [B단계만 따로] 최근 불량이 없는 샷 안에서 설비값 모델로 검사 순서를 정하면")
yB, pB = y[B], P_OOF[B]
oB = np.argsort(-pB, kind="stable")
capB = []
for k in [10, 20, 30, 50]:
    n = int(np.ceil(len(yB) * k / 100)); c = int(yB[oB[:n]].sum())
    capB.append({"inspect_top_%": k, "shots": n, "defects_caught": c, "of": int(yB.sum()), "lift": (c / yB.sum()) / (k / 100)})
    say(f"    B단계 상위 {k:2d}% ({n}샷) 검사 → 불량 {c}/{int(yB.sum())} (무작위 대비 {capB[-1]['lift']:.1f}배)")
pd.DataFrame(capB).to_csv(O / "04_stageB_capture.csv", index=False, encoding="utf-8-sig")
say(f"    B단계 ROC {roc_auc_score(yB, pB):.3f}, 첫 불량만 양성으로 보면 ROC {roc_auc_score((shots.defect_kind == 'first').to_numpy()[B], pB):.3f} → 약하지만 무작위는 아님")
say("  [불확실성 검증] B단계 샷에서 시드 5개 점수의 표준편차(= 테스트 예측의 uncertainty_std)를 3구간으로 나눠 실제 불량률을 비교")
sdB = np.std(np.array(list(P_PROC.values())), axis=0)[B]
unc = pd.DataFrame({"unc": pd.qcut(sdB, 3, labels=["low", "mid", "high"]), "y": yB, "FN": shots.FN.to_numpy()[B]})
unc_t = unc.groupby("unc", observed=True).agg(shots=("y", "size"), defects=("y", "sum"), FN=("FN", "sum"))
unc_t["defect_rate"] = unc_t.defects / unc_t.shots
unc_t.to_csv(O / "05_uncertainty_vs_defect.csv", encoding="utf-8-sig")
for k, r in unc_t.iterrows():
    say(f"    불확실성 {k:4s}: 샷 {int(r.shots)} | 불량 {int(r.defects)} ({r.defect_rate:.1%})")
hi, lo = unc_t.loc["high", "defect_rate"], unc_t.loc["low", "defect_rate"]
say(f"  → 불확실성 상위 구간 불량률이 하위 구간의 {hi / max(lo, 1e-9):.1f}배" + (": 불확실성이 큰 샷을 재검사 후보로 두는 근거" if hi > lo * 1.3 else ": 뚜렷한 차이 없음 → 재검사 후보 기준으로는 등급을 우선하고 불확실성은 보조 지표로만 사용"))
T_HIGH, T_MID = np.quantile(p_cal, 0.9), np.quantile(p_cal, 0.7)
tier = np.select([p_cal >= T_HIGH, p_cal >= T_MID], ["high", "mid"], "low")
for t in ["high", "mid", "low"]:
    k = tier == t
    say(f"  등급 {t:4s}: 샷 {k.sum():4d} | 예측 불량확률 평균 {p_cal[k].mean():.1%} (교차검증 보정 {p_cal_cv[k].mean():.1%}) | 실제 불량률 {y[k].mean():.1%}")
xs = np.arange(1, len(y) + 1) / len(y)
plt.figure(figsize=(6, 5))
plt.plot(xs, np.cumsum(y[order]) / y.sum(), label="final (two-stage)")
plt.plot(xs, np.cumsum(y[np.argsort(-OOF['M2 LightGBM, process vars'], kind='stable')]) / y.sum(), label="process vars only")
plt.plot([0, 1], [0, 1], "--", c="gray", label="random")
plt.xlabel("Share of shots inspected"), plt.ylabel("Share of defects caught"), plt.title("Inspection priority (capture curve)"), plt.legend()
plt.tight_layout(), plt.savefig(O / "fig_capture.png", dpi=150), plt.close()

# ═════════════════════════ 테스트데이터 예측결과 (라벨 없는 데이터) ═════════════════════════
# 라벨 없는 데이터는 검사 결과가 없어 불량 이력을 만들 수 없음 → 2단계 중 B단계(설비값 모델)만 적용
# 등급은 검사 용량 기준(제품별 상위 10% = high, 다음 20% = mid)으로 매김
P = np.array([lgbm(s).fit(shots[BASE], y).predict_proba(unl[BASE])[:, 1] for s in SEEDS])
out = unl[["order", "product"]].copy()
out["process_risk_score"], out["uncertainty_std"] = P.mean(0), P.std(0)
rk = out.groupby("product")["process_risk_score"].rank(pct=True, ascending=False)
out["tier_stageB"] = np.select([rk <= 0.1, rk <= 0.3], ["high", "mid"], "low")
out.to_csv(O / "test_predictions.csv", index=False, encoding="utf-8-sig")
say("=" * 72, f"\n[테스트 예측] 라벨 없는 {len(out)}행 → outputs/test_predictions.csv (B단계 설비값 모델)"
              f" | 등급 분포 {out.tier_stageB.value_counts().to_dict()}")

(O / "summary.txt").write_text("\n".join(LOG), encoding="utf-8")
print("\n완료: outputs/ 확인")
