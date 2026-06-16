"""
===================================================================
Credit Default Prediction  v3  —  LightGBM + XGBoost + CatBoost
===================================================================
Задача : Gini = 2 * ROC-AUC - 1
"""

import gc
import warnings
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import lightgbm as lgb
import xgboost as xgb

from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

warnings.filterwarnings("ignore")


# ════════════════════════════════════════════════════════════════
#  КОНФИГУРАЦИЯ
# ════════════════════════════════════════════════════════════════

PATHS = dict(
    train_data   = "train_data.parquet",
    test_data    = "test_data.parquet",
    train_target = "train_target.csv",
    sample_sub   = "sample_submission_CREDIT.csv",
    output       = "submission.csv",
)

BASE_RATE    = 0.0355
N_FOLDS      = 5
RANDOM_STATE = 42
POS_WEIGHT   = round((1 - BASE_RATE) / BASE_RATE, 1)   # ~27

LGBM_PARAMS = dict(
    objective         = "binary",
    metric            = "auc",
    boosting_type     = "gbdt",
    num_leaves        = 255,
    max_depth         = -1,
    learning_rate     = 0.02,
    n_estimators      = 8000,
    min_child_samples = 20,
    feature_fraction  = 0.70,
    bagging_fraction  = 0.70,
    bagging_freq      = 5,
    reg_alpha         = 0.05,
    reg_lambda        = 0.05,
    scale_pos_weight  = POS_WEIGHT,
    random_state      = RANDOM_STATE,
    n_jobs            = -1,
    verbose           = -1,
)

XGB_PARAMS = dict(
    objective             = "binary:logistic",
    eval_metric           = "auc",
    max_depth             = 8,
    learning_rate         = 0.02,
    n_estimators          = 8000,
    subsample             = 0.70,
    colsample_bytree      = 0.70,
    min_child_weight      = 10,
    scale_pos_weight      = POS_WEIGHT,
    reg_alpha             = 0.05,
    reg_lambda            = 0.05,
    random_state          = RANDOM_STATE,
    n_jobs                = -1,
    verbosity             = 0,
    tree_method           = "hist",
    early_stopping_rounds = 200,   # XGBoost >= 1.6: передаётся в конструктор
)

CATBOOST_PARAMS = dict(
    iterations            = 5000,
    learning_rate         = 0.02,
    depth                 = 8,
    loss_function         = "Logloss",
    eval_metric           = "AUC",
    scale_pos_weight      = POS_WEIGHT,
    random_seed           = RANDOM_STATE,
    thread_count          = -1,
    verbose               = 0,
    early_stopping_rounds = 200,
)

# Колонки разбиты на три прохода:
# каждый читается → агрегируется → удаляется из памяти
COLS_PASS1 = (       # просрочки, утилизация, время
    ["id", "rn",
     "pre_since_opened", "pre_since_confirmed",
     "pre_pterm", "pre_fterm", "pre_till_pclose", "pre_till_fclose",
     "pre_loans5", "pre_loans530", "pre_loans3060",
     "pre_loans6090", "pre_loans90",
     "is_zero_loans5", "is_zero_loans530", "is_zero_loans3060",
     "is_zero_loans6090", "is_zero_loans90",
     "pre_util", "pre_over2limit", "pre_maxover2limit",
     "is_zero_util", "is_zero_over2limit", "is_zero_maxover2limit"]
)

COLS_PASS2 = (       # платёжные последовательности (enc_paym)
    ["id", "rn"]
    + [f"enc_paym_{i}" for i in range(25)]
)

COLS_PASS3 = (       # кредитные метрики и категориальные признаки
    ["id", "rn",
     "pre_loans_credit_limit", "pre_loans_next_pay_summ",
     "pre_loans_outstanding", "pre_loans_total_overdue",
     "pre_loans_max_overdue_sum", "pre_loans_credit_cost_rate",
     "enc_loans_account_holder_type", "enc_loans_credit_status",
     "enc_loans_credit_type", "enc_loans_account_cur",
     "pclose_flag", "fclose_flag"]
)


# ════════════════════════════════════════════════════════════════
#  ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ЗАГРУЗКИ
# ════════════════════════════════════════════════════════════════

def _load_cols(path: str, wanted_cols: list) -> pd.DataFrame:
    """
    Читает только нужные колонки из parquet.
    Сразу приводит к минимальным типам (int8/int16).
    """
    schema_names = set(pq.read_schema(path).names)
    cols = [c for c in wanted_cols if c in schema_names]

    table = pq.ParquetFile(path).read(columns=cols)
    df = table.to_pandas()
    del table

    df["id"] = df["id"].astype("int32")
    for col in df.columns:
        if col == "id":
            continue
        mn, mx = int(df[col].min()), int(df[col].max())
        if -128 <= mn and mx <= 127:
            df[col] = df[col].astype("int8")
        elif -32768 <= mn and mx <= 32767:
            df[col] = df[col].astype("int16")

    mb = df.memory_usage(deep=True).sum() / 1e6
    print(f"    {len(cols)} колонок, {df.shape[0]:,} строк — {mb:.0f} MB")
    return df


def _mb(df: pd.DataFrame) -> str:
    return f"{df.memory_usage(deep=True).sum() / 1e6:.0f} MB"


# ════════════════════════════════════════════════════════════════
#  ПРОХОД 1 — просрочки, утилизация, время
# ════════════════════════════════════════════════════════════════

ZERO_FLAGS   = ["is_zero_loans5", "is_zero_loans530", "is_zero_loans3060",
                "is_zero_loans6090", "is_zero_loans90",
                "is_zero_util", "is_zero_over2limit", "is_zero_maxover2limit"]
OVERDUE_CNTS = ["pre_loans5", "pre_loans530", "pre_loans3060",
                "pre_loans6090", "pre_loans90"]
UTIL_COLS    = ["pre_util", "pre_over2limit", "pre_maxover2limit"]
TIME_COLS    = ["pre_since_opened", "pre_since_confirmed",
                "pre_pterm", "pre_fterm", "pre_till_pclose", "pre_till_fclose"]


def _risk_index(df: pd.DataFrame) -> pd.Series:
    return (
        (1 - df["is_zero_loans90"].astype(np.int16))   * 8 +
        (1 - df["is_zero_loans6090"].astype(np.int16)) * 5 +
        (1 - df["is_zero_loans3060"].astype(np.int16)) * 3 +
        (1 - df["is_zero_loans530"].astype(np.int16))  * 1 +
        df["pre_maxover2limit"].astype(np.int16)
    )


def _pass1(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Загружает колонки просрочек/утилизации, агрегирует, удаляет сырые данные.
    Возвращает:
      feats   — DataFrame с признаками прохода 1
      risk_map — DataFrame {id, worst_rn, last_rn, first_rn}
                 используется в проходах 2 и 3 для фильтрации нужных строк
    """
    print("  Проход 1: просрочки + утилизация")
    df = _load_cols(path, COLS_PASS1)
    avail = set(df.columns)

    def cols(lst):
        return [c for c in lst if c in avail]

    df = df.sort_values(["id", "rn"])
    df["_risk"] = _risk_index(df)

    # ── Строим risk_map: worst / last / first rn для каждого id ──
    g = df.groupby("id")
    risk_map = pd.DataFrame({
        "worst_rn": g["_risk"].idxmax().map(df["rn"]),  # rn худшего кредита
        "last_rn":  g["rn"].max(),
        "first_rn": g["rn"].min(),
    })
    # Сохраняем индексы строк (нужны для выборки worst-row в этом же df)
    idx_worst = g["_risk"].idxmax()
    idx_last  = g["rn"].idxmax()
    idx_first = g["rn"].idxmin()

    # ── Агрегаты по всей истории ──────────────────────────────────
    agg: dict = {"n_credits": ("rn", "max")}
    for c in cols(ZERO_FLAGS):
        agg[f"{c}__min"]  = (c, "min")
        agg[f"{c}__mean"] = (c, "mean")
        agg[f"{c}__std"]  = (c, "std")
    for c in cols(OVERDUE_CNTS):
        agg[f"{c}__max"] = (c, "max")
        agg[f"{c}__sum"] = (c, "sum")
        agg[f"{c}__std"] = (c, "std")
    for c in cols(UTIL_COLS):
        agg[f"{c}__max"]  = (c, "max")
        agg[f"{c}__mean"] = (c, "mean")
        agg[f"{c}__std"]  = (c, "std")
    for c in cols(TIME_COLS):
        agg[f"{c}__min"] = (c, "min")
        agg[f"{c}__max"] = (c, "max")
    df_agg = df.groupby("id").agg(**agg)

    # ── Признаки худшего / последнего / первого кредита ──────────
    worst_cols = cols(["is_zero_loans90", "is_zero_loans6090", "is_zero_loans3060",
                       "pre_loans90", "pre_loans6090", "pre_loans3060",
                       "pre_maxover2limit", "pre_over2limit", "pre_util"])
    last_cols  = cols(["is_zero_loans90", "is_zero_loans6090", "is_zero_loans3060",
                       "pre_util", "pre_maxover2limit", "pre_since_opened"])
    first_cols = cols(["is_zero_loans90", "is_zero_loans6090", "pre_since_opened"])

    df_worst = df.loc[idx_worst].set_index("id")[worst_cols].copy()
    df_worst.columns = [f"worst__{c}" for c in df_worst.columns]

    df_last = df.loc[idx_last].set_index("id")[last_cols].copy()
    df_last.columns = [f"last__{c}" for c in df_last.columns]

    df_first = df.loc[idx_first].set_index("id")[first_cols].copy()
    df_first.columns = [f"first__{c}" for c in df_first.columns]

    # ── Тренд: первая половина истории vs вторая ──────────────────
    trend_cols = cols(["is_zero_loans90", "is_zero_loans6090",
                       "pre_maxover2limit", "pre_util",
                       "pre_loans90", "pre_loans6090"])
    n_cred = df.groupby("id")["rn"].transform("count")
    half   = n_cred // 2
    df["_order"] = df.groupby("id").cumcount()
    agg_first_h = df[df["_order"] <  half].groupby("id")[trend_cols].mean()
    agg_second_h = df[df["_order"] >= half].groupby("id")[trend_cols].mean()
    trend = (agg_second_h - agg_first_h)
    trend.columns = [f"trend_h2h1__{c}" for c in trend.columns]

    # ── Вариативность ─────────────────────────────────────────────
    std_base = df.groupby("id")[trend_cols].std()
    std_base.columns = [f"std__{c}" for c in std_base.columns]

    # ── Окна: последние 3 и 5 кредитов ───────────────────────────
    win_cols = cols(["is_zero_loans90", "is_zero_loans6090",
                     "pre_maxover2limit", "pre_util", "pre_loans90"])
    for n_win, prefix in [(3, "win3"), (5, "win5")]:
        tail = df.sort_values(["id", "rn"]).groupby("id").tail(n_win)
        w_agg = tail.groupby("id")[win_cols].agg(["min", "max", "mean"])
        w_agg.columns = [f"{prefix}__{a}__{b}" for a, b in w_agg.columns]
        df_agg = df_agg.join(w_agg, how="left")

    # ── Сборка прохода 1 ─────────────────────────────────────────
    feats = (
        df_agg
        .join(df_worst,  how="left")
        .join(df_last,   how="left")
        .join(df_first,  how="left")
        .join(trend,     how="left")
        .join(std_base,  how="left")
        .reset_index()
    )

    del df, df_agg, df_worst, df_last, df_first, trend, std_base
    gc.collect()
    print(f"    → {feats.shape[1]} признаков")
    return feats, risk_map


# ════════════════════════════════════════════════════════════════
#  ПРОХОД 2 — платёжные последовательности (enc_paym)
# ════════════════════════════════════════════════════════════════

def _pass2(path: str, risk_map: pd.DataFrame) -> pd.DataFrame:
    """
    Загружает enc_paym_0..24, анализирует каждый как временной ряд,
    агрегирует по клиенту, удаляет сырые данные.
    """
    print("  Проход 2: платёжные последовательности (enc_paym)")
    df = _load_cols(path, COLS_PASS2)
    avail_paym = [c for c in [f"enc_paym_{i}" for i in range(25)] if c in df.columns]
    T = len(avail_paym)

    df = df.sort_values(["id", "rn"])
    ids = df["id"].values
    P   = df[avail_paym].values.astype(np.int16)
    N   = len(P)
    bad = (P > 0).astype(np.int16)
    sev = P.astype(np.float32)

    result = {}

    # Окна: количество плохих месяцев и тяжесть status=3
    for end in [1, 3, 6, 12, 18, min(24, T)]:
        result[f"bad_{end}m"]  = bad[:, :end].sum(axis=1).astype(np.int16)
        result[f"sev_{end}m"]  = sev[:, :end].sum(axis=1)
        result[f"sev3_{end}m"] = (P[:, :end] == 3).sum(axis=1).astype(np.int16)

    # Тренд: последние 6 мес. vs предыдущие 6
    if T >= 12:
        result["trend_6v6_bad"]  = (bad[:, :6].sum(axis=1).astype(float) -
                                    bad[:, 6:12].sum(axis=1).astype(float))
        result["trend_6v6_sev3"] = ((P[:, :6] == 3).sum(axis=1).astype(float) -
                                    (P[:, 6:12] == 3).sum(axis=1).astype(float))

    # Счётчик каждого статуса
    for s in [1, 2, 3]:
        result[f"cnt_status{s}"] = (P == s).sum(axis=1).astype(np.int16)

    # Стрик — максимальная серия подряд идущих плохих месяцев
    max_streak = np.zeros(N, dtype=np.int16)
    cur_streak = np.zeros(N, dtype=np.int16)
    for j in range(T):
        cur_streak = np.where(bad[:, j] == 1, cur_streak + 1, 0).astype(np.int16)
        np.maximum(max_streak, cur_streak, out=max_streak)
    result["max_bad_streak"] = max_streak

    # Флаг восстановления: раньше было плохо, сейчас чисто
    if T >= 6:
        result["recovery_flag"] = (
            (bad[:, 6:].sum(axis=1) > 0) & (bad[:, :3].sum(axis=1) == 0)
        ).astype(np.int16)

    # Переходы: хорошо→плохо и плохо→хорошо
    g2b = np.zeros(N, dtype=np.int16)
    b2g = np.zeros(N, dtype=np.int16)
    for j in range(T - 1):
        prev_bad, curr_bad = bad[:, j + 1], bad[:, j]
        g2b += ((prev_bad == 0) & (curr_bad == 1)).astype(np.int16)
        b2g += ((prev_bad == 1) & (curr_bad == 0)).astype(np.int16)
    result["transitions_g2b"] = g2b
    result["transitions_b2g"] = b2g
    result["net_recovery"]     = (b2g - g2b).astype(np.int16)

    # Позиция первого и последнего плохого месяца
    first_bad = np.full(N, T,  dtype=np.int16)
    last_bad  = np.full(N, -1, dtype=np.int16)
    for j in range(T):
        is_b = bad[:, j] == 1
        first_bad = np.where(is_b & (first_bad > j), j, first_bad)
        last_bad  = np.where(is_b & (last_bad  < j), j, last_bad)
    result["first_bad_month"] = first_bad
    result["last_bad_month"]  = last_bad
    result["bad_span"]        = (last_bad - first_bad).clip(-1)

    # Взвешенная тяжесть (recent-bias)
    weights = np.array([1.0 / (i + 1) for i in range(T)], dtype=np.float32)
    result["weighted_sev"] = (sev * weights).sum(axis=1)

    feat_df = pd.DataFrame(result)
    feat_df["id"] = ids
    agg = feat_df.groupby("id").agg(["max", "mean", "std"])
    agg.columns = [f"paym__{a}__{b}" for a, b in agg.columns]

    # Признаки последнего и худшего кредита по enc_paym
    paym_pick = avail_paym[:5]
    df_last  = df.loc[df.groupby("id")["rn"].idxmax()].set_index("id")[paym_pick].copy()
    df_last.columns = [f"last__{c}" for c in df_last.columns]

    # Худший кредит по worst_rn из risk_map
    worst_rn = risk_map["worst_rn"].rename("worst_rn")
    df_w = df.merge(worst_rn.reset_index(), on="id", how="inner")
    df_w = df_w[df_w["rn"] == df_w["worst_rn"]].drop_duplicates("id").set_index("id")
    df_w = df_w[paym_pick].copy()
    df_w.columns = [f"worst__{c}" for c in df_w.columns]

    feats = agg.join(df_last, how="left").join(df_w, how="left").reset_index()

    del df, feat_df, agg, df_last, df_w, P, bad, sev, result
    gc.collect()
    print(f"    → {feats.shape[1]} признаков")
    return feats


# ════════════════════════════════════════════════════════════════
#  ПРОХОД 3 — кредитные метрики и категориальные признаки
# ════════════════════════════════════════════════════════════════

CAT_COLS    = ["enc_loans_account_holder_type", "enc_loans_credit_status",
               "enc_loans_credit_type", "enc_loans_account_cur"]
CREDIT_COLS = ["pre_loans_credit_limit", "pre_loans_outstanding",
               "pre_loans_max_overdue_sum", "pre_loans_credit_cost_rate",
               "pre_loans_next_pay_summ"]


def _pass3(path: str, risk_map: pd.DataFrame) -> pd.DataFrame:
    """
    Загружает кредитные метрики и категориальные признаки,
    агрегирует, удаляет сырые данные.
    """
    print("  Проход 3: кредитные метрики + категории")
    df = _load_cols(path, COLS_PASS3)
    avail = set(df.columns)

    def cols(lst):
        return [c for c in lst if c in avail]

    df = df.sort_values(["id", "rn"])

    agg: dict = {}
    for c in cols(CAT_COLS):
        agg[f"{c}__min"]  = (c, "min")
        agg[f"{c}__max"]  = (c, "max")
        agg[f"{c}__mean"] = (c, "mean")
    for c in cols(CREDIT_COLS):
        agg[f"{c}__max"]  = (c, "max")
        agg[f"{c}__mean"] = (c, "mean")
        agg[f"{c}__std"]  = (c, "std")
    for c in cols(["pclose_flag", "fclose_flag"]):
        agg[f"{c}__sum"]  = (c, "sum")
        agg[f"{c}__mean"] = (c, "mean")
    df_agg = df.groupby("id").agg(**agg)

    # Признаки худшего / последнего кредита
    worst_rn = risk_map["worst_rn"].rename("worst_rn")
    meta_pick = cols(["enc_loans_credit_status", "enc_loans_credit_type",
                      "pre_loans_credit_limit", "pre_loans_max_overdue_sum"])
    df_w = df.merge(worst_rn.reset_index(), on="id", how="inner")
    df_w = df_w[df_w["rn"] == df_w["worst_rn"]].drop_duplicates("id").set_index("id")
    df_w = df_w[meta_pick].copy()
    df_w.columns = [f"worst__{c}" for c in df_w.columns]

    df_last = df.loc[df.groupby("id")["rn"].idxmax()].set_index("id")
    last_pick = cols(["enc_loans_credit_status", "enc_loans_credit_type",
                      "pre_loans_credit_limit", "pre_loans_outstanding"])
    df_last = df_last[last_pick].copy()
    df_last.columns = [f"last__{c}" for c in df_last.columns]

    feats = (
        df_agg
        .join(df_w,    how="left")
        .join(df_last, how="left")
        .reset_index()
    )

    del df, df_agg, df_w, df_last
    gc.collect()
    print(f"    → {feats.shape[1]} признаков")
    return feats


# ════════════════════════════════════════════════════════════════
#  ГЛАВНАЯ ФУНКЦИЯ ИНЖЕНЕРИИ ПРИЗНАКОВ
# ════════════════════════════════════════════════════════════════

def engineer_features(path: str) -> pd.DataFrame:
    """
    Принимает путь к parquet-файлу (не DataFrame!).
    Выполняет три прохода с разными подмножествами колонок —
    каждый раз сырые данные загружаются и сразу удаляются после агрегации.
    """
    print(f"  Файл: {path}")

    feats1, risk_map = _pass1(path)
    feats2           = _pass2(path, risk_map)
    feats3           = _pass3(path, risk_map)

    # ── Слияние результатов трёх проходов ────────────────────────
    features = feats1.merge(feats2, on="id", how="left") \
                     .merge(feats3, on="id", how="left")

    del feats1, feats2, feats3, risk_map
    gc.collect()

    # ── Производные признаки и взаимодействия ────────────────────
    def f(col, default=0.0):
        return features[col].astype(float) if col in features.columns \
               else pd.Series(default, index=features.index)

    features["overdue_severity"] = (
        (f("is_zero_loans90__min",  1) == 0).astype(int) * 8 +
        (f("is_zero_loans6090__min", 1) == 0).astype(int) * 5 +
        (f("is_zero_loans3060__min", 1) == 0).astype(int) * 3 +
        (f("is_zero_loans530__min",  1) == 0).astype(int) * 1
    )
    sum_ovd = [c for c in features.columns
               if c.endswith("__sum") and any(o in c for o in OVERDUE_CNTS)]
    features["total_overdue_events"] = features[sum_ovd].sum(axis=1)
    features["frac_90d_overdue"]     = 1.0 - f("is_zero_loans90__mean", 1)
    features["severity_x_util"]      = (
        features["overdue_severity"].astype(float) * f("pre_maxover2limit__max") / 19.0
    )
    features["severity_x_ncredits"]  = (
        features["overdue_severity"].astype(float) * np.log1p(f("n_credits"))
    )
    features["instability"] = f("pre_maxover2limit__std") + f("pre_util__std")
    if "pre_till_pclose__max" in features.columns and \
       "pre_till_fclose__max" in features.columns:
        features["close_delay"] = (
            f("pre_till_fclose__max") - f("pre_till_pclose__max")
        )
    if "paym__bad_span__max" in features.columns:
        features["chronic_risk"] = (
            f("paym__bad_span__max") * features["overdue_severity"]
        )

    print(f"  Итого: {features.shape[0]:,} клиентов × {features.shape[1]} признаков")
    return features


# ════════════════════════════════════════════════════════════════
#  МОДЕЛИ
# ════════════════════════════════════════════════════════════════

def run_cv(model_fn, X: pd.DataFrame, y: pd.Series,
           label: str) -> tuple[np.ndarray, list]:
    skf    = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof    = np.zeros(len(y))
    models = []

    for fold, (trn_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X.iloc[trn_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[trn_idx], y.iloc[val_idx]
        model = model_fn(X_tr, y_tr, X_val, y_val)
        oof[val_idx] = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, oof[val_idx])
        print(f"    {label} Fold {fold}: AUC={auc:.5f}  Gini={2*auc-1:.5f}")
        models.append(model)

    oof_auc = roc_auc_score(y, oof)
    print(f"  ► {label} OOF  AUC={oof_auc:.5f}  Gini={2*oof_auc-1:.5f}\n")
    return oof, models


def make_lgbm(X_tr, y_tr, X_val, y_val):
    m = lgb.LGBMClassifier(**LGBM_PARAMS)
    m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
          callbacks=[lgb.early_stopping(200, verbose=False),
                     lgb.log_evaluation(1000)])
    return m


def make_xgb(X_tr, y_tr, X_val, y_val):
    m = xgb.XGBClassifier(**XGB_PARAMS)
    m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return m


def make_catboost(X_tr, y_tr, X_val, y_val):
    m = CatBoostClassifier(**CATBOOST_PARAMS)
    m.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True)
    return m


def rank_blend(oof_list, pred_list, y) -> tuple[np.ndarray, list]:
    n_train, n_test = len(y), len(pred_list[0])
    ginis   = [2 * roc_auc_score(y, o) - 1 for o in oof_list]
    total   = sum(ginis)
    weights = [g / total for g in ginis]

    print("  Веса блендирования (OOF Gini):")
    for name, g, w in zip(["LGB", "XGB", "CAT"], ginis, weights):
        print(f"    {name}: Gini={g:.5f}  вес={w:.3f}")

    blended = sum(w * rankdata(p) / n_test
                  for p, w in zip(pred_list, weights))
    oof_blend = sum(w * rankdata(o) / n_train
                    for o, w in zip(oof_list, weights))
    blend_auc = roc_auc_score(y, oof_blend)
    print(f"\n  ► BLEND OOF  AUC={blend_auc:.5f}  Gini={2*blend_auc-1:.5f}")
    return blended, weights


# ════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════

def main():
    print("=" * 62)
    print("  Credit Default Prediction v3 — LGB + XGB + CatBoost")
    print("=" * 62)

    # ── 1. Feature Engineering (многопроходная загрузка) ─────────
    print("\n[1/5]  Тестовые данные — Feature Engineering")
    test_features = engineer_features(PATHS["test_data"])

    print("\n[2/5]  Обучающие данные — Feature Engineering")
    train_features = engineer_features(PATHS["train_data"])

    # ── 2. Матрицы признаков ─────────────────────────────────────
    print("\n[3/5]  Подготовка матриц признаков")
    target_df = pd.read_csv(PATHS["train_target"])
    print(f"  Метки: {len(target_df):,}, доля дефолтов={target_df['flag'].mean():.4f}")

    train_data = train_features.merge(target_df, on="id", how="inner")
    feat_cols  = [c for c in train_features.columns if c != "id"]

    X_train = train_data[feat_cols].fillna(-1).astype("float32")
    y_train = train_data["flag"].astype("int8")

    for c in feat_cols:
        if c not in test_features.columns:
            test_features[c] = -1
    X_test = test_features[feat_cols].fillna(-1).astype("float32")

    print(f"  X_train: {X_train.shape}  |  X_test: {X_test.shape}")

    # ── 3. Обучение ───────────────────────────────────────────────
    print(f"\n[4/5]  Обучение ({N_FOLDS}-fold CV)")

    print("  — LightGBM —")
    lgb_oof, lgb_models = run_cv(make_lgbm, X_train, y_train, "LGB")
    lgb_imp = pd.concat(
        [pd.Series(m.feature_importances_, index=feat_cols) for m in lgb_models],
        axis=1
    ).mean(axis=1).sort_values(ascending=False)
    print("  Топ-20 признаков LightGBM:")
    for i, (name, score) in enumerate(lgb_imp.head(20).items(), 1):
        print(f"    {i:>2}. {name:<55} {score:.0f}")

    print("\n  — XGBoost —")
    xgb_oof, xgb_models = run_cv(make_xgb, X_train, y_train, "XGB")

    print("\n  — CatBoost —")
    cat_oof, cat_models = run_cv(make_catboost, X_train, y_train, "CAT")

    # ── 4. Блендирование ─────────────────────────────────────────
    lgb_test = np.mean([m.predict_proba(X_test)[:, 1] for m in lgb_models], axis=0)
    xgb_test = np.mean([m.predict_proba(X_test)[:, 1] for m in xgb_models], axis=0)
    cat_test = np.mean([m.predict_proba(X_test)[:, 1] for m in cat_models], axis=0)

    print("\n  — Ранговое блендирование —")
    test_preds, _ = rank_blend(
        oof_list  = [lgb_oof, xgb_oof, cat_oof],
        pred_list = [lgb_test, xgb_test, cat_test],
        y         = y_train,
    )

    # ── 5. Submission ─────────────────────────────────────────────
    print("\n[5/5]  Submission")
    sample_sub = pd.read_csv(PATHS["sample_sub"])
    pred_df    = pd.DataFrame({"id": test_features["id"].values,
                               "flag": test_preds})
    submission = sample_sub[["id"]].merge(pred_df, on="id", how="left")
    submission["flag"] = submission["flag"].fillna(BASE_RATE).round(6)
    submission.to_csv(PATHS["output"], index=False)

    print(f"\n    Сохранён: {PATHS['output']}")
    print(f"     Строк: {len(submission):,}  |  "
          f"Mean: {submission['flag'].mean():.4f}  |  "
          f">0.5: {(submission['flag']>0.5).sum():,}")
    print(f"\n  Первые строки:")
    print(submission.head(10).to_string(index=False))


if __name__ == "__main__":
    main()