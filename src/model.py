"""RI model: logistic regression with class balancing and an F2-chosen alert threshold.

Trained on 1982-2012 storms, threshold chosen on 2013-2018 storms (F2 score, so
missing an RI storm counts more than an extra alert), tested once on 2019-2025.
"""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, precision_score, recall_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from prep import FEATURES


def make_model():
    pre = ColumnTransformer([("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                                                    StandardScaler()), FEATURES)])
    return make_pipeline(pre, LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000))


def fit(train, val):
    m = make_model().fit(train[FEATURES], train.RI)
    p = m.predict_proba(val[FEATURES])[:, 1]
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    f2 = [fbeta_score(val.RI, p >= th, beta=2, zero_division=0) for th in grid]
    return m, float(grid[int(np.argmax(f2))])


def scores(df, flag):
    y = df.RI.values
    caught = df.assign(f=flag)[lambda x: (x.RI == 1) & (x.f == 1)].storm_id.nunique()
    ri_storms = df[df.RI == 1].storm_id.unique()
    fa = df.assign(f=flag)[lambda x: (x.f == 1) & (x.RI == 0) & ~x.storm_id.isin(ri_storms)].storm_id.nunique()
    return {"precision": precision_score(y, flag), "recall": recall_score(y, flag), "f1": f1_score(y, flag),
            "false_alarm_times": int(((flag == 1) & (y == 0)).sum()),
            "ri_storms_caught": f"{caught} of {len(ri_storms)}",
            "non_ri_storms_with_false_alarm": f"{fa} of {df.storm_id.nunique() - len(ri_storms)}"}
