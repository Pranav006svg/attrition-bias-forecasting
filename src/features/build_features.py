"""Feature matrices for the baselines (pre-enrollment fields only).

All vocabularies, category levels and the TF-IDF model are fit on the train
split only and then applied to val/test.

Trial features: categorical design fields, numeric fields, multi-hot MeSH
branches, TF-IDF (300) over eligibility criteria + brief summary.
Arm features: arm type, multi-hot intervention types, number of interventions.
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.data.build_benchmark import ARM_FEATURES, TRIAL_FEATURES

TRIAL_CAT = ["phase", "allocation", "masking", "intervention_model", "primary_purpose",
             "sponsor_class", "sex", "healthy_volunteers"]
TRIAL_NUM = ["n_arms", "min_age_years", "max_age_years", "start_year"]
ARM_CAT = ["arm_type"]
TFIDF_MAX_FEATURES = 300

assert set(TRIAL_CAT + TRIAL_NUM) <= set(TRIAL_FEATURES)
assert set(ARM_CAT) <= set(ARM_FEATURES)


def _as_cat_str(s: pd.Series) -> pd.Series:
    """Categorical values as plain strings; missing -> 'NA' level."""
    return s.astype(object).where(s.notna(), "NA").astype(str)


def _multi_hot(series: pd.Series, vocab: list, prefix: str) -> pd.DataFrame:
    sets = series.fillna("").map(lambda s: set(s.split("|")) if s else set())
    return pd.DataFrame({f"{prefix}{v}": sets.map(lambda x, v=v: int(v in x)) for v in vocab}, index=series.index)


class FeatureBuilder:
    def fit(self, trials_train: pd.DataFrame, arms_train: pd.DataFrame):
        self.cat_levels = {c: sorted(_as_cat_str(trials_train[c]).unique()) for c in TRIAL_CAT}
        self.arm_levels = {c: sorted(_as_cat_str(arms_train[c]).unique()) for c in ARM_CAT}
        self.mesh_vocab = sorted({b for s in trials_train.mesh_branches for b in s.split("|")})
        self.itype_vocab = sorted({t for s in arms_train.intervention_types.fillna("") for t in s.split("|") if t})
        self.tfidf = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, sublinear_tf=True, min_df=5,
                                     stop_words="english", dtype=np.float32)
        self.tfidf.fit(self._text(trials_train))
        return self

    @staticmethod
    def _text(t: pd.DataFrame) -> pd.Series:
        return t.eligibility_text.fillna("") + " " + t.brief_summary.fillna("")

    def trial_matrix(self, t: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(index=t.index)
        for c in TRIAL_CAT:
            X[c] = pd.Categorical(_as_cat_str(t[c]), categories=self.cat_levels[c])
        for c in TRIAL_NUM:
            X[c] = t[c].astype(float)
        X = X.join(_multi_hot(t.mesh_branches, self.mesh_vocab, "mesh_"))
        tf = self.tfidf.transform(self._text(t)).toarray()
        X = X.join(pd.DataFrame(tf, index=t.index, columns=[f"tfidf_{w}" for w in self.tfidf.get_feature_names_out()]))
        X.insert(0, "nct_id", t.nct_id.values)
        return X

    def arm_matrix(self, a: pd.DataFrame, trial_X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(index=a.index)
        for c in ARM_CAT:
            X[c] = pd.Categorical(_as_cat_str(a[c]), categories=self.arm_levels[c])
        X = X.join(_multi_hot(a.intervention_types, self.itype_vocab, "itype_"))
        X["n_interventions"] = a.intervention_names.fillna("").map(lambda s: len(s.split("|")) if s else 0)
        X["nct_id"] = a.nct_id.values
        X = X.merge(trial_X, on="nct_id", how="left").set_index(a.index)
        return X.drop(columns="nct_id")
