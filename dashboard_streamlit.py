#run streamlit app: streamlit run dashboard_streamlit.py
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Dropout Risk Predictor", page_icon="🎓", layout="wide")

MODEL_FILE = os.path.join(os.getcwd(), "dropout_risk_xgb_bundle.pkl")

# ---------- UI: banner + responsive CSS ----------
st.markdown(
    """
<style>
.main .block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 1200px; }

.banner {
  background: #F6F8FB;
  border: 1px solid #E6E9EF;
  border-radius: 14px;
  padding: 14px 16px;
  margin-bottom: 12px;
}
.banner h3 { margin: 0 0 8px 0; }
.banner ol { margin: 0; padding-left: 1.15rem; }
.banner li { margin: 4px 0; }

@media (max-width: 900px) {
  .main .block-container { padding-left: 0.8rem; padding-right: 0.8rem; }
}
</style>

<div class="banner">
  <h3>📌 User Manual</h3>
  <ol>
    <li>Fill the fields you know (others can stay as defaults).</li>
    <li>Click <b>Predict</b> to get <b>AT RISK</b> or <b>NOT AT RISK</b>.</li>
    <li>See <b>Top Reasons</b> (what increases vs decreases risk) after prediction.</li>
    <li>Click <b>Reset to Defaults</b> to clear result + restore default values.</li>
  </ol>
</div>
""",
    unsafe_allow_html=True,
)

# ---------- Load bundle ----------
@st.cache_resource
def load_bundle(path: str):
    return joblib.load(path)


def clip_float(v, lo, hi):
    try:
        v = float(v)
    except Exception:
        v = float(lo)
    return float(min(max(v, float(lo)), float(hi)))


def get_num_cat_cols_from_preprocess(preprocess):
    num_cols, cat_cols = [], []
    try:
        for name, trans, cols in preprocess.transformers_:
            if name == "num":
                num_cols = list(cols)
            elif name == "cat":
                cat_cols = list(cols)
    except Exception:
        pass
    return num_cols, cat_cols


def get_imputer_defaults(preprocess, num_cols, cat_cols):
    num_defaults, cat_defaults = {}, {}
    try:
        num_imp = preprocess.named_transformers_["num"].named_steps["imputer"]
        num_defaults = dict(zip(list(num_cols), list(num_imp.statistics_)))
    except Exception:
        pass
    try:
        cat_imp = preprocess.named_transformers_["cat"].named_steps["imputer"]
        cat_defaults = dict(zip(list(cat_cols), list(cat_imp.statistics_)))
    except Exception:
        pass
    return num_defaults, cat_defaults


def get_ohe_categories(preprocess, cat_cols):
    cats = {}
    try:
        ohe = preprocess.named_transformers_["cat"].named_steps["onehot"]
        for col, cat_list in zip(cat_cols, ohe.categories_):
            cats[col] = list(cat_list)
    except Exception:
        pass
    return cats


def init_state_key(key, default_value, options=None):
    if key not in st.session_state:
        st.session_state[key] = default_value
    if options is not None and len(options) > 0 and st.session_state[key] not in options:
        st.session_state[key] = default_value if default_value in options else options[0]


def reset_all(widget_defaults: dict):
    for k, v in widget_defaults.items():
        st.session_state[k] = v
    st.session_state.pop("pred", None)
    st.session_state.pop("shap_pos", None)
    st.session_state.pop("shap_neg", None)
    st.session_state.pop("shap_err", None)
    st.session_state.pop("shap_explainer", None)
    st.rerun()


# ----------------------------
# SHAP helpers (GROUPED by original feature)
# ----------------------------
def parse_feature_name(feat: str, cat_cols: list):
    # returns: (kind, base_col)
    # num__study_load -> ("num","study_load")
    # cat__imd_band_20-30% -> ("cat","imd_band")
    if feat.startswith("num__"):
        return "num", feat[len("num__"):]
    if feat.startswith("cat__"):
        rest = feat[len("cat__"):]
        for c in sorted(cat_cols, key=len, reverse=True):
            prefix = c + "_"
            if rest.startswith(prefix):
                return "cat", c
        # fallback
        return "cat", rest.split("_")[0]
    return "other", feat


def get_shap_explainer(model):
    import shap
    return shap.TreeExplainer(model)


def compute_grouped_local_shap(
    model,
    X_pp_1row,
    feature_names,
    X_raw_1row: pd.DataFrame,
    label_map: dict,
    cat_cols: list,
    top_n=8,
):
    import numpy as _np

    if "shap_explainer" not in st.session_state:
        st.session_state["shap_explainer"] = get_shap_explainer(model)

    explainer = st.session_state["shap_explainer"]

    sv = explainer.shap_values(X_pp_1row)

    # binary outputs sometimes list [class0, class1]
    if isinstance(sv, list):
        shap_vals = sv[1][0]
    else:
        shap_vals = sv
        if hasattr(shap_vals, "shape") and len(shap_vals.shape) == 2:
            shap_vals = shap_vals[0]

    shap_vals = _np.asarray(shap_vals).ravel()

    # group by base feature
    grouped = {}
    for feat, val in zip(feature_names, shap_vals):
        kind, base = parse_feature_name(feat, cat_cols)
        grouped.setdefault(base, 0.0)
        grouped[base] += float(val)

    rows = []
    raw_row = X_raw_1row.iloc[0].to_dict()

    for base, shap_sum in grouped.items():
        label = label_map.get(base, base.replace("_", " ").title())

        # show chosen category value ONLY (so no confusion)
        if base in cat_cols:
            uv = raw_row.get(base, None)
            if uv is None or (isinstance(uv, float) and np.isnan(uv)) or str(uv).strip() == "":
                reason = f"{label} = (default)"
            else:
                reason = f"{label} = {uv}"
        else:
            reason = label

        rows.append({"Reason": reason, "SHAP": shap_sum})

    df = pd.DataFrame(rows)

    # pick top positive/negative contributions (grouped)
    pos = (
        df[df["SHAP"] > 0]
        .sort_values("SHAP", ascending=False)
        .head(top_n)
        .copy()
    )
    neg = (
        df[df["SHAP"] < 0]
        .sort_values("SHAP", ascending=True)  # most negative first
        .head(top_n)
        .copy()
    )

    # display magnitudes only (clean)
    pos["Impact"] = pos["SHAP"].abs().round(4)
    neg["Impact"] = neg["SHAP"].abs().round(4)
    pos = pos[["Reason", "Impact"]]
    neg = neg[["Reason", "Impact"]]

    return pos.reset_index(drop=True), neg.reset_index(drop=True)


# ---------- Title ----------
st.title("🎓 Dropout Risk Predictor")

if not os.path.exists(MODEL_FILE):
    st.error(f"Model file not found: {MODEL_FILE}")
    st.stop()

bundle = load_bundle(MODEL_FILE)
model = bundle["model"]
preprocess = bundle["preprocess"]
raw_cols = bundle.get("raw_feature_columns", [])
if not raw_cols:
    st.error("Bundle missing raw_feature_columns. Re-save bundle with raw_feature_columns.")
    st.stop()

num_cols = bundle.get("num_cols") or []
cat_cols = bundle.get("cat_cols") or []
if not num_cols or not cat_cols:
    inferred_num, inferred_cat = get_num_cat_cols_from_preprocess(preprocess)
    num_cols = num_cols or inferred_num
    cat_cols = cat_cols or inferred_cat

num_defaults, cat_defaults = get_imputer_defaults(preprocess, num_cols, cat_cols)
cat_options = get_ohe_categories(preprocess, cat_cols)

# ---------- Field definitions ----------
FIELDS = [
    ("gender_std", "cat", "Gender", "Student gender.", None),
    ("region", "cat", "Region", "Student region (OULAD region category).", None),
    ("highest_education", "cat", "Highest Education", "Highest education level.", None),
    ("imd_band", "cat", "IMD Band (Deprivation Level)", "Index of Multiple Deprivation band.", None),
    ("major", "cat", "Major / Field of Study", "Student major/field (if available).", None),
    ("parental_education_level", "cat", "Parental Education Level", "Parents’ education level (if available).", None),
    ("study_environment", "cat", "Study Environment", "Where the student studies most.", None),

    ("study_hours_per_day", "num", "Study Hours per Day", "Average study hours per day.", (0, 16, 0.5)),
    ("study_load", "num", "Study Load (Weekly)", "Overall weekly study load.", (0, 250, 1)),
    ("screen_time", "num", "Screen Time (Hours/Day)", "Daily screen time hours.", (0, 16, 0.5)),

    ("stress_level", "num", "Stress Level (0–10)", "Self-reported stress level.", (0, 10, 1)),
    ("motivation_level", "num", "Motivation Level (0–10)", "Self-reported motivation level.", (0, 10, 1)),
    ("exam_anxiety_score", "num", "Exam Anxiety (0–10)", "Self-reported exam anxiety.", (0, 10, 1)),
    ("attendance_percentage", "num", "Attendance (%)", "Attendance percentage (if available).", (0, 100, 1)),
    ("prior_perf", "num", "Prior Performance (e.g., GPA)", "Prior academic performance.", (0, 4, 0.1)),

    ("n_assess", "num", "Number of Assessments", "How many assessments exist for the student.", (0, 300, 1)),
    ("mean_score", "num", "Mean Assessment Score", "Average assessment score.", (0, 100, 1)),
    ("weighted_score", "num", "Weighted Assessment Score", "Weighted assessment score.", (0, 100, 1)),

    ("total_clicks", "num", "Total VLE Clicks", "Total VLE engagement clicks.", (0, 200000, 10)),
    ("active_days", "num", "Active VLE Days", "Days active on VLE.", (0, 500, 1)),
    ("unique_sites", "num", "Unique VLE Sites Visited", "Distinct VLE sites visited.", (0, 5000, 1)),

    ("studied_credits", "num", "Studied Credits", "Credits studied (OULAD).", (0, 500, 1)),
    ("num_of_prev_attempts", "num", "Previous Attempts", "Number of previous attempts (OULAD).", (0, 50, 1)),
    ("attendance_metric", "num", "Attendance Metric (if used)", "Dataset-specific attendance/engagement metric.", (0, 200000, 10)),
]

label_map = {col: label for col, _, label, _, _ in FIELDS}
help_map = {col: help_text for col, _, _, help_text, _ in FIELDS}

# ---------- Defaults from imputers ----------
widget_defaults = {}
for col, kind, _, _, rng in FIELDS:
    if col not in raw_cols:
        continue
    key = f"w_{col}"

    if kind == "num":
        lo, hi, step = rng
        d = num_defaults.get(col, lo)
        d = clip_float(d, lo, hi)
        widget_defaults[key] = d
        init_state_key(key, d)
    else:
        opts = cat_options.get(col, None)
        d = cat_defaults.get(col, None)
        if opts and len(opts) > 0:
            if d is None or d not in opts:
                d = opts[0]
        if d is None:
            d = ""
        widget_defaults[key] = d
        init_state_key(key, d, options=opts)

# ---------- Buttons ----------
b1, _ = st.columns([1, 5])
with b1:
    if st.button("🔄 Reset to Defaults", use_container_width=True):
        reset_all(widget_defaults)

# ---------- Input form ----------
st.subheader("Student Inputs")

with st.form("student_form", clear_on_submit=False):
    st.markdown("### 🧑‍🎓 Demographics")
    c1, c2, c3 = st.columns(3)
    for col, container in zip(["gender_std", "region", "highest_education"], [c1, c2, c3]):
        if col in raw_cols:
            key = f"w_{col}"
            opts = cat_options.get(col, None)
            with container:
                if opts and len(opts) > 0 and len(opts) <= 100:
                    st.selectbox(label_map[col], opts, key=key, help=help_map.get(col))
                else:
                    st.text_input(label_map[col], key=key, help=help_map.get(col))

    c4, c5, c6 = st.columns(3)
    for col, container in zip(["imd_band", "major", "parental_education_level"], [c4, c5, c6]):
        if col in raw_cols:
            key = f"w_{col}"
            opts = cat_options.get(col, None)
            with container:
                if opts and len(opts) > 0 and len(opts) <= 100:
                    st.selectbox(label_map[col], opts, key=key, help=help_map.get(col))
                else:
                    st.text_input(label_map[col], key=key, help=help_map.get(col))

    if "study_environment" in raw_cols:
        col = "study_environment"
        key = f"w_{col}"
        opts = cat_options.get(col, None)
        if opts and len(opts) > 0 and len(opts) <= 100:
            st.selectbox(label_map[col], opts, key=key, help=help_map.get(col))
        else:
            st.text_input(label_map[col], key=key, help=help_map.get(col))

    st.markdown("### 📚 Study & Wellbeing")
    c7, c8, c9 = st.columns(3)
    for col, container in zip(["study_hours_per_day", "study_load", "screen_time"], [c7, c8, c9]):
        if col in raw_cols:
            key = f"w_{col}"
            lo, hi, step = next(x[4] for x in FIELDS if x[0] == col)
            with container:
                st.number_input(label_map[col], min_value=float(lo), max_value=float(hi), step=float(step), key=key, help=help_map.get(col))

    c10, c11, c12 = st.columns(3)
    for col, container in zip(["stress_level", "motivation_level", "exam_anxiety_score"], [c10, c11, c12]):
        if col in raw_cols:
            key = f"w_{col}"
            lo, hi, step = next(x[4] for x in FIELDS if x[0] == col)
            with container:
                st.number_input(label_map[col], min_value=float(lo), max_value=float(hi), step=float(step), key=key, help=help_map.get(col))

    c13, c14 = st.columns(2)
    for col, container in zip(["attendance_percentage", "prior_perf"], [c13, c14]):
        if col in raw_cols:
            key = f"w_{col}"
            lo, hi, step = next(x[4] for x in FIELDS if x[0] == col)
            with container:
                st.number_input(label_map[col], min_value=float(lo), max_value=float(hi), step=float(step), key=key, help=help_map.get(col))

    st.markdown("### 🧾 Academic / VLE (if available)")
    c15, c16, c17 = st.columns(3)
    for col, container in zip(["n_assess", "mean_score", "weighted_score"], [c15, c16, c17]):
        if col in raw_cols:
            key = f"w_{col}"
            lo, hi, step = next(x[4] for x in FIELDS if x[0] == col)
            with container:
                st.number_input(label_map[col], min_value=float(lo), max_value=float(hi), step=float(step), key=key, help=help_map.get(col))

    c18, c19, c20 = st.columns(3)
    for col, container in zip(["total_clicks", "active_days", "unique_sites"], [c18, c19, c20]):
        if col in raw_cols:
            key = f"w_{col}"
            lo, hi, step = next(x[4] for x in FIELDS if x[0] == col)
            with container:
                st.number_input(label_map[col], min_value=float(lo), max_value=float(hi), step=float(step), key=key, help=help_map.get(col))

    c21, c22, c23 = st.columns(3)
    for col, container in zip(["studied_credits", "num_of_prev_attempts", "attendance_metric"], [c21, c22, c23]):
        if col in raw_cols:
            key = f"w_{col}"
            lo, hi, step = next(x[4] for x in FIELDS if x[0] == col)
            with container:
                st.number_input(label_map[col], min_value=float(lo), max_value=float(hi), step=float(step), key=key, help=help_map.get(col))

    submitted = st.form_submit_button("✅ Predict", use_container_width=True)

# ---------- Predict + SHAP ----------
if submitted:
    row = {c: np.nan for c in raw_cols}

    for col, kind, _, _, _ in FIELDS:
        if col not in raw_cols:
            continue
        key = f"w_{col}"
        v = st.session_state.get(key, None)
        if v is None:
            continue
        if kind == "cat":
            if str(v).strip() == "":
                continue
            row[col] = v
        else:
            row[col] = float(v)

    X_input = pd.DataFrame([row], columns=raw_cols)
    X_pp = preprocess.transform(X_input)

    pred = int(model.predict(X_pp)[0])
    st.session_state["pred"] = pred

    # SHAP: use dense for 1-row to avoid sparse issues
    X_pp_dense = X_pp.toarray() if hasattr(X_pp, "toarray") else X_pp

    try:
        import shap  # noqa

        feature_names = preprocess.get_feature_names_out()
        pos_df, neg_df = compute_grouped_local_shap(
            model=model,
            X_pp_1row=X_pp_dense,
            feature_names=feature_names,
            X_raw_1row=X_input,
            label_map=label_map,
            cat_cols=cat_cols,
            top_n=8,
        )
        st.session_state["shap_pos"] = pos_df
        st.session_state["shap_neg"] = neg_df
        st.session_state.pop("shap_err", None)
    except Exception as e:
        st.session_state["shap_pos"] = None
        st.session_state["shap_neg"] = None
        st.session_state["shap_err"] = repr(e)

# ---------- Output ----------
st.subheader("Prediction")

if "pred" not in st.session_state:
    st.info("Enter inputs and click **Predict**.")
else:
    if st.session_state["pred"] == 1:
        st.error("Prediction: **AT RISK**")
    else:
        st.success("Prediction: **NOT AT RISK**")

    st.markdown("### Explanation (Top Reasons)")
    err = st.session_state.get("shap_err", None)
    pos_df = st.session_state.get("shap_pos", None)
    neg_df = st.session_state.get("shap_neg", None)

    if err is not None:
        st.warning("SHAP explanation not available.")
        st.caption("Install SHAP:  pip install shap")
        st.caption(f"Error: {err}")
    else:
        st.write("⬆️ **Increases risk**")
        if pos_df is not None and len(pos_df) > 0:
            st.dataframe(pos_df, use_container_width=True, hide_index=True)
        else:
            st.caption("No positive contributors found.")

        st.write("⬇️ **Decreases risk**")
        if neg_df is not None and len(neg_df) > 0:
            st.dataframe(neg_df, use_container_width=True, hide_index=True)
        else:
            st.caption("No negative contributors found.")
