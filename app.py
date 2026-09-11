import streamlit as st
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Student Performance AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.stApp {
    background-color: #0f172a;
    color: #f8fafc;
}

h1, h2, h3, h4, label, .stMarkdown {
    color: #f1f5f9 !important;
}

.metric-card {
    background-color: #1e293b;
    padding: 24px;
    border-radius: 14px;
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
    border: 1px solid #334155;
    margin-bottom: 24px;
}

.section-header {
    font-size: 1.25rem;
    font-weight: 700;
    color: #38bdf8 !important;
    margin-bottom: 18px;
}

.result-box {
    background: linear-gradient(135deg, #4338ca 0%, #2563eb 100%);
    color: #ffffff !important;
    padding: 32px;
    border-radius: 16px;
    text-align: center;
    box-shadow: 0 12px 28px -5px rgba(37, 99, 235, 0.4);
    border: 1px solid #4f46e5;
}

.result-score {
    font-size: 3.8rem;
    font-weight: 800;
    margin: 12px 0;
    color: #ffffff !important;
}

div.stButton > button {
    background: linear-gradient(135deg, #4f46e5 0%, #2563eb 100%);
    color: #ffffff !important;
    font-weight: 700;
    font-size: 1.1rem;
    padding: 14px 28px;
    border-radius: 10px;
    border: none;
    width: 100%;
    transition: all 0.3s ease;
}

div.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(79, 70, 229, 0.6);
}

.stSlider label,
.stSelectbox label,
.stRadio label {
    color: #e2e8f0 !important;
    font-weight: 600 !important;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# FILE PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "voting_model.pkl"
DATASET_PATH = BASE_DIR / "dataset.csv"


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"⚠️ Error loading model artifact: {e}")
        return None


# ============================================================
# RECREATE TRAINING PREPROCESSING PIPELINE
# ============================================================

@st.cache_resource
def prepare_training_pipeline():

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"dataset.csv was not found at:\n{DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)

    # 1. Identify numerical and categorical columns
    num_coln = df.select_dtypes(include=["number"]).columns.tolist()
    catg_coln = df.select_dtypes(include=["object"]).columns.tolist()

    # 2. Numerical imputation
    num_imputer = SimpleImputer(strategy="mean")
    if len(num_coln) > 0:
        df[num_coln] = num_imputer.fit_transform(df[num_coln])

    # 3. Categorical imputation
    cat_imputer = SimpleImputer(strategy="most_frequent")
    if len(catg_coln) > 0:
        df[catg_coln] = cat_imputer.fit_transform(df[catg_coln])

    # 4. Ordinal encoding
    ordinal_mapping = {"Low": 0, "Medium": 1, "High": 2}
    oe_coln = [
        "Parental_Involvement",
        "Access_to_Resources",
        "Motivation_Level",
        "Family_Income",
        "Teacher_Quality"
    ]
    for col in oe_coln:
        if col in df.columns:
            df[col] = df[col].map(ordinal_mapping)

    # 5. Parental Education Level
    if "Parental_Education_Level" in df.columns:
        df["Parental_Education_Level"] = df["Parental_Education_Level"].map({
            "High School": 0,
            "College": 1,
            "Postgraduate": 2
        })

    # 6. Binary encoding
    binary_coln = [
        "Extracurricular_Activities",
        "Internet_Access",
        "Learning_Disabilities"
    ]
    for col in binary_coln:
        if col in df.columns:
            df[col] = df[col].map({"Yes": 1, "No": 0})

    # 7. One-Hot Encoding
    excluded_columns = oe_coln + binary_coln + ["Parental_Education_Level"]
    ohe_coln = [col for col in catg_coln if col not in excluded_columns]

    ohe = OneHotEncoder(
        drop="first",
        sparse_output=False,
        handle_unknown="ignore"
    )

    if len(ohe_coln) > 0:
        ohe_encoded = ohe.fit_transform(df[ohe_coln])
        ohe_encoded_df = pd.DataFrame(
            ohe_encoded,
            columns=ohe.get_feature_names_out(ohe_coln),
            index=df.index
        )
        df = pd.concat([df.drop(ohe_coln, axis=1), ohe_encoded_df], axis=1)

    # 8. Extract Target Variable BEFORE dropping columns
    y = df["Exam_Score"]

    # 9. Feature Engineering
    df["Hours_Studied_log"] = np.log1p(df["Hours_Studied"])
    df["Previous_Scores_log"] = np.log1p(df["Previous_Scores"])

    # 10. Create Feature Set X
    drop_columns = [
        "Exam_Score",
        "Gender_Male",
        "School_Type_Public",
        "Sleep_Hours",
        "Distance_from_Home_Moderate",
        "Peer_Influence_Neutral"
    ]
    existing_drop_columns = [col for col in drop_columns if col in df.columns]
    X = df.drop(columns=existing_drop_columns)

    # 11. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=43
    )

    # 12. StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_train)

    return {
        "num_imputer": num_imputer,
        "cat_imputer": cat_imputer,
        "ohe": ohe,
        "ohe_columns": ohe_coln,
        "feature_columns": X.columns.tolist(),
        "scaler": scaler,
        "train_columns": X_train.columns.tolist()
    }


# ============================================================
# LOAD MODEL + PIPELINE
# ============================================================

model = load_model()

try:
    pipeline = prepare_training_pipeline()
except Exception as e:
    st.error(f"❌ Error preparing preprocessing pipeline:\n\n{e}")
    st.stop()


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div style="text-align: center; padding: 15px 0 25px 0;">

<h1 style="color: #f8fafc; font-weight: 800; font-size: 2.6rem;">
🎓 Student Exam Score Predictor
</h1>

<p style="color: #94a3b8; font-size: 1.1rem; max-width: 650px; margin: 0 auto;">
Input student academic telemetry, behavioral metrics,
and socio-economic indicators to forecast final exam
performance using machine learning.
</p>

</div>
""", unsafe_allow_html=True)

st.markdown("---")


# ============================================================
# MAIN COLUMNS
# ============================================================

col1, col2 = st.columns([1.2, 0.8], gap="large")


# ============================================================
# LEFT SIDE - INPUTS
# ============================================================

with col1:
    st.subheader("📋 Student Profile & Performance Inputs")

    # Academic Factors
    with st.container():
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">📚 Core Academic Metrics</div>', unsafe_allow_html=True)

        ac_col1, ac_col2 = st.columns(2)
        with ac_col1:
            hours_studied = st.slider("Hours Studied (per week)", min_value=1, max_value=44, value=20)
            attendance = st.slider("Attendance Rate (%)", min_value=60, max_value=100, value=85)
        with ac_col2:
            previous_scores = st.slider("Previous Exam Score (%)", min_value=40, max_value=100, value=75)
            tutoring_sessions = st.slider("Tutoring Sessions (per month)", min_value=0, max_value=7, value=2)

        st.markdown('</div>', unsafe_allow_html=True)

    # Socio Economic
    with st.container():
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">🏠 Socio-Economic & Resource Access</div>', unsafe_allow_html=True)

        res_col1, res_col2 = st.columns(2)
        with res_col1:
            parental_involvement = st.selectbox("Parental Involvement", ["Low", "Medium", "High"], index=1)
            access_to_resources = st.selectbox("Access to Learning Resources", ["Low", "Medium", "High"], index=1)
            parental_education = st.selectbox("Parental Education Level", ["High School", "College", "Postgraduate"], index=1)

        with res_col2:
            internet_access = st.radio("High-Speed Internet Access", ["Yes", "No"], horizontal=True)
            school_type = st.radio("School Type", ["Public", "Private"], horizontal=True)
            extracurricular = st.radio("Extra-Curricular Participation", ["Yes", "No"], horizontal=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # Lifestyle
    with st.container():
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">⚡ Lifestyle & Demographic Factors</div>', unsafe_allow_html=True)

        life_col1, life_col2 = st.columns(2)
        with life_col1:
            sleep_hours = st.slider("Sleep Duration (hours/night)", min_value=4, max_value=10, value=7)
            motivation_level = st.selectbox("Motivation Level", ["Low", "Medium", "High"], index=1)
        with life_col2:
            physical_activity = st.slider("Physical Activity (hours/week)", min_value=0, max_value=6, value=3)
            gender = st.selectbox("Gender", ["Male", "Female"], index=0)

        st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# RIGHT SIDE - PREDICTION
# ============================================================

with col2:
    st.subheader("🎯 Prediction & Insights")
    st.markdown("<br>", unsafe_allow_html=True)

    predict_clicked = st.button("🚀 Calculate Predicted Exam Score")

    if predict_clicked:
        if model is None:
            st.error("❌ Model artifact could not be loaded.")
        else:
            with st.spinner("Processing telemetry and generating prediction..."):
                try:
                    # STEP 1: CREATE RAW INPUT
                    raw_input = pd.DataFrame([{
                        "Hours_Studied": hours_studied,
                        "Attendance": attendance,
                        "Parental_Involvement": parental_involvement,
                        "Access_to_Resources": access_to_resources,
                        "Extracurricular_Activities": extracurricular,
                        "Sleep_Hours": sleep_hours,
                        "Previous_Scores": previous_scores,
                        "Motivation_Level": motivation_level,
                        "Internet_Access": internet_access,
                        "Tutoring_Sessions": tutoring_sessions,
                        "Family_Income": "Medium",
                        "Teacher_Quality": "Medium",
                        "School_Type": school_type,
                        "Peer_Influence": "Positive",
                        "Physical_Activity": physical_activity,
                        "Learning_Disabilities": "No",
                        "Parental_Education_Level": parental_education,
                        "Distance_from_Home": "Near",
                        "Gender": gender
                    }])

                    # STEP 2: ONE-HOT ENCODING (Apply first on raw categorical columns)
                    ohe_columns = pipeline["ohe_columns"]
                    for col in ohe_columns:
                        if col not in raw_input.columns:
                            raw_input[col] = "Unknown"

                    ohe_input = raw_input[ohe_columns].copy()
                    ohe_encoded = pipeline["ohe"].transform(ohe_input)
                    ohe_encoded_df = pd.DataFrame(
                        ohe_encoded,
                        columns=pipeline["ohe"].get_feature_names_out(ohe_columns),
                        index=raw_input.index
                    )
                    raw_input = pd.concat(
                        [raw_input.drop(columns=ohe_columns, errors="ignore"), ohe_encoded_df],
                        axis=1
                    )

                    # STEP 3: ORDINAL ENCODING
                    ordinal_mapping = {"Low": 0, "Medium": 1, "High": 2}
                    oe_coln = [
                        "Parental_Involvement",
                        "Access_to_Resources",
                        "Motivation_Level",
                        "Family_Income",
                        "Teacher_Quality"
                    ]
                    for col in oe_coln:
                        if col in raw_input.columns:
                            raw_input[col] = raw_input[col].map(ordinal_mapping)

                    # STEP 4: PARENTAL EDUCATION ENCODING
                    if "Parental_Education_Level" in raw_input.columns:
                        raw_input["Parental_Education_Level"] = raw_input["Parental_Education_Level"].map({
                            "High School": 0,
                            "College": 1,
                            "Postgraduate": 2
                        })

                    # STEP 5: BINARY ENCODING
                    binary_coln = [
                        "Extracurricular_Activities",
                        "Internet_Access",
                        "Learning_Disabilities"
                    ]
                    for col in binary_coln:
                        if col in raw_input.columns:
                            raw_input[col] = raw_input[col].map({"Yes": 1, "No": 0})

                    # STEP 6: FEATURE ENGINEERING
                    raw_input["Hours_Studied_log"] = np.log1p(raw_input["Hours_Studied"])
                    raw_input["Previous_Scores_log"] = np.log1p(raw_input["Previous_Scores"])

                    # STEP 7: REMOVE DROPPED COLUMNS
                    drop_columns = [
                        "Gender_Male",
                        "School_Type_Public",
                        "Sleep_Hours",
                        "Distance_from_Home_Moderate",
                        "Peer_Influence_Neutral"
                    ]
                    X_input = raw_input.drop(columns=drop_columns, errors="ignore")

                    # STEP 8: REORDER FEATURES TO MATCH TRAINING
                    expected_features = pipeline["feature_columns"]
                    X_input = X_input.reindex(columns=expected_features, fill_value=0)

                    # STEP 9: STANDARD SCALING
                    X_input_scaled = pipeline["scaler"].transform(X_input)

                    # STEP 10: MODEL PREDICTION
                    raw_pred = float(model.predict(X_input_scaled)[0])
                    score = float(np.clip(raw_pred, 0, 100))

                    # STEP 11: DISPLAY RESULT
                    st.markdown(
                        f"""
                        <div class="result-box">
                            <h3 style="margin: 0; font-size: 1.1rem; opacity: 0.95; text-transform: uppercase; letter-spacing: 1px; color: #ffffff !important;">
                            Predicted Exam Score
                            </h3>
                            <div class="result-score">
                            {score:.1f}
                            <span style="font-size: 2rem;">/ 100</span>
                            </div>
                            <p style="margin: 0; opacity: 0.9; color: #e0e7ff !important;">
                            Raw Model Output: {raw_pred:.2f}
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    # PERFORMANCE INSIGHT
                    st.markdown("<br>", unsafe_allow_html=True)
                    if score >= 85:
                        st.success("🌟 **Outstanding Performance Expectation!** The student is well-positioned for top marks.")
                    elif score >= 70:
                        st.info("📈 **Solid Academic Performance!** Consistent study habits will maintain or boost this outcome.")
                    elif score >= 50:
                        st.warning("⚠️ **Moderate Performance.** Increasing attendance or tutoring frequency is recommended.")
                    else:
                        st.error("🚨 **High Academic Risk.** Targeted intervention required in study hours and core topics.")

                    # KEY INFLUENCING FACTORS
                    st.markdown("### 🔍 Key Influencing Factors")
                    factor_col1, factor_col2 = st.columns(2)
                    with factor_col1:
                        st.metric(
                            label="Weekly Study Commitment",
                            value=f"{hours_studied} hrs",
                            delta="+ High Impact" if hours_studied >= 25 else "- Low Volume"
                        )
                    with factor_col2:
                        st.metric(
                            label="Class Attendance",
                            value=f"{attendance}%",
                            delta="+ Optimal" if attendance >= 80 else "- Critical Loss"
                        )

                    # TECHNICAL DEBUG INFORMATION
                    with st.expander("🔧 Technical Prediction Details"):
                        st.write("Preprocessing completed successfully.")
                        st.write(f"Input features processed: {X_input.shape[1]}")
                        st.write(f"Training features expected: {len(expected_features)}")
                        st.write(f"Raw model prediction: {raw_pred:.4f}")
                        st.write(f"Displayed score (clipped): {score:.4f}")

                except Exception as err:
                    st.error(f"❌ Prediction Error: {err}")
                    with st.expander("Show technical error"):
                        st.exception(err)
    else:
        st.info("👈 Adjust the student parameters on the left and click **Calculate Predicted Exam Score** to view results.")