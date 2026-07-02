import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# ----------------------------------
# PAGE CONFIG
# ----------------------------------

st.set_page_config(
    page_title="AI Sports Talent Scout",
    page_icon="🏏",
    layout="wide"
)

# ----------------------------------
# LOAD MODELS
# ----------------------------------

clf = joblib.load("models/future_potential_model.pkl")
reg = joblib.load("models/auction_value_model.pkl")

role_encoder = joblib.load("models/role_encoder.pkl")
potential_encoder = joblib.load("models/potential_encoder.pkl")

# ----------------------------------
# LOAD DATA
# ----------------------------------

df = pd.read_csv("data/players_featured.csv")

# ----------------------------------
# CUSTOM CSS
# ----------------------------------

st.markdown("""
<style>
.big-title {
    font-size:40px;
    font-weight:bold;
    color:#ffffff;
}
.subtitle {
    font-size:18px;
    color:#cccccc;
}
.metric-card {
    background-color:#1E1E1E;
    padding:15px;
    border-radius:10px;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------
# HEADER
# ----------------------------------

st.markdown(
    "<div class='big-title'>🏏 AI Sports Talent Scout</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>AI-Powered Cricket Talent Identification & Auction Value Prediction System</div>",
    unsafe_allow_html=True
)

st.divider()

# ----------------------------------
# KPI SECTION
# ----------------------------------

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Players", len(df))

with c2:
    st.metric("Features", 16)

with c3:
    st.metric("Classifier Accuracy", "84%")

with c4:
    st.metric("Regression R²", "96%")

st.divider()

# ----------------------------------
# TABS
# ----------------------------------

tab1, tab2, tab3 = st.tabs([
    "🏏 Talent Prediction",
    "📊 Analytics",
    "ℹ️ About Project"
])

# =====================================================
# TAB 1
# =====================================================

with tab1:

    st.subheader("Player Information")

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input(
            "Age",
            18,
            40,
            22
        )

        role = st.selectbox(
            "Role",
            [
                "All-Rounder",
                "Batter",
                "Bowler",
                "Wicketkeeper"
            ]
        )

        matches = st.number_input(
            "Matches",
            0,
            500,
            100
        )

        runs = st.number_input(
            "Runs",
            0,
            10000,
            3000
        )

        batting_average = st.number_input(
            "Batting Average",
            0.0,
            100.0,
            40.0
        )

        strike_rate = st.number_input(
            "Strike Rate",
            0.0,
            250.0,
            130.0
        )

    with col2:

        fifties = st.number_input(
            "Fifties",
            0,
            100,
            10
        )

        hundreds = st.number_input(
            "Hundreds",
            0,
            50,
            5
        )

        wickets = st.number_input(
            "Wickets",
            0,
            500,
            50
        )

        economy = st.number_input(
            "Economy",
            1.0,
            15.0,
            6.0
        )

        catches = st.number_input(
            "Catches",
            0,
            200,
            20
        )

        fitness_score = st.slider(
            "Fitness Score",
            50,
            100,
            80
        )

    if st.button("🚀 Predict Talent"):

        role_encoded = role_encoder.transform([role])[0]

        batting_impact = batting_average * strike_rate
        bowling_impact = wickets / economy
        experience_score = matches * age
        consistency_score = fifties + (2 * hundreds)

        input_data = pd.DataFrame([{
            "age": age,
            "role": role_encoded,
            "matches": matches,
            "runs": runs,
            "batting_average": batting_average,
            "strike_rate": strike_rate,
            "fifties": fifties,
            "hundreds": hundreds,
            "wickets": wickets,
            "economy": economy,
            "catches": catches,
            "fitness_score": fitness_score,
            "batting_impact": batting_impact,
            "bowling_impact": bowling_impact,
            "experience_score": experience_score,
            "consistency_score": consistency_score
        }])

        future_pred = clf.predict(input_data)[0]

        future_potential = (
            potential_encoder.inverse_transform(
                [future_pred]
            )[0]
        )

        auction_value = reg.predict(input_data)[0]

        st.divider()

        st.subheader("📈 Prediction Results")

        rc1, rc2, rc3 = st.columns(3)

        with rc1:
            st.metric(
                "Future Potential",
                future_potential
            )

        with rc2:
            st.metric(
                "Auction Value",
                f"₹ {auction_value:.2f} Cr"
            )

        with rc3:

            scout_rating = min(
                10,
                round(
                    (auction_value / 15) * 10,
                    1
                )
            )

            st.metric(
                "Scout Rating",
                f"{scout_rating}/10"
            )

        st.progress(
            min(
                int(
                    scout_rating * 10
                ),
                100
            )
        )

        st.subheader("📝 Scouting Report")

        insights = []

        if batting_average > 50:
            insights.append(
                "Elite batting consistency."
            )

        if strike_rate > 150:
            insights.append(
                "Explosive scoring ability."
            )

        if wickets > 150:
            insights.append(
                "Match-winning bowling impact."
            )

        if fitness_score > 90:
            insights.append(
                "Exceptional fitness level."
            )

        if catches > 70:
            insights.append(
                "Reliable fielding skills."
            )

        if len(insights) == 0:
            insights.append(
                "Balanced player profile."
            )

        for item in insights:
            st.success(item)

# =====================================================
# TAB 2
# =====================================================

with tab2:

    st.subheader("📊 Dataset Analytics")

    col1, col2 = st.columns(2)

    with col1:

        role_fig = px.bar(
            df["role"].value_counts().reset_index(),
            x="role",
            y="count",
            title="Role Distribution"
        )

        st.plotly_chart(
            role_fig,
            use_container_width=True
        )

    with col2:

        potential_fig = px.bar(
            df["future_potential"].value_counts().reset_index(),
            x="future_potential",
            y="count",
            title="Future Potential Distribution"
        )

        st.plotly_chart(
            potential_fig,
            use_container_width=True
        )

    auction_fig = px.histogram(
        df,
        x="auction_value_cr",
        nbins=30,
        title="Auction Value Distribution"
    )

    st.plotly_chart(
        auction_fig,
        use_container_width=True
    )

    scatter_fig = px.scatter(
        df,
        x="runs",
        y="auction_value_cr",
        color="role",
        title="Runs vs Auction Value"
    )

    st.plotly_chart(
        scatter_fig,
        use_container_width=True
    )

# =====================================================
# TAB 3
# =====================================================

with tab3:

    st.subheader("ℹ️ About Project")

    st.markdown("""
### Project Objective

Identify future cricket stars and estimate
their market value using Machine Learning.

### Dataset

- 3000 Synthetic Cricket Players
- 16 Engineered Features
- Talent Score System
- Auction Value Estimation

### Machine Learning Models

#### Classification
- Random Forest Classifier
- Predicts:
    - Low
    - Medium
    - High
    - Elite

#### Regression
- Random Forest Regressor
- Predicts Auction Value

### Model Performance

- Classification Accuracy: 84%
- Regression R² Score: 96%

### Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Plotly
- Streamlit

### Author

AI Sports Talent Scout Project
""")