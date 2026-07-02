# AI Sports Talent Scout - Updated Version
# Replace your existing app.py with this file

import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="AI Sports Talent Scout",
    page_icon="🏏",
    layout="wide"
)

clf = joblib.load(
    "models/future_potential_model.pkl"
)

reg = joblib.load(
    "models/auction_value_model.pkl"
)

role_encoder = joblib.load(
    "models/role_encoder.pkl"
)

potential_encoder = joblib.load(
    "models/potential_encoder.pkl"
)

df = pd.read_csv(
    "data/players_featured.csv"
)

st.markdown(
    """
    # 🏏 AI Sports Talent Scout

    ### AI-Powered Cricket Talent Identification & Auction Value Prediction
    """
)


st.markdown(
    """
    # 🏏 AI Sports Talent Scout

    ### AI-Powered Cricket Talent Identification & Auction Value Prediction
    """
)
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "Players",
        len(df)
    )

with c2:
    st.metric(
        "Features",
        16
    )

with c3:
    st.metric(
        "Classifier Accuracy",
        "84%"
    )

with c4:
    st.metric(
        "Regression R²",
        "96%"
    )

    c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "Players",
        len(df)
    )

with c2:
    st.metric(
        "Features",
        16
    )

with c3:
    st.metric(
        "Classifier Accuracy",
        "84%"
    )

with c4:
    st.metric(
        "Regression R²",
        "96%"
    )

    tab1, tab2, tab3 = st.tabs(
    [
        "🏏 Talent Prediction",
        "📊 Analytics",
        "ℹ️ About Project"
    ]
)
    

    with tab1:
        if st.button("🚀 Predict Talent"):
            role_encoded = role_encoder.transform(
    [role]
)[0]

batting_impact = batting_average * strike_rate

bowling_impact = wickets / economy

experience_score = matches * age

consistency_score = (
    fifties
    + (2 * hundreds)
)



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


future_pred = clf.predict(
    input_data
)[0]

probs = clf.predict_proba(
    input_data
)[0]

future_potential = (
    potential_encoder
    .inverse_transform(
        [future_pred]
    )[0]
)

auction_value = reg.predict(
    input_data
)[0]




r1, r2, r3 = st.columns(3)

with r1:
    st.metric(
        "Future Potential",
        future_potential
    )

with r2:
    st.metric(
        "Auction Value",
        f"₹ {auction_value:.2f} Cr"
    )

with r3:

    scout_rating = min(
        10,
        round(
            auction_value / 1.5,
            1
        )
    )

    st.metric(
        "Scout Rating",
        f"{scout_rating}/10"
    )





    st.subheader(
    "🎯 Prediction Confidence"
)

prob_df = pd.DataFrame({

    "Potential":
        potential_encoder.classes_,

    "Probability":
        probs

})

fig = px.bar(
    prob_df,
    x="Potential",
    y="Probability",
    color="Potential"
)

st.plotly_chart(
    fig,
    use_container_width=True
)



st.subheader(
    "🏏 Player Skill Radar"
)

categories = [

    "Batting",
    "Bowling",
    "Fielding",
    "Fitness",
    "Experience"

]

values = [

    min(
        100,
        batting_average * 1.5
    ),

    min(
        100,
        wickets / 3
    ),

    min(
        100,
        catches
    ),

    fitness_score,

    min(
        100,
        matches / 2
    )

]

radar = go.Figure()

radar.add_trace(

    go.Scatterpolar(

        r=values,
        theta=categories,
        fill="toself"

    )

)

st.plotly_chart(
    radar,
    use_container_width=True
)


st.subheader(
    "📊 Player vs Dataset Average"
)

comparison_df = pd.DataFrame({

    "Metric":[

        "Runs",
        "Strike Rate",
        "Fitness Score"

    ],

    "Player":[

        runs,
        strike_rate,
        fitness_score

    ],

    "Dataset Average":[

        round(
            df["runs"].mean(),
            2
        ),

        round(
            df["strike_rate"].mean(),
            2
        ),

        round(
            df["fitness_score"].mean(),
            2
        )

    ]

})

st.dataframe(
    comparison_df,
    use_container_width=True
)




report = f"""
AI Sports Talent Scout Report

Role: {role}

Future Potential:
{future_potential}

Auction Value:
₹ {auction_value:.2f} Cr

Scout Rating:
{scout_rating}/10
"""

st.download_button(
    "📄 Download Report",
    report,
    file_name="player_report.txt"
)



with tab2:
    st.subheader(
    "🔥 Correlation Heatmap"
)

corr = (
    df
    .select_dtypes(
        include="number"
    )
    .corr()
)

heatmap = ff.create_annotated_heatmap(

    z=corr.values,
    x=list(corr.columns),
    y=list(corr.columns)

)

st.plotly_chart(
    heatmap,
    use_container_width=True
)

st.subheader(
    "🏆 Top 10 Elite Talents"
)

top_players = (

    df
    .sort_values(
        "auction_value_cr",
        ascending=False
    )
    .head(10)

)

st.dataframe(

    top_players[
        [

            "player_id",
            "role",
            "talent_score",
            "auction_value_cr"

        ]
    ],

    use_container_width=True

)

with tab3:
    st.markdown(
"""
## About Project

AI Sports Talent Scout is an end-to-end Machine Learning application designed to identify future cricket stars.

### Features

- Future Potential Prediction
- Auction Value Prediction
- Radar Skill Analysis
- Probability Confidence Analysis
- Talent Leaderboard

### Technologies

- Python
- Pandas
- NumPy
- Scikit-Learn
- Streamlit
- Plotly

### Model Performance

- Random Forest Classifier
- Accuracy: 84%

- Random Forest Regressor
- R² Score: 96%
"""
)
    
    
