
import pandas as pd
import streamlit as st
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Cricket Player Finder",
    page_icon="🏏",
    layout="wide"
)

# -----------------------------
# Load player statistics
# -----------------------------
@st.cache_data
def load_players():
    file_path = Path("processed_data/player_statistics.csv")
    return pd.read_csv(file_path)

players = load_players()

# -----------------------------
# Prepare recommendation engine
# -----------------------------
feature_columns = [
    "total_runs",
    "balls_faced",
    "fours",
    "sixes",
    "strike_rate",
    "batting_average",
    "runs_conceded",
    "wickets",
    "legal_balls",
    "economy"
]

@st.cache_data
def calculate_similarity(data):
    features = data[feature_columns].copy()
    features = features.apply(pd.to_numeric, errors="coerce")
    features = features.fillna(0)
    features = features.replace([float("inf"), float("-inf")], 0)

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    return cosine_similarity(scaled_features)

similarity_matrix = calculate_similarity(players)

# -----------------------------
# Header
# -----------------------------
st.title("🏏 Cricket Player Finder")
st.markdown(
    "### Discover IPL players with similar statistical profiles"
)
st.write(
    "Choose a player to find five players with similar "
    "batting and bowling statistics."
)

st.divider()

# -----------------------------
# Player selection
# -----------------------------
player_names = sorted(players["player"].dropna().unique())

selected_player = st.selectbox(
    "Search or select an IPL player",
    player_names
)

selected_rows = players[
    players["player"] == selected_player
]

if not selected_rows.empty:
    player_index = selected_rows.index[0]
    player = players.iloc[player_index]

    st.subheader(f"Statistics: {selected_player}")

    # Key statistics
    col1, col2, col3 = st.columns(3)

    col1.metric("Total Runs", f"{player['total_runs']:.0f}")
    col2.metric("Batting Average", f"{player['batting_average']:.2f}")
    col3.metric("Strike Rate", f"{player['strike_rate']:.2f}")

    col4, col5, col6 = st.columns(3)

    col4.metric("Wickets", f"{player['wickets']:.0f}")
    col5.metric("Economy Rate", f"{player['economy']:.2f}")
    col6.metric("Balls Faced", f"{player['balls_faced']:.0f}")

    st.divider()

    # -----------------------------
    # Recommendations
    # -----------------------------
    st.subheader("🎯 Top 5 Similar Players")

    scores = list(enumerate(similarity_matrix[player_index]))
    scores = [
        (index, score)
        for index, score in scores
        if index != player_index
        and players.iloc[index]["player"] != selected_player
    ]

    scores.sort(key=lambda item: item[1], reverse=True)
    top_five = scores[:5]

    if not top_five:
        st.warning("No similar players were found.")
    else:
        results = []

        for rank, (index, score) in enumerate(top_five, start=1):
            similar_player = players.iloc[index]

            results.append({
                "Rank": rank,
                "Player": similar_player["player"],
                "Similarity Score": round(float(score), 3),
                "Total Runs": int(similar_player["total_runs"]),
                "Batting Average": round(
                    float(similar_player["batting_average"]), 2
                ),
                "Strike Rate": round(
                    float(similar_player["strike_rate"]), 2
                ),
                "Wickets": int(similar_player["wickets"]),
                "Economy": round(float(similar_player["economy"]), 2)
            })

        results_df = pd.DataFrame(results)
        st.dataframe(results_df, hide_index=True, use_container_width=True)

        st.caption(
            "Similarity scores measure similarity between statistical "
            "profiles. They are not probabilities of match performance."
        )

        # Comparison chart
        st.subheader("📊 Runs Comparison")

        chart_data = results_df.set_index("Player")[["Total Runs"]]
        st.bar_chart(chart_data)

st.divider()
st.caption("Cricket Player Finder | IPL statistical similarity project")
