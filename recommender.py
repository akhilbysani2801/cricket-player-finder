
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

DATA_PATH = Path("processed_data/player_statistics.csv")

BATTING_FEATURES = [
    "total_runs",
    "balls_faced",
    "fours",
    "sixes",
    "strike_rate",
    "batting_average",
]

BOWLING_FEATURES = [
    "runs_conceded",
    "wickets",
    "legal_balls",
    "economy",
]


# --------------------------------------------------
# 2. LOAD AND CLEAN PLAYER DATA
# --------------------------------------------------

def load_players():
    """Load player statistics and clean invalid values."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"File not found: {DATA_PATH}. Run main.py first."
        )

    players = pd.read_csv(DATA_PATH)

    required_columns = [
        "player",
        *BATTING_FEATURES,
        *BOWLING_FEATURES,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in players.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns in player_statistics.csv: "
            f"{missing_columns}"
        )

    players = players.dropna(subset=["player"]).copy()
    players["player"] = players["player"].astype(str).str.strip()

    players = players[players["player"] != ""].copy()

    numeric_columns = BATTING_FEATURES + BOWLING_FEATURES

    for column in numeric_columns:
        players[column] = pd.to_numeric(
            players[column], errors="coerce"
        )

    players[numeric_columns] = players[numeric_columns].replace(
        [np.inf, -np.inf], np.nan
    )

    # Replace missing numeric statistics with zero for display
    # and feature preparation. This is a simplifying assumption:
    # unknown values should ideally be distinguished from true zeros.
    players[numeric_columns] = players[numeric_columns].fillna(0)

    # Remove rows with invalid negative count statistics.
    count_columns = [
        "total_runs",
        "balls_faced",
        "fours",
        "sixes",
        "runs_conceded",
        "wickets",
        "legal_balls",
    ]

    valid_rows = (players[count_columns] >= 0).all(axis=1)
    players = players.loc[valid_rows].copy()

    # Remove duplicate player names so each player appears once.
    # If duplicates have conflicting statistics, investigate them
    # in the source dataset before relying on the recommendations.
    players = players.drop_duplicates(
        subset=["player"], keep="first"
    ).reset_index(drop=True)

    # Zero bowling figures are retained for players who did not bowl.
    players["has_bowling_sample"] = players["legal_balls"] > 0

    if players.empty:
        raise ValueError("No valid player records are available.")

    return players


# --------------------------------------------------
# 3. BUILD SIMILARITY MATRICES
# --------------------------------------------------

def build_similarity_matrix(data, feature_columns):
    """Scale features and calculate cosine similarity."""

    features = data[feature_columns].copy()

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    return cosine_similarity(scaled_features)


def prepare_recommender(players):
    """Build batting, bowling and overall similarity matrices."""

    batting_similarity = build_similarity_matrix(
        players, BATTING_FEATURES
    )

    bowling_similarity = build_similarity_matrix(
        players, BOWLING_FEATURES
    )

    # Overall similarity combines the batting and bowling scores.
    overall_similarity = (
        0.5 * batting_similarity
        + 0.5 * bowling_similarity
    )

    return {
        "batting": batting_similarity,
        "bowling": bowling_similarity,
        "overall": overall_similarity,
    }


# --------------------------------------------------
# 4. RECOMMEND SIMILAR PLAYERS
# --------------------------------------------------

def recommend_similar_players(
    players,
    model,
    player_index,
    mode="Overall",
    number=5,
):
    """
    Return the top similar players.

    Supported modes: Batting, Bowling, Overall.
    """

    if not 0 <= player_index < len(players):
        raise IndexError("Player index is out of range.")

    if number < 1:
        raise ValueError("number must be greater than zero.")

    mode = mode.strip().lower()

    if mode not in {"batting", "bowling", "overall"}:
        raise ValueError(
            "Mode must be Batting, Bowling or Overall."
        )

    target = players.iloc[player_index]
    similarity_matrix = model[mode]

    results = []

    for candidate_index, candidate in players.iterrows():

        # Never recommend the selected player.
        if candidate_index == player_index:
            continue

        # Avoid duplicate-name recommendations.
        if candidate["player"] == target["player"]:
            continue

        batting_score = float(
            model["batting"][player_index, candidate_index]
        )

        bowling_score = float(
            model["bowling"][player_index, candidate_index]
        )

        overall_score = float(
            model["overall"][player_index, candidate_index]
        )

        score = float(
            similarity_matrix[player_index, candidate_index]
        )

        # Explain whether both players have bowling records.
        target_bowled = bool(target["has_bowling_sample"])
        candidate_bowled = bool(candidate["has_bowling_sample"])

        if target_bowled and candidate_bowled:
            bowling_note = "Both have bowling records"
        elif not target_bowled and not candidate_bowled:
            bowling_note = "Neither has recorded legal balls"
        else:
            bowling_note = "Bowling experience differs"

        results.append({
            "player": candidate["player"],
            "similarity_score": score,
            "batting_similarity": batting_score,
            "bowling_similarity": bowling_score,
            "overall_similarity": overall_score,
            "bowling_note": bowling_note,
            "total_runs": candidate["total_runs"],
            "balls_faced": candidate["balls_faced"],
            "fours": candidate["fours"],
            "sixes": candidate["sixes"],
            "strike_rate": candidate["strike_rate"],
            "batting_average": candidate["batting_average"],
            "runs_conceded": candidate["runs_conceded"],
            "wickets": candidate["wickets"],
            "legal_balls": candidate["legal_balls"],
            "overs_bowled": (
                candidate["legal_balls"] / 6
            ),
            "economy": candidate["economy"],
        })

    results_df = pd.DataFrame(results)

    if results_df.empty:
        return results_df

    results_df = results_df.sort_values(
        "similarity_score",
        ascending=False,
    )

    return results_df.head(number).reset_index(drop=True)


# --------------------------------------------------
# 5. COMMAND-LINE TEST
# --------------------------------------------------

if __name__ == "__main__":

    players = load_players()
    model = prepare_recommender(players)

    print("=" * 55)
    print("CRICKET PLAYER FINDER - RECOMMENDATION ENGINE")
    print("=" * 55)
    print("Total unique players:", len(players))

    print("\nPlayers with bowling records:",
          int(players["has_bowling_sample"].sum()))

    print("Players without recorded legal bowling balls:",
          int((~players["has_bowling_sample"]).sum()))

    # Select a test player.
    test_player_name = "DA Warner"

    matches = players.index[
        players["player"].str.casefold()
        == test_player_name.casefold()
    ]

    if len(matches) == 0:
        print(f"\n{test_player_name} was not found.")
        print("Example players:")
        print(players["player"].head(15).to_string(index=False))

    else:
        selected_index = int(matches[0])

        for mode in ["Batting", "Bowling", "Overall"]:

            print(f"\n{'-' * 55}")
            print(f"TOP 5 {mode.upper()} RECOMMENDATIONS")
            print("-" * 55)

            recommendations = recommend_similar_players(
                players=players,
                model=model,
                player_index=selected_index,
                mode=mode,
                number=5,
            )

            if recommendations.empty:
                print("No recommendations available.")
            else:
                display_columns = [
                    "player",
                    "similarity_score",
                    "batting_similarity",
                    "bowling_similarity",
                    "wickets",
                    "economy",
                ]

                print(
                    recommendations[display_columns].to_string(
                        index=False,
                        float_format=lambda value: f"{value:.3f}",
                    )
                )

    print("\nRecommendation engine test completed.")
