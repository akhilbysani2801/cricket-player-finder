
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Load the player statistics
file_path = Path("processed_data/player_statistics.csv")
players = pd.read_csv(file_path)

# Step 2: Select statistics used to compare players
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

# Step 3: Clean invalid or missing values
features = players[feature_columns].copy()
features = features.apply(pd.to_numeric, errors="coerce")
features = features.fillna(0)
features = features.replace([float("inf"), float("-inf")], 0)

# Step 4: Scale the statistics
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Step 5: Calculate similarity between every pair of players
similarity_matrix = cosine_similarity(scaled_features)

# Step 6: Recommend the five most similar players
def recommend_players(player_name, number=5):
    matching_players = players[
        players["player"].str.casefold() == player_name.casefold()
    ]

    if matching_players.empty:
        print(f"Player '{player_name}' was not found.")
        print("Try a name from the available player list.")
        return

    player_index = matching_players.index[0]

    similarity_scores = list(enumerate(similarity_matrix[player_index]))

    # Exclude the selected player
    similarity_scores = [
        (index, score)
        for index, score in similarity_scores
        if index != player_index
    ]

    # Sort from highest similarity to lowest
    similarity_scores.sort(key=lambda item: item[1], reverse=True)
    top_players = similarity_scores[:number]

    result_columns = [
        "player",
        "total_runs",
        "batting_average",
        "strike_rate",
        "wickets",
        "economy"
    ]

    print(f"\nPlayers most similar to {player_name}:")
    print("-" * 75)

    for rank, (index, score) in enumerate(top_players, start=1):
        player = players.iloc[index]

        print(f"\n{rank}. {player['player']}")
        print(f"   Similarity score: {score:.3f}")

        for column in result_columns[1:]:
            print(f"   {column.replace('_', ' ').title()}: "
                  f"{player[column]:.2f}")

# Step 7: Display some player names for testing
print("CRICKET PLAYER FINDER")
print("=" * 40)
print("Total players:", len(players))
print("\nExample player names:")
print(players["player"].head(15).to_string(index=False))

# Step 8: Test the recommender
recommend_players("DA Warner")
