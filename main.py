
import pandas as pd
from pathlib import Path

# Step 1: Find all match CSV files
data_folder = Path(".")
csv_files = [
    file for file in data_folder.glob("*.csv")
    if not file.name.endswith("_info.csv")
    and file.name != "all_matches.csv"
]

print("Number of match files:", len(csv_files))

# Step 2: Load and combine all matches
match_data = []

for i, file in enumerate(csv_files, start=1):
    try:
        df = pd.read_csv(file)
        match_data.append(df)
    except Exception as error:
        print(f"Could not load {file.name}: {error}")

    if i % 100 == 0:
        print(f"Loaded {i} match files...")

if not match_data:
    raise ValueError("No match data was loaded.")

data = pd.concat(match_data, ignore_index=True)

print("\nAll match data combined.")
print("Total deliveries:", len(data))
print("Total columns:", len(data.columns))

# Step 3: Fill missing numeric event values with zero
for column in ["runs_off_bat", "extras", "wides", "noballs",
               "byes", "legbyes", "penalty"]:
    if column in data.columns:
        data[column] = data[column].fillna(0)

# Step 4: Calculate batting statistics
data["total_runs"] = data["runs_off_bat"]

# A legal ball faced excludes wides
data["ball_faced"] = (data["wides"] == 0).astype(int)

batting = data.groupby("striker").agg(
    total_runs=("runs_off_bat", "sum"),
    balls_faced=("ball_faced", "sum"),
    fours=("runs_off_bat", lambda x: (x == 4).sum()),
    sixes=("runs_off_bat", lambda x: (x == 6).sum())
)

# Strike rate = runs per 100 balls faced
batting["strike_rate"] = (
    batting["total_runs"] / batting["balls_faced"].replace(0, pd.NA)
) * 100

# Step 5: Calculate dismissals for batting average
dismissal_types = [
    "bowled", "caught", "caught and bowled", "lbw",
    "stumped", "hit wicket", "run out", "retired out",
    "obstructing the field"
]

dismissals = data[
    data["player_dismissed"].notna()
    & data["wicket_type"].isin(dismissal_types)
].groupby("player_dismissed").size()

batting["dismissals"] = dismissals.reindex(
    batting.index, fill_value=0
)

batting["batting_average"] = (
    batting["total_runs"] /
    batting["dismissals"].replace(0, pd.NA)
)

# Step 6: Calculate bowling statistics
# Exclude byes, legbyes and penalty from bowler-conceded runs.
data["bowler_runs_conceded"] = (
    data["runs_off_bat"] + data["wides"] + data["noballs"]
)

# Wickets credited to the bowler
bowler_dismissals = ["bowled", "caught", "caught and bowled",
                     "lbw", "stumped", "hit wicket"]

data["bowler_wicket"] = (
    data["wicket_type"].isin(bowler_dismissals)
).astype(int)

# Legal deliveries exclude wides and no-balls
data["legal_balls"] = (
    (data["wides"] == 0) & (data["noballs"] == 0)
).astype(int)

bowling = data.groupby("bowler").agg(
    runs_conceded=("bowler_runs_conceded", "sum"),
    wickets=("bowler_wicket", "sum"),
    legal_balls=("legal_balls", "sum")
)

bowling["overs_bowled"] = bowling["legal_balls"] / 6

bowling["economy"] = (
    bowling["runs_conceded"] /
    bowling["overs_bowled"].replace(0, pd.NA)
)

# Step 7: Combine batting and bowling statistics
players = batting.join(bowling, how="outer").fillna(0)

# Step 8: Save the player statistics
output_folder = Path("processed_data")
output_folder.mkdir(exist_ok=True)

output_file = output_folder / "player_statistics.csv"
players.to_csv(output_file, index_label="player")

print("\nPlayer statistics created!")
print("Number of players:", len(players))
print("Saved to:", output_file)

print("\nSample player statistics:")
print(players.head(10).to_string())
