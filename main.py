
import pandas as pd
from pathlib import Path


# ==================================================
# 1. CONFIGURATION
# ==================================================

DATA_FOLDER = Path(".")
OUTPUT_FOLDER = Path("processed_data")

OUTPUT_FOLDER.mkdir(exist_ok=True)

CAREER_FILE = OUTPUT_FOLDER / "player_statistics.csv"
SEASON_FILE = OUTPUT_FOLDER / "player_season_statistics.csv"
TEAM_FILE = OUTPUT_FOLDER / "player_team_history.csv"


# ==================================================
# 2. LOAD ORIGINAL IPL MATCH FILES
# ==================================================

csv_files = sorted([
    file
    for file in DATA_FOLDER.glob("*.csv")
    if not file.name.endswith("_info.csv")
    and file.name != "all_matches.csv"
    and file.name != "player_statistics.csv"
])

print("=" * 60)
print("CRICKET PLAYER FINDER - DATA PROCESSING")
print("=" * 60)

print("Match files found:", len(csv_files))

if not csv_files:
    raise FileNotFoundError(
        "No original match CSV files found in the project folder."
    )

match_data = []

for index, file in enumerate(csv_files, start=1):
    try:
        df = pd.read_csv(file)
        match_data.append(df)
    except Exception as error:
        print(f"Could not load {file.name}: {error}")

    if index % 100 == 0:
        print(f"Loaded {index} files...")

if not match_data:
    raise ValueError("No match data could be loaded.")

data = pd.concat(match_data, ignore_index=True)

print("Total delivery records:", len(data))


# ==================================================
# 3. VALIDATE COLUMNS
# ==================================================

required_columns = [
    "match_id",
    "season",
    "batting_team",
    "bowling_team",
    "striker",
    "non_striker",
    "bowler",
    "runs_off_bat",
    "wides",
    "noballs",
    "byes",
    "legbyes",
    "penalty",
    "wicket_type",
    "player_dismissed",
    "other_wicket_type",
    "other_player_dismissed",
]

missing_columns = [
    column
    for column in required_columns
    if column not in data.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

# Fielding columns are optional because some datasets
# may not include them.
fielder_columns = [
    column
    for column in ["fielder_1", "fielder_2", "fielder_3"]
    if column in data.columns
]


# ==================================================
# 4. CLEAN DATA
# ==================================================

numeric_columns = [
    "runs_off_bat",
    "wides",
    "noballs",
    "byes",
    "legbyes",
    "penalty",
]

if "extras" in data.columns:
    numeric_columns.append("extras")

for column in numeric_columns:
    data[column] = pd.to_numeric(
        data[column],
        errors="coerce",
    ).fillna(0).clip(lower=0)

text_columns = [
    "striker",
    "non_striker",
    "bowler",
    "batting_team",
    "bowling_team",
    "wicket_type",
    "player_dismissed",
    "other_wicket_type",
    "other_player_dismissed",
    *fielder_columns,
]

for column in text_columns:
    data[column] = (
        data[column]
        .fillna("")
        .astype(str)
        .str.strip()
    )

data["wicket_type"] = data["wicket_type"].str.lower()
data["other_wicket_type"] = (
    data["other_wicket_type"].str.lower()
)

data["season"] = pd.to_numeric(
    data["season"],
    errors="coerce",
)

data = data.dropna(
    subset=["season", "match_id"]
).copy()

data["season"] = data["season"].astype(int)

# Remove rows with no usable match or season identifier.
data = data[
    data["match_id"].astype(str).str.strip().ne("")
].copy()

data["total_runs"] = data["runs_off_bat"]

# A wide is not a ball faced. A no-ball is counted as faced.
data["ball_faced"] = (
    data["wides"] == 0
).astype(int)

# Standard credited bowler dismissals.
bowler_dismissal_types = {
    "bowled",
    "caught",
    "caught and bowled",
    "lbw",
    "stumped",
    "hit wicket",
}

# Dismissals counted against the batter's average.
batter_dismissal_types = {
    "bowled",
    "caught",
    "caught and bowled",
    "lbw",
    "stumped",
    "hit wicket",
    "run out",
    "retired out",
    "obstructing the field",
    "hit the ball twice",
    "handled the ball",
    "timed out",
}

data["bowler_runs_conceded"] = (
    data["runs_off_bat"]
    + data["wides"]
    + data["noballs"]
)

data["bowler_wickets"] = (
    data["wicket_type"]
    .isin(bowler_dismissal_types)
    .astype(int)
    +
    data["other_wicket_type"]
    .isin(bowler_dismissal_types)
    .astype(int)
)

data["legal_balls"] = (
    (data["wides"] == 0)
    & (data["noballs"] == 0)
).astype(int)


# ==================================================
# 5. HELPER: CALCULATE BATTING STATISTICS
# ==================================================

def calculate_batting(frame, group_columns):
    """Calculate batting statistics for career or season groups."""

    batting_data = frame[
        frame["striker"] != ""
    ].copy()

    if batting_data.empty:
        return pd.DataFrame()

    batting = batting_data.groupby(
        group_columns,
        dropna=False,
    ).agg(
        total_runs=("total_runs", "sum"),
        balls_faced=("ball_faced", "sum"),
        fours=(
            "runs_off_bat",
            lambda values: (values == 4).sum(),
        ),
        sixes=(
            "runs_off_bat",
            lambda values: (values == 6).sum(),
        ),
    )

    batting["strike_rate"] = (
        batting["total_runs"]
        .div(
            batting["balls_faced"].replace(0, float("nan"))
        )
        * 100
    ).fillna(0)

    first_dismissal = (
        frame["player_dismissed"].ne("")
        & frame["wicket_type"].isin(
            batter_dismissal_types
        )
    )

    first_counts = (
        frame.loc[first_dismissal]
        .groupby(
            group_columns + ["player_dismissed"],
            dropna=False,
        )
        .size()
    )

    second_dismissal = (
        frame["other_player_dismissed"].ne("")
        & frame["other_wicket_type"].isin(
            batter_dismissal_types
        )
    )

    second_counts = (
        frame.loc[second_dismissal]
        .groupby(
            group_columns + ["other_player_dismissed"],
            dropna=False,
        )
        .size()
    )

    # Map dismissal counts back to the player/group keys.
    dismissal_counts = {}

    for key, count in first_counts.items():
        group_key = key[:-1]
        player_name = key[-1]
        dismissal_counts[(group_key, player_name)] = (
            dismissal_counts.get((group_key, player_name), 0)
            + count
        )

    for key, count in second_counts.items():
        group_key = key[:-1]
        player_name = key[-1]
        dismissal_counts[(group_key, player_name)] = (
            dismissal_counts.get((group_key, player_name), 0)
            + count
        )

    dismissal_values = []

    for index_value in batting.index:
        if not isinstance(index_value, tuple):
            index_value = (index_value,)

        group_key = tuple(index_value)
        player_name = group_key[-1]
        season_or_career_key = group_key[:-1]

        dismissal_values.append(
            dismissal_counts.get(
                (season_or_career_key, player_name),
                0,
            )
        )

    batting["dismissals"] = dismissal_values

    batting["batting_average"] = (
        batting["total_runs"]
        .div(
            batting["dismissals"].replace(0, float("nan"))
        )
    ).fillna(0)

    return batting


# ==================================================
# 6. HELPER: CALCULATE BOWLING STATISTICS
# ==================================================

def calculate_bowling(frame, group_columns):
    """Calculate bowling statistics for career or season groups."""

    bowling_data = frame[
        frame["bowler"] != ""
    ].copy()

    if bowling_data.empty:
        return pd.DataFrame()

    bowling = bowling_data.groupby(
        group_columns,
        dropna=False,
    ).agg(
        runs_conceded=("bowler_runs_conceded", "sum"),
        wickets=("bowler_wickets", "sum"),
        legal_balls=("legal_balls", "sum"),
    )

    bowling["overs_bowled"] = (
        bowling["legal_balls"] / 6
    )

    bowling["economy"] = (
        bowling["runs_conceded"]
        .div(
            bowling["overs_bowled"].replace(
                0, float("nan")
            )
        )
    ).fillna(0)

    bowling["bowling_average"] = (
        bowling["runs_conceded"]
        .div(
            bowling["wickets"].replace(
                0, float("nan")
            )
        )
    ).fillna(0)

    return bowling


# ==================================================
# 7. CAREER STATISTICS
# ==================================================

print("\nCalculating career statistics...")

career_batting = calculate_batting(
    data,
    ["striker"],
)

career_bowling = calculate_bowling(
    data,
    ["bowler"],
)

career_batting.index.name = "player"
career_bowling.index.name = "player"

players = career_batting.join(
    career_bowling,
    how="outer",
)

players = players.fillna(0)

career_columns = [
    "total_runs",
    "balls_faced",
    "fours",
    "sixes",
    "strike_rate",
    "dismissals",
    "batting_average",
    "runs_conceded",
    "wickets",
    "legal_balls",
    "overs_bowled",
    "economy",
    "bowling_average",
]

players = players.reindex(
    columns=career_columns,
    fill_value=0,
)

players.to_csv(CAREER_FILE, index=True)

print("Career statistics saved:", CAREER_FILE)
print("Career player records:", len(players))


# ==================================================
# 8. SEASON-WISE PLAYER STATISTICS
# ==================================================

print("\nCalculating season-wise statistics...")

season_batting = calculate_batting(
    data,
    ["season", "striker"],
)

season_bowling = calculate_bowling(
    data,
    ["season", "bowler"],
)

# Standardise the player key before joining.
season_batting.index = season_batting.index.set_names(
    ["season", "player"]
)

season_bowling.index = season_bowling.index.set_names(
    ["season", "player"]
)

season_stats = season_batting.join(
    season_bowling,
    how="outer",
).fillna(0)

season_stats = season_stats.reset_index()

season_columns = [
    "season",
    "player",
    "total_runs",
    "balls_faced",
    "fours",
    "sixes",
    "strike_rate",
    "dismissals",
    "batting_average",
    "runs_conceded",
    "wickets",
    "legal_balls",
    "overs_bowled",
    "economy",
    "bowling_average",
]

season_stats = season_stats.reindex(
    columns=season_columns,
)

# Include players who appeared in a match but had no recorded
# batting or bowling delivery, such as a non-striker.
appearance_candidates = []

for player_column in [
    "striker",
    "non_striker",
    "bowler",
]:
    subset = data[
        data[player_column].ne("")
    ][["season", player_column]].copy()

    subset.columns = ["season", "player"]
    appearance_candidates.append(subset)

appearances = pd.concat(
    appearance_candidates,
    ignore_index=True,
).drop_duplicates()

season_keys = season_stats[["season", "player"]]

missing_appearance_keys = (
    appearances.merge(
        season_keys,
        on=["season", "player"],
        how="left",
        indicator=True,
    )
)

missing_appearance_keys = missing_appearance_keys[
    missing_appearance_keys["_merge"] == "left_only"
][["season", "player"]]

if not missing_appearance_keys.empty:
    zero_rows = missing_appearance_keys.copy()

    for column in season_columns:
        if column not in ["season", "player"]:
            zero_rows[column] = 0

    season_stats = pd.concat(
        [season_stats, zero_rows[season_columns]],
        ignore_index=True,
    )

season_stats = season_stats.sort_values(
    ["player", "season"]
).reset_index(drop=True)

season_stats.to_csv(
    SEASON_FILE,
    index=False,
)

print("Season statistics saved:", SEASON_FILE)
print("Player-season records:", len(season_stats))


# ==================================================
# 9. PLAYER TEAM HISTORY
# ==================================================

print("\nBuilding player team history...")

appearance_frames = []


def add_appearances(
    player_column,
    team_column,
    role,
):
    """Create player-team-season appearance records."""

    subset = data[
        data[player_column].ne("")
        & data[team_column].ne("")
    ][
        [
            "match_id",
            "season",
            player_column,
            team_column,
        ]
    ].copy()

    subset = subset.rename(
        columns={
            player_column: "player",
            team_column: "team",
        }
    )

    subset["role"] = role

    appearance_frames.append(subset)


# Batting players belong to the batting team.
add_appearances(
    "striker",
    "batting_team",
    "batter",
)

add_appearances(
    "non_striker",
    "batting_team",
    "non-striker",
)

# Bowlers and fielders belong to the bowling team.
add_appearances(
    "bowler",
    "bowling_team",
    "bowler",
)

for fielder_column in fielder_columns:
    add_appearances(
        fielder_column,
        "bowling_team",
        "fielder",
    )

# Dismissed batters belong to the batting team.
add_appearances(
    "player_dismissed",
    "batting_team",
    "dismissed batter",
)

add_appearances(
    "other_player_dismissed",
    "batting_team",
    "dismissed batter",
)

all_appearances = pd.concat(
    appearance_frames,
    ignore_index=True,
)

all_appearances = all_appearances[
    all_appearances["player"].ne("")
    & all_appearances["team"].ne("")
].copy()

# Remove duplicate player-team-match records so each match counts once.
all_appearances = all_appearances.drop_duplicates(
    subset=[
        "match_id",
        "season",
        "player",
        "team",
    ]
)

team_history = all_appearances.groupby(
    ["player", "team", "season"],
    as_index=False,
).agg(
    matches_played=("match_id", "nunique"),
)

team_history = team_history.sort_values(
    ["player", "season", "team"]
).reset_index(drop=True)

team_history.to_csv(
    TEAM_FILE,
    index=False,
)

print("Team history saved:", TEAM_FILE)
print("Player-team-season records:", len(team_history))


# ==================================================
# 10. DATA QUALITY SUMMARY
# ==================================================

print("\n" + "=" * 60)
print("DATA PROCESSING COMPLETED")
print("=" * 60)

print("Career players:", len(players))
print(
    "Players with bowling records:",
    int((players["legal_balls"] > 0).sum()),
)
print(
    "Players with zero legal bowling balls:",
    int((players["legal_balls"] == 0).sum()),
)

print("Seasons covered:", sorted(data["season"].unique().tolist()))
print("Unique teams:", data["batting_team"].nunique())

print("\nGenerated files:")
print("1.", CAREER_FILE)
print("2.", SEASON_FILE)
print("3.", TEAM_FILE)

print("\nSample season statistics:")
print(season_stats.head(10).to_string(index=False))

print("\nSample team history:")
print(team_history.head(10).to_string(index=False))

print("\nAll processing finished.")
