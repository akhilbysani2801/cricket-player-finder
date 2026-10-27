
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
    file for file in DATA_FOLDER.glob("*.csv")
    if not file.name.endswith("_info.csv")
    and file.name not in {
        "all_matches.csv",
        "player_statistics.csv",
        "player_season_statistics.csv",
        "player_team_history.csv",
    }
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
        match_data.append(pd.read_csv(file))
    except Exception as error:
        print(f"Could not load {file.name}: {error}")

    if index % 100 == 0:
        print(f"Processed loading for {index} files...")

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
    column for column in required_columns
    if column not in data.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

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
        data[column], errors="coerce"
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
    data["season"], errors="coerce"
)

data = data.dropna(
    subset=["season", "match_id"]
).copy()

data["season"] = data["season"].astype(int)

data = data[
    data["match_id"].astype(str).str.strip().ne("")
].copy()

data["total_runs"] = data["runs_off_bat"]

# A wide is not a ball faced. A no-ball is counted as faced.
data["ball_faced"] = (data["wides"] == 0).astype(int)

# Standard bowler-credited dismissal types.
bowler_dismissal_types = {
    "bowled",
    "caught",
    "caught and bowled",
    "lbw",
    "stumped",
    "hit wicket",
}

# Dismissals counted against a batter's average.
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
# 5. CALCULATE BATTING STATISTICS
# ==================================================

def calculate_batting(frame, group_columns):
    """
    Calculate batting statistics for career or season groups.
    Dismissal records are grouped using the same keys as batting.
    """

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
            batting["balls_faced"].replace(
                0, float("nan")
            )
        )
        * 100
    ).fillna(0)

    # IMPORTANT:
    # Do not select group_columns + ["player_dismissed"].
    # For career grouping this would select "striker" twice
    # after renaming, causing the Pandas grouping error.

    player_column = group_columns[-1]
    preceding_columns = group_columns[:-1]
    dismissal_frames = []

    # First recorded dismissed batter.
    first = frame.loc[
        frame["player_dismissed"].ne("")
        & frame["wicket_type"].isin(
            batter_dismissal_types
        ),
        preceding_columns + ["player_dismissed"],
    ].copy()

    if not first.empty:
        first = first.rename(
            columns={
                "player_dismissed": player_column
            }
        )
        dismissal_frames.append(first)

    # Additional dismissed batter, when recorded.
    second = frame.loc[
        frame["other_player_dismissed"].ne("")
        & frame["other_wicket_type"].isin(
            batter_dismissal_types
        ),
        preceding_columns + ["other_player_dismissed"],
    ].copy()

    if not second.empty:
        second = second.rename(
            columns={
                "other_player_dismissed": player_column
            }
        )
        dismissal_frames.append(second)

    # Count dismissals by exactly the same keys used for batting.
    if dismissal_frames:
        dismissal_events = pd.concat(
            dismissal_frames,
            ignore_index=True,
        )

        dismissal_counts = dismissal_events.groupby(
            group_columns,
            dropna=False,
        ).size()

        batting["dismissals"] = (
            dismissal_counts
            .reindex(batting.index)
            .fillna(0)
            .astype(int)
        )
    else:
        batting["dismissals"] = 0

    # Zero dismissals means undefined batting average.
    # Do not replace NaN with zero.
    batting["batting_average"] = (
        batting["total_runs"]
        .div(
            batting["dismissals"].replace(
                0, float("nan")
            )
        )
    )

    return batting


# ==================================================
# 6. CALCULATE BOWLING STATISTICS
# ==================================================

def calculate_bowling(frame, group_columns):
    """Calculate bowling statistics by career or season."""

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
    )

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

# Fill counting statistics only.
# Preserve undefined batting and bowling averages.
career_count_columns = [
    "total_runs",
    "balls_faced",
    "fours",
    "sixes",
    "dismissals",
    "runs_conceded",
    "wickets",
    "legal_balls",
    "overs_bowled",
]

for column in career_count_columns:
    if column in players.columns:
        players[column] = players[column].fillna(0)

for column in ["strike_rate", "economy"]:
    if column in players.columns:
        players[column] = players[column].fillna(0)

players = players.reset_index()

career_columns = [
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

players = players.reindex(
    columns=career_columns,
)

players.to_csv(
    CAREER_FILE,
    index=False,
)

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

season_batting.index = season_batting.index.set_names(
    ["season", "player"]
)

season_bowling.index = season_bowling.index.set_names(
    ["season", "player"]
)

season_stats = season_batting.join(
    season_bowling,
    how="outer",
)

season_count_columns = [
    "total_runs",
    "balls_faced",
    "fours",
    "sixes",
    "dismissals",
    "runs_conceded",
    "wickets",
    "legal_balls",
    "overs_bowled",
]

for column in season_count_columns:
    if column in season_stats.columns:
        season_stats[column] = (
            season_stats[column].fillna(0)
        )

for column in ["strike_rate", "economy"]:
    if column in season_stats.columns:
        season_stats[column] = (
            season_stats[column].fillna(0)
        )

# Keep batting_average and bowling_average NaN when undefined.
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

# Include recorded appearances by players without a batting
# or bowling row, such as a non-striker.
appearance_candidates = []

for player_column in [
    "striker",
    "non_striker",
    "bowler",
]:
    subset = data.loc[
        data[player_column].ne(""),
        ["season", player_column],
    ].copy()

    subset.columns = ["season", "player"]
    appearance_candidates.append(subset)

appearances = pd.concat(
    appearance_candidates,
    ignore_index=True,
).drop_duplicates()

existing_keys = season_stats[
    ["season", "player"]
]

missing_keys = appearances.merge(
    existing_keys,
    on=["season", "player"],
    how="left",
    indicator=True,
)

missing_keys = missing_keys.loc[
    missing_keys["_merge"] == "left_only",
    ["season", "player"],
]

if not missing_keys.empty:
    zero_rows = missing_keys.copy()

    for column in season_columns:
        if column not in ["season", "player"]:
            zero_rows[column] = 0

    season_stats = pd.concat(
        [
            season_stats,
            zero_rows[season_columns],
        ],
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


def add_appearances(player_column, team_column):
    """Record players associated with teams in each match."""

    subset = data.loc[
        data[player_column].ne("")
        & data[team_column].ne(""),
        [
            "match_id",
            "season",
            player_column,
            team_column,
        ],
    ].copy()

    subset = subset.rename(
        columns={
            player_column: "player",
            team_column: "team",
        }
    )

    appearance_frames.append(subset)


# Batters belong to the batting team.
add_appearances("striker", "batting_team")
add_appearances("non_striker", "batting_team")

# Bowlers and fielders belong to the bowling team.
add_appearances("bowler", "bowling_team")

for fielder_column in fielder_columns:
    add_appearances(
        fielder_column,
        "bowling_team",
    )

# Dismissed batters belong to the batting team.
add_appearances(
    "player_dismissed",
    "batting_team",
)

add_appearances(
    "other_player_dismissed",
    "batting_team",
)

all_appearances = pd.concat(
    appearance_frames,
    ignore_index=True,
)

all_appearances = all_appearances[
    all_appearances["player"].ne("")
    & all_appearances["team"].ne("")
].copy()

# Count each player-team-match only once.
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
    "Players with no legal bowling balls:",
    int((players["legal_balls"] == 0).sum()),
)

print(
    "Seasons covered:",
    sorted(data["season"].unique().tolist()),
)

print(
    "Unique batting teams:",
    data["batting_team"].nunique(),
)

print("\nGenerated files:")
print("1.", CAREER_FILE)
print("2.", SEASON_FILE)
print("3.", TEAM_FILE)

print("\nSample career batting averages:")
print(
    players[
        [
            "player",
            "total_runs",
            "dismissals",
            "batting_average",
        ]
    ].head(10).to_string(index=False)
)

print("\nSample season statistics:")
print(season_stats.head(10).to_string(index=False))

print("\nAll processing finished.")
