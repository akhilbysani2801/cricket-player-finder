
from pathlib import Path

import pandas as pd
import streamlit as st

from recommender import (
    load_players,
    prepare_recommender,
    recommend_similar_players,
)


# =========================================================
# 1. CONFIGURATION
# =========================================================

DATA_DIR = Path("processed_data")

CAREER_FILE = DATA_DIR / "player_statistics.csv"
SEASON_FILE = DATA_DIR / "player_season_statistics.csv"
TEAM_FILE = DATA_DIR / "player_team_history.csv"

st.set_page_config(
    page_title="Cricket Player Finder",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# 2. CUSTOM DESIGN
# =========================================================

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(
            135deg,
            #dce5ef 0%,
            #edf1f6 100%
        );
    }

    [data-testid="stSidebar"] {
        background-color: #eaf0f6;
        border-right: 1px solid #cbd6e2;
    }

    .main-title {
        color: #173b5e;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #50677e;
        font-size: 1rem;
        margin-bottom: 1.4rem;
    }

    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #d5e0eb;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 2px 7px rgba(25, 55, 85, 0.05);
    }

    div[data-testid="stMetricLabel"] {
        color: #50677e;
    }

    div[data-testid="stMetricValue"] {
        color: #173b5e;
    }

    h1, h2, h3 {
        color: #173b5e;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 3. LOAD DATA
# =========================================================

@st.cache_data
def load_dashboard_data():
    if not CAREER_FILE.exists():
        raise FileNotFoundError(
            f"Missing file: {CAREER_FILE}. Run python main.py first."
        )

    if not SEASON_FILE.exists():
        raise FileNotFoundError(
            f"Missing file: {SEASON_FILE}. Run python main.py first."
        )

    if not TEAM_FILE.exists():
        raise FileNotFoundError(
            f"Missing file: {TEAM_FILE}. Run python main.py first."
        )

    career = pd.read_csv(CAREER_FILE)
    season = pd.read_csv(SEASON_FILE)
    teams = pd.read_csv(TEAM_FILE)

    # Handle the player index written by main.py.
    if "player" not in career.columns:
        first_column = career.columns[0]
        career = career.rename(
            columns={first_column: "player"}
        )

    required_career = [
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

    for column in required_career:
        if column not in career.columns:
            career[column] = 0

    required_season = [
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

    missing_season = [
        column
        for column in required_season
        if column not in season.columns
    ]

    if missing_season:
        raise ValueError(
            "Season CSV is missing columns: "
            + ", ".join(missing_season)
        )

    required_teams = [
        "player",
        "team",
        "season",
        "matches_played",
    ]

    missing_teams = [
        column
        for column in required_teams
        if column not in teams.columns
    ]

    if missing_teams:
        raise ValueError(
            "Team history CSV is missing columns: "
            + ", ".join(missing_teams)
        )

    # Standardise names and numeric columns.
    for frame in [career, season, teams]:
        frame["player"] = (
            frame["player"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    career = career[career["player"] != ""].copy()
    season = season[season["player"] != ""].copy()
    teams = teams[teams["player"] != ""].copy()

    numeric_columns = [
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

    for column in numeric_columns:
        career[column] = pd.to_numeric(
            career[column], errors="coerce"
        ).fillna(0)

        season[column] = pd.to_numeric(
            season[column], errors="coerce"
        ).fillna(0)

    season["season"] = pd.to_numeric(
        season["season"], errors="coerce"
    )

    teams["season"] = pd.to_numeric(
        teams["season"], errors="coerce"
    )

    teams["matches_played"] = pd.to_numeric(
        teams["matches_played"], errors="coerce"
    ).fillna(0)

    career = career.drop_duplicates(
        subset=["player"]
    ).reset_index(drop=True)

    season = season.dropna(
        subset=["season"]
    ).copy()

    season["season"] = season["season"].astype(int)

    teams = teams.dropna(
        subset=["season"]
    ).copy()

    teams["season"] = teams["season"].astype(int)

    return career, season, teams


@st.cache_resource
def load_recommendation_engine():
    players = load_players()
    model = prepare_recommender(players)
    return players, model


try:
    career_data, season_data, team_data = load_dashboard_data()
    recommender_players, recommender_model = (
        load_recommendation_engine()
    )
except Exception as error:
    st.error(f"Could not load project data: {error}")
    st.info(
        "Check your processed_data folder and run "
        "`python main.py` if necessary."
    )
    st.stop()


# =========================================================
# 4. HELPER FUNCTIONS
# =========================================================

def number(value):
    """Format a number without unnecessary decimal places."""
    try:
        value = float(value)
        if pd.isna(value):
            return "0"
        return f"{value:,.0f}"
    except (TypeError, ValueError):
        return "0"


def decimal(value, places=2):
    """Format a decimal value."""
    try:
        value = float(value)
        if pd.isna(value):
            return "0"
        return f"{value:,.{places}f}"
    except (TypeError, ValueError):
        return "0"


def get_career(player_name):
    result = career_data[
        career_data["player"] == player_name
    ]

    if result.empty:
        return None

    return result.iloc[0]


def get_seasons(player_name):
    result = season_data[
        season_data["player"] == player_name
    ].copy()

    return result.sort_values("season")


def get_teams(player_name):
    result = team_data[
        team_data["player"] == player_name
    ].copy()

    return result.sort_values(
        ["season", "team"]
    )


def get_best_batting_season(player_name):
    seasons = get_seasons(player_name)

    if seasons.empty:
        return None

    return seasons.sort_values(
        ["total_runs", "season"],
        ascending=[False, True],
    ).iloc[0]


def get_best_bowling_season(player_name):
    seasons = get_seasons(player_name)

    if seasons.empty:
        return None

    # Only consider seasons in which the player bowled.
    seasons = seasons[seasons["legal_balls"] > 0]

    if seasons.empty:
        return None

    return seasons.sort_values(
        ["wickets", "season"],
        ascending=[False, True],
    ).iloc[0]


def career_team_summary(player_name):
    history = get_teams(player_name)

    if history.empty:
        return {
            "teams": [],
            "latest_team": "Not available",
            "team_count": 0,
        }

    team_names = history["team"].dropna().unique().tolist()

    latest_season = history["season"].max()

    latest_teams = history.loc[
        history["season"] == latest_season,
        "team",
    ].dropna().unique().tolist()

    latest_team = ", ".join(latest_teams)

    return {
        "teams": team_names,
        "latest_team": latest_team or "Not available",
        "team_count": len(team_names),
    }


def show_player_metrics(player_name):
    row = get_career(player_name)

    if row is None:
        st.warning("No career statistics found.")
        return

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Career Runs", number(row["total_runs"]))
    col2.metric("Strike Rate", decimal(row["strike_rate"]))
    col3.metric("Batting Average", decimal(row["batting_average"]))
    col4.metric("Wickets", number(row["wickets"]))

    col5, col6, col7, col8 = st.columns(4)

    col5.metric("Fours", number(row["fours"]))
    col6.metric("Sixes", number(row["sixes"]))
    col7.metric("Economy", decimal(row["economy"]))
    col8.metric("Runs Conceded", number(row["runs_conceded"]))


# =========================================================
# 5. SIDEBAR NAVIGATION
# =========================================================

st.sidebar.markdown("## 🏏 Cricket Player Finder")
st.sidebar.caption("IPL Performance & Recommendation Analytics")

page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Player Explorer",
        "Compare Players",
        "Team History",
        "Similar Players",
    ],
)

all_players = sorted(
    career_data["player"].dropna().unique().tolist()
)

if not all_players:
    st.error("No player records found.")
    st.stop()

st.sidebar.divider()
st.sidebar.caption(
    f"{len(all_players):,} player records loaded"
)


# =========================================================
# 6. OVERVIEW
# =========================================================

if page == "Overview":

    st.markdown(
        '<div class="main-title">🏏 Cricket Player Finder</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">'
        'Explore IPL careers, seasonal performance, team history, '
        'and data-driven player recommendations.'
        '</div>',
        unsafe_allow_html=True,
    )

    total_players = career_data["player"].nunique()
    total_runs = career_data["total_runs"].sum()
    total_wickets = career_data["wickets"].sum()
    total_teams = team_data["team"].nunique()

    seasons = sorted(
        season_data["season"].dropna().unique().tolist()
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Players", number(total_players))
    c2.metric("Runs in Dataset", number(total_runs))
    c3.metric("Wickets in Dataset", number(total_wickets))
    c4.metric("Teams Recorded", number(total_teams))

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("🏆 Top Run Scorers")

        top_batters = career_data.nlargest(
            10, "total_runs"
        )[
            ["player", "total_runs", "strike_rate"]
        ].rename(
            columns={
                "player": "Player",
                "total_runs": "Runs",
                "strike_rate": "Strike Rate",
            }
        )

        st.bar_chart(
            top_batters.set_index("Player")["Runs"]
        )

        st.dataframe(
            top_batters,
            hide_index=True,
            use_container_width=True,
        )

    with right:
        st.subheader("🎯 Top Wicket Takers")

        top_bowlers = career_data.nlargest(
            10, "wickets"
        )[
            ["player", "wickets", "economy"]
        ].rename(
            columns={
                "player": "Player",
                "wickets": "Wickets",
                "economy": "Economy",
            }
        )

        st.bar_chart(
            top_bowlers.set_index("Player")["Wickets"]
        )

        st.dataframe(
            top_bowlers,
            hide_index=True,
            use_container_width=True,
        )

    st.divider()
    st.subheader("📅 Season Coverage")

    if seasons:
        st.write(
            f"Seasons available in the processed dataset: "
            f"{int(min(seasons))}–{int(max(seasons))}"
        )

        season_runs = (
            season_data.groupby("season")["total_runs"]
            .sum()
            .sort_index()
        )

        st.line_chart(season_runs)

    st.caption(
        "Overview totals describe the records available in your "
        "dataset; they are not necessarily official league totals."
    )


# =========================================================
# 7. PLAYER EXPLORER
# =========================================================

elif page == "Player Explorer":

    st.title("🔎 Player Explorer")
    st.write(
        "Explore a player's career statistics, best seasons, "
        "and year-by-year progress."
    )

    selected_player = st.selectbox(
        "Select a player",
        all_players,
        key="explorer_player",
    )

    row = get_career(selected_player)

    if row is not None:
        st.subheader(selected_player)
        show_player_metrics(selected_player)

        team_summary = career_team_summary(selected_player)

        st.markdown("### Career Snapshot")

        a, b, c = st.columns(3)

        a.metric(
            "Teams Recorded",
            team_summary["team_count"],
        )

        b.metric(
            "Most Recent Team in Dataset",
            team_summary["latest_team"],
        )

        player_teams = get_teams(selected_player)

        if not player_teams.empty:
            a_seasons = player_teams["season"].nunique()
        else:
            a_seasons = 0

        c.metric("Seasons with Team Records", a_seasons)

        st.caption(
            "Team details are inferred from match appearances. "
            "The most recent recorded team is not a guarantee "
            "of the player's current official team."
        )

        st.divider()
        st.subheader("🏆 Best Seasons")

        best_batting = get_best_batting_season(
            selected_player
        )

        best_bowling = get_best_bowling_season(
            selected_player
        )

        left, right = st.columns(2)

        with left:
            st.markdown("**Best season by runs**")

            if best_batting is not None:
                st.metric(
                    "Season",
                    int(best_batting["season"]),
                )

                x, y, z = st.columns(3)

                x.metric(
                    "Runs",
                    number(best_batting["total_runs"]),
                )
                y.metric(
                    "Strike Rate",
                    decimal(best_batting["strike_rate"]),
                )
                z.metric(
                    "Average",
                    decimal(best_batting["batting_average"]),
                )
            else:
                st.info("No batting season data found.")

        with right:
            st.markdown("**Best season by wickets**")

            if best_bowling is not None:
                st.metric(
                    "Season",
                    int(best_bowling["season"]),
                )

                x, y, z = st.columns(3)

                x.metric(
                    "Wickets",
                    number(best_bowling["wickets"]),
                )
                y.metric(
                    "Economy",
                    decimal(best_bowling["economy"]),
                )
                z.metric(
                    "Bowling Average",
                    decimal(best_bowling["bowling_average"]),
                )
            else:
                st.info("No bowling season data found.")

        st.divider()
        st.subheader("📈 Season-by-Season Performance")

        player_seasons = get_seasons(selected_player)

        if player_seasons.empty:
            st.info("No season records found for this player.")
        else:
            chart_tab1, chart_tab2, chart_tab3 = st.tabs(
                ["Batting", "Bowling", "Season Table"]
            )

            with chart_tab1:
                st.markdown("**Runs by season**")
                st.line_chart(
                    player_seasons.set_index("season")[
                        ["total_runs"]
                    ]
                )

                st.markdown("**Strike rate by season**")
                st.line_chart(
                    player_seasons.set_index("season")[
                        ["strike_rate"]
                    ]
                )

                st.markdown("**Batting average by season**")
                st.line_chart(
                    player_seasons.set_index("season")[
                        ["batting_average"]
                    ]
                )

            with chart_tab2:
                bowling_seasons = player_seasons[
                    player_seasons["legal_balls"] > 0
                ]

                if bowling_seasons.empty:
                    st.info("No bowling records available.")
                else:
                    st.markdown("**Wickets by season**")
                    st.line_chart(
                        bowling_seasons.set_index("season")[
                            ["wickets"]
                        ]
                    )

                    st.markdown("**Economy by season**")
                    st.line_chart(
                        bowling_seasons.set_index("season")[
                            ["economy"]
                        ]
                    )

            with chart_tab3:
                display = player_seasons[
                    [
                        "season",
                        "total_runs",
                        "strike_rate",
                        "batting_average",
                        "fours",
                        "sixes",
                        "wickets",
                        "economy",
                        "overs_bowled",
                    ]
                ].rename(
                    columns={
                        "season": "Season",
                        "total_runs": "Runs",
                        "strike_rate": "Strike Rate",
                        "batting_average": "Batting Average",
                        "fours": "Fours",
                        "sixes": "Sixes",
                        "wickets": "Wickets",
                        "economy": "Economy",
                        "overs_bowled": "Overs",
                    }
                )

                st.dataframe(
                    display,
                    hide_index=True,
                    use_container_width=True,
                )


# =========================================================
# 8. COMPARE PLAYERS
# =========================================================

elif page == "Compare Players":

    st.title("⚖️ Compare Players")
    st.write(
        "Compare the career statistics and season trends "
        "of any two players."
    )

    col1, col2 = st.columns(2)

    with col1:
        player_a = st.selectbox(
            "Select Player A",
            all_players,
            index=0,
            key="compare_a",
        )

    default_b = 1 if len(all_players) > 1 else 0

    with col2:
        player_b = st.selectbox(
            "Select Player B",
            all_players,
            index=default_b,
            key="compare_b",
        )

    if player_a == player_b:
        st.warning(
            "Choose two different players for a useful comparison."
        )
    else:
        data_a = get_career(player_a)
        data_b = get_career(player_b)

        if data_a is not None and data_b is not None:
            st.subheader("Career Statistics")

            comparison = pd.DataFrame(
                {
                    player_a: data_a,
                    player_b: data_b,
                }
            )

            comparison_metrics = [
                "total_runs",
                "strike_rate",
                "batting_average",
                "fours",
                "sixes",
                "runs_conceded",
                "wickets",
                "economy",
                "bowling_average",
                "overs_bowled",
            ]

            comparison = comparison.loc[
                comparison_metrics
            ]

            comparison.index = [
                "Runs",
                "Strike Rate",
                "Batting Average",
                "Fours",
                "Sixes",
                "Runs Conceded",
                "Wickets",
                "Economy",
                "Bowling Average",
                "Overs Bowled",
            ]

            st.dataframe(
                comparison,
                use_container_width=True,
            )

            st.subheader("📊 Visual Comparison")

            metric_choice = st.selectbox(
                "Choose a metric",
                [
                    "total_runs",
                    "strike_rate",
                    "batting_average",
                    "fours",
                    "sixes",
                    "wickets",
                    "economy",
                    "bowling_average",
                ],
                format_func=lambda value: {
                    "total_runs": "Career Runs",
                    "strike_rate": "Strike Rate",
                    "batting_average": "Batting Average",
                    "fours": "Fours",
                    "sixes": "Sixes",
                    "wickets": "Wickets",
                    "economy": "Economy",
                    "bowling_average": "Bowling Average",
                }[value],
            )

            chart_data = pd.DataFrame(
                {
                    "Player": [player_a, player_b],
                    "Value": [
                        data_a[metric_choice],
                        data_b[metric_choice],
                    ],
                }
            ).set_index("Player")

            st.bar_chart(chart_data)

            st.divider()
            st.subheader("📈 Compare Season Trends")

            seasons_a = get_seasons(player_a)
            seasons_b = get_seasons(player_b)

            trend_metric = st.selectbox(
                "Season trend metric",
                [
                    "total_runs",
                    "strike_rate",
                    "batting_average",
                    "wickets",
                    "economy",
                ],
                format_func=lambda value: {
                    "total_runs": "Runs",
                    "strike_rate": "Strike Rate",
                    "batting_average": "Batting Average",
                    "wickets": "Wickets",
                    "economy": "Economy",
                }[value],
            )

            trend_a = seasons_a[
                ["season", trend_metric]
            ].rename(
                columns={trend_metric: player_a}
            )

            trend_b = seasons_b[
                ["season", trend_metric]
            ].rename(
                columns={trend_metric: player_b}
            )

            trend = pd.merge(
                trend_a,
                trend_b,
                on="season",
                how="outer",
            ).sort_values("season")

            trend = trend.set_index("season")

            st.line_chart(trend)

            st.caption(
                "Missing values indicate that a player has no "
                "record for that season in the processed dataset."
            )


# =========================================================
# 9. TEAM HISTORY
# =========================================================

elif page == "Team History":

    st.title("🏟️ Player Team History")
    st.write(
        "See the teams recorded for a player across IPL seasons."
    )

    selected_player = st.selectbox(
        "Select a player",
        all_players,
        key="team_history_player",
    )

    history = get_teams(selected_player)

    if history.empty:
        st.info("No team history found for this player.")
    else:
        summary = career_team_summary(selected_player)

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Different Teams Recorded",
            summary["team_count"],
        )

        c2.metric(
            "Seasons with Team Records",
            history["season"].nunique(),
        )

        c3.metric(
            "Matches Recorded",
            number(history["matches_played"].sum()),
        )

        st.caption(
            "Matches are counted separately for each team-season. "
            "The same player may have records for multiple teams "
            "in a season; these records should not be added together "
            "to estimate unique career matches."
        )

        st.subheader("Team Appearances by Season")

        display_history = history[
            [
                "season",
                "team",
                "matches_played",
            ]
        ].rename(
            columns={
                "season": "Season",
                "team": "Team",
                "matches_played": "Matches Recorded",
            }
        )

        st.dataframe(
            display_history,
            hide_index=True,
            use_container_width=True,
        )

        st.subheader("Teams Represented")

        team_summary = (
            history.groupby("team", as_index=False)
            .agg(
                seasons=("season", "nunique"),
                matches_recorded=("matches_played", "sum"),
                first_season=("season", "min"),
                last_season=("season", "max"),
            )
            .sort_values(
                ["first_season", "team"]
            )
        )

        team_summary = team_summary.rename(
            columns={
                "team": "Team",
                "seasons": "Seasons Recorded",
                "matches_recorded": "Team-Match Appearances",
                "first_season": "First Season",
                "last_season": "Last Season",
            }
        )

        st.dataframe(
            team_summary,
            hide_index=True,
            use_container_width=True,
        )

        st.bar_chart(
            history.groupby("season")[
                "matches_played"
            ].sum()
        )

        st.caption(
            "Team membership is inferred from delivery records. "
            "Players who were only on the bench and never appeared "
            "in a recorded delivery may not be included."
        )


# =========================================================
# 10. SIMILAR PLAYERS
# =========================================================

elif page == "Similar Players":

    st.title("🤝 Similar Player Recommendations")
    st.write(
        "Find players with similar statistical profiles "
        "using the existing recommendation engine."
    )

    selected_player = st.selectbox(
        "Choose a player",
        sorted(
            recommender_players["player"]
            .dropna()
            .unique()
            .tolist()
        ),
        key="similar_player",
    )

    mode = st.selectbox(
        "Recommendation type",
        ["Overall", "Batting", "Bowling"],
    )

    number_of_recommendations = st.slider(
        "Number of recommendations",
        min_value=1,
        max_value=10,
        value=5,
    )

    if st.button("Find Similar Players", type="primary"):
        player_names = (
            recommender_players["player"]
            .astype(str)
            .tolist()
        )

        if selected_player not in player_names:
            st.error(
                "The selected player is not available "
                "in the recommendation model."
            )
        else:
            player_index = player_names.index(selected_player)

            try:
                recommendations = recommend_similar_players(
                    recommender_players,
                    recommender_model,
                    player_index,
                    mode=mode,
                    number=number_of_recommendations,
                )

                if recommendations is None or recommendations.empty:
                    st.info(
                        "No recommendations were returned."
                    )
                else:
                    st.success(
                        f"Players most similar to {selected_player}"
                    )

                    st.dataframe(
                        recommendations,
                        hide_index=True,
                        use_container_width=True,
                    )

            except Exception as error:
                st.error(
                    f"Recommendation error: {error}"
                )

    st.caption(
        "Similarity scores depend on the selected recommendation "
        "mode and the features used by recommender.py."
    )


# =========================================================
# 11. FOOTER
# =========================================================

st.sidebar.divider()
st.sidebar.caption("Cricket Player Finder")
st.sidebar.caption("IPL Analytics & ML Recommendation System")
