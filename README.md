
# 🏏 Cricket Player Finder

A machine learning project that recommends five IPL players with similar statistical profiles using player performance data, feature scaling, and cosine similarity.

## Features

- Search and select IPL players.
- View batting and bowling statistics.
- Recommend the five most statistically similar players.
- Compare player runs using a bar chart.
- Interactive web interface built with Streamlit.

## Tech Stack

- Python
- Pandas and NumPy
- Scikit-learn
- Streamlit
- Matplotlib

## Machine Learning Approach

1. Collect historical IPL ball-by-ball data.
2. Calculate player-level batting and bowling statistics.
3. Scale numerical features using StandardScaler.
4. Calculate cosine similarity between player profiles.
5. Display the top five similar players.

Similarity scores measure statistical resemblance, not the probability of future match performance.

## Project Structure

```text
cricket-player-finder/
├── app.py
├── main.py
├── recommender.py
├── requirements.txt
├── .gitignore
├── README.md
└── processed_data/
    └── player_statistics.csv
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd cricket-player-finder
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Prepare the dataset

Place the required IPL match CSV files in the location expected by `main.py`, then run:

```powershell
python main.py
```

This generates player statistics in `processed_data/`.

### 5. Launch the application

```powershell
python -m streamlit run app.py
```

Open the local URL displayed in the terminal.

## Future Improvements

- Improve batting average and bowling economy calculations.
- Add player-role and season filters.
- Improve similarity features and validate recommendations.
- Deploy the application online.

## Disclaimer

This project compares player statistics for educational and analytical purposes. Similarity does not guarantee similar future performance.

## Author

Akhil Bysani
