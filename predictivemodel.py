import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# --- 1. Load the Statcast Data ---
try:
    df = pd.read_csv('stats.csv')
except FileNotFoundError:
    print("Error: 'stats.csv' not found. Make sure it's in the same folder as the script.")
    exit()

print("Successfully loaded 'stats.csv' data.\n")

# --- 2. Data Cleaning and Preparation ---
df.rename(columns={'last_name, first_name': 'player_name'}, inplace=True)
df['woba'] = pd.to_numeric(df['woba'], errors='coerce')
df['xwoba'] = pd.to_numeric(df['xwoba'], errors='coerce')
# Add batting_avg to the cleaning process
df['batting_avg'] = pd.to_numeric(df['batting_avg'], errors='coerce')
df.dropna(subset=['woba', 'xwoba', 'pa', 'year', 'player_id', 'batting_avg'], inplace=True)

# --- 3. The Backtesting Loop ---

years_to_test = sorted(df['year'].unique())
if not years_to_test:
    print("No valid year data found in the CSV.")
    exit()

backtest_summary = []

print("="*70)
print("       Backtesting the Sabermetric 'Unlucky Player' Model")
print("="*70)
print("This analysis finds the 10 most 'unlucky' players from each season and")
print("checks their wOBA the following season to see if they improved.\n")

test_range = years_to_test[:-1] if len(years_to_test) > 1 else []

for year in test_range:
    current_year_data = df[df['year'] == year].copy()
    next_year_data = df[df['year'] == year + 1].copy()
    qualified_hitters = current_year_data[current_year_data['pa'] >= 400].copy()

    if qualified_hitters.empty:
        print(f"\n--- No qualified hitters found for {year}. Skipping. ---")
        continue

    qualified_hitters['luck_diff'] = qualified_hitters['woba'] - qualified_hitters['xwoba']
    unlucky_players = qualified_hitters.sort_values(by='luck_diff', ascending=True).head(10)

    results = []
    for _, player in unlucky_players.iterrows():
        player_id, player_name, unlucky_woba = player['player_id'], player['player_name'], player['woba']
        next_year_stats = next_year_data[next_year_data['player_id'] == player_id].iloc[0] if not next_year_data[next_year_data['player_id'] == player_id].empty else None

        if next_year_stats is not None and pd.notna(next_year_stats['woba']):
            results.append({"Player": player_name, f"{year}_wOBA": unlucky_woba, f"{year+1}_wOBA": next_year_stats['woba'], "Change": next_year_stats['woba'] - unlucky_woba})
        else:
            results.append({"Player": player_name, f"{year}_wOBA": unlucky_woba, f"{year+1}_wOBA": "N/A", "Change": "N/A"})

    print(f"\n--- Analysis for {year} -> {year+1} ---")
    if results:
        results_df = pd.DataFrame(results)
        valid_changes = pd.to_numeric(results_df['Change'], errors='coerce').dropna()
        avg_change = valid_changes.mean()
        positive_regression_count = (valid_changes > 0).sum()

        print(results_df.to_string(index=False))
        print(f"\nSummary for {year}:")
        print(f"  - {positive_regression_count} out of {len(valid_changes)} players saw their wOBA improve the next year.")
        print(f"  - Average wOBA Change: {avg_change:+.3f} points.")

        backtest_summary.append({'year': year, 'avg_woba_change': avg_change})

# --- 4. Visualize the Backtest Accuracy ---
if backtest_summary:
    summary_df = pd.DataFrame(backtest_summary)
    summary_df['year_transition'] = summary_df['year'].apply(lambda y: f"{int(y)}→{int(y)+1}")

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 7))

    colors = ['green' if x > 0 else 'red' for x in summary_df['avg_woba_change']]
    plt.bar(summary_df['year_transition'], summary_df['avg_woba_change'], color=colors)

    plt.title('Model Backtest: Average wOBA Change for "Unlucky" Players in the Following Season', fontsize=16, fontweight='bold')
    plt.ylabel('Average Change in wOBA', fontsize=12)
    plt.xlabel('Season Transition', fontsize=12)
    plt.xticks(rotation=45)
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    plt.figtext(0.5, 0.01, 'A positive bar shows that, on average, the unlucky cohort improved the next year.', ha='center', fontsize=10)

    print("\n" + "="*70)
    print("Displaying backtest visualization. Close the plot window to continue...")
    print("="*70 + "\n")
    plt.show()

# --- 5. Overall Performance and Prediction for Next Year ---
if backtest_summary:
    summary_df = pd.DataFrame(backtest_summary)
    overall_avg_change = summary_df['avg_woba_change'].mean()

    print("\n" + "="*70)
    print("           Overall Model Performance & Prediction")
    print("="*70 + "\n")
    print(f"Overall Average wOBA Change (All Years): {overall_avg_change:+.3f} points")

    # --- Linear Regression to predict the next year's trend ---
    X = summary_df[['year']]
    y = summary_df['avg_woba_change']

    model = LinearRegression()
    model.fit(X, y)

    latest_year = df['year'].max()
    next_year_prediction = model.predict([[latest_year]])[0]

    print(f"Linear Regression Estimate for {int(latest_year)}→{int(latest_year)+1} Change: {next_year_prediction:+.3f} points\n")

# --- 6. Identify the Unluckiest Players from the MOST RECENT Season ---
latest_year = df['year'].max()
print(f"\n" + "="*70)
print(f"       Top 10 Most 'Unlucky' Hitters from {int(latest_year)}")
print(f"       (Potential Bounce-Back Candidates for {int(latest_year) + 1})")
print("="*70 + "\n")

latest_year_data = df[df['year'] == latest_year].copy()
qualified_latest = latest_year_data[latest_year_data['pa'] >= 400].copy()

if not qualified_latest.empty:
    qualified_latest['luck_diff'] = qualified_latest['woba'] - qualified_latest['xwoba']
    unlucky_latest = qualified_latest.sort_values(by='luck_diff', ascending=True).head(10)
    # Add batting_avg to the final output table
    print(unlucky_latest[['player_name', 'batting_avg', 'woba', 'xwoba', 'luck_diff']].to_string(index=False))
    print("\n")
else:
    print(f"No qualified hitters found for {int(latest_year)} to analyze.")

print("Analysis Complete.\n")
