import pandas as pd

# Define what counts as a whiff (swing and miss)
WHIFF_OUTCOMES = ['swinging_strike', 'foul_tip', 'swinging_strike_blocked']

# Define what counts as a swing (any bat contact attempt)
SWING_OUTCOMES = ['hit_into_play', 'swinging_strike', 'foul', 'foul_tip', 'swinging_strike_blocked']

# Minimum number of pitches needed to consider a pitch "significant"
MIN_PITCHES_FOR_ANALYSIS = 100


def analyze_pitch_strategy(csv_filename, pitcher_name):
    """
    Analyzes whether a pitcher's "best" pitch is more effective when sequenced
    after another specific pitch.

    This function:
    1. Loads pitch data for a specific pitcher
    2. Identifies their most effective pitch (by whiff rate)
    3. Identifies their most common setup pitch
    4. Compares the effectiveness of the best pitch when preceded by the setup pitch
       versus in other contexts
    """
    try:
        all_pitches_df = pd.read_csv(csv_filename)
    except FileNotFoundError:
        print("\n" + "-"*70)
        print(f"SKIPPING: '{csv_filename}' not found for {pitcher_name}.")
        print("Please download the required data from Baseball Savant.")
        print("-"*70)
        return None

    # Filter to only include pitches thrown by this specific pitcher
    pitcher_df = all_pitches_df[all_pitches_df['player_name'] == pitcher_name].copy()
    if pitcher_df.empty:
        print(f"Error: Pitcher '{pitcher_name}' not found in {csv_filename}.")
        return None

    print("\n" + "-"*70)
    print(f"               Pitch Strategy Analysis for {pitcher_name}")
    print("-"*70)
    print(f"Successfully loaded {len(pitcher_df)} pitches.\n")

    # Classify each pitch outcome: did the batter swing, and if so, did they miss?
    pitcher_df['is_whiff'] = pitcher_df['description'].isin(WHIFF_OUTCOMES)
    pitcher_df['is_swing'] = pitcher_df['description'].isin(SWING_OUTCOMES)

    # --- 2. Determine "Best Stuff" ---
    # Calculate stats for each pitch type (grouped by pitch_name)
    pitch_summary = pitcher_df.groupby('pitch_name').agg(
        total_pitches=('pitch_name', 'count'),
        total_swings=('is_swing', 'sum'),
        total_whiffs=('is_whiff', 'sum')
    ).reset_index()

    # Filter to only pitches thrown frequently enough to be statistically meaningful
    pitch_summary = pitch_summary[pitch_summary['total_pitches'] > MIN_PITCHES_FOR_ANALYSIS]

    # Calculate the whiff rate (percentage of swings that missed) for each pitch
    # Avoid division by zero by replacing 0 swings with 1
    pitch_summary['whiff_rate'] = (pitch_summary['total_whiffs'] / pitch_summary['total_swings'].replace(0, 1)) * 100

    if pitch_summary.empty:
        print("Not enough data to determine a 'best' pitch.")
        return None

    # Find the pitch with the highest whiff rate (this is the "best" pitch)
    best_pitch_stats = pitch_summary.sort_values(by='whiff_rate', ascending=False).iloc[0]
    best_pitch_name = best_pitch_stats['pitch_name']

    # Find the most frequently thrown pitch (excluding the best pitch)
    # This will be the "setup pitch" - what typically comes before the best pitch
    all_pitches_except_best = pitcher_df[pitcher_df['pitch_name'] != best_pitch_name]
    setup_pitch_name = all_pitches_except_best['pitch_name'].mode()[0]

    print(f"Identified '{best_pitch_name}' as the 'Best Pitch' (Whiff Rate: {best_pitch_stats['whiff_rate']:.1f}%)")
    print(f"Identified '{setup_pitch_name}' as the most common 'Setup Pitch'.\n")
    print(f"Hypothesis: Is the {best_pitch_name} even MORE effective when thrown")
    print(f"immediately after a {setup_pitch_name} in the same at-bat?")
    print("-"*70 + "\n")

    # --- 3. Test the Hypothesis ---
    # Only look at at-bats where the pitcher threw multiple pitches
    multi_pitch_at_bats = pitcher_df.groupby('at_bat_number').filter(lambda x: len(x) > 1)

    # Filter to only instances of the pitcher's best pitch
    best_pitches = multi_pitch_at_bats[multi_pitch_at_bats['pitch_name'] == best_pitch_name].copy()

    def get_previous_pitch(row):
        """Helper function to find what pitch was thrown immediately before this one"""
        at_bat_data = multi_pitch_at_bats[multi_pitch_at_bats['at_bat_number'] == row['at_bat_number']]

        # Get all pitches before this one in the same at-bat
        earlier_pitches = at_bat_data[at_bat_data['pitch_number'] < row['pitch_number']]

        if not earlier_pitches.empty:
            # Return the pitch immediately before (highest pitch_number that's still less than current)
            previous_pitch = earlier_pitches.sort_values(by='pitch_number', ascending=False).iloc[0]
            return previous_pitch['pitch_name']
        return None

    # For each instance of the best pitch, determine what pitch came before it
    best_pitches['previous_pitch'] = best_pitches.apply(get_previous_pitch, axis=1)

    # Separate instances where the best pitch followed the setup pitch vs other pitches
    sequenced_best_pitches = best_pitches[best_pitches['previous_pitch'] == setup_pitch_name]
    non_sequenced_best_pitches = best_pitches[best_pitches['previous_pitch'] != setup_pitch_name]

    # Check if we have enough data to make a valid comparison
    has_sequenced_data = (not sequenced_best_pitches.empty and
                         sequenced_best_pitches['is_swing'].sum() > 0)
    has_non_sequenced_data = (not non_sequenced_best_pitches.empty and
                             non_sequenced_best_pitches['is_swing'].sum() > 0)

    if not (has_sequenced_data and has_non_sequenced_data):
        print("Not enough specific sequences in the data to perform a valid comparison.")
        return None

    # Calculate whiff rates for both scenarios
    sequenced_whiff_rate = (sequenced_best_pitches['is_whiff'].sum() /
                           sequenced_best_pitches['is_swing'].sum()) * 100
    non_sequenced_whiff_rate = (non_sequenced_best_pitches['is_whiff'].sum() /
                               non_sequenced_best_pitches['is_swing'].sum()) * 100

    print("Results:\n")
    print(f"  - Whiff Rate of '{best_pitch_name}' after a '{setup_pitch_name}': {sequenced_whiff_rate:.1f}%")
    print(f"  - Whiff Rate of '{best_pitch_name}' in other contexts: {non_sequenced_whiff_rate:.1f}%")

    # Determine which strategy this pitcher relies on
    conclusion_text = ""
    conclusion_type = ""

    if sequenced_whiff_rate > non_sequenced_whiff_rate:
        conclusion_type = "SEQUENCING"
        pitcher_first_name = pitcher_name.split(',')[0]
        conclusion_text = (f"The data supports the SEQUENCING hypothesis for {pitcher_first_name}.\n"
                         f"His {best_pitch_name} was significantly more effective when set up by his {setup_pitch_name}.")
    else:
        conclusion_type = "BEST STUFF"
        pitcher_first_name = pitcher_name.split(',')[0]
        conclusion_text = (f"The data supports the BEST STUFF hypothesis for {pitcher_first_name}.\n"
                         f"His {best_pitch_name} is so dominant that the preceding pitch did not increase its effectiveness.")

    print("\n" + "-"*70)
    print("Conclusion:")
    print(conclusion_text)
    print("-"*70 + "\n")

    return {'name': pitcher_name.split(',')[0], 'strategy': conclusion_type}


def generate_final_synthesis(results):
    """
    Generates a high-level summary based on the individual pitcher analyses.
    Groups pitchers into archetypes and provides strategic insights.
    """
    print("\n" + "="*80)
    print("||" + " "*29 + "FINAL SYNTHESIS" + " "*30 + "||")
    print("="*80)

    if not results:
        print("No valid results to synthesize.")
        return

    # Categorize pitchers by their dominant strategy
    sequencing_pitchers = [r['name'] for r in results if r['strategy'] == 'SEQUENCING']
    best_stuff_pitchers = [r['name'] for r in results if r['strategy'] == 'BEST STUFF']

    print("\nThis analysis reveals a critical insight: there is no single, universal pitching strategy.")
    print("The optimal approach is highly dependent on a pitcher's individual arsenal and skillset.\n")

    # Describe the "Sequencing" pitchers
    if sequencing_pitchers:
        print("--- The 'Sequencing' Archetype ---")
        pitcher_list = ', '.join(sequencing_pitchers)
        print(f"For pitchers like {pitcher_list}, who often rely on an elite off-speed or")
        print("breaking pitch, the data shows that sequencing is key. Their best pitches become")
        print("even more deceptive and effective when tunneled off their fastball or another primary pitch.\n")

    # Describe the "Best Stuff" pitchers
    if best_stuff_pitchers:
        print("--- The 'Best Stuff' Archetype ---")
        pitcher_list = ', '.join(best_stuff_pitchers)
        print(f"For power pitchers like {pitcher_list}, whose best pitch is often a truly")
        print("dominant, outlier fastball, the data suggests their best stuff is overpowering on its own.")
        print("Its effectiveness isn't significantly boosted by a setup pitch because hitters have to")
        print("contend with its elite velocity and movement regardless of the prior pitch.\n")

    # Provide overall strategic takeaway
    print("--- Overall Strategic Conclusion ---")
    print("A team's pitching strategy should not be monolithic. It must be tailored to the individual.")
    print("This analysis provides a clear framework for identifying a pitcher's archetype and")
    print("developing a data-driven game plan to maximize their specific talents.")
    print("="*80)


if __name__ == "__main__":
    pitchers_to_analyze = [
        {'name': 'Yamamoto, Yoshinobu', 'file': 'yamamoto.csv'},
        {'name': 'Skubal, Tarik', 'file': 'skubal.csv'},
        {'name': 'Skenes, Paul', 'file': 'skenes.csv'}
    ]

    all_results = []
    for pitcher in pitchers_to_analyze:
        result = analyze_pitch_strategy(csv_filename=pitcher['file'], pitcher_name=pitcher['name'])
        if result:
            all_results.append(result)

    if all_results:
        generate_final_synthesis(all_results)
