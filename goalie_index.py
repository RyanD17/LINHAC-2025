#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Goalie Index Analysis
=====================
This script builds a goalie performance index based on:
1. Expected goals against (xG)
2. Rebound quality control
3. Actual vs expected save performance

Author: Cascade AI Assistant
Date: 2025-03-24
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import re

# Set style for plots
plt.style.use('fivethirtyeight')
sns.set_palette('colorblind')

def load_and_preprocess_data(file_path):
    """
    Load and preprocess the hockey data from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Preprocessed shot data
    """
    print(f"Loading data from: {file_path}")
    
    try:
        # Load data
        df = pd.read_csv(file_path)
        
        # Print event type distribution
        print("\nEvent type distribution:")
        print(df['eventname'].value_counts().head(10))
        
        # Define shot-related events
        # Expanded to include more event types
        shot_events = [
            'shot', 'goal', 'shotblock', 'shotattempt', 'shotmiss', 
            'shotwide', 'shotdistant', 'shotclose', 'shotrebound'
        ]
        
        # Filter to only include shot-related events
        # We're more inclusive now to capture more data
        shot_df = df[df['eventname'].str.contains('shot|goal', case=False, na=False)].copy()
        
        if shot_df.empty:
            print("No shot events found in data")
            return pd.DataFrame()
        
        print(f"\nFound {len(shot_df)} shot events")
        
        # Check for key columns
        required_columns = ['teamgoalieid', 'opposinggoalieid', 'eventname', 'outcome']
        missing_columns = [col for col in required_columns if col not in shot_df.columns]
        
        if missing_columns:
            print(f"Warning: Missing required columns: {missing_columns}")
            # Try to find alternative columns if available
            if 'teamgoalieid' in missing_columns and 'teamgoalie' in shot_df.columns:
                shot_df['teamgoalieid'] = shot_df['teamgoalie']
                missing_columns.remove('teamgoalieid')
            if 'opposinggoalieid' in missing_columns and 'opposinggoalie' in shot_df.columns:
                shot_df['opposinggoalieid'] = shot_df['opposinggoalie']
                missing_columns.remove('opposinggoalieid')
            
            # Check for goalie IDs - if missing, we'll generate synthetic ones
            if not all(col in df.columns for col in ['teamgoalieid', 'opposinggoalieid']):
                print("Warning: Missing required columns: ['teamgoalieid', 'opposinggoalieid']")
                print("\nGenerating synthetic goalie IDs based on team IDs...")
                
                # Create synthetic goalie IDs based on team IDs
                unique_teams = pd.concat([shot_df['teamid'], shot_df['opposingteamid']]).unique()
                team_to_goalie = {team_id: int(str(team_id) + '001') for team_id in unique_teams}
                
                # Assign synthetic goalie IDs
                shot_df['teamgoalieid'] = shot_df['teamid'].map(team_to_goalie)
                shot_df['opposinggoalieid'] = shot_df['opposingteamid'].map(team_to_goalie)
                
                print(f"Created {len(team_to_goalie)} synthetic goalie IDs")
        
        # Identify unique goalies
        if 'teamgoalieid' in shot_df.columns:
            team_goalies = shot_df['teamgoalieid'].dropna().unique()
            print(f"\nUnique team goalies: {len(team_goalies)}")
        else:
            team_goalies = []
        
        if 'opposinggoalieid' in shot_df.columns:
            opposing_goalies = shot_df['opposinggoalieid'].dropna().unique()
            print(f"Unique opposing goalies: {len(opposing_goalies)}")
        else:
            opposing_goalies = []
        
        # Sample of event names for understanding the data
        print("\nSample event names in filtered data:")
        print(shot_df['eventname'].value_counts())
        
        # Add defensive context if available
        if 'defenderswithin3m' in shot_df.columns:
            # Fill missing values with median
            shot_df['defenderswithin3m'] = shot_df['defenderswithin3m'].fillna(
                shot_df['defenderswithin3m'].median())
        else:
            # Create a synthetic defensive pressure metric if not available
            if 'xcoord' in shot_df.columns and 'ycoord' in shot_df.columns:
                # Shots closer to net typically have more defensive pressure
                distance_to_net = np.sqrt(
                    (shot_df['xcoord'] - 200)**2 + (shot_df['ycoord'] - 100)**2
                ) / 200  # Normalize by max distance
                shot_df['defenderswithin3m'] = 3 - 2 * distance_to_net.clip(0, 1)
            else:
                # If no location data, use a default value
                shot_df['defenderswithin3m'] = 1.0
        
        # Improve defensive pressure tracking
        if 'defenderswithin3m' in shot_df.columns:
            print("\nProcessing defensive pressure data...")
            # Try to convert to numeric if possible
            try:
                shot_df['defenderswithin3m'] = pd.to_numeric(shot_df['defenderswithin3m'], errors='coerce')
                # Fill NA values with a reasonable default (avoiding inplace=True warning)
                shot_df['defenderswithin3m'] = shot_df['defenderswithin3m'].fillna(0)
            except:
                print("Warning: Could not convert defenderswithin3m to numeric values")
                # If conversion fails, create a numeric version
                shot_df['defenderswithin3m_numeric'] = 0
                
        # Handle defensive_pressure if it exists but might be categorical
        if 'defensive_pressure' in shot_df.columns:
            # Check if it contains categorical values like 'low', 'medium', 'high'
            unique_values = shot_df['defensive_pressure'].unique()
            if all(isinstance(x, str) for x in unique_values if pd.notna(x)):
                print("\nConverting categorical defensive pressure to numeric values...")
                # Map categorical values to numeric
                pressure_map = {'low': 0.33, 'medium': 0.67, 'high': 1.0}
                # Apply mapping - use regex to handle repeated strings like 'lowlowlow'
                shot_df['defensive_pressure_numeric'] = shot_df['defensive_pressure'].apply(
                    lambda x: pressure_map.get(
                        re.match(r'^(low|medium|high)', str(x).lower()).group(1) if pd.notna(x) and re.match(r'^(low|medium|high)', str(x).lower()) else None, 
                        0.5
                    )
                )
            else:
                # Try to convert to numeric
                try:
                    shot_df['defensive_pressure'] = pd.to_numeric(shot_df['defensive_pressure'], errors='coerce')
                    shot_df['defensive_pressure'].fillna(0, inplace=True)
                except:
                    print("Warning: Could not process defensive_pressure values")
        
        # Handle xG data - use actual xG when available, do not generate synthetic xG
        if 'xg_allattempts' in shot_df.columns and not shot_df['xg_allattempts'].isna().all():
            print("\nUsing xg_allattempts data from dataset")
            shot_df['xg_allattempts'] = shot_df['xg_allattempts']
        else:
            print("\nNo xg_allattempts data found - will use actual goals instead of xG for calculations")
            # Just set to actual goals - we'll handle this downstream
            shot_df['xg_allattempts'] = np.nan
        
        # Improve shot quality classification
        shot_df['shot_quality'] = 'medium'
        
        # High-danger shots (high xG)
        if 'xg_allattempts' in shot_df.columns and not shot_df['xg_allattempts'].isna().all():
            high_xg_threshold = shot_df['xg_allattempts'].quantile(0.7)
            shot_df.loc[shot_df['xg_allattempts'] >= high_xg_threshold, 'shot_quality'] = 'high'
            
            # Low-danger shots (low xG)
            low_xg_threshold = shot_df['xg_allattempts'].quantile(0.3)
            shot_df.loc[shot_df['xg_allattempts'] <= low_xg_threshold, 'shot_quality'] = 'low'
        else:
            # If no xG data, default classification based on whether it's a goal
            shot_df['shot_quality'] = 'medium'
            shot_df.loc[shot_df['is_goal'] == 1, 'shot_quality'] = 'high'
        
        # Identify defensive pressure levels
        if 'defenderswithin3m' in shot_df.columns:
            shot_df['defensive_pressure'] = 'medium'
            # High defensive pressure
            shot_df.loc[shot_df['defenderswithin3m'] >= 2, 'defensive_pressure'] = 'high'
            # Low defensive pressure
            shot_df.loc[shot_df['defenderswithin3m'] <= 1, 'defensive_pressure'] = 'low'
        
        # Add timestamp to each shot for sequential analysis
        if 'clock' in shot_df.columns and 'period' in shot_df.columns:
            # Convert period and clock to seconds
            shot_df['period_seconds'] = shot_df['period'].fillna(1) * 1200  # 20 minutes per period
            
            # Convert clock (MM:SS) to seconds
            def clock_to_seconds(clock_str):
                if pd.isna(clock_str):
                    return 0
                try:
                    parts = clock_str.split(':')
                    if len(parts) == 2:
                        return int(parts[0]) * 60 + int(parts[1])
                    return 0
                except:
                    return 0
                    
            shot_df['clock_seconds'] = shot_df['clock'].apply(clock_to_seconds)
            shot_df['game_seconds'] = shot_df['period_seconds'] - shot_df['clock_seconds']
        else:
            # Assign a sequential timestamp if no clock data
            shot_df['game_seconds'] = range(len(shot_df))
        
        # Identify rebounds using game_seconds and gameid
        shot_df = identify_rebounds(shot_df)
        
        return shot_df
    
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()  # Return empty DataFrame on error

def identify_rebounds(shot_df):
    """
    Identify rebounds from shot data with improved accuracy.
    
    Args:
        shot_df (pd.DataFrame): DataFrame containing shot data
        
    Returns:
        pd.DataFrame: Shot data with rebound indicators
    """
    # Sort shots by game, period, and time
    if 'gameid' in shot_df.columns and 'game_seconds' in shot_df.columns:
        shot_df = shot_df.sort_values(['gameid', 'game_seconds'])
    
    # Initialize rebound columns
    shot_df['is_rebound'] = False
    shot_df['rebound_from'] = np.nan
    shot_df['rebound_distance'] = np.nan
    shot_df['rebound_danger'] = 'none'  # Will classify danger level of rebound
    
    # Track previous shot info for rebound detection
    prev_game = None
    prev_time = None
    prev_team = None
    prev_idx = None
    
    # Time threshold for rebounds in seconds
    rebound_time_threshold = 3.0  # seconds
    
    # Loop through shots chronologically
    for i, (idx, row) in enumerate(shot_df.iterrows()):
        current_game = row.get('gameid', None)
        current_time = row.get('game_seconds', None)
        current_team = row.get('teamid', None)
        
        # If this is from the same game as previous shot and within time threshold
        if (prev_game is not None and 
            current_game == prev_game and 
            current_time is not None and 
            prev_time is not None and 
            current_time - prev_time <= rebound_time_threshold):
            
            # Classify as rebound
            shot_df.at[idx, 'is_rebound'] = True
            shot_df.at[idx, 'rebound_from'] = prev_idx
            
            # Calculate rebound danger level based on shot quality
            if 'shot_quality' in shot_df.columns:
                prev_quality = shot_df.at[prev_idx, 'shot_quality'] if prev_idx in shot_df.index else 'medium'
                current_quality = row['shot_quality']
                
                # Higher danger if shot quality increased
                if (prev_quality == 'low' and current_quality in ['medium', 'high']) or \
                   (prev_quality == 'medium' and current_quality == 'high'):
                    shot_df.at[idx, 'rebound_danger'] = 'high'
                # Medium danger if quality stayed the same
                elif prev_quality == current_quality:
                    shot_df.at[idx, 'rebound_danger'] = 'medium'
                # Low danger if quality decreased
                else:
                    shot_df.at[idx, 'rebound_danger'] = 'low'
            else:
                shot_df.at[idx, 'rebound_danger'] = 'medium'  # Default
            
            # Calculate rebound distance if coordinates available
            if all(col in shot_df.columns for col in ['xcoord', 'ycoord']):
                if prev_idx in shot_df.index:
                    prev_x = shot_df.at[prev_idx, 'xcoord']
                    prev_y = shot_df.at[prev_idx, 'ycoord']
                    curr_x = row['xcoord']
                    curr_y = row['ycoord']
                    
                    rebound_distance = np.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
                    shot_df.at[idx, 'rebound_distance'] = rebound_distance
        
        # Update previous shot info
        prev_game = current_game
        prev_time = current_time
        prev_team = current_team
        prev_idx = idx
    
    # Count rebounds for reporting
    rebound_count = shot_df['is_rebound'].sum()
    print(f"\nFound {rebound_count} rebounds ({rebound_count/len(shot_df)*100:.2f}% of shots)")
    
    return shot_df

def calculate_goalie_metrics(shot_df):
    """
    Calculate goalie metrics with improved calculations and additions:
    - Improved xG calibration
    - Defensive context consideration
    - Shot quality classifications
    - Enhanced rebound analysis
    
    Args:
        shot_df (pd.DataFrame): DataFrame containing shot data
        
    Returns:
        pd.DataFrame: DataFrame containing goalie metrics
    """
    print("Calculating goalie metrics...")
    
    # For each team's goalie and opposing team's goalie, calculate metrics
    goalie_metrics = []
    
    team_goalies = shot_df['teamgoalieid'].dropna().unique() if 'teamgoalieid' in shot_df.columns else []
    opposing_goalies = shot_df['opposinggoalieid'].dropna().unique() if 'opposinggoalieid' in shot_df.columns else []
    
    # Create tqdm progress bar for processing goalies
    total_goalies = len(team_goalies) + len(opposing_goalies)
    progress_bar = tqdm(total=total_goalies, desc="Processing goalies")
    
    # Process team goalies
    print("\nProcessing team goalies...")
    if 'teamgoalieid' in shot_df.columns:
        team_shots = shot_df[shot_df['teamgoalieid'].notna()]
        print(f"Found {len(team_shots)} shots on team goalies")
        
        print(f"Processing {len(team_goalies)} team goalies")
        for goalie_id in team_goalies:
            goalie_shots = team_shots[team_shots['teamgoalieid'] == goalie_id]
            
            if len(goalie_shots) > 0:
                print(f"Goalie {goalie_id} outcome counts: {goalie_shots['outcome'].value_counts()}")
                metrics = calculate_single_goalie_metrics(goalie_shots, goalie_id, 'team')
                goalie_metrics.append(metrics)
            
            progress_bar.update(1)
    
    # Process opposing goalies
    print("\nProcessing opposing goalies...")
    if 'opposinggoalieid' in shot_df.columns:
        opposing_shots = shot_df[shot_df['opposinggoalieid'].notna()]
        print(f"Found {len(opposing_shots)} shots on opposing goalies")
        
        print(f"Processing {len(opposing_goalies)} opposing goalies")
        for goalie_id in opposing_goalies:
            goalie_shots = opposing_shots[opposing_shots['opposinggoalieid'] == goalie_id]
            
            if len(goalie_shots) > 0:
                print(f"Goalie {goalie_id} outcome counts: {goalie_shots['outcome'].value_counts()}")
                metrics = calculate_single_goalie_metrics(goalie_shots, goalie_id, 'opposing')
                goalie_metrics.append(metrics)
            
            progress_bar.update(1)
    
    progress_bar.close()
    
    # Convert to DataFrame
    goalie_metrics_df = pd.DataFrame(goalie_metrics)
    
    # Calculate goalie index with improved formula
    if not goalie_metrics_df.empty:
        weight_gsae = 2.0  # Weight for Goals Saved Above Expected
        weight_save_pct = 3.0  # Weight for Save Percentage
        weight_rebound = -5.0  # Weight for Dangerous Rebound Control (negative because lower is better)
        weight_def_adj = 1.0  # Weight for defensive context adjustment
        
        # Add defensive context adjustment
        if 'avg_defensive_pressure' in goalie_metrics_df.columns:
            # Normalize to range 0-1 where 1 is highest pressure
            max_pressure = goalie_metrics_df['avg_defensive_pressure'].max()
            min_pressure = goalie_metrics_df['avg_defensive_pressure'].min()
            
            if max_pressure > min_pressure:
                goalie_metrics_df['defensive_adjustment'] = (
                    (goalie_metrics_df['avg_defensive_pressure'] - min_pressure) / 
                    (max_pressure - min_pressure)
                )
            else:
                goalie_metrics_df['defensive_adjustment'] = 0.5
        else:
            goalie_metrics_df['defensive_adjustment'] = 0.5  # Default neutral value
        
        # Calculate the new goalie index
        goalie_metrics_df['goalie_index'] = (
            (goalie_metrics_df['goals_saved_above_expected'] * weight_gsae) +
            (goalie_metrics_df['save_percentage'] * weight_save_pct) +
            (goalie_metrics_df['dangerous_rebound_percentage'] * weight_rebound) +
            (goalie_metrics_df['defensive_adjustment'] * weight_def_adj)
        )
    
    return goalie_metrics_df

def calculate_single_goalie_metrics(goalie_shots, goalie_id, goalie_type):
    """
    Calculate metrics for a single goalie with improved definitions.
    
    Args:
        goalie_shots (pd.DataFrame): Shots against this goalie
        goalie_id (int): ID of the goalie
        goalie_type (str): 'team' or 'opposing'
        
    Returns:
        dict: Dictionary containing goalie metrics
    """
    # Count total shots
    total_shots = len(goalie_shots)
    
    # Count outcomes if available
    outcome_success_count = 0
    if 'outcome' in goalie_shots.columns:
        # Success means the shot was saved (opposing goalie) or went in (team goalie)
        outcome_map = {
            'team': {'successful': 'goal', 'failed': 'save'},
            'opposing': {'successful': 'save', 'failed': 'goal'}
        }
        
        # Count successful outcomes based on goalie type
        if goalie_type in outcome_map:
            success_value = outcome_map[goalie_type]['successful']
            outcome_success_count = (goalie_shots['outcome'] == success_value).sum()
            if outcome_success_count == 0:
                # Try alternate method: check if 'successful' is literally in the outcome column
                outcome_success_count = (goalie_shots['outcome'] == 'successful').sum()
        else:
            # Fallback if goalie_type is unknown
            outcome_success_count = (goalie_shots['outcome'] == 'successful').sum()
    
    # Calculate save percentage based on goalie type
    if goalie_type == 'opposing':
        # For opposing goalies, success means saving the shot
        save_count = outcome_success_count
        save_percentage = save_count / total_shots if total_shots > 0 else 0
    else:
        # For team goalies, success means the shot went in
        # So save percentage is 1 - success rate
        save_count = total_shots - outcome_success_count
        save_percentage = save_count / total_shots if total_shots > 0 else 0
    
    # Calculate xG-based metrics
    xg_against = 0
    goals_saved_above_expected = 0
    
    # Use xg_allattempts if available, otherwise fall back to regular xG
    xg_field = 'xg_allattempts' if 'xg_allattempts' in goalie_shots.columns else 'xg'
    
    if xg_field in goalie_shots.columns and not goalie_shots[xg_field].isna().all():
        # Sum xG for all shots
        xg_against = goalie_shots[xg_field].sum()
        
        # For opposing goalies, calculate how many goals they saved above expected
        if goalie_type == 'opposing':
            # Goals should be the number of failed saves
            goals_allowed = total_shots - save_count
            # Positive GSAE means fewer goals allowed than expected (good)
            goals_saved_above_expected = xg_against - goals_allowed
        else:
            # For team goalies, we want to know how many goals they allowed above expected
            # Goals should be the number of successful shots (goals scored)
            goals_scored = outcome_success_count
            # Negative GSAE means more goals allowed than expected (bad)
            goals_saved_above_expected = goals_scored - xg_against
    else:
        # No xG data available - don't generate synthetic values
        # For opposing goalies, goals allowed should equal expected goals
        # This makes GSAE = 0 (neutral) when no xG data is available
        if goalie_type == 'opposing':
            goals_allowed = total_shots - save_count
            xg_against = goals_allowed  # Set expected goals equal to actual goals
        else:
            goals_scored = outcome_success_count
            xg_against = goals_scored  # Set expected goals equal to actual goals
        
        goals_saved_above_expected = 0  # Neutral value when no xG data
        print(f"Note: No xG data for goalie {goalie_id}. GSAE set to 0.")
    
    # Calculate rebound-related metrics
    rebound_shots = goalie_shots[goalie_shots['is_rebound'] == True] if 'is_rebound' in goalie_shots.columns else pd.DataFrame()
    rebound_count = len(rebound_shots)
    
    # Calculate rebound percentage
    rebound_percentage = rebound_count / total_shots if total_shots > 0 else 0
    
    # Calculate dangerous rebound percentage with improved definition
    dangerous_rebound_count = 0
    
    if 'rebound_danger' in goalie_shots.columns:
        dangerous_rebound_count = len(goalie_shots[(goalie_shots['is_rebound'] == True) & (goalie_shots['rebound_danger'] == 'high')])
    else:
        # Fallback if rebound_danger column is missing
        dangerous_rebound_count = len(rebound_shots)
    
    dangerous_rebound_percentage = dangerous_rebound_count / total_shots if total_shots > 0 else 0
    
    # Track shots and saves by danger level
    high_danger_shots = 0
    high_danger_saves = 0
    medium_danger_shots = 0
    medium_danger_saves = 0
    low_danger_shots = 0
    low_danger_saves = 0
    
    # Track shot quality statistics if available
    if 'shot_quality' in goalie_shots.columns:
        # High danger shots
        high_danger_shots_df = goalie_shots[goalie_shots['shot_quality'] == 'high']
        high_danger_shots = len(high_danger_shots_df)
        
        # Medium danger shots
        medium_danger_shots_df = goalie_shots[goalie_shots['shot_quality'] == 'medium']
        medium_danger_shots = len(medium_danger_shots_df)
        
        # Low danger shots
        low_danger_shots_df = goalie_shots[goalie_shots['shot_quality'] == 'low']
        low_danger_shots = len(low_danger_shots_df)
        
        # Calculate saves by danger level for opposing goalies
        if goalie_type == 'opposing':
            if 'outcome' in goalie_shots.columns:
                # For opposing goalies, saves are successful outcomes
                high_danger_saves = (high_danger_shots_df['outcome'] == 'save').sum()
                medium_danger_saves = (medium_danger_shots_df['outcome'] == 'save').sum()
                low_danger_saves = (low_danger_shots_df['outcome'] == 'save').sum()
        else:
            # For team goalies, saves are failed outcomes (not goals)
            if 'outcome' in goalie_shots.columns:
                high_danger_saves = high_danger_shots - (high_danger_shots_df['outcome'] == 'goal').sum()
                medium_danger_saves = medium_danger_shots - (medium_danger_shots_df['outcome'] == 'goal').sum()
                low_danger_saves = low_danger_shots - (low_danger_shots_df['outcome'] == 'goal').sum()
    
    # Calculate save percentages by danger level
    high_danger_save_percentage = high_danger_saves / high_danger_shots if high_danger_shots > 0 else 0
    medium_danger_save_percentage = medium_danger_saves / medium_danger_shots if medium_danger_shots > 0 else 0
    low_danger_save_percentage = low_danger_saves / low_danger_shots if low_danger_shots > 0 else 0
    
    # Calculate average defensive pressure if available
    avg_defensive_pressure = 0
    defensive_adjustment = 0
    
    # First try the numeric version we created in preprocessing
    if 'defensive_pressure_numeric' in goalie_shots.columns:
        avg_defensive_pressure = goalie_shots['defensive_pressure_numeric'].mean()
        # Calculate defensive adjustment
        league_avg_pressure = 0.5  # Average value (between low 0.33 and high 1.0)
        defensive_adjustment = -(avg_defensive_pressure - league_avg_pressure) * 0.05
    
    # Check if we have original defensive pressure data and it's numeric
    elif 'defensive_pressure' in goalie_shots.columns:
        try:
            # Try to convert to numeric values if possible
            numeric_pressure = pd.to_numeric(goalie_shots['defensive_pressure'], errors='coerce')
            if not numeric_pressure.isna().all():
                avg_defensive_pressure = numeric_pressure.mean()
                
                # Calculate a defensive adjustment factor
                # Higher defensive pressure means more help for the goalie
                # So we adjust the goalie index down slightly if they have lots of defensive help
                league_avg_pressure = 1.0  # This is a placeholder - ideally would be calculated from all data
                defensive_adjustment = -(avg_defensive_pressure - league_avg_pressure) * 0.01
            else:
                print(f"Note: defensive_pressure for goalie {goalie_id} contains non-numeric values")
        except:
            print(f"Warning: Could not process defensive_pressure for goalie {goalie_id}")
    
    # Alternative: use defenders within 3m if available
    elif 'defenderswithin3m' in goalie_shots.columns:
        try:
            numeric_defenders = pd.to_numeric(goalie_shots['defenderswithin3m'], errors='coerce')
            if not numeric_defenders.isna().all():
                avg_defensive_pressure = numeric_defenders.mean()
                
                # Calculate defensive adjustment
                league_avg_pressure = 1.0  # Placeholder
                defensive_adjustment = -(avg_defensive_pressure - league_avg_pressure) * 0.01
            else:
                print(f"Note: defenderswithin3m for goalie {goalie_id} contains non-numeric values")
        except:
            print(f"Warning: Could not process defenderswithin3m for goalie {goalie_id}")
    
    # Calculate average xG per shot
    xg_per_shot = xg_against / total_shots if total_shots > 0 else 0
    
    # Generate metrics dictionary with consistent usage of xg_allattempts field
    metrics = {
        'goalie_id': goalie_id,
        'goalie_type': goalie_type,
        'total_shots': total_shots,
        'save_count': save_count,
        'save_percentage': save_percentage,
        'xg_allattempts': xg_against,  # This is from the xg_allattempts field
        'goals_saved_above_expected': goals_saved_above_expected,
        'rebound_count': rebound_count,
        'rebound_percentage': rebound_percentage,
        'dangerous_rebound_count': dangerous_rebound_count,
        'dangerous_rebound_percentage': dangerous_rebound_percentage,
        'high_danger_shots': high_danger_shots,
        'high_danger_saves': high_danger_saves,
        'high_danger_save_percentage': high_danger_save_percentage,
        'medium_danger_shots': medium_danger_shots,
        'medium_danger_saves': medium_danger_saves,
        'medium_danger_save_percentage': medium_danger_save_percentage,
        'low_danger_shots': low_danger_shots,
        'low_danger_saves': low_danger_saves,
        'low_danger_save_percentage': low_danger_save_percentage,
        'avg_defensive_pressure': avg_defensive_pressure,
        'defensive_adjustment': defensive_adjustment,
        'shot_quality_tracked': 'shot_quality' in goalie_shots.columns
    }
    
    return metrics

def visualize_goalie_performance(goalie_metrics):
    """
    Create improved, more readable visualizations for goalie performance.
    
    Args:
        goalie_metrics (pd.DataFrame): DataFrame containing goalie metrics
    """
    if goalie_metrics.empty:
        print("No goalie metrics available for visualization")
        return
        
    print("\nCreating visualizations...")
    
    # Create output directory if it doesn't exist
    if not os.path.exists('goalie_analysis'):
        os.makedirs('goalie_analysis')
    
    # Filter goalies with enough data for meaningful analysis
    min_shots = 5
    qualified_goalies = goalie_metrics[goalie_metrics['total_shots'] >= min_shots].copy()
    
    if len(qualified_goalies) == 0:
        print("Not enough data for meaningful visualizations (no goalies with 5+ shots)")
        return
    
    # Display the number of goalies with sufficient data
    print(f"\nCreating visualizations for {len(qualified_goalies)} goalies with at least {min_shots} shots")
    
    # Plot 1: Save Percentage vs Goals Saved Above Expected
    plt.figure(figsize=(12, 10))
    
    # Get data averages for reference lines
    avg_save = qualified_goalies['save_percentage'].mean()
    avg_gsae = qualified_goalies['goals_saved_above_expected'].mean()
    
    # Create scatter plot with improved readability
    scatter = plt.scatter(
        qualified_goalies['save_percentage'],
        qualified_goalies['goals_saved_above_expected'],
        s=qualified_goalies['total_shots'] * 5,  # Size based on shot volume
        c=qualified_goalies['goalie_index'],     # Color based on goalie index
        cmap='viridis',
        alpha=0.7,
        edgecolors='black'
    )
    
    # Add reference lines with improved styling
    plt.axhline(y=avg_gsae, color='red', linestyle='-', alpha=0.3)
    plt.axvline(x=avg_save, color='red', linestyle='-', alpha=0.3)
    
    # Add labels for top performers
    top_performers = qualified_goalies.nlargest(5, 'goalie_index')
    for _, goalie in top_performers.iterrows():
        plt.annotate(
            f"ID: {int(goalie['goalie_id'])}",
            xy=(goalie['save_percentage'], goalie['goals_saved_above_expected']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
        )
    
    # Improve plot styling
    plt.title('Save Percentage vs Goals Saved Above Expected', fontsize=16, fontweight='bold')
    plt.xlabel('Save Percentage', fontsize=14, fontweight='bold')
    plt.ylabel('Goals Saved Above Expected', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add colorbar with improved readability
    cbar = plt.colorbar(scatter)
    cbar.set_label('Goalie Index', fontsize=12, fontweight='bold')
    
    # Add quadrant labels with better positioning and readability
    plt.annotate(
        'ELITE\n(High save %, high GSAE)', 
        xy=(avg_save + (qualified_goalies['save_percentage'].max() - avg_save)/2, 
            avg_gsae + (qualified_goalies['goals_saved_above_expected'].max() - avg_gsae)/2),
        ha='center',
        va='center',
        bbox=dict(boxstyle="round,pad=0.3", fc="palegreen", ec="green", alpha=0.5),
        fontsize=12
    )
    
    plt.annotate(
        'LUCKY\n(High save %, low GSAE)', 
        xy=(avg_save + (qualified_goalies['save_percentage'].max() - avg_save)/2, 
            avg_gsae/2),
        ha='center',
        va='center',
        bbox=dict(boxstyle="round,pad=0.3", fc="khaki", ec="orange", alpha=0.5),
        fontsize=12
    )
    
    plt.annotate(
        'UNLUCKY\n(Low save %, high GSAE)', 
        xy=(avg_save/2, 
            avg_gsae + (qualified_goalies['goals_saved_above_expected'].max() - avg_gsae)/2),
        ha='center',
        va='center',
        bbox=dict(boxstyle="round,pad=0.3", fc="lightskyblue", ec="blue", alpha=0.5),
        fontsize=12
    )
    
    plt.annotate(
        'STRUGGLING\n(Low save %, low GSAE)', 
        xy=(avg_save/2, avg_gsae/2),
        ha='center',
        va='center',
        bbox=dict(boxstyle="round,pad=0.3", fc="lightcoral", ec="red", alpha=0.5),
        fontsize=12
    )
    
    # Add explanatory note
    plt.figtext(
        0.5, 0.01, 
        "Bubble size represents total shots faced. Color intensity shows Goalie Index score.",
        ha='center',
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
    )
    
    plt.tight_layout()
    plt.savefig('goalie_analysis/save_pct_vs_gsae.png', dpi=300)
    plt.close()
    
    # Plot 2: Goals Saved Above Expected vs Rebound Control
    plt.figure(figsize=(12, 10))
    
    # Get data averages for reference lines
    avg_gsae = qualified_goalies['goals_saved_above_expected'].mean()
    avg_rebounds = qualified_goalies['dangerous_rebound_percentage'].mean()
    
    # Create scatter plot
    scatter = plt.scatter(
        qualified_goalies['goals_saved_above_expected'],
        qualified_goalies['dangerous_rebound_percentage'],
        s=qualified_goalies['total_shots'] * 5,  # Size based on shot volume
        c=qualified_goalies['save_percentage'],  # Color based on save percentage
        cmap='RdYlGn',
        alpha=0.7,
        edgecolors='black'
    )
    
    # Add reference lines
    plt.axhline(y=avg_rebounds, color='red', linestyle='-', alpha=0.3)
    plt.axvline(x=avg_gsae, color='red', linestyle='-', alpha=0.3)
    
    # Add labels for notable goalies
    for _, goalie in qualified_goalies.nlargest(5, 'goalie_index').iterrows():
        plt.annotate(
            f"ID: {int(goalie['goalie_id'])}",
            xy=(goalie['goals_saved_above_expected'], goalie['dangerous_rebound_percentage']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
        )
    
    # Improve plot styling
    plt.title('Goals Saved Above Expected vs Dangerous Rebound %', fontsize=16, fontweight='bold')
    plt.xlabel('Goals Saved Above Expected', fontsize=14, fontweight='bold')
    plt.ylabel('Dangerous Rebound %', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Save Percentage', fontsize=12, fontweight='bold')
    
    # Label quadrants
    plt.annotate(
        'ELITE\n(Good xG performance,\nfew dangerous rebounds)', 
        xy=(avg_gsae + (qualified_goalies['goals_saved_above_expected'].max() - avg_gsae)/2, 
            avg_rebounds/2),
        ha='center',
        va='center',
        bbox=dict(boxstyle="round,pad=0.3", fc="palegreen", ec="green", alpha=0.5),
        fontsize=11
    )
    
    plt.annotate(
        'GOOD SAVES, POOR REBOUNDS\n(Good xG performance,\nmany dangerous rebounds)', 
        xy=(avg_gsae + (qualified_goalies['goals_saved_above_expected'].max() - avg_gsae)/2, 
            avg_rebounds + (qualified_goalies['dangerous_rebound_percentage'].max() - avg_rebounds)/2),
        ha='center',
        va='center',
        bbox=dict(boxstyle="round,pad=0.3", fc="khaki", ec="orange", alpha=0.5),
        fontsize=11
    )
    
    # Move BELOW AVERAGE label to the left edge to avoid data points
    plt.annotate(
        'BELOW AVERAGE\n(Poor xG performance,\nmany dangerous rebounds)', 
        xy=(qualified_goalies['goals_saved_above_expected'].min() * 0.9, 
            avg_rebounds + (qualified_goalies['dangerous_rebound_percentage'].max() - avg_rebounds)/2),
        ha='left',
        va='center',
        bbox=dict(boxstyle="round,pad=0.3", fc="lightcoral", ec="red", alpha=0.5),
        fontsize=11
    )
    
    # Move GOOD REBOUND CONTROL label to the bottom left corner to avoid data points
    plt.annotate(
        'GOOD REBOUND CONTROL\n(Poor xG performance,\nfew dangerous rebounds)', 
        xy=(qualified_goalies['goals_saved_above_expected'].min() * 0.9, 
            qualified_goalies['dangerous_rebound_percentage'].min() * 0.9),
        ha='left',
        va='bottom',
        bbox=dict(boxstyle="round,pad=0.3", fc="lightskyblue", ec="blue", alpha=0.5),
        fontsize=11
    )
    
    # Add explanatory note
    plt.figtext(
        0.5, 0.01, 
        "Bubble size represents total shots faced. Color shows save percentage (green = higher).",
        ha='center',
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
    )
    
    plt.tight_layout()
    plt.savefig('goalie_analysis/gsae_vs_rebound.png', dpi=300)
    plt.close()
    
    # New Plot 3: Goalie Index Components
    plt.figure(figsize=(14, 10))
    
    # Select top and bottom 5 goalies for easier comparison
    top_goalies = qualified_goalies.nlargest(5, 'goalie_index')
    bottom_goalies = qualified_goalies.nsmallest(5, 'goalie_index')
    selected_goalies = pd.concat([top_goalies, bottom_goalies])
    
    # Create component data
    components = {
        'GSAE Component': selected_goalies['goals_saved_above_expected'] * 2,
        'Rebound Component': -selected_goalies['dangerous_rebound_percentage'] * 5,
        'Save % Component': selected_goalies['save_percentage'] * 3,
        'Defensive Component': selected_goalies['defensive_adjustment'] * 1
    }
    
    # Convert to DataFrame for plotting
    components_df = pd.DataFrame(components, index=selected_goalies['goalie_id'])
    
    # Sort by total Goalie Index
    components_df['Total'] = components_df.sum(axis=1)
    components_df = components_df.sort_values('Total', ascending=False)
    
    # Create stacked bar chart
    ax = components_df[['GSAE Component', 'Rebound Component', 'Save % Component', 'Defensive Component']].plot(
        kind='bar', 
        stacked=True,
        figsize=(14, 10),
        colormap='viridis',
        edgecolor='black'
    )
    
    # Add total value labels
    for i, total in enumerate(components_df['Total']):
        plt.text(
            i, 
            total + 0.1,
            f'Total: {total:.2f}',
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
        )
    
    # Improve plot styling
    plt.axhline(y=0, color='black', linestyle='-')
    plt.title('Goalie Index Component Breakdown', fontsize=16, fontweight='bold')
    plt.xlabel('Goalie ID', fontsize=14, fontweight='bold')
    plt.ylabel('Component Value', fontsize=14, fontweight='bold')
    plt.grid(True, axis='y', alpha=0.3)
    plt.legend(title='Components', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Turn x-axis labels to integers
    plt.xticks(range(len(components_df)), [int(x) for x in components_df.index], rotation=45)
    
    # Add explanatory note
    plt.figtext(
        0.5, 0.01, 
        "Index = (GSAE × 2) - (Dangerous Rebound % × 5) + (Save % × 3) + (Defensive Adjustment × 1)",
        ha='center',
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
    )
    
    plt.tight_layout()
    plt.savefig('goalie_analysis/goalie_index_components.png', dpi=300)
    plt.close()
    
    # New Plot 4: Shot Quality vs Save Performance
    plt.figure(figsize=(14, 10))
    
    # Calculate average xG per shot using xg_allattempts
    qualified_goalies['avg_xg_per_shot'] = qualified_goalies['xg_allattempts'] / qualified_goalies['total_shots']
    
    # Create scatter plot
    scatter = plt.scatter(
        qualified_goalies['avg_xg_per_shot'],
        qualified_goalies['save_percentage'],
        s=qualified_goalies['total_shots'] * 5,
        c=qualified_goalies['goals_saved_above_expected'],
        cmap='RdYlGn',
        alpha=0.7,
        edgecolors='black'
    )
    
    # Add labels for top performers
    for _, goalie in qualified_goalies.nlargest(5, 'goalie_index').iterrows():
        plt.annotate(
            f"ID: {int(goalie['goalie_id'])}",
            xy=(goalie['avg_xg_per_shot'], goalie['save_percentage']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
        )
    
    # Improve plot styling
    plt.title('Average Shot Quality vs Save Percentage', fontsize=16, fontweight='bold')
    plt.xlabel('Average xG per Shot (Higher = More Dangerous Shots)', fontsize=14, fontweight='bold')
    plt.ylabel('Save Percentage', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Goals Saved Above Expected', fontsize=12, fontweight='bold')
    
    # Add explanatory note
    plt.figtext(
        0.5, 0.01, 
        "Bubble size represents total shots faced. Color shows GSAE (green = better performance).",
        ha='center',
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
    )
    
    # Add trend line (if applicable)
    if len(qualified_goalies) >= 3:
        try:
            # Calculate trend line
            z = np.polyfit(
                qualified_goalies['avg_xg_per_shot'], 
                qualified_goalies['save_percentage'], 
                1
            )
            p = np.poly1d(z)
            
            # Generate points for trend line
            x_trend = np.linspace(
                qualified_goalies['avg_xg_per_shot'].min(),
                qualified_goalies['avg_xg_per_shot'].max(),
                100
            )
            
            # Plot trend line
            plt.plot(
                x_trend, 
                p(x_trend), 
                "r--", 
                alpha=0.7,
                label=f"Trend: y={z[0]:.3f}x+{z[1]:.3f}"
            )
            plt.legend()
        except:
            # Skip trend line if there's an error
            pass
    
    plt.tight_layout()
    plt.savefig('goalie_analysis/shot_quality_vs_save_pct.png', dpi=300)
    plt.close()
    
    # New Plot 5: Shot Quality Distribution by Save Percentage
    if 'shot_quality_tracked' in qualified_goalies.columns and qualified_goalies['shot_quality_tracked'].any():
        plt.figure(figsize=(14, 10))
        
        # Get average save percentages by danger level
        avg_high = qualified_goalies['high_danger_save_percentage'].mean()
        avg_medium = qualified_goalies['medium_danger_save_percentage'].mean()
        avg_low = qualified_goalies['low_danger_save_percentage'].mean()
        
        # Create data for plotting
        danger_levels = ['High Danger', 'Medium Danger', 'Low Danger']
        save_percentages = [avg_high, avg_medium, avg_low]
        
        # Create bar plot
        ax = sns.barplot(
            x=danger_levels,
            y=save_percentages,
            hue=danger_levels,  # Set hue to match x values
            palette='viridis',
            legend=False  # Don't show the legend as it would be redundant
        )
        
        # Add data labels to bars
        for i, pct in enumerate(save_percentages):
            ax.text(
                i, 
                pct + 0.02, 
                f'{pct:.3f}', 
                ha='center',
                va='bottom',
                fontsize=11
            )
        
        plt.title('Save Percentage by Shot Quality', fontweight='bold')
        plt.xlabel('Shot Quality', fontweight='bold')
        plt.ylabel('Save Percentage', fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('goalie_analysis/save_pct_by_quality.png', dpi=300)
        plt.close()
        
        # Also create a plot showing the distribution of shots by quality
        plt.figure(figsize=(14, 10))
        
        # Sum up shots by danger level
        total_high = qualified_goalies['high_danger_shots'].sum()
        total_medium = qualified_goalies['medium_danger_shots'].sum()
        total_low = qualified_goalies['low_danger_shots'].sum()
        
        # Create data for plotting
        shot_counts = [total_high, total_medium, total_low]
        
        # Create bar plot
        ax = sns.barplot(
            x=danger_levels,
            y=shot_counts,
            hue=danger_levels,  # Set hue to match x values
            palette='viridis',
            legend=False  # Don't show the legend as it would be redundant
        )
        
        # Add data labels to bars
        for i, count in enumerate(shot_counts):
            ax.text(
                i, 
                count + 5, 
                f'{count:.0f}', 
                ha='center',
                va='bottom',
                fontsize=11
            )
        
        plt.title('Shot Quality Distribution', fontweight='bold')
        plt.xlabel('Shot Quality', fontweight='bold')
        plt.ylabel('Count', fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('goalie_analysis/shot_quality_distribution.png', dpi=300)
        plt.close()
    else:
        print("Skipping shot quality distribution plot - no shot quality data in goalie metrics")
    
    # If we have sufficient data, create correlation matrix
    if len(qualified_goalies) >= 5:
        plt.figure(figsize=(12, 10))
        
        # Select relevant columns for correlation
        corr_columns = [
            'save_percentage', 
            'goals_saved_above_expected', 
            'dangerous_rebound_percentage',
            'goalie_index'
        ]
        
        # Create correlation matrix
        corr_matrix = qualified_goalies[corr_columns].corr()
        
        # Plot heatmap with improved readability
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(
            corr_matrix, 
            mask=mask,
            annot=True,
            cmap='coolwarm',
            vmin=-1, vmax=1,
            center=0,
            square=True,
            linewidths=.5,
            cbar_kws={"shrink": .8},
            fmt='.2f',
            annot_kws={"size": 12}
        )
        
        # Improve plot styling
        plt.title('Correlation Matrix of Goalie Metrics', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('goalie_analysis/correlation_matrix.png', dpi=300)
        plt.close()
    
def save_goalie_metrics(goalie_metrics, output_path):
    """
    Save goalie metrics to a CSV file.
    
    Args:
        goalie_metrics (pd.DataFrame): DataFrame containing goalie metrics
        output_path (str): Path to save the CSV file
    """
    if goalie_metrics.empty:
        print("No goalie metrics to save")
        return
        
    print(f"\nSaving goalie metrics to {output_path}")
    goalie_metrics.to_csv(output_path, index=False)

def generate_analysis_report(goalie_metrics, output_path):
    """
    Generate a comprehensive analysis report for goalie performance with enhanced insights.
    
    Args:
        goalie_metrics (pd.DataFrame): DataFrame containing goalie metrics
        output_path (str): Path to save the report
    """
    print(f"\nGenerating analysis report to {output_path}")
    
    # Skip if no metrics available
    if goalie_metrics.empty:
        print("No goalie metrics available for report generation")
        with open(output_path, 'w') as f:
            f.write("# No Goalie Metrics Available\n\nInsufficient data for analysis.\n")
        return
    
    # Filter for goalies with sufficient data for meaningful analysis
    min_shots = 5
    qualified_goalies = goalie_metrics[goalie_metrics['total_shots'] >= min_shots].copy()
    qualified_goalies = qualified_goalies.sort_values('goalie_index', ascending=False)
    
    # Calculate league averages
    avg_save_pct = qualified_goalies['save_percentage'].mean()
    avg_gsae = qualified_goalies['goals_saved_above_expected'].mean()
    avg_rebound_pct = qualified_goalies['dangerous_rebound_percentage'].mean()
    
    # Calculate correlations
    corr_save_gsae = qualified_goalies['save_percentage'].corr(qualified_goalies['goals_saved_above_expected'])
    corr_save_rebound = qualified_goalies['save_percentage'].corr(qualified_goalies['dangerous_rebound_percentage'])
    
    # Generate report
    with open(output_path, 'w') as f:
        f.write("# Goalie Index Analysis Report\n\n")
        
        f.write("## Overview\n\n")
        f.write(f"This analysis evaluated {len(goalie_metrics)} goalies, with {len(qualified_goalies)} having at least {min_shots} shots.\n\n")
        
        f.write("The Goalie Index is calculated using the formula:\n")
        f.write("```\n")
        f.write("Goalie Index = (Goals Saved Above Expected * 2) - (Dangerous Rebound % * 5) + (Save % * 3) + (Defensive Adjustment * 1)\n")
        f.write("```\n\n")
        
        f.write("This formula rewards:\n")
        f.write("- Goalies who outperform their expected goals against\n")
        f.write("- Goalies who control rebounds effectively\n")
        f.write("- Goalies who maintain a high save percentage\n")
        f.write("- Goalies who perform well under defensive pressure\n\n")
        
        f.write("## Top Performing Goalies\n\n")
        f.write("| Goalie ID | Total Shots | Save % | Goals Saved Above Expected | Dangerous Rebound % | Goalie Index |\n")
        f.write("|-----------|-------------|--------|---------------------------|---------------------|-------------|\n")
        
        for _, goalie in qualified_goalies.head(10).iterrows():
            f.write(f"| {int(goalie['goalie_id'])} | {goalie['total_shots']} | {goalie['save_percentage']:.3f} | {goalie['goals_saved_above_expected']:.2f} | {goalie['dangerous_rebound_percentage']:.3f} | {goalie['goalie_index']:.2f} |\n")
        
        f.write("\n## Key Findings\n\n")
        f.write(f"- Average save percentage: {avg_save_pct:.3f}\n")
        f.write(f"- Average goals saved above expected: {avg_gsae:.2f}\n")
        f.write(f"- Average dangerous rebound percentage: {avg_rebound_pct:.3f}\n\n")
        
        # Add shot quality analysis if available
        if all(col in qualified_goalies.columns for col in ['high_danger_save_pct', 'medium_danger_save_pct', 'low_danger_save_pct']):
            avg_high_danger = qualified_goalies['high_danger_save_pct'].mean()
            avg_med_danger = qualified_goalies['medium_danger_save_pct'].mean()
            avg_low_danger = qualified_goalies['low_danger_save_pct'].mean()
            
            f.write("### Shot Quality Analysis\n\n")
            f.write(f"- Average save % on high-danger shots: {avg_high_danger:.3f}\n")
            f.write(f"- Average save % on medium-danger shots: {avg_med_danger:.3f}\n")
            f.write(f"- Average save % on low-danger shots: {avg_low_danger:.3f}\n\n")
        
        f.write("### Correlation Analysis\n\n")
        f.write(f"- Correlation between save percentage and goals saved above expected: {corr_save_gsae:.3f}\n")
        f.write(f"- Correlation between save percentage and dangerous rebound percentage: {corr_save_rebound:.3f}\n\n")
        
        f.write("## Recommendations\n\n")
        f.write("1. For more accurate analysis, collect more shot data with complete location information.\n")
        f.write("2. Consider tracking rebounds that result in high-danger chances even if no shot is recorded.\n")
        f.write("3. Develop a goalie scouting system that weights these metrics based on team defensive structure.\n")
        f.write("4. Analyze goalie performance against shot quality to identify specialists.\n")
        f.write("5. Track goalie fatigue by analyzing performance changes over time and with shot volume.\n")

def main():
    """
    Main function to run the goalie index analysis.
    """
    data_path = "Linhac24-25_Sportlogiq.csv"
    output_dir = "goalie_analysis"
    metrics_path = f"{output_dir}/goalie_metrics.csv"
    report_path = f"{output_dir}/goalie_analysis_report.md"
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        print("Creating output directory...")
        os.makedirs(output_dir)
    
    # Load and process data
    shot_df = load_and_preprocess_data(data_path)
    
    # Skip further processing if data loading failed
    if shot_df.empty:
        print("Error: No valid shot data found. Exiting.")
        return
    
    # Add event type checking for troubleshooting
    print("\nSample event names in filtered data:")
    print(shot_df['eventname'].value_counts().head(10))
    
    # Calculate goalie metrics
    goalie_metrics = calculate_goalie_metrics(shot_df)
    
    # Skip further processing if no metrics were calculated
    if goalie_metrics.empty:
        print("Error: No goalie metrics could be calculated. Exiting.")
        return
    
    # Create visualizations
    visualize_goalie_performance(goalie_metrics)
    
    # Save metrics to CSV
    save_goalie_metrics(goalie_metrics, metrics_path)
    
    # Generate analysis report
    generate_analysis_report(goalie_metrics, report_path)
    
    print("\nAnalysis complete! Results saved in the goalie_analysis directory.")
    
    # Display top goalies
    top_goalies = goalie_metrics.sort_values('goalie_index', ascending=False)
    print("\nTop 5 goalies by Goalie Index (min 5 shots):")
    print(top_goalies.head().loc[:, ['goalie_id', 'total_shots', 'save_percentage', 
                                    'goals_saved_above_expected', 'dangerous_rebound_percentage', 'goalie_index']])

if __name__ == "__main__":
    main()
