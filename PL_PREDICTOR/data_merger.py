import pandas as pd
import os

def combine_all_seasons(data_folder='data/'):
    """Combine all individual season files into one dataset"""
    
    all_data = []
    
    # Get all CSV files that look like season data
    season_files = [f for f in os.listdir(data_folder) 
                   if f.endswith('.csv') and len(f) == 6]
    
    print(f"Found {len(season_files)} season files")
    
    for file in sorted(season_files):
        df = pd.read_csv(os.path.join(data_folder, file))
        
        # Extract season year from filename
        season_year = file.replace('.csv', '')
        
        # Create proper football season format
        start_year = f"20{season_year}"
        end_year = str(int(start_year) + 1)
        df['season'] = f"{start_year}-{end_year[-2:]}"
        
        # ADD THIS LINE - append to list
        all_data.append(df)
        print(f"Loaded {file}: {len(df)} matches → {df['season'].iloc[0]}")
    
    # Combine all seasons
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Save combined dataset
    combined_df.to_csv(os.path.join(data_folder, 'combined_all_seasons.csv'), index=False)
    
    print(f"\n✅ Combined dataset saved: {len(combined_df)} total matches")
    return combined_df

# Run this
combine_all_seasons()
