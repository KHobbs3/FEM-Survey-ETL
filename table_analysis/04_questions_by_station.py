import pandas as pd
from src.config import *
import re
import numpy as np
from datetime import datetime
import geopandas as gpd
import glob
from src.recodes import apply_writein_recodes

data=pd.read_csv(DIR_WEIGHTED_DATA)
data=apply_writein_recodes(data)   # "Autre" write-ins -> reviewed recodes (occupation, radio_language, ...)
survey = pd.read_excel(DIR_META_DATA, sheet_name='survey')
choices = pd.read_excel(DIR_META_DATA, sheet_name='choices')

def get_unsplit_answers(data,var):
    df=split_dribar_weighted(data,"",var)
    df=df.fillna(0)
    df=df.apply(lambda col: col.astype(str) + '%' )
    df.reset_index(inplace=True)
    df['Combined_Column'] = df.apply(lambda row: ' \n '.join(row), axis=1)
    merged_str="\n\n".join(df.Combined_Column)+"\n\n"
    return merged_str


def value_counts_multiple_answers_weighted(var: pd.Series, weights: pd.Series) -> pd.Series:
    var_name = var.name
    df = pd.DataFrame({'answer': var, 'weight': weights})
    df = df.dropna(subset=['answer'])          
    df['answer'] = df['answer'].astype(str).str.split('|')
    df = df.explode('answer', ignore_index=True) 
    out = df.groupby('answer')['weight'].sum().sort_values(ascending=False)
    out.rename({"answer":var_name})
    out.index.name=var_name
    return out


def get_raw_counts(data, var):
    """Get raw (unweighted) count of responses per answer"""
    df = pd.DataFrame({'answer': data[var], 'weight': data['combined_weight_adjusted']})
    df = df.dropna(subset=['answer'])
    df['answer'] = df['answer'].astype(str).str.split('|')
    df = df.explode('answer', ignore_index=True)
    out = df.groupby('answer').size().sort_values(ascending=False)
    out.index.name = var
    return out


def get_weighted_counts(data, var):
    """Get weighted count (effective population) per answer"""
    df = pd.DataFrame({'answer': data[var], 'weight': data['combined_weight_adjusted']})
    df = df.dropna(subset=['answer'])
    df['answer'] = df['answer'].astype(str).str.split('|')
    df = df.explode('answer', ignore_index=True)
    out = df.groupby('answer')['weight'].sum().sort_values(ascending=False)
    out.index.name = var
    return out


def get_unsplit_answers_with_counts(data, var):
    """Get answer percentages with raw and weighted counts - formatted for Streamlit viz"""
    # Get percentages
    df_pct = split_dribar_weighted(data, "", var)
    df_pct = df_pct.fillna(0)
    df_pct.reset_index(inplace=True)
    
    # Get raw counts
    raw_counts = get_raw_counts(data, var)
    
    # Get weighted counts
    weighted_counts = get_weighted_counts(data, var)
    
    # Format output: Answer\nX% (with N and Weighted N in display)
    # The regex in page_radio.py expects: ([^\n%]+?)\s*\n\s*([\d.]+)%
    combined_rows = []
    for _, row in df_pct.iterrows():
        answer_label = row[0]  # Answer name
        pct_value = row[1]      # Percentage
        
        if answer_label in raw_counts.index:
            raw_n = int(raw_counts[answer_label])
            weighted_n = round(weighted_counts[answer_label], 1)
            
            # Format: "Answer (N=45, Weighted N=120.3)\n45.2%"
            # The visualizer will extract: "Answer (N=45, Weighted N=120.3)" and "45.2"
            formatted_answer = f"{answer_label} (N={raw_n}, Weighted N={weighted_n})"
            combined_rows.append(f"{formatted_answer}\n{pct_value}%")
    
    # Join with double newline between entries
    merged_str = "\n\n".join(combined_rows) + "\n\n"
    return merged_str


def split_dribar_weighted(df, split_var, question):
  if df[question].nunique()==0:
     return pd.DataFrame()
  if split_var!="No Split" and split_var!="":
    ret=df.groupby([split_var]).apply(lambda x: value_counts_multiple_answers_weighted(x[question],x["combined_weight_adjusted"])).reset_index().rename(columns={"weight":"Count"})
    if ret.shape[0]==1:
          var=ret[split_var].values[0]
          st.warning(f"The selected split is not possible, all interviews are {split_var} ="+var)
          return pd.DataFrame()
    ret=pd.pivot(ret,columns=split_var,index=question,values="Count")#.fillna(0)
    sums=pd.DataFrame(df.groupby(split_var).sum())["combined_weight_adjusted"]
    return round(ret.div(sums,axis=1)*100,2)
  
  ret=round(100*value_counts_multiple_answers_weighted(df[question],df["combined_weight_adjusted"])/df["combined_weight_adjusted"].sum(),2)
  return ret.reset_index().rename(columns={"weight":question,question:""}).set_index("")


TABLE_COLUMNS=[
    'Question',
    'All Answers',
    'Unweighted Count',
    'Weighted Count',
    'USE',
    'AGE_GROUP',
    'GENDER',
    'URBAN_RURAL'
    ]

QUESTIONS=[
    'station_most_listened',
    'station_listened_yesterday',
    'station_past_7_days',
    'media_type',
    'radio_consumption_method',
    'favourite_radio_format',
    'radio_ad_saturation',
    'radio_language',
    'radio_when',
    'radio_what',
    'radio_trust',
    'radio_influence',
    'known_contraceptive_options',
    'decision_confident_3',
    'decision_confident',
    'time_before_preferred_pregnancy',
    'willingness_to_travel',
    'occupation',
    'life_goals',
    'forming_beliefs',
    'role_models',
    'likeable_traits',
    'happiness',
    'satisfaction'
    ]


# ============================================================================
# ADD STATION LABELS TO DATA
# ============================================================================

# Path to station files
STATION_PATH = DIR_STATION

# Create GeoDataFrame from data
data_gdf = gpd.GeoDataFrame(
    data,
    geometry=gpd.points_from_xy(data['gps-Longitude'], data['gps-Latitude']),
    crs='epsg:4326'
)

# Initialize station label column
if 'station_label' not in data.columns:
    data['station_label'] = None

# Convert to projected CRS for accurate distance calculations
data_gdf_projected = data_gdf.to_crs('epsg:32632')

# Collect all station polygons for later nearest-neighbor search
all_stations = []

# Iterate through each station file
for station_file in glob.glob(STATION_PATH + "*.gpkg"):
    # Read the station polygon file
    station_gdf = gpd.read_file(station_file)
    
    # Ensure CRS matches
    if station_gdf.crs != data_gdf.crs:
        station_gdf = station_gdf.to_crs(data_gdf.crs)
    
    # Convert to projected CRS for distance calculations
    station_gdf_projected = station_gdf.to_crs('epsg:32632')
    
    # Extract station name from filename
    station_name = station_file.split('/')[-1].replace('.gpkg', '')
    
    # Add station name to the GeoDataFrame
    station_gdf_projected['station_name'] = station_name
    
    # Store for later nearest-neighbor search
    all_stations.append(station_gdf_projected)
    
    # Perform spatial join to find points within polygons
    points_in_station = gpd.sjoin(
        data_gdf_projected, 
        station_gdf_projected, 
        how='inner', 
        predicate='within'
    )
    
    # Update the station_label for matching points
    if len(points_in_station) > 0:
        data.loc[points_in_station.index, 'station_label'] = station_name
    print(f"Assigned {len(points_in_station)} points to station: {station_name}")

print(f"\nTotal points labeled (within polygons): {data['station_label'].notna().sum()}")
print(f"Total points unlabeled: {data['station_label'].isna().sum()}")

# Snap unmatched points to nearest station
unmatched_indices = data[data['station_label'].isna()].index
unmatched_points = data_gdf_projected.loc[unmatched_indices]

# Distance threshold in meters (1 km)
MAX_DISTANCE = 1000

if len(unmatched_points) > 0 and len(all_stations) > 0:
    print(f"\nSnapping {len(unmatched_points)} unmatched points to nearest stations (within {MAX_DISTANCE}m)...")
    
    # Combine all station polygons into one GeoDataFrame
    all_stations_gdf = pd.concat(all_stations, ignore_index=True)
    
    # Reset index to ensure clean integer indexing
    all_stations_gdf = all_stations_gdf.reset_index(drop=True)
    
    snapped_count = 0
    
    # For each unmatched point, find nearest station
    for idx in unmatched_indices:
        point_geom = data_gdf_projected.loc[idx, 'geometry']
        
        # Calculate distance to all stations
        all_stations_gdf['distance'] = all_stations_gdf.geometry.distance(point_geom)
        
        # Get the position of minimum distance
        min_dist_position = all_stations_gdf['distance'].argmin()
        nearest_station = all_stations_gdf.iloc[min_dist_position]
        
        # Only snap if within threshold distance
        if nearest_station['distance'] <= MAX_DISTANCE:
            data.loc[idx, 'station_label'] = nearest_station['station_name']
            snapped_count += 1
    
    print(f"Snapped {snapped_count} points to nearest stations.")
    print(f"{len(unmatched_points) - snapped_count} points were beyond {MAX_DISTANCE}m threshold.")

# Final summary
print(f"\nTotal points labeled: {data['station_label'].notna().sum()}")
print(f"Total points unlabeled: {data['station_label'].isna().sum()}")

# ============================================================================
# CREATE STATION-BASED TABLES
# ============================================================================

# Get list of stations
STATIONS = [station for station in data['station_label'].dropna().unique()]
TABLE_STATION_COLUMNS = ['Question', *STATIONS]

# Radio questions to analyze by station
RADIO_QUESTIONS = [
    'station_most_listened',
    'station_listened_yesterday',
    'station_past_7_days',
    'media_type',
    'radio_consumption_method',
    'favourite_radio_format',
    'radio_ad_saturation',
    'radio_language',
    'radio_when',
    'radio_what',
    'radio_trust',
    'radio_influence'
]

# ============================================================================
# POPULATE: Radio Questions / Station Table (WEIGHTED with N counts)
# ============================================================================
table_radio_station = pd.DataFrame(columns=TABLE_STATION_COLUMNS)

for question_var in RADIO_QUESTIONS:
    row = [question_var]
    
    for station in STATIONS:
        station_filter = data[data['station_label'] == station]
        
        # Use weighted calculations with counts - formatted for viz
        if len(station_filter) > 0:
            station_answers = get_unsplit_answers_with_counts(station_filter, question_var)
        else:
            station_answers = "No data"
        
        row.append(station_answers)
    
    table_radio_station.loc[len(table_radio_station)] = row

# Map question names to readable text
table_radio_station.Question = table_radio_station.Question.map(
    dict(zip(survey.name, survey.hint))
)

# ============================================================================
# ADD STATION SUMMARY STATISTICS
# ============================================================================

# Create summary table with weighted counts
station_summary = pd.DataFrame({
    'Station': STATIONS,
    'Unweighted_Count': [len(data[data['station_label'] == s]) for s in STATIONS],
    'Weighted_Count': [data[data['station_label'] == s]['combined_weight_adjusted'].sum() 
                       for s in STATIONS],
    'Avg_Weight': [data[data['station_label'] == s]['combined_weight_adjusted'].mean() 
                   for s in STATIONS]
}).round(2)

print("\nStation Coverage Summary (Weighted):")
print(station_summary)

# ============================================================================
# EXPORT TO EXCEL
# ============================================================================

today_date = datetime.today()
date_string = today_date.strftime("%y_%m_%d")

table=pd.DataFrame(columns=TABLE_COLUMNS)
with pd.ExcelWriter(DIR_FINAL / f"Benin_question_table_by_station_{date_string}.xlsx",
                    engine="openpyxl") as writer:
        
    # NEW: Station-based table (formatted for Streamlit viz)
    table_radio_station.to_excel(writer, sheet_name="Radio_By_Station", 
                                  index=False, freeze_panes=(1, 0))
    
    # NEW: Station summary
    station_summary.to_excel(writer, sheet_name="Station_Summary", 
                            index=False, freeze_panes=(1, 0))

print(f"\nExported tables to: Benin_question_table_by_station_{date_string}.xlsx")
print("  - Radio_By_Station (by radio station with N counts - formatted for viz)")
print("  - Station_Summary (weighted counts)")

dupes = data[data.duplicated(subset=['household_id'], keep=False)]
print(len(dupes))
dupes.to_csv(DIR_FINAL / "04_station_overlaps.csv", index=False)
