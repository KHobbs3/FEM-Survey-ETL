import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import hashlib
import re
import warnings
from .config import COUNTRY, CLUSTER_KEY
warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================================
# WEIGHT CALCULATION FOR THREE-STAGE SAMPLING WITH CONVENIENCE STAGE 2
# ============================================================================
# Stage 1: Stratified PPS cluster sampling (PROBABILITY)
# Stage 2: Convenience household selection (NON-PROBABILITY) 
# Stage 3: Random respondent selection within household (PROBABILITY)
# ============================================================================

def create_household_id(data):
    """
    Create household ID from deviceid + lat + lon + alt
    Handles cases where same household visited multiple times
    """
    print("Creating household IDs...")
    
    # Check if coordinates exist
    required_cols = ['deviceid', 'KEY', 'gps-Latitude', 'gps-Longitude', 'gps-Altitude']
    if not all(col in data.columns for col in required_cols):
        print("⚠️  WARNING: Missing required columns for household ID")
        print(f"   Required: {required_cols}")
        print(f"   Available: {[col for col in required_cols if col in data.columns]}")
        print("   Creating simple sequential household IDs")
        data['household_id'] = range(len(data))
        return data
    
    # Create household ID
    data = data.copy()
    data['household_id'] = data[required_cols].apply(
        lambda row: hashlib.md5(  # Use MD5 algorithm
            ",".join(row.astype(str)).encode()  # Concatenate values with "," separator
            ).hexdigest(),  # Convert hash to hex string
            axis=1  # Apply row-wise
        )

    n_households = data['household_id'].nunique()
    n_respondents = len(data)
    print(f"Check: {n_households == len(data)}")
    print(f" Created {n_households} unique households")
    print(f"  {n_respondents} respondents across {n_households} households")
    print(f"  Average: {n_respondents/n_households:.2f} respondents per household")
    
    return data


def parse_n(x):
    """
    Simple function of parse_hh_member_column(): the last value in the space-separated
    household-member index list, minus 1. Uses the last *token* rather than the last
    character so lists reaching 10+ ("... 9 10") and numeric cells (3.0) parse correctly.
    Returns NaN for blanks/unparseable values (filled with 1 by the caller).
    """
    tokens = str(x).split()
    if pd.isna(x) or not tokens:
        return np.nan
    try:
        return int(float(tokens[-1])) - 1
    except ValueError:
        return np.nan


def parse_hh_member_column(data, hh_member_col='hh_member_'):
    """
    Parse the household member column to extract number of eligible adults
    
    The column format appears to be: "2 3 4" where the last digit gives n
    where n = number of eligible household members
    
    So "2 3 4" means 4 eligible adults in household
    """
    print(f"\nParsing household member column: {hh_member_col}")
    
    data = data.copy()
    
    # Extract last digit and subtract 1
    data['eligible_adults_in_hh'] = (
        data[hh_member_col].astype(str).str[-1].astype(int)
    )
    

    # Fill blanks with 1
    data['eligible_adults_in_hh'] = data['eligible_adults_in_hh'].str.replace("", np.nan).fillna(1).astype(int)
    
    return data


# def calculate_exact_inclusion_probability(settlements_chosen, k_draws, inclusion_probability='proportion'):
#     """
#     Calculate exact inclusion probability for with-replacement PPS
#     inclusion_probability: column with probability of selection used in PPS.
#     For k draws with replacement:
#     π_i = 1 - (1 - p_i)^k
#     where p_i = N_i / N_total (selection probability per draw)
#     """
#     # Make a copy to avoid modifying original
#     settlements_chosen = settlements_chosen.copy()
    
#     # Dtype check - if not numeric, try to convert from percentage string
#     if not pd.api.types.is_numeric_dtype(settlements_chosen[inclusion_probability]):
#         print(f"⚠️  Column '{inclusion_probability}' is not numeric. Attempting to convert from percentage string...")
#         try:
#             # Remove '%' symbol, convert to float, divide by 100
#             settlements_chosen[inclusion_probability] = (
#                 settlements_chosen[inclusion_probability]
#                 .astype(str)
#                 .str.replace('%', '', regex=False)
#                 .astype(float) / 100
#             )
#             print(f"✓ Successfully converted '{inclusion_probability}' to numeric probabilities")
#         except Exception as e:
#             raise ValueError(f"Could not convert column '{inclusion_probability}' to numeric: {e}")
    
#     # Additional validation
#     assert (settlements_chosen[inclusion_probability] >= 0).all(), \
#         "Inclusion probabilities must be non-negative"
#     assert (settlements_chosen[inclusion_probability] <= 1).all(), \
#         f"Inclusion probabilities must be ≤ 1. Max value found: {settlements_chosen[inclusion_probability].max()}"
    
#     # Assuming 'inclusion_probability' is the per-draw probability
#     # (i.e., N_i / N_total)
#     per_draw_prob = settlements_chosen[inclusion_probability]
    
#     # Exact inclusion probability over k draws
#     settlements_chosen['cluster_inclusion_prob_exact'] = (
#         1 - (1 - per_draw_prob) ** k_draws
#     )
    
#     # Weight = 1 / inclusion probability
#     settlements_chosen['cluster_weight_exact'] = (
#         1.0 / settlements_chosen['cluster_inclusion_prob_exact']
#     )
    
#     # Compare to naive calculation
#     settlements_chosen['inclusion_prob_naive'] = np.minimum(
#         1.0, 
#         k_draws * per_draw_prob
#     )
    
#     print("\nComparison of inclusion probability calculations:")
#     print("\nPer-draw prob | Exact (k draws) | Naive (k × p)")
#     for idx in settlements_chosen.head().index:
#         p = per_draw_prob[idx]
#         exact = settlements_chosen.loc[idx, 'cluster_inclusion_prob_exact']
#         naive = settlements_chosen.loc[idx, 'inclusion_prob_naive']
#         print(f"{p:.4f}        | {exact:.4f}          | {naive:.4f}")
    
#     return settlements_chosen


def calculate_three_stage_weights(settlements_eligible, settlements_chosen, data,
                                  avg_household_size=5.5, allocation='unequal', k_draws=None, inclusion_probability='proportion',
                                  na_province_path=None, cluster_key='cluster_key', use_stratum_weight=True,
                                  certainty_units=False):
    # use_stratum_weight: multiply the cluster weight by (N_h/N)/(k_h/K). False for stratified PPS, where
    # 1/(1-(1-p)^k_h) already reflects each stratum's allocation (True double-corrects; kept for Benin).
    # certainty_units: treat clusters with k_h*p_i >= 1 as selected with certainty (weight 1) and compute
    # the others' probabilities from the rest of the stratum. Without it such a cluster gets weight ~1 but
    # still counts in the others' p_i, so the stratum total is overstated (False keeps Benin unchanged).
    # k_draws is unused (kept for old call sites): inclusion probabilities use the number of
    # clusters surveyed in each stratum. na_province_path: optional CSV for respondents left
    # without a stratum after the merge (row-level -- point it at the validation folder).
    # cluster_key: column present in both `data` and `settlements_chosen` that identifies the
    # cluster (load_sampling_frames() builds it from geom_id or the settlement name).
    # settlements_chosen must carry each cluster's frame 'proportion' (load_sampling_frames does).
    """
    Calculate weights for three-stage stratified cluster sampling
    
    DESIGN:
    -------
    Stage 1: Stratified PPS cluster sampling (9 per stratum)
    Stage 2: Convenience household selection (non-probability)
    Stage 3: Random selection of 1 eligible adult per household
    
    LIMITATIONS:
    ------------
    - Stage 2 is convenience-based, so household weights are approximations
    - We estimate total households from population / avg household size
    - Results should be interpreted with appropriate caution
    
    Parameters:
    -----------
    settlements_eligible : DataFrame with all eligible clusters
    settlements_chosen : DataFrame with sampled clusters  
    data : DataFrame with individual responses
    hh_member_col : str, column name containing household member count
    avg_household_size : float, for estimating total households in cluster
    
    Returns:
    --------
    data_weighted : DataFrame with all weight components
    weights_summary : DataFrame with weight breakdown by cluster
    """
    
    print("="*70)
    print("THREE-STAGE SAMPLING DESIGN WEIGHT CALCULATION")
    print("="*70)
    print("\nDESIGN SUMMARY:")
    print("  Stage 1: Stratified PPS cluster sampling (probability-based)")
    print("  Stage 2: Convenience household selection (NON-probability)")
    print("  Stage 3: Random adult selection within household (probability-based)")
    print("\n⚠️  IMPORTANT: Stage 2 convenience sampling limits inference validity")
    print("="*70)
    
    # ========================================================================
    # STAGE 1 WEIGHT: CLUSTER SELECTION
    # ========================================================================
    print("\n" + "="*70)
    print("STAGE 1: CLUSTER SELECTION WEIGHTS")
    print("="*70)

            
    # Population by stratum
    strata_counts = settlements_chosen.groupby('province').size()
    stratum_pops = settlements_eligible.groupby('province')['population'].sum()
    total_pop = stratum_pops.sum()
    total_clusters = len(settlements_chosen)
    stratum_proportions = stratum_pops / total_pop
    
    print("\nPopulation distribution:")
    for province, pop in stratum_pops.items():
        prop = stratum_proportions[province]
        print(f"  {province}: {pop:,} ({prop:.1%})")

    # Compute stratum weights depending on allocation performed
    if allocation == 'equal':
        k_per_stratum = strata_counts.iloc[0]
        n_strata = settlements_eligible['province'].nunique()
        
        print(f"\nClusters per stratum: {k_per_stratum}")
        print(f"Total clusters: {total_clusters}")
        print(f"Number of strata: {n_strata}")
        
        # Calculate stratum weights (for equal allocation)
        stratum_weights = stratum_proportions * n_strata

        # Add weights back to dataframe
        settlements_chosen['stratum_weights'] = pd.Series(stratum_weights)
        
        print("\nStratum weights (corrects for equal allocation):")
        for province, weight in stratum_weights.items():
            print(f"  {province}: {weight:.3f}")

    elif allocation == 'unequal':
        # For UNEQUAL allocation:
        stratum_weights = {}
        unsampled = set(settlements_eligible['province'].dropna()) - set(strata_counts.index)
        if unsampled:
            print(f"⚠️  Eligible strata with no surveyed cluster (not represented in weights): {sorted(unsampled)}")
        for province in strata_counts.index:
            k_h = strata_counts[province]  # Actual clusters selected in this stratum
            N_h = stratum_pops[province]   # Population in this stratum
            
            # Stratum weight = (Population proportion) / (Sample proportion)
            stratum_weights[province] = (N_h / total_pop) / (k_h / total_clusters)

        # Add weights back to dataframe
        settlements_chosen['stratum_weights'] = settlements_chosen.province.map(lambda x: stratum_weights[x])
        
        print("\nStratum weights (corrects for UNequal allocation):")
        print(settlements_chosen.groupby('province')['stratum_weights'].mean())
    
    else:
        print('Please designation allocation="unequal" or "equal".')

    if not use_stratum_weight:
        settlements_chosen['stratum_weights'] = 1.0
        print("\nStratum weights not applied (= 1): the cluster weight below already accounts for each "
              "stratum's number of clusters")

    # Calculate inclusion probabilities WITHIN each stratum
    for province in settlements_chosen['province'].unique():
        mask = settlements_chosen['province'] == province
        k_h = strata_counts[province]
        
        # Stratum total from settlements_eligible; each surveyed cluster's own proportion was
        # taken from its frame row by load_sampling_frames (names can repeat, so no lookup by key)
        stratum_data = settlements_eligible[settlements_eligible['province'] == province]
        stratum_prop_sum = stratum_data['proportion'].sum()

        # Re-normalized proportion within the stratum
        settlements_chosen.loc[mask, 'proportion_within_stratum'] = (
            settlements_chosen.loc[mask, 'proportion'] / stratum_prop_sum
        )
        
        # Certainty units: a cluster with k_h * p_i >= 1 is selected for sure (inclusion prob 1) and
        # represents only itself; the other clusters are k_h - n_cert draws from the rest of the stratum.
        # Repeated because removing one large cluster can push another over the threshold.
        certain = pd.Series(False, index=settlements_chosen.index[mask])
        k_rest, rest_sum = k_h, stratum_prop_sum
        while certainty_units:
            new = ~certain & (k_rest * settlements_chosen.loc[mask, 'proportion'] / rest_sum >= 1)
            if not new.any():
                break
            certain |= new
            k_rest = k_h - certain.sum()
            rest_sum = stratum_prop_sum - settlements_chosen.loc[certain[certain].index, 'proportion'].sum()
        if certain.any():
            print(f"  {province}: {certain.sum()} certainty cluster(s) (k*p >= 1) -> inclusion prob 1, holding "
                  f"{1 - rest_sum / stratum_prop_sum:.0%} of the stratum; other {k_rest} clusters drawn from the rest")
        settlements_chosen.loc[mask, 'certainty_unit'] = certain

        # Calculate inclusion probability with k_h draws (k_rest from the rest of the stratum if certainty units)
        p_i = settlements_chosen.loc[mask, 'proportion'] / rest_sum
        settlements_chosen.loc[mask, 'inclusion_prob_exact'] = (1 - (1 - p_i) ** k_rest).where(~certain, 1.0)
        
        # Cluster weight = 1 / inclusion probability
        settlements_chosen.loc[mask, 'cluster_weight'] = (
            1.0 / settlements_chosen.loc[mask, 'inclusion_prob_exact']
        )

    print("\nCluster weights by stratum:")
    print(settlements_chosen.groupby('province')['cluster_weight'].agg(['mean', 'min', 'max']))

    # Combined Stage 1 weight
    settlements_chosen['stage1_weight'] = (
        settlements_chosen['stratum_weights'] * 
        settlements_chosen['cluster_weight']
    )

    print(f"\nStage 1 weights:")
    print(f"  Mean: {settlements_chosen['stage1_weight'].mean():.2f}")
    print(f"  Range: [{settlements_chosen['stage1_weight'].min():.2f}, "
          f"{settlements_chosen['stage1_weight'].max():.2f}]")
    
    
    # ========================================================================
    # STAGE 2 WEIGHT: HOUSEHOLD SELECTION (Approximation)
    # ========================================================================
    print("\n" + "="*70)
    print("STAGE 2: HOUSEHOLD SELECTION WEIGHTS (APPROXIMATION)")
    print("="*70)
    print("\n⚠️  NOTE: Convenience sampling used - these are approximations")

    settlements_chosen_working = settlements_chosen.copy()
    settlements_chosen_working = settlements_chosen_working.set_index(cluster_key)

    # Count unique households actually visited per cluster
    households_visited = data.groupby(cluster_key)['household_id'].nunique()
    settlements_chosen_working['households_visited'] = households_visited

    # Estimate total households in cluster. avg_household_size: one number, or a Series by cluster
    # (e.g. urban vs rural household size); clusters missing from the Series get its mean.
    if np.isscalar(avg_household_size):
        print(f"   Assuming average household size: {avg_household_size}")
        settlements_chosen_working['household_size'] = avg_household_size
    else:
        sizes = settlements_chosen_working.index.map(avg_household_size)
        settlements_chosen_working['household_size'] = pd.Series(sizes, index=settlements_chosen_working.index).fillna(avg_household_size.mean())
        print("   Household size by cluster: "
              + str(settlements_chosen_working['household_size'].value_counts().sort_index().to_dict()))
    settlements_chosen_working['total_households_est'] = (
        settlements_chosen_working['population'] / settlements_chosen_working['household_size']
    )
    
    # Stage 2 weight (approximation)
    settlements_chosen_working['stage2_weight'] = (
        settlements_chosen_working['total_households_est'] / 
        settlements_chosen_working['households_visited']
    )
    
    settlements_chosen_working = settlements_chosen_working.reset_index()
    
    print(f"\nHouseholds visited per cluster:")
    print(f"  Mean: {settlements_chosen_working['households_visited'].mean():.1f}")
    print(f"  Range: [{settlements_chosen_working['households_visited'].min()}, "
          f"{settlements_chosen_working['households_visited'].max()}]")
    
    print(f"\nStage 2 weights:")
    print(f"  Mean: {settlements_chosen_working['stage2_weight'].mean():.2f}")
    print(f"  Range: [{settlements_chosen_working['stage2_weight'].min():.2f}, "
          f"{settlements_chosen_working['stage2_weight'].max():.2f}]")
    
    # ========================================================================
    # STAGE 3 WEIGHT: RESPONDENT SELECTION WITHIN HOUSEHOLD
    # ========================================================================
    print("\n" + "="*70)
    print("STAGE 3: RESPONDENT SELECTION WEIGHTS")
    print("="*70)
    
    # Stage 3 weight = number of eligible adults in household
    # (since we randomly selected 1 from n eligible adults)
    data['stage3_weight'] = data['eligible_adults_in_hh']
    
    print(f"\nStage 3 weights (# eligible adults per HH):")
    print(f"  Mean: {data['stage3_weight'].mean():.2f}")
    print(f"  Median: {data['stage3_weight'].median():.0f}")
    print(f"  Range: [{data['stage3_weight'].min()}, {data['stage3_weight'].max()}]")
    
    print("\nDistribution:")
    stage3_dist = data['stage3_weight'].value_counts().sort_index()
    for n_eligible, count in stage3_dist.items():
        pct = (count / len(data)) * 100
        print(f"  {int(n_eligible)} eligible: {count} households ({pct:.1f}%)")
    
    # ========================================================================
    # COMBINE ALL THREE STAGES
    # ========================================================================
    print("\n" + "="*70)
    print("COMBINING WEIGHTS")
    print("="*70)

    # Merge Stage 1 and Stage 2 weights
    weights = settlements_chosen_working[list(dict.fromkeys([
            'Name', 'stratum_weights',
            'cluster_weight','stage1_weight', 'stage2_weight', 'household_size', 'province', cluster_key
        ]))].copy()

    # Drop residual weights from data (but keep the join key)
    data.drop(columns = [col for col in weights.columns if col in data.columns and col != cluster_key],
              errors='ignore', inplace=True)

    # Merge into individual data
    # Ensure keys are clean
    data[cluster_key] = data[cluster_key].astype(str).str.strip()
    weights[cluster_key] = weights[cluster_key].astype(str).str.strip()

    # Check for duplicates before merge
    dupes = weights[weights[cluster_key].duplicated()]
    if len(dupes) > 0:
        print(f"⚠️  Removing {len(dupes)} duplicate {cluster_key}s from weights")
        weights = weights.drop_duplicates(subset=[cluster_key], keep='first')

    # Perform merge with validation
    data_weighted = data.merge(weights, on=cluster_key, how='left')

    # Validate merge result
    if len(data_weighted) != len(data):
        print(f"⚠️  ERROR: Merge expanded data from {len(data)} to {len(data_weighted)} rows!")
        print(f"   This indicates many-to-many matches. Check for duplicates.")
        
        # Find problematic locations
        merged_counts = data_weighted.groupby(cluster_key).size()
        original_counts = data.groupby(cluster_key).size()
        mismatches = merged_counts[merged_counts != original_counts]
        
        if len(mismatches) > 0:
            print(f"\n   Locations with mismatched row counts:")
            for loc in mismatches.index:
                print(f"     {loc}: {original_counts[loc]} → {merged_counts[loc]}")
    else:
        print(f"✓ Merge successful: {len(data_weighted)} rows")

    # Check for missing weights
    if data_weighted['stage1_weight'].isna().any():
        print("⚠️  WARNING: Some respondents missing Stage 1 weights")
    if data_weighted['stage2_weight'].isna().any():
        print("⚠️  WARNING: Some respondents missing Stage 2 weights")

    # Combined weight (all three stages)
    data_weighted['combined_weight'] = (
        data_weighted['stage1_weight'] * 
        data_weighted['stage2_weight'] * 
        data_weighted['stage3_weight']
    )

    print("\nWeight component means:")
    print(f"  Stage 1 (cluster):     {data_weighted['stage1_weight'].mean():.2f}")
    print(f"  Stage 2 (household):   {data_weighted['stage2_weight'].mean():.2f}")
    print(f"  Stage 3 (respondent):  {data_weighted['stage3_weight'].mean():.2f}")
    print(f"  Combined:              {data_weighted['combined_weight'].mean():.2f}")
    
    # ========================================================================
    # NON-RESPONSE ADJUSTMENT
    # ========================================================================
    print("\n" + "="*70)
    print("NON-RESPONSE ADJUSTMENT")
    print("="*70)
    
    # Draw-attempt adjustment: BENIN ONLY (kept so Benin's weights are unchanged).
    # Not applied elsewhere because stage 3 (= eligible adults) is already the full
    # within-household weight: members are drawn at random until an available one is
    # found, so the respondent is a random pick among available members however many
    # draws it took. Dividing by attempts / eligible adults would count household size
    # twice and upweight households whose first draw succeeded. Household non-response
    # is absorbed by stage 2 (cluster households / households interviewed).
    if COUNTRY == 'Benin' and 'random_draw_attempts' in data_weighted.columns:
        # Household-level response rate. This adjusts Stage 3 weight for non-response.
        data_weighted['household_response_rate'] = data_weighted.apply(
            lambda x: x['random_draw_attempts'] /  x['eligible_adults_in_hh'], axis = 1
        )
        # Cap at 100%: a handful of rows show more draw attempts than eligible adults
        # recorded, which isn't possible -- the user attributed this to enumerator error
        # (2026-08-19) rather than a real >100% response rate, so it's capped rather than
        # left to inflate combined_weight_adjusted for those rows.
        data_weighted['household_response_rate'] = data_weighted['household_response_rate'].clip(upper=1.0)

        # BENIN-ONLY adjustments below (Borgou day 1, Plateau language non-response) are
        # gated on COUNTRY so they never fire on another country's province names.
        # Borgou day 1 (2026-07-16): enumerators weren't accurately recording the number
        # of eligible household members that day, so those rows' own response rate isn't
        # trustworthy and shouldn't inform the Borgou adjustment -- but they should still
        # receive the same adjustment as the rest of Borgou (confirmed by the user
        # 2026-08-19; day-1 shows a narrower eligible_adults_in_hh range (only 1-2, never
        # 3+) and 100% single-attempt draws vs. the rest of Borgou).
        if COUNTRY == 'Benin' and 'starttime' in data_weighted.columns and 'province' in data_weighted.columns:
            start_date = pd.to_datetime(data_weighted['starttime'], format='mixed', dayfirst=True).dt.date
            borgou_day1 = (data_weighted['province'] == 'Borgou') & (start_date == pd.Timestamp('2026-07-16').date())
            if borgou_day1.any():
                borgou_rest = (data_weighted['province'] == 'Borgou') & (~borgou_day1)
                borgou_pooled_rate = data_weighted.loc[borgou_rest, 'household_response_rate'].mean()
                print(f"\nBorgou day 1 (2026-07-16): {borgou_day1.sum()} respondents excluded from "
                      f"computing the Borgou non-response adjustment; assigned the rest-of-Borgou "
                      f"pooled rate ({borgou_pooled_rate:.1%}) instead of their own.")
                data_weighted.loc[borgou_day1, 'household_response_rate'] = borgou_pooled_rate

        # Plateau non-response due to language barrier: 8 households in Plateau reached
        # eligibility screening and were excluded specifically because the drawn respondent
        # didn't speak the survey's administration language (Fon) -- confirmed as genuine
        # non-response, not ineligibility, per the user 2026-08-19 ("they were nonresponsive
        # because they could not speak the survey language"). These households were dropped
        # as >90%-missing in 00_clean_data.ipynb and never appear in `data` at all, so they
        # can't be individually weighted -- instead, every COMPLETED Plateau respondent's
        # weight is inflated to also stand in for the households that couldn't be surveyed,
        # so Plateau's weighted totals still reflect its true population.
        # Source: data/4_validation_checks/00_unfinished_submissions.csv, rows with
        # province==8 (Plateau) and language_1==0. Recompute this count if that file changes.
        PLATEAU_LANGUAGE_NONRESPONSE_N = 8
        if COUNTRY == 'Benin' and 'province' in data_weighted.columns:
            plateau_mask = data_weighted['province'] == 'Plateau'
            n_completed_plateau = plateau_mask.sum()
            if n_completed_plateau > 0:
                plateau_language_response_rate = n_completed_plateau / (n_completed_plateau + PLATEAU_LANGUAGE_NONRESPONSE_N)
                print(f"\nPlateau language non-response: {PLATEAU_LANGUAGE_NONRESPONSE_N} households excluded "
                      f"because the drawn respondent didn't speak the survey language, on top of "
                      f"{n_completed_plateau} completed interviews. Inflating completed Plateau respondents' "
                      f"weight by {1/plateau_language_response_rate:.3f}x ({plateau_language_response_rate:.1%} "
                      f"effective response rate) so they also represent the households that couldn't be surveyed.")
                data_weighted.loc[plateau_mask, 'household_response_rate'] *= plateau_language_response_rate

        # Calculate average response rate by stratum
        stratum_response_rates = data_weighted.groupby(['province'])['household_response_rate'].mean()

        print("\nAverage response rates by stratum:")
        for province, rate in stratum_response_rates.items():
            print(f"  {province}: {rate:.1%}")

        # If response rates vary significantly by stratum (>10% difference),
        # you may need stratum-level non-response adjustment
        max_diff = stratum_response_rates.max() - stratum_response_rates.min()
        if max_diff > 0.10:
            print(f"\n⚠️  Large variation in response rates across strata (diff={max_diff:.1%})")
            print("   This may explain remaining discrepancies in weighted proportions")


        data_weighted['combined_weight_adjusted'] = (
            data_weighted['combined_weight'] / data_weighted['household_response_rate']
        )
        
        print(f"\nOverall response rate: {data_weighted['household_response_rate'].mean():.1%}")
        print(f"Range: [{data_weighted['household_response_rate'].min():.1%}, {data_weighted['household_response_rate'].max():.1%}]")

    else:
        print("\nNo draw-attempt adjustment: stage 3 already covers within-household selection, "
              "and stage 2 covers household non-response")
        data_weighted['combined_weight_adjusted'] = data_weighted['combined_weight']

    
    
    # ========================================================================
    # NORMALIZE WEIGHTS
    # ========================================================================
    print("\n" + "="*70)
    print("NORMALIZING WEIGHTS")
    print("="*70)
    
    n = len(data_weighted)
    data_weighted['normalized_weight'] = (
        data_weighted['combined_weight_adjusted'] * n / 
        data_weighted['combined_weight_adjusted'].sum()
    )
    
    print(f"\nWeights normalized to sum to sample size: {n}")
    print(f"Sum of normalized weights: {data_weighted['normalized_weight'].sum():.2f}")
    
    # ========================================================================
    # DIAGNOSTICS
    # ========================================================================
    print("\n" + "="*70)
    print("WEIGHT DIAGNOSTICS")
    print("="*70)
    
    print("\nNormalized weight statistics:")
    print(f"  Mean:   {data_weighted['normalized_weight'].mean():.4f} (should be 1.0)")
    print(f"  Median: {data_weighted['normalized_weight'].median():.4f}")
    print(f"  Std:    {data_weighted['normalized_weight'].std():.4f}")
    print(f"  Min:    {data_weighted['normalized_weight'].min():.4f}")
    print(f"  Max:    {data_weighted['normalized_weight'].max():.4f}")
    
    # Effective sample size
    n_effective = (data_weighted['normalized_weight'].sum()**2 / 
                   (data_weighted['normalized_weight']**2).sum())
    deff = n / n_effective
    
    print(f"\nDesign Effect: {deff:.2f}")
    print(f"Effective sample size: {n_effective:.0f} (of {n} actual)")
    print(f"Efficiency: {(n_effective/n)*100:.1f}%")
    
    # Coefficient of variation
    cv = data_weighted['normalized_weight'].std() / data_weighted['normalized_weight'].mean()
    print(f"Coefficient of Variation: {cv:.3f}")
    
    if cv > 0.5:
        print("  ⚠️  High weight variation")
    if deff > 2.0:
        print("  ⚠️  High design effect")
    
    # By stratum
    print("\nWeight statistics by stratum:")
    stratum_stats = data_weighted.groupby('province').agg({
        'normalized_weight': ['count', 'mean', 'median', 'std', 'min', 'max']
    })['normalized_weight']
    print(stratum_stats.round(3))
    
    # Check stratum proportions
    print("\nSample vs Population proportions (design weights, before any post-stratification):")
    na_province = data_weighted.province.isna()
    if na_province.any():
        print(f"⚠️  {na_province.sum()} respondents have no stratum after the merge")
        if na_province_path is not None:
            data_weighted.loc[na_province].to_csv(na_province_path, index=False)
    for province in data_weighted['province'].dropna().unique():
        print(province)

        # Population proportion
        pop_prop = stratum_proportions[province]
        
        # Sample proportion (unweighted)
        sample_prop = (data_weighted['province'] == province).mean()
        
        # Weighted proportion (should match population)
        weighted_sum = data_weighted.loc[
            data_weighted['province'] == province, 'normalized_weight'
        ].sum()
        weighted_prop = weighted_sum / data_weighted['normalized_weight'].sum()
        
        print(f"\n  {province}:")
        print(f"    Population:  {pop_prop:.1%}")
        print(f"    Sample (unweighted): {sample_prop:.1%}")
        print(f"    Sample (weighted):   {weighted_prop:.1%}")
        
        diff = abs(weighted_prop - pop_prop)
        if diff < 0.02:
            print(f"    ✓ Good match (diff={diff:.2%})")
        else:
            print(f"    ⚠️  Large discrepancy (diff={diff:.2%})")
    
    # ========================================================================
    # STAGE CONTRIBUTION TO DESIGN EFFECT
    # ========================================================================
    print("\n" + "="*70)
    print("DESIGN EFFECT DECOMPOSITION BY STAGE")
    print("="*70)
    
    for stage_name, stage_col in [
        ('Stage 1 (Cluster)', 'stage1_weight'),
        ('Stage 2 (Household)', 'stage2_weight'),
        ('Stage 3 (Respondent)', 'stage3_weight')
    ]:
        stage_w = data_weighted[stage_col]  # not `weights` -- that's the cluster table returned below
        n_eff_stage = (stage_w.sum()**2) / (stage_w**2).sum()
        deff_stage = n / n_eff_stage
        
        print(f"\n{stage_name}:")
        print(f"  DEFF: {deff_stage:.3f}")
        print(f"  Effective n: {n_eff_stage:.0f}")
        print(f"  Efficiency loss: {((n-n_eff_stage)/n)*100:.1f}%")
    
    print("\n" + "="*70)
    
    return data_weighted, weights


def check_stage_weights(settlements_eligible, settlements_chosen, data_weighted, cluster_key='cluster_key',
                        target_clusters=None, out_path=None):
    """Checks that stage 1 x stage 2 behave as PPS requires. Prints aggregates by province only.

    Under PPS the cluster weight is ~ N_h / (k_h * MOS_i) and the household weight is
    HH_i / m_i = MOS_i / (hh_size * m_i), so their product should collapse to
    N_h / (k_h * hh_size * m_i): the frame population cancels. `ratio` below is the actual
    product over that expectation -- ~1 everywhere if the two stages use the same size measure
    and the cluster/household counts line up. It can only exceed 1 through the
    1/(1-(1-p)^k) form for large clusters (reported as `nonlinear`).

    target_clusters: {province: clusters allocated} (sampling/config.R, before the x2 for replacements).
    out_path: optional CSV of the per-cluster table (cluster names, populations, weights -- no respondents).
    Returns the per-cluster table.
    """
    print("=" * 70)
    print("STAGE 1 x STAGE 2 CHECKS")
    print("=" * 70)

    N_h = settlements_eligible.groupby('province')['population'].sum()
    P_h = settlements_eligible.groupby('province')['proportion'].sum()
    k_h = settlements_chosen.groupby('province').size()

    c = settlements_chosen[[cluster_key, 'province', 'population', 'proportion']].copy()
    c[cluster_key] = c[cluster_key].astype(str).str.strip()
    by_cluster = data_weighted.groupby(cluster_key)
    c = c.join(by_cluster[['stage1_weight', 'stage2_weight', 'household_size']].first(), on=cluster_key)
    c['m_households'] = c[cluster_key].map(by_cluster['household_id'].nunique())
    c['n_respondents'] = c[cluster_key].map(by_cluster.size())
    c['mean_stage3'] = c[cluster_key].map(by_cluster['stage3_weight'].mean())

    c['N_h'], c['k_h'] = c['province'].map(N_h), c['province'].map(k_h)
    p_pop = c['population'] / c['N_h']
    p_mos = c['proportion'] / c['province'].map(P_h)
    c['size_measure_ratio'] = p_mos / p_pop                # 1 = stage 1 MOS is the stage 2 population
    c['k_p'] = c['k_h'] * p_mos
    c['nonlinear'] = c['k_p'] / (1 - (1 - p_mos) ** c['k_h'])
    c['w12'] = c['stage1_weight'] * c['stage2_weight']
    # With certainty units (calculate_three_stage_weights(certainty_units=True)) a certainty cluster stands
    # for its own population and the rest share N_h minus it over k_h minus the certainty count
    cert = (settlements_chosen['certainty_unit'].eq(True).to_numpy()
            if 'certainty_unit' in settlements_chosen.columns else np.zeros(len(c), bool))
    cert_pop = c['population'].where(cert, 0).groupby(c['province']).transform('sum')
    cert_n = pd.Series(cert, index=c.index).groupby(c['province']).transform('sum')
    represents = np.where(cert, c['population'], (c['N_h'] - cert_pop) / (c['k_h'] - cert_n))
    c['w12_expected'] = represents / (c['household_size'] * c['m_households'])
    # Counted as the weights treated them; without certainty_units, clusters with k*p >= 1
    c['certainty_unit'] = cert if 'certainty_unit' in settlements_chosen.columns else (c['k_p'] >= 1).to_numpy()
    c['ratio'] = c['w12'] / c['w12_expected']

    # 1. Size measure
    dev = (c['size_measure_ratio'] - 1).abs().max()
    print(f"\n1. Stage 1 size measure vs stage 2 population: max relative difference {dev:.2e}"
          + ("  ✓ same quantity" if dev < 1e-6 else "  ⚠️  DIFFERENT quantities -- stages cannot cancel"))

    # 2. Clusters per stratum
    # The allocation is for reference only: k_h is what was surveyed, which is what the weights need
    print("\n2. Clusters used as k_h (distinct surveyed clusters) -- allocation shown for reference:")
    for prov in k_h.index:
        target = (target_clusters or {}).get(prov)
        flag = "" if target is None else f"  (allocated {target})"
        print(f"  {prov}: {k_h[prov]}{flag}")

    # 3. Product of stages 1 and 2
    def cv(s):
        return s.std() / s.mean() if len(s) > 1 else np.nan
    summary = c.groupby('province').agg(
        clusters=('ratio', 'size'),
        cv_w12=('w12', cv),
        cv_w12_x_take=('ratio', cv),
        ratio_min=('ratio', 'min'), ratio_median=('ratio', 'median'), ratio_max=('ratio', 'max'),
        certainty=('certainty_unit', 'sum'),
        stage2_below_1=('stage2_weight', lambda s: int((s < 1).sum())),
        households=('m_households', 'sum'), respondents=('n_respondents', 'sum'),
        mean_stage3=('mean_stage3', 'mean'),
    )
    print("\n3. Cluster-level product stage1 x stage2, by province")
    print("   cv_w12: CV of the product (also moves with households per cluster, 12-51)")
    print("   cv_w12_x_take / ratio_*: product / [N_h/(k_h*hh_size*m_i)] -- should be ~0 / ~1")
    print("   certainty: certainty clusters (weight 1); stage2_below_1: frame population < households visited")
    print(summary.round(3).to_string())

    # 4. Where the province totals come from
    ht = (c['stage1_weight'] * c['population']).groupby(c['province']).sum() / N_h
    w = data_weighted
    shares = pd.DataFrame({
        'population': N_h / N_h.sum(),
        'w1_x_w2': (w['stage1_weight'] * w['stage2_weight']).groupby(w['province']).sum(),
        'w12_expected_x_w3': (w[cluster_key].astype(str).str.strip().map(c.set_index(cluster_key)['w12_expected'])
                               * w['stage3_weight']).groupby(w['province']).sum(),
        'all_stages': w['combined_weight'].groupby(w['province']).sum(),
    })
    for col in shares.columns[1:]:
        shares[col] = shares[col] / shares[col].sum()
    print("\n4. Horvitz-Thompson population estimate / frame population (sum w1*pop_i / N_h; ~1 expected):")
    print(ht.round(3).to_string())
    print("\n   Province shares: population vs weighted with different stages")
    print(shares.round(3).to_string())

    if out_path is not None:
        c.to_csv(out_path, index=False)
        print(f"\nPer-cluster table saved to {out_path}")
    return c


def poststratify(data_weighted, controls, cell_col='province', weight_col='combined_weight_adjusted',
                 cluster_col='cluster_key', min_respondents=20, min_clusters=2):
    """Post-stratification adjustment w_pst = w * N_c / N^_c (Statistics Canada 7.1.4.1), where
    N^_c is the sum of `weight_col` over respondents in cell c. The formula is the same for PPS
    weights as for SRS; only N^_c changes. `controls` is {cell: population count} for the same
    population the sample frame covers -- only the ratios between cells matter, since weights
    are normalized afterwards.

    Returns a Series of post-stratified weights normalized to the sample size. Prints one line
    per cell (counts and factors only) and warns about cells too thin to post-stratify on.
    """
    controls = pd.Series(controls, dtype=float)
    cells = data_weighted[cell_col]
    missing = sorted(set(cells.dropna()) - set(controls.index))
    if missing or cells.isna().any():
        raise ValueError(f"No control total for cells {missing}"
                         + (f" and {cells.isna().sum()} respondents have no {cell_col}" if cells.isna().any() else ""))

    w = data_weighted[weight_col]
    N_hat = w.groupby(cells).sum()
    controls = controls[N_hat.index]
    # Put both on the same scale (shares), so counts of people and sums of weights compare
    factor = (controls / controls.sum()) / (N_hat / N_hat.sum())

    print("=" * 70)
    print(f"POST-STRATIFICATION ON {cell_col}")
    print("=" * 70)
    n = cells.value_counts()
    k = data_weighted.groupby(cell_col)[cluster_col].nunique() if cluster_col in data_weighted.columns else None
    for cell in N_hat.index:
        thin = n[cell] < min_respondents or (k is not None and k[cell] < min_clusters)
        print(f"  {cell}: n={n[cell]}" + (f", clusters={k[cell]}" if k is not None else "")
              + f" | weighted share {N_hat[cell] / N_hat.sum():.1%} -> control {controls[cell] / controls.sum():.1%}"
              + f" | factor {factor[cell]:.3f}" + ("  ⚠️  thin cell: collapse with a neighbour" if thin else ""))

    w_pst = w * cells.map(factor)
    w_pst = w_pst * len(w_pst) / w_pst.sum()
    deff = len(w_pst) * (w_pst ** 2).sum() / w_pst.sum() ** 2
    print(f"\n  Kish DEFF after post-stratification: {deff:.2f} (effective n {len(w_pst) / deff:.0f})")
    return w_pst


def _to_number(series):
    """'12,345' / ' 1.2% ' / 0.012 -> float. Percent signs are dropped without rescaling:
    proportions are only ever used relative to each other within a stratum."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    cleaned = series.astype(str).str.replace(r"[,%\s]", "", regex=True).replace({"": np.nan, "nan": np.nan})
    return pd.to_numeric(cleaned, errors="coerce")


_FRENCH_ACCENTS = set("éèêëàâäçôöîïùûüÉÈÊÀÂÇÔÎÙÛ")


def csv_encoding(path):
    """Encoding of a CSV: UTF-8 (with or without BOM), else whichever of cp1252 (Excel on
    Windows) or mac_roman (Excel on Mac) yields more French accented letters. Both 8-bit
    codecs decode any bytes without error, so guessing one silently garbles names
    (e.g. Mac 'é' = 0x8e reads as 'Ž' in cp1252)."""
    raw = open(path, "rb").read()
    try:
        raw.decode("utf-8")
        return "utf-8-sig"
    except UnicodeDecodeError:
        pass
    score = lambda enc: sum(ch in _FRENCH_ACCENTS for ch in raw.decode(enc, errors="replace"))
    return max(["cp1252", "mac_roman"], key=score)


def decode_csv(path):
    """Text of a CSV whose lines may mix encodings (sheets pasted together from different tools):
    each line is read as UTF-8 if it is valid UTF-8, else as whichever of cp1252 / mac_roman gives
    more French accented letters -- so 'é' comes out right on every line."""
    raw = open(path, "rb").read()
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
    lines = []
    for line in raw.split(b"\n"):
        try:
            lines.append(line.decode("utf-8"))
        except UnicodeDecodeError:
            lines.append(max((line.decode(enc, errors="replace") for enc in ("cp1252", "mac_roman")),
                             key=lambda s: sum(ch in _FRENCH_ACCENTS for ch in s)))
    return "\n".join(lines)


def read_table(path):
    """Read a sampling file as a plain DataFrame, whatever its format (CSV, Excel, or a
    GIS file such as .gpkg/.geojson). Geometry is dropped -- weighting doesn't need it."""
    from pathlib import Path
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        # Only blank cells are missing: a settlement literally named "NA" must survive
        import io
        df = pd.read_csv(io.StringIO(decode_csv(path)), keep_default_na=False, na_values=[""])
        df.columns = df.columns.str.strip()
        return df
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    import geopandas as gpd
    return pd.DataFrame(gpd.read_file(path)).drop(columns="geometry", errors="ignore")


def _surveyed_names(chosen, key):
    """Name the surveyed clusters: 'Name' (display) and a unique 'cluster_key'. Rows repeating a cluster
    (same name and region, e.g. several visit days) are kept once; a name used in two regions gets the
    region appended. Shared by load_sampling_frames (02b) and load_surveyed_clusters (02a)."""
    from .locations import normalize_place
    n_blank = chosen[key].isna().sum()
    if n_blank:
        print(f"Surveyed clusters: ignoring {n_blank} rows with no {key}")
    chosen = chosen.loc[chosen[key].notna()].copy()
    # A cluster literally named "NA" turns into a missing value whenever a CSV is re-read
    chosen[key] = chosen[key].astype(str).str.strip().replace({"NA": "Unnamed cluster (NA)"})
    if key != "Name" and "Name" in chosen.columns:
        chosen["Name"] = chosen["Name"].fillna(chosen[key]).astype(str).str.strip()
    else:
        chosen["Name"] = chosen[key]
    chosen["cluster_key"] = chosen[key].map(normalize_place)
    region = (chosen["province"].fillna("").astype(str).str.strip() if "province" in chosen.columns
              else pd.Series("", index=chosen.index))
    same = pd.DataFrame({"k": chosen["cluster_key"], "r": region.map(normalize_place)}).duplicated()
    if same.any():
        print(f"Surveyed clusters: {same.sum()} rows repeat a cluster already listed (e.g. several visit days) -- kept once")
        chosen, region = chosen.loc[~same], region.loc[~same]
    repeated = chosen["cluster_key"].duplicated(keep=False)
    if repeated.any():
        chosen.loc[repeated, "Name"] = chosen.loc[repeated, "Name"] + " (" + region[repeated] + ")"
        chosen.loc[repeated, "cluster_key"] = (chosen.loc[repeated, key] + " " + region[repeated]).map(normalize_place)
    return chosen


def load_surveyed_clusters(path, rename=None, key=None):
    """The surveyed clusters' polygons straight from a GIS file (e.g. DIR_SETTLEMENTS_SAMPLED), named exactly
    as load_sampling_frames names them -- so 02a can place respondents without loading the eligible frame."""
    import geopandas as gpd
    from pathlib import Path
    key = key or CLUSTER_KEY
    chosen = gpd.read_file(path).rename(columns=rename or {})
    if key not in chosen.columns:
        raise KeyError(f"{Path(path).name} has no '{key}' column; available columns: "
                       f"{[c for c in chosen.columns if c != 'geometry']}. Map it in SETTLEMENTS_CHOSEN_RENAME")
    return _surveyed_names(chosen, key)


def _parse_point(value):
    """'6.1234, 1.2345' (lat, lon) -> (lat, lon); None if there aren't two decimal numbers."""
    nums = re.findall(r"-?\d+\.\d+", str(value))
    return (float(nums[0]), float(nums[1])) if len(nums) >= 2 else None


def load_sampling_frames(eligible_path, chosen_path, eligible_rename=None, chosen_rename=None,
                         exclude_clusters=None, strata_map=None, min_population=None, key=None,
                         with_geometry=None, point_col=None, snap_m=1000, utm_crs=None):
    """Load the eligible sampling frame and the surveyed clusters in the shape
    calculate_three_stage_weights() expects, and check they are consistent.

    The frame is rebuilt as the sample draw saw it:
      exclude_clusters: clusters removed from the frame (e.g. out of bounds), given by `key`
      strata_map:       {admin area in the frame: stratum}; areas not listed are dropped
      min_population:   clusters below this population are dropped

    Each surveyed cluster is linked to ONE row of the rebuilt frame, which supplies its stratum
    ('province'), population, PPS proportion and polygon. In order:
      1. location -- the surveyed cluster's polygon (if the file is a GIS file) or its GPS point
                     (`point_col`, "lat, lon") lies inside a frame polygon, or within `snap_m` of one
      2. name     -- `key` matches the frame exactly (accents/case/punctuation ignored), or names one of
                     the villages in a grouped frame name ("Assahoun" -> "Kévé; Assahoun"), or a whole
                     word of one ("Agou" -> "Agou Gadzépé; Agou Tavie")
    Where several frame rows fit, the surveyed file's own region label picks one; anything still
    ambiguous or unmatched stops the loader with a list. The linked frame row's name becomes the
    cluster's 'Name' and 'cluster_key' (the surveyed file's own name is kept as 'surveyed_name').

    with_geometry: keep the frame's polygons, so `chosen` is a GeoDataFrame of the surveyed clusters'
    own polygons (02a). Default: whenever the frame is a GIS file.
    Defaults for key / exclude_clusters / point_col / utm_crs come from src/config.py.

    Returns (eligible, chosen). Prints cluster names / counts only -- no respondent data.
    """
    from pathlib import Path
    from . import config
    from .locations import normalize_place

    key = key or CLUSTER_KEY
    exclude_clusters = getattr(config, "EXCLUDED_CLUSTERS", []) if exclude_clusters is None else exclude_clusters
    point_col = point_col or getattr(config, "SETTLEMENTS_CHOSEN_POINT_COL", None)
    utm_crs = utm_crs or getattr(config, "UTM_CRS", "epsg:32631")
    is_gis = lambda path: Path(path).suffix.lower() not in (".csv", ".xlsx", ".xls")
    if with_geometry is None:
        with_geometry = is_gis(eligible_path)
    if with_geometry or is_gis(chosen_path):
        import geopandas as gpd
    eligible = (gpd.read_file(eligible_path) if with_geometry else read_table(eligible_path)).rename(columns=eligible_rename or {})
    chosen = (gpd.read_file(chosen_path) if is_gis(chosen_path) else read_table(chosen_path)).rename(columns=chosen_rename or {})

    for label, df, required, setting in [("eligible frame", eligible, [key, "province", "population"], "SETTLEMENTS_ELIGIBLE_RENAME"),
                                         ("surveyed clusters", chosen, [key], "SETTLEMENTS_CHOSEN_RENAME")]:
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise KeyError(f"{label} is missing {missing}; available columns: {list(df.columns)}. "
                           f"Map them in {setting} in src/config.py (clusters link on '{key}', see CLUSTER_KEY)")

    # A frame straight out of the preselection step has population but no PPS 'proportion'.
    # Proportions are renormalised within each stratum, so population itself is equivalent --
    # valid only if the sample draw used population as the size measure.
    if "proportion" not in eligible.columns:
        print("Eligible frame has no 'proportion' column -- using population as the PPS size measure")
        eligible["proportion"] = eligible["population"]

    chosen = _surveyed_names(chosen, key)
    # A cluster literally named "NA" turns into a missing value whenever a CSV is re-read
    eligible[key] = eligible[key].astype(str).str.strip().replace({"NA": "Unnamed cluster (NA)"})
    eligible["cluster_key"] = eligible[key].map(normalize_place)
    eligible["province"] = eligible["province"].astype(str).str.strip()
    eligible["population"] = _to_number(eligible["population"])
    eligible["proportion"] = _to_number(eligible["proportion"])

    # Only fully identical rows are duplicates: distinct settlements can share a name
    dup = eligible.drop(columns="geometry", errors="ignore").duplicated()
    if dup.any():
        print(f"Eligible frame: dropping {dup.sum()} exact duplicate rows")
        eligible = eligible.loc[~dup]

    # Every village named in the frame, before the rebuild (to explain unmatched clusters)
    villages = lambda name: [v for v in (normalize_place(p) for p in str(name).split(";")) if isinstance(v, str)]
    frame_names_before = set(eligible["cluster_key"]) | {v for n in eligible[key] for v in villages(n)}

    # Rebuild the frame as the draw saw it. Order doesn't matter: each step only removes rows.
    excluded = {normalize_place(str(c).strip()) for c in (exclude_clusters or [])}
    if excluded:
        drop = eligible["cluster_key"].isin(excluded)
        not_found = sorted(excluded - set(eligible["cluster_key"]))
        print(f"Eligible frame: excluding {drop.sum()} rows for {len(excluded)} listed clusters"
              + (f" (not in the frame: {not_found})" if not_found else ""))
        eligible = eligible.loc[~drop]
    if strata_map is not None:
        strata = eligible["province"].map(strata_map)
        dropped = eligible.loc[strata.isna(), "province"].value_counts()
        if len(dropped):
            print(f"Eligible frame: dropping clusters in areas outside the strata: {dropped.to_dict()}")
        eligible = eligible.assign(admin_area=eligible["province"], province=strata).loc[strata.notna()]
    if min_population is not None:
        small = eligible["population"] < min_population
        print(f"Eligible frame: dropping {small.sum()} clusters below the minimum population ({min_population:.1f})")
        eligible = eligible.loc[~small]
    eligible = eligible.copy()

    # ---- Link each surveyed cluster to one frame row ----
    for col in ("province", "population", "proportion"):
        if col in chosen.columns:
            chosen = chosen.rename(columns={col: f"{col}_surveyed_file"})
    chosen = chosen.reset_index(drop=True)
    frame = eligible.reset_index(drop=True)
    labels = (chosen["province_surveyed_file"].astype(str).str.strip().map(lambda v: (strata_map or {}).get(v, v))
              .map(normalize_place) if "province_surveyed_file" in chosen.columns else pd.Series(pd.NA, index=chosen.index))
    frame_region = frame["province"].map(normalize_place)

    def pick(rows, i):
        """One frame row from candidates: the only one, else the only one in the surveyed region."""
        rows = list(dict.fromkeys(rows))
        if len(rows) <= 1:
            return (rows[0] if rows else None), rows
        same = [r for r in rows if isinstance(labels[i], str) and frame_region[r] == labels[i]]
        return (same[0] if len(same) == 1 else None), rows

    link, how, candidates = {}, {}, {}

    # 1. Location: the surveyed polygon / GPS point inside (or near) a frame polygon
    if with_geometry:
        pts = None
        if "geometry" in chosen.columns and chosen.geometry.notna().any():
            pts = gpd.GeoSeries(chosen.geometry, crs=chosen.crs).representative_point()
        elif point_col and point_col in chosen.columns:
            parsed = chosen[point_col].map(_parse_point)
            ok = parsed.notna()
            pts = gpd.GeoSeries(gpd.points_from_xy([p[1] if p else None for p in parsed],
                                                   [p[0] if p else None for p in parsed]), crs="epsg:4326")
            pts[~ok] = None
            print(f"Surveyed clusters: {ok.sum()} of {len(chosen)} rows have a usable '{point_col}'")
        if pts is not None:
            pts = gpd.GeoDataFrame(geometry=pts).dropna().to_crs(frame.crs)
            hits = pts.sjoin(frame[["geometry"]], predicate="intersects")
            for i, grp in hits.groupby(level=0):
                row, rows = pick(grp["index_right"], i)
                candidates[i] = rows
                if row is not None:
                    link[i], how[i] = row, "location"
            near = pts.loc[[i for i in pts.index if i not in link and i not in candidates]]
            if len(near):
                nn = gpd.sjoin_nearest(near.to_crs(utm_crs), frame[["geometry"]].to_crs(utm_crs),
                                       distance_col="dist_m", max_distance=snap_m)
                for i, grp in nn.groupby(level=0):
                    link[i], how[i] = grp["index_right"].iloc[0], "location (within %d m)" % snap_m

    # 2. Name: exact, one of a grouped name's villages, or a whole word of one
    by_key, by_village = {}, {}
    for r, (k, n) in enumerate(zip(frame["cluster_key"], frame[key])):
        by_key.setdefault(k, []).append(r)
        for v in villages(n):
            by_village.setdefault(v, []).append(r)
    for i, k in chosen["cluster_key"].items():
        if i in link or not isinstance(k, str):
            continue
        rows = by_key.get(k) or by_village.get(k) or [
            r for v, rs in by_village.items() if f" {k} " in f" {v} " for r in rs]
        row, rows = pick(rows, i)
        if row is not None:
            link[i], how[i] = row, "name"
        elif rows:
            candidates[i] = rows

    # ---- Report and assemble ----
    problems = []
    ambiguous = [i for i in chosen.index if i not in link and len(candidates.get(i, [])) > 1]
    if ambiguous:
        options = lambda i: [f"{frame.at[r, key]} ({frame.at[r, 'province']})" for r in candidates[i]][:5]
        problems.append("surveyed clusters that match more than one frame settlement (add the region to the "
                        "surveyed-clusters file, or exclude the wrong one):\n      " + "\n      ".join(
                            f"{chosen.at[i, 'Name']}: {options(i)}" for i in ambiguous))
    unmatched = [i for i in chosen.index if i not in link and i not in ambiguous]
    if unmatched:
        why = []
        for i in unmatched:
            k = chosen.at[i, "cluster_key"]
            reason = ("on the exclusion list" if k in excluded else
                      "not in the frame file at all" if k not in frame_names_before else
                      "outside the strata or below the minimum population")
            why.append(f"{chosen.at[i, 'Name']} ({str(k)[:10]}...): {reason}")
        problems.append("surveyed clusters missing from the rebuilt frame (no cluster weight):\n      "
                        + "\n      ".join(why))
    if frame["proportion"].isna().any():
        problems.append(f"{frame['proportion'].isna().sum()} eligible rows with missing/non-numeric proportion")
    if problems:
        raise ValueError("Sampling files are inconsistent:\n  - " + "\n  - ".join(problems))

    print("Surveyed clusters linked to the frame by: " + str(pd.Series(how).value_counts().to_dict()))
    # The surveyed file keeps its own names (so 02a can name clusters from it alone); the frame row
    # supplies stratum, population, proportion and polygon. Its name is kept as 'frame_name'.
    chosen = chosen.drop(columns=["geometry"], errors="ignore")
    chosen["_frame_row"] = chosen.index.map(link)
    frame_cols = ["province", "population", "proportion"] + (["geometry"] if with_geometry else [])
    chosen = chosen.join(frame[frame_cols + [key]].rename(columns={key: "frame_name"}), on="_frame_row")
    # Several surveyed clusters pointing at one frame settlement (e.g. "Kévé; Assahoun" and "Assahoun"):
    # kept once; key_aliases maps the merged names' keys to the kept one, so respondents can follow
    dup = chosen["_frame_row"].duplicated()
    aliases = {}
    if dup.any():
        kept = chosen.loc[~dup].set_index("_frame_row")
        aliases = dict(zip(chosen.loc[dup, "cluster_key"], chosen.loc[dup, "_frame_row"].map(kept["cluster_key"])))
        print("Surveyed clusters: merged names that are the same frame settlement: "
              + str({n: kept.at[r, "Name"] for n, r in zip(chosen.loc[dup, "Name"], chosen.loc[dup, "_frame_row"])}))
        chosen = chosen.loc[~dup]
    chosen = chosen.drop(columns=["_frame_row"]).reset_index(drop=True)

    # Labels in the surveyed-clusters file are informational only -- report disagreements
    if "province_surveyed_file" in chosen.columns:
        label = chosen["province_surveyed_file"].astype(str).str.strip().map(lambda v: (strata_map or {}).get(v, v))
        diff = chosen.loc[label != chosen["province"], "Name"]
        if len(diff):
            print(f"Note: region label in the surveyed-clusters file differs from the frame for {diff.tolist()} "
                  "(frame used)")

    if with_geometry:
        chosen = gpd.GeoDataFrame(chosen, geometry="geometry", crs=eligible.crs)
    chosen.attrs["key_aliases"] = aliases
    print(f"Eligible frame: {len(eligible)} clusters in {eligible['province'].nunique()} strata | "
          f"Surveyed: {len(chosen)} clusters in {chosen['province'].nunique()} strata")
    return eligible, chosen

# ============================================================================
# USAGE EXAMPLE
# ============================================================================

"""
# Load data
settlements_eligible = pd.read_csv("niger_preselected_minpop_733.csv", index_col=0)
settlements_chosen = pd.read_csv("settlements_chosen.csv")
data = pd.read_csv("your_cleaned_data.csv")

# Clean population column
settlements_chosen["population"] = (
    settlements_chosen["population"].replace({",":""}, regex=True).astype(int)
)

# Calculate three-stage weights
# IMPORTANT: Specify the correct column name for household members
data_weighted, weights_summary = calculate_three_stage_weights(
    settlements_eligible,
    settlements_chosen,
    data,
    hh_member_col='hh_member_',  # Replace with your actual column name
    avg_household_size=5.5  # Adjust based on Niger demographics
)

# Save results
data_weighted.to_csv("data_with_three_stage_weights.csv", index=False)
weights_summary.to_csv("weight_summary_by_cluster.csv", index=False)

# Check weight distribution
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.hist(data_weighted['normalized_weight'], bins=50, edgecolor='black')
plt.axvline(1.0, color='red', linestyle='--', label='Mean (should be 1.0)')
plt.xlabel('Normalized Weight')
plt.ylabel('Frequency')
plt.title('Distribution of Survey Weights')
plt.legend()
plt.savefig('weight_distribution.png', dpi=300, bbox_inches='tight')
"""