import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import hashlib
import re
import warnings
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
    Simple function of parse_hh_member_column()
    
    """
    try:
        return int(str(x)[-1])-1
    except ValueError:
        return x


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
                                  avg_household_size=5.5, allocation='unequal', k_draws=27, inclusion_probability='proportion'):
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
        for province in settlements_eligible['province'].unique():
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

    # Calculate inclusion probabilities WITHIN each stratum
    for province in settlements_chosen['province'].unique():
        mask = settlements_chosen['province'] == province
        k_h = strata_counts[province]
        
        # Get proportions for this stratum from settlements_eligible
        stratum_data = settlements_eligible[settlements_eligible['province'] == province]
        stratum_prop_sum = stratum_data['proportion'].sum()
        
        # Create mapping of geom_id to re-normalized proportion
        prop_within = (stratum_data.set_index('geom_id')['proportion'] / stratum_prop_sum).to_dict()
        
        # Apply to settlements_chosen
        settlements_chosen.loc[mask, 'proportion_within_stratum'] = (
            settlements_chosen.loc[mask, 'geom_id'].map(prop_within)
        )
        
        # Calculate inclusion probability with k_h draws
        p_i = settlements_chosen.loc[mask, 'proportion_within_stratum']
        settlements_chosen.loc[mask, 'inclusion_prob_exact'] = 1 - (1 - p_i) ** k_h
        
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
    print(f"   Assuming average household size: {avg_household_size}")
    
    settlements_chosen_working = settlements_chosen.copy()
    settlements_chosen_working = settlements_chosen_working.set_index('sample_number')
    
    # Count unique households actually visited per cluster
    households_visited = data.groupby('location')['household_id'].nunique()
    settlements_chosen_working['households_visited'] = households_visited

    # Estimate total households in cluster
    settlements_chosen_working['total_households_est'] = (
        settlements_chosen_working['population'] / avg_household_size
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
    weights = settlements_chosen_working[[
            'sample_number', 'stratum_weights',
            'cluster_weight','stage1_weight', 'stage2_weight', 'province', 'geom_id'
        ]]

    # Drop residual weights from data
    data.drop(columns = [col for col in weights.columns if col in data.columns], errors='ignore', inplace=True)

    # Merge into individual data
    # Ensure keys are clean
    data['location'] = data['location'].astype(str).str.strip()
    weights['sample_number'] = weights['sample_number'].astype(str).str.strip()

    # Check for duplicates before merge
    dupes = weights[weights['sample_number'].duplicated()]
    if len(dupes) > 0:
        print(f"⚠️  Removing {len(dupes)} duplicate sample_numbers from weights")
        weights = weights.drop_duplicates(subset=['sample_number'], keep='first')

    # Perform merge with validation
    data_weighted = data.merge(
        weights, 
        left_on='location', 
        right_on='sample_number',
        how='left'
    )

    # Validate merge result
    if len(data_weighted) != len(data):
        print(f"⚠️  ERROR: Merge expanded data from {len(data)} to {len(data_weighted)} rows!")
        print(f"   This indicates many-to-many matches. Check for duplicates.")
        
        # Find problematic locations
        merged_counts = data_weighted.groupby('location').size()
        original_counts = data.groupby('location').size()
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
    
    # If random_draw_attempts = within-household contact attempts:
    if 'random_draw_attempts' in data_weighted.columns:
        # Household-level response rate. This adjusts Stage 3 weight for non-response.
        data_weighted['household_response_rate'] = data_weighted.apply(
            lambda x: x['random_draw_attempts'] /  x['eligible_adults_in_hh'], axis = 1
        )
        # Cap at 100%: a handful of rows show more draw attempts than eligible adults
        # recorded, which isn't possible -- the user attributed this to enumerator error
        # (2026-08-19) rather than a real >100% response rate, so it's capped rather than
        # left to inflate combined_weight_adjusted for those rows.
        data_weighted['household_response_rate'] = data_weighted['household_response_rate'].clip(upper=1.0)

        # Borgou day 1 (2026-07-16): enumerators weren't accurately recording the number
        # of eligible household members that day, so those rows' own response rate isn't
        # trustworthy and shouldn't inform the Borgou adjustment -- but they should still
        # receive the same adjustment as the rest of Borgou (confirmed by the user
        # 2026-08-19; day-1 shows a narrower eligible_adults_in_hh range (only 1-2, never
        # 3+) and 100% single-attempt draws vs. the rest of Borgou).
        if 'starttime' in data_weighted.columns and 'province' in data_weighted.columns:
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
        if 'province' in data_weighted.columns:
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
        print("\n⚠️  No random_draw_attempts column found - skipping non-response adjustment")
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
    print("\nSample vs Population proportions (checking stratum weights):")
    data_weighted.loc[data_weighted.province.isna()].to_csv("data/2_cleaned/1_weighted_na_province.csv")
    for province in data_weighted['province'].unique():
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
        weights = data_weighted[stage_col]
        n_eff_stage = (weights.sum()**2) / (weights**2).sum()
        deff_stage = n / n_eff_stage
        
        print(f"\n{stage_name}:")
        print(f"  DEFF: {deff_stage:.3f}")
        print(f"  Effective n: {n_eff_stage:.0f}")
        print(f"  Efficiency loss: {((n-n_eff_stage)/n)*100:.1f}%")
    
    print("\n" + "="*70)
    
    return data_weighted, weights


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