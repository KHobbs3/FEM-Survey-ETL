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


def pps_inclusion_probabilities(sizes, k, certainty_units=False):
    """
    Inclusion probability of every cluster in ONE stratum's frame after k PPS draws.

    pi_i = 1 - (1 - p_i)^k, with p_i = size_i / stratum total.

    With certainty_units=True, a cluster with k * p_i >= 1 is selected with certainty
    (pi_i = 1). The other k - n_cert draws are PPS draws from the rest of the stratum,
    so p_i is recomputed over the non-certainty clusters; this repeats until no new
    cluster crosses the threshold.

    Returns (pi, certain): two Series on the index of `sizes`.
    """
    sizes = pd.Series(sizes, dtype=float)
    certain = pd.Series(False, index=sizes.index)
    if certainty_units:
        while True:
            n_rest = k - certain.sum()
            rest = sizes[~certain]
            if n_rest <= 0 or rest.empty:
                break
            new = rest.index[n_rest * rest / rest.sum() >= 1]
            if len(new) == 0:
                break
            certain[new] = True

    n_rest = k - certain.sum()
    pi = pd.Series(1.0, index=sizes.index)
    if (~certain).any():
        p = sizes[~certain] / sizes[~certain].sum()
        pi[~certain] = 1 - (1 - p) ** n_rest
    return pi, certain


def household_response_rate(data):
    """
    Within-household response rate per respondent: random_draw_attempts / eligible_adults_in_hh,
    capped at 1, with the Borgou day-1 and Plateau language rules below. Used by the
    non-response adjustment (NONRESPONSE_ADJUSTMENT) and 05_province_response_rate.py.
    Returns a Series aligned with data.
    """
    rate = data['random_draw_attempts'] / data['eligible_adults_in_hh']
    # Cap at 100%: a handful of rows show more draw attempts than eligible adults
    # recorded, which isn't possible -- the user attributed this to enumerator error
    # (2026-08-19) rather than a real >100% response rate, so it's capped rather than
    # left to inflate combined_weight_adjusted for those rows.
    rate = rate.clip(upper=1.0)

    # Borgou day 1 (2026-07-16): enumerators weren't accurately recording the number
    # of eligible household members that day, so those rows' own response rate isn't
    # trustworthy and shouldn't inform the Borgou adjustment -- but they should still
    # receive the same adjustment as the rest of Borgou (confirmed by the user
    # 2026-08-19; day-1 shows a narrower eligible_adults_in_hh range (only 1-2, never
    # 3+) and 100% single-attempt draws vs. the rest of Borgou).
    if 'starttime' in data.columns and 'province' in data.columns:
        start_date = pd.to_datetime(data['starttime'], format='mixed', dayfirst=True).dt.date
        borgou_day1 = (data['province'] == 'Borgou') & (start_date == pd.Timestamp('2026-07-16').date())
        if borgou_day1.any():
            borgou_rest = (data['province'] == 'Borgou') & (~borgou_day1)
            borgou_pooled_rate = rate.loc[borgou_rest].mean()
            print(f"\nBorgou day 1 (2026-07-16): {borgou_day1.sum()} respondents excluded from "
                  f"computing the Borgou non-response adjustment; assigned the rest-of-Borgou "
                  f"pooled rate ({borgou_pooled_rate:.1%}) instead of their own.")
            rate.loc[borgou_day1] = borgou_pooled_rate

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
    if 'province' in data.columns:
        plateau_mask = data['province'] == 'Plateau'
        n_completed_plateau = plateau_mask.sum()
        if n_completed_plateau > 0:
            plateau_language_response_rate = n_completed_plateau / (n_completed_plateau + PLATEAU_LANGUAGE_NONRESPONSE_N)
            print(f"\nPlateau language non-response: {PLATEAU_LANGUAGE_NONRESPONSE_N} households excluded "
                  f"because the drawn respondent didn't speak the survey language, on top of "
                  f"{n_completed_plateau} completed interviews. Inflating completed Plateau respondents' "
                  f"weight by {1/plateau_language_response_rate:.3f}x ({plateau_language_response_rate:.1%} "
                  f"effective response rate) so they also represent the households that couldn't be surveyed.")
            rate.loc[plateau_mask] *= plateau_language_response_rate
    return rate


def calculate_three_stage_weights(settlements_eligible, settlements_chosen, data,
                                  avg_household_size=5.5, allocation='unequal', k_draws=27, inclusion_probability='proportion',
                                  use_stratum_weight=True, certainty_units=False, nonresponse_adjustment=False):
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
    avg_household_size : float, for estimating total households in cluster
    inclusion_probability : str, settlements_eligible column used as the PPS size measure
    use_stratum_weight : bool (config.USE_STRATUM_WEIGHT). False = stage 1 is the PPS
        cluster weight alone. The cluster weight already uses k_h, so each stratum's
        weighted total is proportional to its population; multiplying by
        (N_h/N)/(k_h/K) corrects for the allocation a second time.
    certainty_units : bool (config.PPS_CERTAINTY_UNITS). True = clusters with
        k_h * p_i >= 1 get inclusion probability 1 (see pps_inclusion_probabilities).
    nonresponse_adjustment : bool (config.NONRESPONSE_ADJUSTMENT). True = divide by
        random_draw_attempts / eligible_adults_in_hh (with the Borgou day-1 and Plateau
        language rules). False = combined_weight_adjusted is the three-stage weight:
        stage 3 already lets the respondent stand for every eligible adult listed.

    Returns:
    --------
    data_weighted : DataFrame with all weight components
    cluster_weights : DataFrame with weight breakdown by cluster (input to check_stage_weights)
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
    stratum_pops = settlements_eligible.groupby('province')['population'].sum()
    total_pop = stratum_pops.sum()
    stratum_proportions = stratum_pops / total_pop

    # k_h = distinct clusters actually surveyed in the stratum (rows of settlements_chosen
    # with respondents), not the allocation or every row drawn (which may include unused
    # replacements). config.CLUSTERS_ALLOCATED holds the allocation, for reference only.
    surveyed = (settlements_chosen['sample_number'].astype(str).str.strip()
                .isin(data['location'].astype(str).str.strip()))
    strata_counts = settlements_chosen[surveyed].groupby('province')['geom_id'].nunique()
    total_clusters = strata_counts.sum()
    if (~surveyed).any():
        print(f"\n⚠️  {(~surveyed).sum()} settlements_chosen rows have no respondents and are not "
              f"counted in k_h (rows per stratum: {settlements_chosen.groupby('province').size().to_dict()})")
    print(f"\nk_h (distinct surveyed clusters): {strata_counts.to_dict()}")

    print("\nPopulation distribution:")
    for province, pop in stratum_pops.items():
        prop = stratum_proportions[province]
        print(f"  {province}: {pop:,} ({prop:.1%})")

    # Compute stratum weights depending on allocation performed
    if not use_stratum_weight:
        settlements_chosen['stratum_weights'] = 1.0
        print("\nStratum weight NOT applied (USE_STRATUM_WEIGHT=False): stage 1 = PPS cluster weight only.")
        print("  The cluster weight already uses k_h, so each stratum's weighted total is")
        print("  proportional to its population without an allocation correction.")

    elif allocation == 'equal':
        k_per_stratum = strata_counts.iloc[0]
        n_strata = settlements_eligible['province'].nunique()

        print(f"\nClusters per stratum: {k_per_stratum}")
        print(f"Total clusters: {total_clusters}")
        print(f"Number of strata: {n_strata}")

        # Calculate stratum weights (for equal allocation)
        stratum_weights = stratum_proportions * n_strata

        # Add weights back to dataframe
        settlements_chosen['stratum_weights'] = settlements_chosen['province'].map(stratum_weights)

        print("\nStratum weights (corrects for equal allocation):")
        for province, weight in stratum_weights.items():
            print(f"  {province}: {weight:.3f}")

    elif allocation == 'unequal':
        # For UNEQUAL allocation:
        stratum_weights = {}
        for province, k_h in strata_counts.items():  # Actual clusters surveyed in this stratum
            N_h = stratum_pops[province]   # Population in this stratum

            # Stratum weight = (Population proportion) / (Sample proportion)
            stratum_weights[province] = (N_h / total_pop) / (k_h / total_clusters)

        # Add weights back to dataframe
        settlements_chosen['stratum_weights'] = settlements_chosen.province.map(stratum_weights)

        print("\nStratum weights (corrects for UNequal allocation):")
        print(settlements_chosen.groupby('province')['stratum_weights'].mean())

    else:
        print('Please designation allocation="unequal" or "equal".')

    # Calculate inclusion probabilities WITHIN each stratum
    settlements_chosen['certainty_unit'] = False
    for province in settlements_chosen['province'].unique():
        mask = settlements_chosen['province'] == province
        k_h = strata_counts.get(province, 0)
        settlements_chosen.loc[mask, 'k_h'] = k_h
        if k_h == 0:
            continue

        # Size measure for this stratum from settlements_eligible, re-normalized within stratum
        stratum_data = settlements_eligible[settlements_eligible['province'] == province].set_index('geom_id')
        sizes = stratum_data[inclusion_probability]
        prop_within = sizes / sizes.sum()
        pi, certain = pps_inclusion_probabilities(sizes, k_h, certainty_units)

        # Apply to settlements_chosen
        geoms = settlements_chosen.loc[mask, 'geom_id']
        settlements_chosen.loc[mask, 'proportion_within_stratum'] = geoms.map(prop_within)
        settlements_chosen.loc[mask, 'inclusion_prob_exact'] = geoms.map(pi)
        settlements_chosen.loc[mask, 'certainty_unit'] = geoms.map(certain).fillna(False).astype(bool)
        if certain.any():
            print(f"\n{province}: {certain.sum()} certainty cluster(s) (k_h*p_i >= 1) with "
                  f"{prop_within[certain].sum():.0%} of the stratum; remaining {k_h - certain.sum()} "
                  f"draws spread over the rest.")

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

    cluster_weights = settlements_chosen_working.copy()

    # Merge Stage 1 and Stage 2 weights
    weights = settlements_chosen_working[[
            'sample_number', 'stratum_weights',
            'cluster_weight','stage1_weight', 'stage2_weight', 'province', 'geom_id'
        ]].copy()

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
    if not nonresponse_adjustment:
        print("\nSkipped (nonresponse_adjustment=False): stage 3 already lets the respondent "
              "represent every eligible adult listed, and stage 2 covers households that "
              "didn't take part.")
        data_weighted['combined_weight_adjusted'] = data_weighted['combined_weight']
    elif 'random_draw_attempts' in data_weighted.columns:
        # Household-level response rate. This adjusts Stage 3 weight for non-response.
        data_weighted['household_response_rate'] = household_response_rate(data_weighted)

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
    if data_weighted.province.isna().any():
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
        w = data_weighted[stage_col]
        n_eff_stage = (w.sum()**2) / (w**2).sum()
        deff_stage = n / n_eff_stage
        
        print(f"\n{stage_name}:")
        print(f"  DEFF: {deff_stage:.3f}")
        print(f"  Effective n: {n_eff_stage:.0f}")
        print(f"  Efficiency loss: {((n-n_eff_stage)/n)*100:.1f}%")
    
    print("\n" + "="*70)
    
    return data_weighted, cluster_weights


def kish_deff(w):
    """Kish design effect from unequal weights: n * sum(w^2) / sum(w)^2."""
    w = pd.Series(w).dropna()
    return len(w) * (w ** 2).sum() / w.sum() ** 2


def _shares(weights, groups):
    return weights.groupby(groups).sum() / weights.sum()


def compare_stratum_weight(settlements_eligible, settlements_chosen, data, **kwargs):
    """
    By stratum: population share, unweighted share, and weighted share with and without
    the stratum weight (USE_STRATUM_WEIGHT on/off). Everything else (non-response, Borgou
    day 1, Plateau language) is the same in both columns. Prints aggregates only.
    """
    import contextlib
    import io

    kwargs.pop('use_stratum_weight', None)
    out = {}
    for label, flag in [('weighted, with stratum weight', True), ('weighted, cluster weight only', False)]:
        with contextlib.redirect_stdout(io.StringIO()):
            dw, _ = calculate_three_stage_weights(settlements_eligible.copy(), settlements_chosen.copy(),
                                                  data.copy(), use_stratum_weight=flag, **kwargs)
        out[label] = _shares(dw['normalized_weight'], dw['province'])
    pop = settlements_eligible.groupby('province')['population'].sum()
    table = pd.DataFrame({
        'population': pop / pop.sum(),
        'unweighted': dw['province'].value_counts(normalize=True),
        **out,
    }).fillna(0)
    print("Share of the total by stratum (%):")
    print((table * 100).round(1).to_string())
    for label in out:
        print(f"  max |{label} - population|: {(table[label] - table['population']).abs().max() * 100:.1f} pts")
    return table


# ============================================================================
# STAGE CHECKS
# ============================================================================

def check_stage_weights(cluster_weights, data_weighted, settlements_eligible, avg_household_size=5.5,
                        clusters_allocated=None, size_col='proportion', out_dir=None, tol=0.05):
    """
    Diagnostic for the three weight stages, printed by stratum (no respondent rows).

    Under PPS, stage 1 x stage 2 = (1/pi_i) * (M_i/hh_size)/m_i ~= N_h / (k_h * hh_size * m_i):
    the cluster's population M_i cancels, but only if stage 1's size measure is the same
    quantity as stage 2's population. Prints per stratum:
      - whether the stage 1 size measure (p_i * N_h) matches the stage 2 population
      - k_h (distinct surveyed clusters) next to the allocation (reference only)
      - ratio of w1*w2 to N_h/(k_h*hh_size*m_i) (~1; certainty clusters give k_h*p_i > 1)
      - certainty clusters (k_h*p_i >= 1) and clusters with stage 2 weight < 1
      - mean stage 3
      - Horvitz-Thompson population estimate (sum of M_i/pi_i) / frame population
      - province shares: population, w1*w2, w12_expected*w3, all stages
    Writes the per-cluster table to out_dir/02b_stage_weight_check_clusters.csv.

    cluster_weights: second return value of calculate_three_stage_weights.
    Returns (by_stratum, by_cluster).
    """
    cw = cluster_weights.copy()
    cw['sample_number'] = cw['sample_number'].astype(str).str.strip()
    stratum_pops = settlements_eligible.groupby('province')['population'].sum()

    # Stage 1 size measure on the population scale vs stage 2's population
    frame = settlements_eligible.set_index('geom_id')
    size_share = frame[size_col] / frame.groupby('province')[size_col].transform('sum')
    cw['frame_population'] = cw['geom_id'].map(frame['population'])
    cw['stage1_size_as_population'] = cw['geom_id'].map(size_share) * cw['province'].map(stratum_pops)
    cw['size_ratio'] = cw['population'] / cw['stage1_size_as_population']

    cw['k_h_p_i'] = cw['k_h'] * cw['proportion_within_stratum']
    cw['w1'] = cw['cluster_weight']
    cw['w12'] = cw['w1'] * cw['stage2_weight']
    cw['w12_expected'] = cw['province'].map(stratum_pops) / (
        cw['k_h'] * avg_household_size * cw['households_visited'])
    cw['w12_ratio'] = cw['w12'] / cw['w12_expected']
    cw['ht_population'] = cw['w1'] * cw['population']

    d = data_weighted.copy()
    d['location'] = d['location'].astype(str).str.strip()
    by_loc = d.groupby('location').agg(respondents=('location', 'size'),
                                       mean_stage3=('stage3_weight', 'mean'))
    cw = cw.join(by_loc, on='sample_number')
    surveyed = cw[cw['respondents'].fillna(0) > 0]

    name_cols = [c for c in ['name', 'settlement', 'village', 'admin3', 'commune'] if c in cw.columns]
    name = name_cols[0] if name_cols else 'sample_number'
    alloc = pd.Series(clusters_allocated or {}, dtype=float)

    rows = {}
    for prov, g in surveyed.groupby('province'):
        rows[prov] = {
            'size_measure_matches': bool(((g['size_ratio'] - 1).abs() <= tol).all()),
            'size_ratio_min': g['size_ratio'].min(),
            'size_ratio_max': g['size_ratio'].max(),
            'k_h': g['geom_id'].nunique(),
            'allocated': alloc.get(prov, np.nan),
            'w12_ratio_mean': g['w12_ratio'].mean(),
            'w12_ratio_min': g['w12_ratio'].min(),
            'w12_ratio_max': g['w12_ratio'].max(),
            'certainty_clusters': ", ".join(g.loc[g['k_h_p_i'] >= 1, name].astype(str)),
            'stage2_below_1': ", ".join(g.loc[g['stage2_weight'] < 1, name].astype(str)),
            'mean_stage3': d.loc[d['province'] == prov, 'stage3_weight'].mean(),
            'ht_pop_over_frame': g['ht_population'].sum() / stratum_pops[prov],
        }
    by_stratum = pd.DataFrame(rows).T

    # Province shares under each combination of stages (respondent-level)
    d['w12'] = d['location'].map(surveyed.set_index('sample_number')['w12'])
    d['w12_expected'] = d['location'].map(surveyed.set_index('sample_number')['w12_expected'])
    all_col = 'normalized_weight_design' if 'normalized_weight_design' in d.columns else 'combined_weight_adjusted'
    shares = pd.DataFrame({
        'population': stratum_pops / stratum_pops.sum(),
        'unweighted': d['province'].value_counts(normalize=True),
        'w1*w2': _shares(d['w12'], d['province']),
        'w12_expected*w3': _shares(d['w12_expected'] * d['stage3_weight'], d['province']),
        'all stages': _shares(d[all_col], d['province']),
    })

    print("=" * 70)
    print("STAGE WEIGHT CHECK (by stratum)")
    print("=" * 70)
    with pd.option_context('display.width', 200, 'display.max_columns', 20, 'display.max_colwidth', 60):
        print("\nStage 1 size measure vs stage 2 population (ratio should be 1):")
        print(by_stratum[['size_measure_matches', 'size_ratio_min', 'size_ratio_max']].to_string())
        print("\nClusters: k_h (distinct surveyed) vs allocation (reference only):")
        print(by_stratum[['k_h', 'allocated']].to_string())
        print("\nw1*w2 / [N_h/(k_h*hh_size*m_i)] (should be ~1):")
        print(by_stratum[['w12_ratio_mean', 'w12_ratio_min', 'w12_ratio_max']].astype(float).round(3).to_string())
        print("\nCertainty clusters (k_h*p_i >= 1) and clusters with stage 2 weight < 1:")
        print(by_stratum[['certainty_clusters', 'stage2_below_1']].to_string())
        print("\nMean stage 3 (eligible adults per household) and HT population / frame:")
        print(by_stratum[['mean_stage3', 'ht_pop_over_frame']].astype(float).round(3).to_string())
        print(f"\nProvince shares (%); 'all stages' = {all_col}:")
        print((shares * 100).round(1).to_string())

    if not by_stratum['size_measure_matches'].all():
        print("\n⚠️  Stage 1 size measure differs from stage 2 population in: "
              f"{list(by_stratum.index[~by_stratum['size_measure_matches'].astype(bool)])} -- the "
              "population does not cancel, so w1*w2 depends on cluster size.")

    by_cluster = cw[[c for c in [*name_cols, 'sample_number', 'geom_id', 'province', 'population',
                                 'frame_population', 'stage1_size_as_population', 'size_ratio', 'k_h',
                                 'proportion_within_stratum', 'k_h_p_i', 'certainty_unit',
                                 'inclusion_prob_exact', 'w1', 'households_visited', 'stage2_weight',
                                 'w12', 'w12_expected', 'w12_ratio', 'ht_population', 'respondents',
                                 'mean_stage3'] if c in cw.columns]]
    if out_dir is not None:
        from pathlib import Path
        path = Path(out_dir) / "02b_stage_weight_check_clusters.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        by_cluster.to_csv(path, index=False)
        print(f"\nPer-cluster table written to {path}")
    return by_stratum, by_cluster


# ============================================================================
# POST-STRATIFICATION
# ============================================================================

def poststratify(data, weight_col, cell_col, controls, cluster_col='location', min_n=20, min_clusters=2):
    """
    Post-stratify weights to known cell totals (Statistics Canada, Survey Methods and
    Practices, 7.1.4.1): w_post = w * N_c / N^_c, where N^_c is the sum of the design
    weights in cell c and N_c is the control total. The same formula applies to PPS
    weights as to SRS. The result is renormalised to mean 1.

    controls: Series (or dict) of control totals indexed by cell. Raises ValueError if a
    respondent has no cell or a cell in the data has no control total.
    Returns (weights Series aligned with data, summary DataFrame by cell). The printed
    factor is scaled so 1 = cell already at its population share.
    """
    controls = pd.Series(controls, dtype=float)
    cells = data[cell_col]
    if cells.isna().any():
        raise ValueError(f"{cells.isna().sum()} respondents have no '{cell_col}' -- cannot post-stratify")
    missing = sorted(set(cells.unique()) - set(controls.dropna().index))
    if missing:
        raise ValueError(f"No control total for {cell_col} cell(s): {missing}")
    w = data[weight_col].astype(float)
    if w.isna().any():
        raise ValueError(f"{w.isna().sum()} respondents have no '{weight_col}'")

    n_hat = w.groupby(cells).sum()
    unused = sorted(set(controls.index) - set(n_hat.index))
    if unused:
        print(f"⚠️  Control cells with no respondents (left out of the totals): {unused}")
    N = controls.reindex(n_hat.index)
    factor = N / n_hat
    w_post = w * cells.map(factor)
    w_post = w_post * len(w_post) / w_post.sum()

    summary = pd.DataFrame({
        'respondents': cells.value_counts(),
        'clusters': data.groupby(cell_col)[cluster_col].nunique() if cluster_col in data.columns else np.nan,
        'population_share': N / N.sum(),
        'sample_share': cells.value_counts(normalize=True),
        'design_share': n_hat / n_hat.sum(),
        'factor': factor * w.sum() / N.sum(),
        'final_share': w_post.groupby(cells).sum() / w_post.sum(),
    })
    thin = summary[(summary['respondents'] < min_n) | (summary['clusters'] < min_clusters)]

    print("=" * 70)
    print(f"POST-STRATIFICATION ON '{cell_col}'")
    print("=" * 70)
    out = summary.copy()
    for c in ['population_share', 'sample_share', 'design_share', 'final_share']:
        out[c] = (out[c] * 100).round(1)
    print(out.round({'factor': 3}).to_string())
    if len(thin):
        print(f"\n⚠️  Thin cells (<{min_n} respondents or <{min_clusters} clusters): {list(thin.index)} -- "
              "their factors rest on very little data.")
    print(f"\nKish DEFF: {kish_deff(w):.2f} (design) -> {kish_deff(w_post):.2f} (post-stratified)")
    return w_post, summary


def apply_poststratification(data_weighted, settlements_eligible, on='province', **kwargs):
    """
    Keep the design-only weight as normalized_weight_design and, if `on` is set
    (config.POSTSTRATIFY_ON), replace normalized_weight with the weight post-stratified to
    settlements_eligible population totals by `on`. on=None leaves normalized_weight as is.
    """
    data_weighted = data_weighted.copy()
    data_weighted['normalized_weight_design'] = data_weighted['normalized_weight']
    data_weighted['combined_weight_design'] = data_weighted['combined_weight_adjusted']
    if on is None:
        print("Post-stratification skipped (POSTSTRATIFY_ON=None): normalized_weight is the design weight.")
        return data_weighted, None
    controls = settlements_eligible.groupby(on)['population'].sum()
    data_weighted['normalized_weight'], summary = poststratify(
        data_weighted, 'normalized_weight_design', on, controls, **kwargs)
    # The pipeline (pipeline_output config WEIGHT_COL) weights by combined_weight_adjusted,
    # so apply the same per-cell factor there too -- otherwise post-stratification would
    # never reach the dashboard figures.
    data_weighted['combined_weight_adjusted'] = (
        data_weighted['combined_weight_design']
        * data_weighted['normalized_weight'] / data_weighted['normalized_weight_design'])
    return data_weighted, summary


# ============================================================================
# USAGE (Benin run: 02b_calculate_weights_KH.ipynb)
# ============================================================================

"""
from src.config import USE_STRATUM_WEIGHT, PPS_CERTAINTY_UNITS, POSTSTRATIFY_ON, CLUSTERS_ALLOCATED

data_weighted, cluster_weights = calculate_three_stage_weights(
    settlements_eligible, settlements_chosen, data,
    avg_household_size=5.5,
    use_stratum_weight=USE_STRATUM_WEIGHT,
    certainty_units=PPS_CERTAINTY_UNITS,
)
check_stage_weights(cluster_weights, data_weighted, settlements_eligible, avg_household_size=5.5,
                    clusters_allocated=CLUSTERS_ALLOCATED, out_dir="data/4_validation_checks")
data_weighted, ps_summary = apply_poststratification(data_weighted, settlements_eligible, on=POSTSTRATIFY_ON)
"""