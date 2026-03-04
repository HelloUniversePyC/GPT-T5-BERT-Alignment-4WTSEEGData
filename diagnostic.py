import pandas as pd
import numpy as np
from helpers.constants import *

step = 20
sub_num = 2
wrong = []
for sub_num, _ in DIRECTORYS.items():
    if sub_num in [16,17]:
        continue
    print("="*80)
    print(f"CHECKING PICKLE FILE FOR SUBJECT {sub_num}")
    print("="*80)

    # Load the pickle
    features_df = pd.read_pickle(f"pickle_features/S{sub_num}_{step}_feature_df.pkl")

    print(f"\nFeatures DataFrame Info:")
    print(f"  Shape: {features_df.shape}")
    print(f"  Columns: {features_df.columns.tolist()}")

    print(f"\nTime Range:")
    print(f"  Min time_start_ms: {features_df['time_start_ms'].min():.1f}")
    print(f"  Max time_start_ms: {features_df['time_start_ms'].max():.1f}")
    print(f"  Min time_end_ms: {features_df['time_end_ms'].min():.1f}")
    print(f"  Max time_end_ms: {features_df['time_end_ms'].max():.1f}")

    print(f"\nConsolidation Window Check:")
    print(f"  Need data up to: 5100ms")
    print(f"  Have data up to: {features_df['time_end_ms'].max():.1f}ms")
    print(f"  Status: {'✅ SUFFICIENT' if features_df['time_end_ms'].max() >= 5100 else '❌ INSUFFICIENT'}")

    # Check a sample trial
    print(f"\nSample Trial Analysis:")
    sample_trial_idx = features_df['trial_idx'].iloc[0]
    sample_trial = features_df[features_df['trial_idx'] == sample_trial_idx]
    print(f"  Trial {sample_trial_idx}:")
    print(f"    Number of windows: {len(sample_trial)}")
    print(f"    Time span: {sample_trial['time_start_ms'].min():.1f} - {sample_trial['time_end_ms'].max():.1f}ms")
    print(f"    Duration: {sample_trial['time_end_ms'].max() - sample_trial['time_start_ms'].min():.1f}ms")

    # Check consolidation window specifically
    OFFSET_MS = 600
    consolidation_start = 3500 + OFFSET_MS  # 4100
    consolidation_end = 4500 + OFFSET_MS    # 5100

    mask = (features_df['time_start_ms'] >= consolidation_start) & \
        (features_df['time_end_ms'] <= consolidation_end)
    consolidation_windows = features_df[mask]

    print(f"\nConsolidation Window (4100-5100ms):")
    print(f"  Windows found: {len(consolidation_windows)}")
    print(f"  Unique trials: {consolidation_windows['trial_idx'].nunique() if len(consolidation_windows) > 0 else 0}")
    print(f"  Total trials in dataset: {features_df['trial_idx'].nunique()}")

    if len(consolidation_windows) == 0:
        wrong.append(sub_num)
        print(f"\n❌ PROBLEM: No windows in consolidation period!")
        print(f"   Your features_df was created with insufficient epoch extraction")
        print(f"   You need to delete this pickle and regenerate with post_time=5.5")
    else:
        print(f"\n✅ GOOD: Consolidation window has data")

    print("\n" + "="*80)

print("Wrong subjects ", wrong)