import numpy as np
#from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
#import tkinter as tk
from tqdm import tqdm
import torch
import pickle
import time
from helpers.ridge import RidgePerElectrode
from helpers.Subject import Subject
from helpers.helpers import *
from helpers.constants import *
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer, AutoModel
import numpy as np
import gzip
import os
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split,KFold
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm
import gc 
import glob
#from helpers.Sub_test import Mne_Subject

def create_trial_df_from_aligned_triggers(subject, aligned_trials_df, triggers_df):
    """
    Recreate trial_df format from aligned triggers for regression pipeline.
    
    Parameters:
    -----------
    subject : Subject object
        Your Subject instance with aligned triggers
    aligned_trials_df : pd.DataFrame
        Output from align_triggers_with_word_timing() or align_triggers_with_word_timing_var1()
    triggers_df : pd.DataFrame
        Full trigger dataframe with all trigger information
        
    Returns:
    --------
    trial_df : pd.DataFrame
        DataFrame matching your original format with columns:
        [Type, number, system_timePreOnset, system_timePostTrigger, sentence, 
         block_num, stim_num, response, Var1, Var2, ...]
    """
    
    # Prepare triggers_df - get word triggers only
    word_pattern = r'WORD(\d)_(VIS|AUD)'
    triggers_df['word_position'] = triggers_df['Type'].str.extract(word_pattern)[0]
    word_triggers = triggers_df[triggers_df['word_position'].notna()].copy()
    word_triggers['word_position'] = word_triggers['word_position'].astype(int)
    
    # Group into trials (matching align_triggers logic)
    trials = []
    current_trial = []
    expected_position = 1
    
    for idx, row in word_triggers.iterrows():
        word_pos = row['word_position']
        
        if word_pos == 1:
            if len(current_trial) == 4:
                trials.append(current_trial)
            current_trial = [row]
            expected_position = 2
        elif word_pos == expected_position and len(current_trial) > 0:
            current_trial.append(row)
            expected_position += 1
            
            if len(current_trial) == 4:
                trials.append(current_trial)
                current_trial = []
                expected_position = 1
        else:
            current_trial = []
            if word_pos == 1:
                current_trial = [row]
                expected_position = 2
            else:
                expected_position = 1
    
    if len(current_trial) == 4:
        trials.append(current_trial)
    
    print(f"Found {len(trials)} complete trials")
    
    # Build trial_df - one row per trigger (all triggers from trial)
    trial_rows = []
    
    for trial_idx, trial_words in enumerate(trials):
        # Get full sentence from first word
        word1 = trial_words[0]
        
        # Extract sentence
        if 'sentence' in word1 and isinstance(word1['sentence'], dict):
            if 'sen_field' in word1['sentence']:
                sen_field = word1['sentence']['sen_field']
                if isinstance(sen_field, dict):
                    sentence = ' '.join([
                        sen_field.get('w1', ''),
                        sen_field.get('w2', ''),
                        sen_field.get('w3', ''),
                        sen_field.get('w4', '')
                    ]).strip()
                else:
                    sentence = ''
            else:
                sentence = ''
        else:
            sentence = ''
        
        # Add all 4 word triggers for this trial
        for word_row in trial_words:
            trial_row = {
                'Type': word_row['Type'],
                'number': word_row['number'],
                'system_timePreOnset': word_row['system_timePreOnset'],
                'system_timePostTrigger': word_row['system_timePostTrigger'],
                'sentence': sentence,  # Full sentence for all words in trial
                'block_num': word_row.get('block_num', -1),
                'stim_num': word_row.get('stim_num', -1),
                'response': word_row.get('response', {}),
                'Var1': word_row.get('Var1', -1),
                'Var2': word_row.get('Var2', -1),
                'trial_idx': trial_idx  # Add trial index
            }
            trial_rows.append(trial_row)
    
    trial_df = pd.DataFrame(trial_rows)
    
    print(f"Created trial_df with {len(trial_df)} rows ({len(trials)} trials × ~4 words)")
    print(f"Columns: {trial_df.columns.tolist()}")
    
    return trial_df


def save_trial_df_for_subject(subject_num, subject,step, output_dir='.'):
    """
    Complete pipeline to create and save trial_df for a subject.
    
    Usage:
    ------
    sub = Subject(...)
    sub.load_behav_mat()
    sub.convert_behav_mats()
    sub.load_neuro_mat()
    sub.convert_neuro_mat()
    sub.pre_process_gamma(do_trial_averaging=False)
    
    # For subjects where system_time works:
    aligned_trials = sub.align_triggers_with_word_timing()
    
    # OR for subjects where Var1 is needed:
    aligned_trials = sub.align_triggers_with_word_timing_var1()
    
    # Then save:
    save_trial_df_for_subject(8, sub)
    """
    
    # Get triggers dataframe
    triggers_df = subject.prepare_trigger_dataframe()
    triggers_df = subject.correct_trigger_errors(triggers_df)
    
    # Get aligned trials (should already be in subject.aligned_trials_df)
    if not hasattr(subject, 'aligned_trials_df'):
        raise ValueError("Subject doesn't have aligned_trials_df. Run align_triggers first!")
    
    aligned_trials_df = subject.aligned_trials_df
    
    if len(aligned_trials_df) == 0:
        raise ValueError("No aligned trials found!")
    
    # Create trial_df
    trial_df = create_trial_df_from_aligned_triggers(
        subject, 
        aligned_trials_df, 
        triggers_df
    )
    
    # Save as .npy with allow_pickle=True (matches your format)
    output_path = f"{output_dir}/S{subject_num}_trial_df.npy"
    np.save(output_path, trial_df.to_dict('records'), allow_pickle=True)
    
    print(f"\n✅ Saved trial_df to: {output_path}")
    print(f"   Format: .npy with allow_pickle=True")
    print(f"   Rows: {len(trial_df)}")
    print(f"   Unique sentences: {trial_df['sentence'].nunique()}")
    
    # Also save as pickle for easier loading
    pickle_path = f"{output_dir}/S{subject_num}_{step}_trial_df.pkl"
    trial_df.to_pickle(pickle_path)
    print(f"   Also saved as pickle: {pickle_path}")
    
    return trial_df

def verify_trial_df(subject_num):
    """
    Load and verify a trial_df matches expected format.
    """
    # Load
    trial_df = pd.DataFrame(np.load(f"S{subject_num}_trial_df.npy", allow_pickle=True))
    
    print(f"\n{'='*60}")
    print(f"VERIFICATION: Subject {subject_num} trial_df")
    print('='*60)
    print(f"Shape: {trial_df.shape}")
    print(f"Columns: {trial_df.columns.tolist()}")
    print(f"\nFirst 5 rows:")
    print(trial_df.head())
    print(f"\nUnique sentences: {trial_df.iloc[:, 4].nunique()}")
    
    return trial_df

#Each timestamp sample is 1ms 
def main():
    step = 20 
    for sub_num, directory in DIRECTORYS.items():

        if sub_num in [1,2,3,4,5,6,8,9,10,11,12,13,14,15,16,17]: #16 is also saying corrupted, as is 14,, issue with hdr class dict for subject 17
            continue
        print(f"\n{'='*80}")
        start_time = time.time()
        print(f"PROCESSING SUBJECT {sub_num}")
        print(f"{'='*80}\n")
        
        subject = f"S{sub_num}"
        sub = Subject(subject=subject, subNum=sub_num, record_direc=directory)

        # Load data
        print("Loading Behavioral Data")
        sub.load_behav_mat() #Load Trigger Frame
        sub.convert_behav_mats() #Pre-process trigger frame
        print("Loading Neurological Data")
        #sub.load_neuro_mat() # Loading in record data and HDR table
        sub.load_neuro_mat_robust(chunk_size = 100) #noticing that subject 2 has much larger record data than other subjects, # timestamps starts with 2 not 1
        sub.convert_neuro_mat() #Prep-process HDR table
        #sub.record_filter = np.load(f"sub_{sub_num}_{step}_record_filter.npy")
        print("Pre-processing Gamma band signals")
        sub.pre_process_gamma_memory_conserve_fixed(do_trial_averaging=False) 
        #np.save(f"sub_{sub_num}_{step}_record_filter.npy", sub.record_filter, allow_pickle = True)
        np.save(f"sentences_subs/S{sub_num}_sentences_iter.npy", sub.trigger_frame['sentence'].to_dict(), allow_pickle = True)
        # Step 1: Align triggers with word timing
        print("\n=== Step 1: Align Triggers ===")
        aligned_trials = sub.align_triggers_with_word_timing(debug=True, diagnose_bounds=True)
        print(f"Found {len(aligned_trials)} trials")

        # Step 2: Extract epochs locked to word1
        print("\n=== Step 2: Extract Epochs ===")
        sub.extract_epochs(lock_to="word1", pre_time=0.2, post_time=5.5)
        print(f"Epochs shape: {sub.epochs.shape}")
        print(f"Time range: {sub.epoch_times.min()*1000:.1f} to {sub.epoch_times.max()*1000:.1f} ms")
        
        # Step 3: Extract sliding window features
        print("\n=== Step 3: Extract Sliding Window Features ===")
        features = sub.extract_sliding_window_features(
            window_ms=200, 
            step_ms=step, #Lets try with 100 ms step now that we are scaling the data
            freq_range=(30, 150), #This should be going up high gamma activity
            baseline_window=(-200, 0)
        )

        print(f"\n✅ Success! Features shape: {features.shape}")
        print(f"Columns: {features.columns.tolist()[:10]}...")
        
        # Save trial data
        trial_df = save_trial_df_for_subject(sub_num, sub, step)
        trial_df.to_pickle(f"S{sub_num}_{step}_trial_df.pkl")

        #Save features df
        print("\n=== Step 4: Saving features df")
        file_path = f"pickle_features/S{sub_num}_{step}_feature_df.pkl"

        # Optional: Ensure directory exists to avoid FileNotFoundError
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        try:
            sub.features_df.to_pickle(file_path)
            print(f"Successfully saved: {file_path}")
        except Exception as e:
            print(f"Error saving {file_path}: {e}")
            break
        
        print(f"\nCleaning up memory for Subject {sub_num}...")
        
        
        # Delete the Subject object and all its data
        del sub
        del trial_df
        del features
        del aligned_trials
        
        # Force garbage collection 
        #How to do manual garbage collection in python
        gc.collect()
        
        print(f"✓ Memory cleaned for Subject {sub_num}\n")
        end_time = time.time()
        print(f"Subject run in {(end_time -start_time)/60} minutes")
        # =============================================

    print("\n" + "="*80)
    print("ALL SUBJECTS COMPLETED")
    print("="*80)
if __name__ == "__main__":
    main()