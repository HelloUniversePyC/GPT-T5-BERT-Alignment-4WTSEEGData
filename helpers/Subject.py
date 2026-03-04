import wheel
import pandas as pd
import numpy as np
import sklearn as skl
from mat4py import loadmat
import os
import scipy.io
import csv
import struct
import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from mne_bids import BIDSPath, read_raw_bids
from scipy.signal import butter, filtfilt
from mne.time_frequency import tfr_multitaper
import seaborn as sb
from mpl_toolkits.mplot3d import Axes3D
import mne
from mne.viz import plot_alignment, snapshot_brain_montage
import mne_bids
import shutil
from mne.channels import make_standard_montage
from sklearn.model_selection import train_test_split
import tempfile #creation of temporay cache file for memory mapping
from scipy.signal import iirnotch, filtfilt, butter, sosfiltfilt, decimate, sosfilt
from joblib import Parallel, delayed
import numpy as np
from tqdm import tqdm
from transformers import GPT2Tokenizer, GPT2Model
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import sys
import gc
from helpers.constants import *
import datetime

class Subject(object):
    """Class to Create Subject Objects with appropriate metadata and prep-processing"""
    def __init__(self, subject, subNum, record_direc):
        self.subInit = subject
        self.subNum = subNum
        self.record_direc = record_direc
        
    def load_behav_mat(self):
        """Used to load in matlab files for behavioral experiment"""
        #Initialize Paths
        self.Gen_Path = f"{EXPANSION_PATH}/Sub-Mat-Converted/{self.subInit}/"
        Curr_State = self.Gen_Path+self.subInit+"-CurrState.mat"
        Trig_Sentence = self.Gen_Path+self.subInit+"-sentence.mat"
        Trigger = self.Gen_Path+self.subInit+"-trig.mat"
        
        #Loading in neccesary raw files
        self.curr_raw = scipy.io.loadmat(Curr_State,squeeze_me=True, struct_as_record = False, simplify_cells = True)
        self.trig_raw = scipy.io.loadmat(Trigger, squeeze_me=True, struct_as_record = False, simplify_cells = True)
        self.sentence_raw = scipy.io.loadmat(Trig_Sentence, squeeze_me=True, struct_as_record = False, simplify_cells = True)
        
        #Creating seperate data frames of the nested struct fields
        self.presentation_matrix_frame = pd.DataFrame(self.curr_raw['curr_state_export']['presentation_matrix'])
        self.trigger_frame = pd.DataFrame(self.trig_raw['trigg_list'])
        self.file_frame = pd.DataFrame(list(self.curr_raw['curr_state_export']['filenames']))
        
    def convert_behav_mats(self):
        """Used to create processed dataframes from raw matlab files"""
        #First handle curr_state
        curr_state_slice = self.curr_raw['curr_state_export']
        #Create seperate dictionary of all columns that can be resolved to just a constant
        self.constants = list(curr_state_slice.keys())
        #remove the class type fields
        self.constants.remove('trigger_list')
        self.constants.remove('presentation_matrix')
        self.constants.remove('blockList')
        self.constants.remove('filenames')
        #Use dictionary comprehension to only keep primitive type data in final dictionary
        self.constants_frame = {key: curr_state_slice[key] for key in self.constants}

        #Handle trigger_frame
        #replacing MATLAB opaque sentences with properly parsed ones
        sentence_dict = self.sentence_raw['sentence_work']
        self.trigger_frame['sentence'] = sentence_dict
    

    def diagnose_mat_file(self, file_path):
        """
        Diagnose issues with .mat files
        """
        print(f"\n{'='*60}")
        print(f"DIAGNOSING: {file_path}")
        print('='*60)
        
        #Check if file exists
        if not os.path.exists(file_path):
            print("❌ FILE NOT FOUND")
            return False
        
        # Check file size
        file_size = os.path.getsize(file_path) / (1024**3)  # GB
        print(f"✓ File exists: {file_size:.2f} GB")
        
        # Try to detect format
        with open(file_path, 'rb') as f:
            magic = f.read(4)
            print(f"File signature: {magic}")
        
        # Try scipy.io.loadmat (for MATLAB v7.2 and earlier)
        print("\n1. Trying scipy.io.loadmat...")
        try:
            data = scipy.io.loadmat(file_path)
            print("✓ SUCCESS with scipy.io.loadmat")
            print(f"  Keys: {list(data.keys())}")
            return 'scipy'
        except Exception as e:
            print(f"❌ Failed: {str(e)[:100]}")
        
        # Try h5py (for MATLAB v7.3+)
        print("\n2. Trying h5py...")
        try:
            with h5py.File(file_path, 'r') as f:
                print("✓ SUCCESS with h5py")
                print(f"  Keys: {list(f.keys())}")
                if 'record' in f:
                    print(f"  'record' shape: {f['record'].shape}")
                    print(f"  'record' dtype: {f['record'].dtype}")
                return 'h5py'
        except Exception as e:
            print(f"❌ Failed: {str(e)[:100]}")
        
        # File is corrupted or unreadable
        print("\n❌ FILE IS CORRUPTED OR INCOMPATIBLE")
        print("Possible issues:")
        print("  - File transfer error (incomplete download)")
        print("  - Unsupported MATLAB version")
        print("  - File corruption")
        print("  - Not a valid .mat file")
        
        return False
    #cpu bound <- fork, limited by processing power<- matrix ops, io bound<- thread, limited by waiting for data (ex: <- http requests), read/writing files
    #Try memory mapping
  
    def load_neuro_mat_robust(self, load_record=True, chunk_size=200_000):
        """
        Load neural data safely into a memory-mapped array 
        Automatically cleans up previous subjects' cache directories to prevent disk space issues
        """
        # Clean up previous subjects' caches before creating this subject's cache
        base_cache_dir = os.path.join(tempfile.gettempdir(), "neural_cache")
        
        if os.path.exists(base_cache_dir):
            print(f"Checking for old cache directories to clean up...")
            deleted_count = 0
            freed_space_gb = 0
            
            for cache_subdir in os.listdir(base_cache_dir):
                cache_path = os.path.join(base_cache_dir, cache_subdir)
                
                # Skip if it's the current subject's cache
                if cache_subdir == self.subInit:
                    continue
                
                # Calculate size before deletion
                if os.path.isdir(cache_path):
                    dir_size = sum(
                        os.path.getsize(os.path.join(dirpath, filename))
                        for dirpath, dirnames, filenames in os.walk(cache_path)
                        for filename in filenames
                    )
                    freed_space_gb += dir_size / (1024**3)
                    
                    try:
                        shutil.rmtree(cache_path)
                        deleted_count += 1
                        print(f"  ✓ Deleted cache: {cache_subdir} ({dir_size / (1024**3):.2f} GB)")
                    except Exception as e:
                        print(f"  ⚠️ Could not delete {cache_subdir}: {e}")
            
            if deleted_count > 0:
                print(f"✓ Cleaned up {deleted_count} old cache director{'y' if deleted_count == 1 else 'ies'}, freed {freed_space_gb:.2f} GB")
            else:
                print(f"  No old caches to clean up")
        
        # Now create this subject's cache directory
        if not hasattr(self, "cache_dir"):
            self.cache_dir = os.path.join(
                tempfile.gettempdir(),
                "neural_cache",
                self.subInit
            )
            os.makedirs(self.cache_dir, exist_ok=True)

        hdr_path = self.Gen_Path + self.subInit + "-hdr.mat"
        print(f"Loading HDR from: {hdr_path}")

        self.hdr_raw = scipy.io.loadmat(
            hdr_path,
            squeeze_me=True,
            struct_as_record=False,
            simplify_cells=True
        )
        print("HDR loaded")

        if not load_record:
            return

        record_path = f"{EXPANSION_PATH}/DATA/{self.record_direc}/neuralMatfile/FILE1.mat"
        print(f"\nLoading neural data from: {record_path}")

        file_format = self.diagnose_mat_file(record_path)
        if not file_format:
            raise OSError("MAT file unreadable or corrupted")

        if file_format == "scipy":
            mat = scipy.io.loadmat(record_path)
            shape = mat["record"].shape
            dtype = np.float32
            record_reader = lambda sl: mat["record"][sl]

        elif file_format == "h5py":
            f = h5py.File(record_path, "r")
            dset = f["record"]
            shape = dset.shape
            dtype = np.float32
            record_reader = lambda sl: dset[sl]

        else:
            raise ValueError("Unknown MAT format")

        print(f"Original dataset shape: {shape}")
        if shape[0] > shape[1]:
            # Likely (samples, channels) - need transpose
            n_samples, n_channels = shape[0], shape[1]
            need_transpose = True
            print(f"Detected MATLAB format: {n_samples} samples × {n_channels} channels")
            print("Will transpose to (channels, samples)")
        else:
            # Already (channels, samples)
            n_channels, n_samples = shape[0], shape[1]
            need_transpose = False
            print(f"Detected format: {n_channels} channels × {n_samples} samples")
            print("No transpose needed")

        mmap_path = os.path.join(self.cache_dir, f"{self.subInit}_record.dat")

        self.record = np.memmap(
            mmap_path,
            mode="w+",
            dtype=dtype,
            shape=(n_channels, n_samples)  # Always (channels, samples)
        )

        if need_transpose:
            print("Loading data with transpose...")
            for start in range(0, n_samples, chunk_size):
                end = min(start + chunk_size, n_samples)
                
                # Read chunk: (chunk_samples, channels)
                # Transpose to: (channels, chunk_samples)
                chunk = record_reader(
                    (slice(start, end), slice(None))
                ).astype(np.float32, copy=False)
                
                self.record[:, start:end] = chunk.T
                
                if start % (chunk_size * 10) == 0:
                    print(f"  Loaded {100 * end / n_samples:.1f}%")
        else:
            print("Loading data without transpose...")
            for start in range(0, n_samples, chunk_size):
                end = min(start + chunk_size, n_samples)
                
                # Read chunk: (channels, chunk_samples)
                chunk = record_reader(
                    (slice(None), slice(start, end))
                ).astype(np.float32, copy=False)
                
                # No transpose needed
                self.record[:, start:end] = chunk
                
                if start % (chunk_size * 10) == 0:
                    print(f"  Loaded {100 * end / n_samples:.1f}%")
                del chunk
                gc.collect()

        print(f"✓ Memmap created: {self.record.shape} (channels × samples)")
        print(f"  Duration: {n_samples/1000:.1f}s assuming 1000 Hz")  # Will verify with hdr later

    def load_neuro_mat(self, load_record = True, chunk_size = 100000):
        """Load in neurological files with chunking"""
        # Load in hdr
        hdr_path = self.Gen_Path + self.subInit + "-hdr.mat"
        self.hdr_raw = scipy.io.loadmat(hdr_path, squeeze_me=True, struct_as_record=False, simplify_cells=True)
        if load_record:
            record_path = f"{EXPANSION_PATH}/DATA/{self.record_direc}/neuralMatfile/FILE1.mat"

            print(f"Loading neural data from: {record_path}")

            with h5py.File(record_path, 'r') as f:
                dataset = f['record']
                print(f"Dataset shape: {dataset.shape}")
                print(f"Dataset dtype: {dataset.dtype}")
                
                # Pre-allocate the full array
                self.record = np.zeros(dataset.shape, dtype=dataset.dtype)
                
                # Load in chunks
                n_samples = dataset.shape[1] 
                
                print(f"Loading in chunks of {chunk_size} samples...")
                for start_idx in range(0, n_samples, chunk_size):
                    end_idx = min(start_idx + chunk_size, n_samples)
                    
                    # Load chunk and store it
                    self.record[:, start_idx:end_idx] = dataset[:, start_idx:end_idx]
                    
                    # Progress indicator
                    if start_idx % 1000000 == 0:
                        progress = (end_idx / n_samples) * 100
                        print(f"  Loaded {end_idx}/{n_samples} samples ({progress:.1f}%)")
                del dataset 
                gc.collect()

                
                print(f"Loaded neural data: {self.record.shape}")
                print(f"Memory usage: {self.record.nbytes / 1e9:.2f} GB")
                

    def convert_neuro_mat(self):
        """Convert neurological mat files to dataframes, handles non-class columns"""
        prim_keys = ['ver', 'patientID', 'recordID', 'startdate', 'starttime', 'bytes', 'records', 'duration', 'ns', 'label', 'labelNew']
        hdr_class_dict = {key: value for key, value in self.hdr_raw['hdr'].items() if key not in prim_keys}
        hdr_class_dict.pop('labelNew2', None)
        self.hdr_frame = pd.DataFrame(hdr_class_dict)
        self.hdr_prim_dict= {key: value for key, value in self.hdr_raw['hdr'].items() if key in prim_keys}
        #placing labels in a seperate list
        self.labels = list(self.hdr_raw['hdr']['label'])
    def pre_process_gamma_memory_conserve_fixed(
        self,
        do_trial_averaging=False,
        time_chunk_sec=10,
        n_threads=4
    ):
        """
        Bandpass + bipolar montage using memmaps and time chunking.
        """
        fs = int(self.hdr_frame.frequency[0])
        n_channels, n_samples = self.record.shape

        sos_gamma = butter(
            5,
            [30 / (fs / 2), 150 / (fs / 2)],
            btype="band",
            output="sos"
        )

        chunk_len = int(time_chunk_sec * fs)
        print(f"Processing in {time_chunk_sec}s chunks ({chunk_len} samples)")
        
        out_path = os.path.join(self.cache_dir, f"{self.subInit}_gamma_filtered.dat")
        filtered_mm = np.memmap(
            out_path,
            mode="w+",
            dtype=np.float32,
            shape=(n_channels - 1, n_samples)
        )

        def filter_single_channel(ch_idx, raw_chunk):
            """Process ONE channel in-place - no copies"""
            # Compute bipolar for this channel only
            bipolar_signal = raw_chunk[ch_idx] - raw_chunk[ch_idx + 1]
            
            # Apply notch filter
            notched = self.notch_filter_scipy(bipolar_signal, fs)
            
            # Apply gamma filter and return
            return sosfilt(sos_gamma, notched).astype(np.float32)

        print("Starting chunked bipolar + filtering...")
        
        for start in tqdm(range(0, n_samples, chunk_len), desc="Time chunks"):
            end = min(start + chunk_len, n_samples)

            # Load raw chunk (SHARED across all channels)
            raw_chunk = self.record[:, start:end]
            
            # Process each CHANNEL separately (no large intermediate arrays)
            for ch in range(n_channels - 1):
                filtered_mm[ch, start:end] = filter_single_channel(ch, raw_chunk)
            
            # Explicit cleanup
            del raw_chunk
            
            if start % (chunk_len * 5) == 0:
                print(f"  {100 * end / n_samples:.1f}% complete")
                gc.collect()

        print("✓ Filtering complete")

        if do_trial_averaging:
            print("Performing trial averaging...")
            
            timeFIXBSL = -400
            timeWord = 1100
            idxFIXBSL = int(np.ceil(timeFIXBSL * fs / 1000))
            idxWord = int(np.ceil(timeWord * fs / 1000))
            sizeTime = idxWord - idxFIXBSL

            triggers = self.trigger_frame["Var1"].values
            valid_triggers = triggers[
                (triggers + idxWord) < n_samples
            ]

            averaged = np.zeros((filtered_mm.shape[0], sizeTime), dtype=np.float32)

            for ch in range(filtered_mm.shape[0]):
                acc = np.zeros(sizeTime, dtype=np.float32)
                for t in valid_triggers:
                    acc += filtered_mm[ch, t + idxFIXBSL : t + idxWord]
                averaged[ch] = acc / len(valid_triggers)

            self.record_filter = averaged
            print("✓ Trial averaging complete")
        else:
            self.record_filter = filtered_mm
            del filtered_mm
            gc.collect()

        return self.record_filter

    def pre_process_gamma(self, do_trial_averaging=False, batch_size=20):
        """
        Apply filtering and bipolar montage to neural data with batched processing
        """
        self.record = self.record.transpose().astype(np.float32)
        fs = int(self.hdr_frame.frequency[0])

        sos_gamma = butter(5, [30 / (fs / 2), 150 / (fs / 2)], btype='band', output='sos') #this should go up to 150 <- we are removing high gamma signals otherwise
        print("Variables initialized")

        # Bipolar montage 
        bipolar_signals = (self.record[:-1, :] - self.record[1:, :]).astype(np.float32)

        def filter_chan(chan): #smoothing
            """This Python function filter_chan takes a channel of data (chan), first applies a specific notch filter 
            (likely removing power line noise using mne) and converts it to float32, then applies a zero-phase Butterworth bandpass/lowpass filter 
            (using sosfiltfilt with predefined sos_gamma coefficients), also converting the final result to float32, effectively cleaning up a single data stream by removing specific 
            frequencies and then smoothing it. """

            notch = self.notch_filter_mne(chan, fs).astype(np.float32)
            return sosfiltfilt(sos_gamma, notch).astype(np.float32)

        # Process in batches to avoid memory issues
        print("Applying notch filter and low-pass filter...")
        n_channels = bipolar_signals.shape[0]
        filtered_signals = np.zeros_like(bipolar_signals, dtype=np.float32)

        for batch_start in tqdm(range(0, n_channels, batch_size), desc="Processing batches"):
            batch_end = min(batch_start + batch_size, n_channels)
            
            # Process this batch in parallel
            batch_filtered = np.array(
                Parallel(n_jobs=-1)(
                    delayed(filter_chan)(bipolar_signals[i, :])
                    for i in range(batch_start, batch_end)
                )
            )
            
            # Store results
            filtered_signals[batch_start:batch_end, :] = batch_filtered
            
            # Force garbage collection between batches
            del batch_filtered
            import gc
            gc.collect()

        print("Bipolar Montage and Filtering Complete")

        if do_trial_averaging:
            print("Performing trial averaging...")
            timeFIXBSL = -400
            timeWord = 1100
            idxFIXBSL = int(np.ceil(timeFIXBSL * fs / 1000))
            idxWord = int(np.ceil(timeWord * fs / 1000))
            sizeTime = idxWord - idxFIXBSL

            triggers = self.trigger_frame['Var1'].values
            valid_triggers = triggers[(triggers + idxWord) < filtered_signals.shape[1]]

            filtered_data = np.zeros((filtered_signals.shape[0], sizeTime), dtype=np.float32)
            print("Averaging time windows...")
            for elec_idx in tqdm(range(filtered_signals.shape[0]), desc="Averaging Time Windows"):
                trig_indices = valid_triggers[:, None] + np.arange(idxFIXBSL, idxWord)
                elec_data = filtered_signals[elec_idx, trig_indices]
                filtered_data[elec_idx, :] = elec_data.mean(axis=0)

            self.record_filter = pd.DataFrame(filtered_data, columns=np.arange(sizeTime))
            print("Trial averaging complete - data reduced to averaged trials")
        else:
            self.record_filter = filtered_signals
            print(f"Filtering complete - preserved full data: {filtered_signals.shape}")

        return self.record_filter
            
    def notch_filter_scipy(self, signal, fs, notch_freq=60):
        """Lighter-weight notch filter"""
        Q = 30  # Quality factor
        b, a = iirnotch(notch_freq, Q, fs)
        return filtfilt(b, a, signal, padlen = 0).astype(np.float32)
    
    def get_experiment_start_time(self):
        """
        Extract experiment start time from curr_state data
        Similar to original: exp_params['exp_start']
        """
        try:
            if hasattr(self, 'curr_raw') and 'curr_state_export' in self.curr_raw:
                curr_state = self.curr_raw['curr_state_export']
                
                # Look for experiment start time in various possible field names
                start_time_candidates = ['exp_start', 'experiment_start', 'start_time', 'exp_start_time']
                
                for field in start_time_candidates:
                    if field in curr_state:
                        exp_start_str = str(curr_state[field])
                        if isinstance(curr_state[field], np.ndarray):
                            exp_start_str = str(curr_state[field][0])
                        
                        exp_start_time = datetime.datetime.strptime(exp_start_str, '%d-%b-%Y %H:%M:%S')
                        return exp_start_time
                        
        except Exception as e:
            print(f"Could not extract experiment start time: {e}")
            return None
            
        print("Warning: Experiment start time not found")
        return None
    
    """
    Trial Explaination:
        A trial corresponds to a single experimental event, usually defined by a trigger (e.g., stimulus onset, word presentation, fixation, consolidation phase).

    aligned_triggers_df already contains:

    Trial identifiers

    Trigger timestamps

    Experimental conditions

    Possibly subject/session metadata

So one row ≈ one experimental trial.<- rows are trials in trial_Df

Trial features are compact numerical summaries of SEEG activity extracted from a specific time window around each experimental event, for each electrode.<- generate summaries from aligned trials
    """
            
    def extract_trial_features(self, aligned_triggers_df, 
                            time_window='consolidation', 
                            window_start_ms=4100, window_duration_ms=1000):
        """
        Extract trial features from aligned triggers using specified time windows
        
        Parameters
        ----------
        aligned_triggers_df : pd.DataFrame
            Output from align_triggers_simple() or align_triggers_extensive()
        time_window : str
            Type of analysis window ('consolidation', 'fixation', 'words', 'custom')
        window_start_ms : int
            Start time relative to trigger in milliseconds (for custom windows)  
        window_duration_ms : int
            Duration of analysis window in milliseconds (for custom windows)
            
        Returns
        -------
        trial_df : pd.DataFrame
            DataFrame with trial features for each electrode
        """
        if not hasattr(self, 'record_filter'):
            raise ValueError("Filtered data not available. Call pre_process_gamma() first.")
            
        fs = int(self.hdr_frame.frequency[0])
        record_data = self.record_filter
        
        # Define time windows based on experimental paradigm
        time_windows = {
            'fixation': (0, 600),           
            'words': (600, 4100),           
            'consolidation': (4100, 5100),  
            'answer': (5100, 6100),         
            'custom': (window_start_ms, window_start_ms + window_duration_ms)
        }
        
        if time_window not in time_windows:
            raise ValueError(f"time_window must be one of {list(time_windows.keys())}")
            
        start_ms, end_ms = time_windows[time_window]
        start_samples = int(start_ms * fs / 1000)
        end_samples = int(end_ms * fs / 1000)
        
        print(f"Extracting features from {time_window} window: {start_ms}-{end_ms}ms")
        print(f"Sample range: {start_samples}-{end_samples}")
        
        # ✅ CREATE TRIAL_IDX TO SENTENCE MAPPING
        # Get sentence information from trigger_frame
        trial_to_sentence = {}
        if hasattr(self, 'trigger_frame') and 'sentence' in self.trigger_frame.columns:
            for idx, row in aligned_triggers_df.iterrows():
                trial_idx = row.get('trial_idx', idx)
                # Try to get sentence from trigger_frame using the original index
                if idx < len(self.trigger_frame):
                    sentence_data = self.trigger_frame.iloc[idx]['sentence']
                    
                    # Extract full sentence text
                    if isinstance(sentence_data, dict) and 'sen_field' in sentence_data:
                        sen_field = sentence_data['sen_field']
                        if isinstance(sen_field, dict):
                            # Construct sentence from w1, w2, w3, w4
                            words = [
                                sen_field.get('w1', ''),
                                sen_field.get('w2', ''),
                                sen_field.get('w3', ''),
                                sen_field.get('w4', '')
                            ]
                            sentence = ' '.join([w for w in words if w]).strip()
                            trial_to_sentence[trial_idx] = sentence
        
        print(f"Mapped {len(trial_to_sentence)} trials to sentences")
        
        # Extract features for each trial
        trial_records = []
        skipped_trials = 0
        
        for trial_idx, (orig_idx, row) in enumerate(aligned_triggers_df.iterrows()):
            # Get trigger sample index
            trigger_sample = None
            
            if 'word1_onset_sample' in row and pd.notna(row['word1_onset_sample']):
                trigger_sample = int(row['word1_onset_sample'])
            elif 'matched_events' in row and pd.notna(row['matched_events']):
                trigger_sample = int(row['matched_events'])
            elif 'observed_start_time' in row and pd.notna(row['observed_start_time']):
                trigger_sample = int(row['observed_start_time'])
            
            if trigger_sample is None:
                print(f"No valid trigger timing found for trial {trial_idx}")
                skipped_trials += 1
                continue
                
            # Calculate analysis window indices
            trial_start = trigger_sample + start_samples
            trial_end = trigger_sample + end_samples
            
            # Check bounds
            if trial_start < 0 or trial_end > record_data.shape[1]:
                feature_vec = np.full(record_data.shape[0], np.nan)
                skipped_trials += 1
                print(f"Trial {trial_idx} out of bounds: {trial_start}-{trial_end}, max: {record_data.shape[1]}")
            else:
                # Extract trial data and compute mean
                trial_data = record_data[:, trial_start:trial_end]
                feature_vec = np.mean(trial_data, axis=1)
                
            # Create trial record
            trial_record = {
                "trial": trial_idx,
                "original_trigger_idx": orig_idx,
                "trigger_sample": trigger_sample,
                "analysis_window": time_window,
                "window_start_ms": start_ms,
                "window_end_ms": end_ms,
                **{f"electrode_{i}": feature_vec[i] for i in range(record_data.shape[0])}
            }
            
            # ✅ ADD SENTENCE
            if trial_idx in trial_to_sentence:
                trial_record['sentence'] = trial_to_sentence[trial_idx]
            else:
                trial_record['sentence'] = None
            
            # Add behavioral data if available
            for col in ['sentence_type', 'modality', 'trigger_type']:
                if col in row:
                    trial_record[col] = row[col]
            
            trial_records.append(trial_record)
        
        print(f"Extracted features for {len(trial_records)} trials")
        print(f"Skipped {skipped_trials} trials due to boundary issues")
        
        self.trial_df = pd.DataFrame(trial_records)
        
        # ✅ VERIFY SENTENCE COLUMN EXISTS
        if 'sentence' in self.trial_df.columns:
            non_null_sentences = self.trial_df['sentence'].notna().sum()
            print(f"✅ Added 'sentence' column: {non_null_sentences}/{len(self.trial_df)} trials have sentences")
        
        return self.trial_df
    
   
        
    def prepare_trigger_dataframe(self):
        """
        Prepare trigger dataframe from curr_state trigger_list
        Recreates the original notebook's trigger processing
        Keeps all columns from trigger_list.
        """
        if not hasattr(self, 'curr_raw'):
            raise ValueError("Behavioral data not loaded. Call load_behav_mat() first.")

        # Extract trigger_list from curr_state
        curr_state = self.curr_raw['curr_state_export']
        if 'trigger_list' in curr_state:
            trigger_list = curr_state['trigger_list']
        else:
            # Fallback to existing trigger_frame
            return self.trigger_frame.copy()

        # If trigger_list is a structured numpy array, keep all columns
        if hasattr(trigger_list, 'dtype') and hasattr(trigger_list.dtype, 'names'):
            all_names = trigger_list.dtype.names  # Keep all column names
            triggers_data = [
                [trigger[name].item() if hasattr(trigger[name], 'item') else trigger[name] 
                for name in all_names]
                for trigger in trigger_list
            ]
            triggers_df = pd.DataFrame(triggers_data, columns=all_names)
        else:
            # Use existing trigger_frame if trigger_list format is different
            triggers_df = self.trigger_frame.copy()

        return triggers_df
    
    def correct_trigger_errors(self, triggers_df):
        """
        Apply known corrections to trigger data - matches original notebook, Issues identified from Victor's notebook
        """
        triggers_df = triggers_df.copy()
        
        # Correct GIF trigger numbers (should be 1, not 2) - from original
        if 'Type' in triggers_df.columns and 'number' in triggers_df.columns:
            triggers_df.loc[triggers_df['Type'] == 'GIF', 'number'] = 1
        
        # Remove duplicate SW_HASH entries - from original  
        if 'Type' in triggers_df.columns:
            duplicate_sw_hash = triggers_df[
                (triggers_df['Type'] == 'SW_HASH') & 
                (triggers_df['Type'].shift(1) == 'SW_HASH')
            ]
            triggers_df = triggers_df.drop(duplicate_sw_hash.index)
            triggers_df = triggers_df.reset_index(drop=True)
            
        return triggers_df
    def align_triggers_from_system_time(self):
        """
        Alternate trigger alignment method using system time if Var1 errors are present
        """
        triggers_df = self.prepare_trigger_dataframe()
        triggers_df = self.correct_trigger_errors(triggers_df)
        
        fs = int(self.hdr_frame['frequency'][0])
        
        # Convert system time (seconds) to sample indices
        # Assuming recording started at the same time as experiment
        triggers_df['event_samples'] = (triggers_df['system_timePreOnset'] * fs).astype(int)
        
        print(f"System time range: {triggers_df['system_timePreOnset'].min():.2f}s to {triggers_df['system_timePreOnset'].max():.2f}s")
        print(f"Converted to samples: {triggers_df['event_samples'].min()} to {triggers_df['event_samples'].max()}")
        
        # Filter valid events within recording bounds
        data_samples = self.record_filter.shape[1]
        valid_mask = (triggers_df['event_samples'] >= 0) & (triggers_df['event_samples'] < data_samples)
        
        num_invalid = (~valid_mask).sum()
        print(f"⚠️ {num_invalid} events fall outside recording bounds and will be excluded")
        
        triggers_df = triggers_df[valid_mask].reset_index(drop=True)
        print(f"Retained {len(triggers_df)} valid triggers")
        
        return triggers_df
        
    def align_triggers_extensive(self, verbose=False):
        """
        Extensive trigger alignment with verification - recreates original notebook logic
        """
        # Prepare trigger dataframe
        triggers_df = self.prepare_trigger_dataframe()
        triggers_df = self.correct_trigger_errors(triggers_df)
        
        # Add computed columns - matches original
        if 'system_timePreOnset' in triggers_df.columns and 'system_timePostTrigger' in triggers_df.columns:
            triggers_df['prev_trigger_delay'] = (
                triggers_df['system_timePreOnset'] - 
                triggers_df['system_timePostTrigger'].shift(1)
            )
            triggers_df['trigger_duration'] = (
                triggers_df['system_timePostTrigger'] - 
                triggers_df['system_timePreOnset']
            )
        
        # For now, use simple matching since we don't have actual events extracted
        # This can be enhanced if you extract events from your neural data
        print("Using simple trigger matching (enhance with event extraction for full verification)")
        return self.align_triggers_simple(triggers_df)
        
    def align_triggers_with_word_timing_var1(self, debug=False):
        """
        Align triggers using Var1 (sample indices) instead of system_time.
        Use this for subjects where system_timePreOnset is unreliable.
        """
        triggers_df = self.prepare_trigger_dataframe()
        triggers_df = self.correct_trigger_errors(triggers_df)
        
        if 'Var1' not in self.trigger_frame.columns:
            print("⚠️ Error: No Var1 column found in trigger_frame")
            self.aligned_trials_df = pd.DataFrame()
            return self.aligned_trials_df
        
        if debug:
            print("\n=== Using Var1 for alignment ===")
            print(f"Columns: {triggers_df.columns.tolist()}")
        
        fs = int(self.hdr_frame['frequency'][0])
        data_samples = self.record_filter.shape[1]
        
        # Get Var1 sample indices
        triggers_df['event_samples'] = self.trigger_frame['Var1'].values
        
        # Filter valid events
        valid_mask = (triggers_df['event_samples'] >= 0) & (triggers_df['event_samples'] < data_samples)
        num_invalid = (~valid_mask).sum()
        
        if num_invalid > 0:
            print(f"⚠️ {num_invalid} events fall outside recording bounds, excluding them")
        
        triggers_df = triggers_df[valid_mask].reset_index(drop=True)
        print(f"Found {len(triggers_df)} valid triggers from Var1")
        print(f"Sample range: {triggers_df['event_samples'].min()} to {triggers_df['event_samples'].max()}")
        
        # Extract word events
        word_pattern = r'WORD(\d)_(VIS|AUD)'
        triggers_df['word_position'] = triggers_df['Type'].str.extract(word_pattern)[0]
        triggers_df['modality_type'] = triggers_df['Type'].str.extract(word_pattern)[1]
        
        word_triggers = triggers_df[triggers_df['word_position'].notna()].copy()
        word_triggers['word_position'] = word_triggers['word_position'].astype(int)
        
        if len(word_triggers) == 0:
            print("ERROR: No word triggers found!")
            self.aligned_trials_df = pd.DataFrame()
            return self.aligned_trials_df
        
        print(f"Found {len(word_triggers)} word triggers")
        
        # Group into trials (1→2→3→4 sequences)
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
                if debug and word_pos != 1:
                    print(f"Unexpected sequence at index {idx}: got {word_pos}, expected {expected_position}")
                current_trial = []
                if word_pos == 1:
                    current_trial = [row]
                    expected_position = 2
                else:
                    expected_position = 1
        
        if len(current_trial) == 4:
            trials.append(current_trial)
        
        print(f"Identified {len(trials)} complete trials (with all 4 words)")
        
        if len(trials) == 0:
            print("⚠️ No complete trials found")
            self.aligned_trials_df = pd.DataFrame()
            return self.aligned_trials_df
        
        # Build trial dataframe
        trial_records = []
        skipped_trials = 0
        
        # Baseline buffer (400ms before WORD1)
        baseline_buffer_ms = 400
        baseline_buffer_samples = int(baseline_buffer_ms * fs / 1000)
        
        for trial_idx, trial_words in enumerate(trials):
            word1 = trial_words[0]
            word4 = trial_words[3]
            
            # Get sample indices from Var1
            word1_sample = int(word1['event_samples'])
            word4_sample = int(word4['event_samples'])
            
            # Trial boundaries: 400ms before WORD1 to 1000ms after WORD4
            trial_start_sample = word1_sample - baseline_buffer_samples
            
            # Estimate WORD4 duration (typical word display ~300ms)
            word4_duration_samples = int(0.3 * fs)
            trial_end_sample = word4_sample + word4_duration_samples + int(1.0 * fs)  # +1000ms buffer
            
            # Validate bounds
            if trial_start_sample < 0 or trial_end_sample >= data_samples:
                if debug and trial_idx < 5:
                    print(f"Trial {trial_idx} out of bounds: {trial_start_sample}-{trial_end_sample}, max: {data_samples}")
                skipped_trials += 1
                continue
            
            # Extract sentence info
            sentence_type = 'Unknown'
            modality = word1['modality_type']
            
            if 'sentence' in word1 and isinstance(word1['sentence'], dict):
                if 'sen_field' in word1['sentence']:
                    sen_field = word1['sentence']['sen_field']
                    if isinstance(sen_field, dict):
                        sentence_type = sen_field.get('sentenceType', 'Unknown')
                        if 'modality' in sen_field:
                            modality = sen_field['modality']
            
            # Store trial info
            trial_record = {
                'trial_idx': trial_idx,
                'trial_start_sample': trial_start_sample,
                'trial_end_sample': trial_end_sample,
                'trial_duration_ms': (trial_end_sample - trial_start_sample) * 1000 / fs,
                'word1_onset_sample': word1_sample,
                'word1_onset_in_trial': word1_sample - trial_start_sample,
                'word4_offset_sample': word4_sample,
                'sentence_type': sentence_type,
                'modality': modality,
            }
            
            # Add all word timings
            for i, word in enumerate(trial_words, 1):
                trial_record[f'word{i}_onset'] = int(word['event_samples'])
                # Estimate offset (onset + typical duration)
                trial_record[f'word{i}_offset'] = int(word['event_samples']) + word4_duration_samples
            
            trial_records.append(trial_record)
        
        self.aligned_trials_df = pd.DataFrame(trial_records)
        print(f"✅ Successfully aligned {len(self.aligned_trials_df)} valid trials using Var1")
        if skipped_trials > 0:
            print(f"⚠️ Skipped {skipped_trials} trials due to boundary issues")
        
        return self.aligned_trials_df
     
    def compute_multitaper_power(self, trial_data, baseline_data, 
                             window_samples, step_samples, fs, freq_range):
        """
        Compute multitaper power spectrograms with baseline normalization
        """
        n_electrodes = trial_data.shape[0]
        trial_length = trial_data.shape[1]
        
        # Calculate number of windows
        n_windows = (trial_length - window_samples) // step_samples + 1
        
        sos_gamma = butter(5, [freq_range[0]/(fs/2), freq_range[1]/(fs/2)], 
                        btype='band', output='sos')
        
        # Filter baseline data for all electrodes
        baseline_filtered = np.zeros_like(baseline_data)
        for elec in range(n_electrodes):
            baseline_filtered[elec, :] = sosfiltfilt(sos_gamma, baseline_data[elec, :])
        
        # Compute baseline statistics (vectorized)
        baseline_power = np.mean(baseline_filtered**2, axis=1)  # Shape: (n_electrodes,)
        baseline_std = np.std(baseline_filtered**2, axis=1)
        baseline_std[baseline_std == 0] = 1e-10  # Avoid division by zero
        
        # Filter trial data for all electrodes
        trial_filtered = np.zeros_like(trial_data)
        for elec in range(n_electrodes):
            trial_filtered[elec, :] = sosfiltfilt(sos_gamma, trial_data[elec, :])
        features_list = []
        
        for win_idx in range(n_windows):
            start_idx = win_idx * step_samples
            end_idx = start_idx + window_samples
            
            if end_idx > trial_length:
                break
            
            # Extract window for ALL electrodes at once
            window_filtered = trial_filtered[:, start_idx:end_idx]
            
            # Compute power for ALL electrodes (vectorized!)
            window_power = np.mean(window_filtered**2, axis=1)  # Shape: (n_electrodes,)
            
            # Z-score (vectorized!)
            z_power = (window_power - baseline_power) / baseline_std
            
            # Build feature dict
            window_features = {
                'time_window_idx': win_idx,
                'time_start_ms': start_idx * 1000 / fs,
                'time_end_ms': end_idx * 1000 / fs,
            }
            
            # Add electrode-specific features
            for elec in range(n_electrodes):
                window_features[f'electrode_{elec}_power'] = window_power[elec]
                window_features[f'electrode_{elec}_z_power'] = z_power[elec]
            
            features_list.append(window_features)
        
        return features_list
    def compute_power_from_filtered(self, trial_filtered, baseline_filtered, 
                               window_samples, step_samples, fs):
        """
        Compute power from ALREADY FILTERED data.
        Dynamically adapts to number of electrodes in the data.
        
        """
        # CRITICAL: Get actual number of electrodes from data
        n_electrodes = trial_filtered.shape[0]
        trial_length = trial_filtered.shape[1]
        
        # Calculate number of windows
        n_windows = (trial_length - window_samples) // step_samples + 1
        
        # ========================================
        # Pre-compute baseline statistics for ALL electrodes
        # ========================================
        # baseline_filtered shape: (n_electrodes, n_baseline_samples)
        baseline_power = np.mean(baseline_filtered**2, axis=1)  # Shape: (n_electrodes,)
        baseline_std = np.std(baseline_filtered**2, axis=1)     # Shape: (n_electrodes,)
        
        # Avoid division by zero
        baseline_std = np.where(baseline_std == 0, 1e-10, baseline_std)
        
        # ========================================
        # Process each time window
        # ========================================
        features_list = []
        
        for win_idx in range(n_windows):
            start_idx = win_idx * step_samples
            end_idx = start_idx + window_samples
            
            if end_idx > trial_length:
                break
            
            # Extract window for ALL electrodes at once
            window_filtered = trial_filtered[:, start_idx:end_idx]
            # window_filtered shape: (n_electrodes, window_samples)
            
            # Compute power for ALL electrodes (vectorized)
            window_power = np.mean(window_filtered**2, axis=1)  # Shape: (n_electrodes,)
            
            # Z-score normalize using baseline (vectorized)
            z_power = (window_power - baseline_power) / baseline_std  # Shape: (n_electrodes,)
            
            # Build feature dictionary for this time window
            window_features = {
                'time_window_idx': win_idx,
                'time_start_ms': start_idx * 1000 / fs,
                'time_end_ms': end_idx * 1000 / fs,
            }
            
            # Add features for each electrode
            for elec in range(n_electrodes):
                window_features[f'electrode_{elec}_power'] = window_power[elec]
                window_features[f'electrode_{elec}_z_power'] = z_power[elec]
            
            features_list.append(window_features)
        
        return features_list
   
    """
    Sliding Window:
    A sliding window in signal processing is a technique that moves a fixed-size "window" or frame across a data stream (like an audio or sensor signal) to analyze localized segments<- 
    take many aggregate measurements over continuous timeseries in window steps
    """
    def extract_sliding_window_features(self, window_ms=200, step_ms=5, 
                            freq_range=(30, 150), baseline_window=(-200, 0)):
        """
        Extract features using sliding window with gamma band filtering.
        """
        if not hasattr(self, 'aligned_trials_df'):
            raise ValueError("Trials not aligned. Call align_triggers_with_word_timing() first")
        
        fs = int(self.hdr_frame['frequency'][0])
        record_data = self.record_filter
        
        n_electrodes = record_data.shape[0]
        n_samples = record_data.shape[1]
        
        print(f"Data info: {n_electrodes} electrodes, {n_samples} samples ({n_samples/fs:.1f}s)")
        
        # Convert time parameters to samples
        window_samples = int(window_ms * fs / 1000)
        step_samples = int(step_ms * fs / 1000)
        baseline_start = int(baseline_window[0] * fs / 1000)
        baseline_end = int(baseline_window[1] * fs / 1000)
        
        print(f"Settings: {window_ms}ms window, {step_ms}ms step, {freq_range[0]}-{freq_range[1]} Hz")
        
        # Create filter ONCE (reuse for all electrodes)
        sos_gamma = butter(5, [freq_range[0]/(fs/2), freq_range[1]/(fs/2)], 
                        btype='band', output='sos')
        
        print("Processing trials (filtering on-the-fly)...")
        
        all_trial_features = []
        
        for _, trial in tqdm(self.aligned_trials_df.iterrows(), 
                            total=len(self.aligned_trials_df),
                            desc="Processing trials"):
            
            trial_start = trial['trial_start_sample']
            trial_end = trial['trial_end_sample']
            word1_onset = trial['word1_onset_sample']
            
            trial_data = record_data[:, trial_start:trial_end]
            
            trial_filtered = np.zeros_like(trial_data, dtype=np.float32)
            for elec in range(n_electrodes):
                trial_filtered[elec, :] = sosfiltfilt(sos_gamma, trial_data[elec, :])
            
            # Compute baseline window (relative to WORD1 onset within trial)
            baseline_start_idx = word1_onset - trial_start + baseline_start
            baseline_end_idx = word1_onset - trial_start + baseline_end
            
            # Validate baseline bounds
            if baseline_start_idx < 0 or baseline_end_idx > trial_filtered.shape[1]:
                if baseline_start_idx < 0:
                    print(f"⚠️ Trial {trial['trial_idx']}: baseline starts before trial (skipping)")
                continue
            
            # Extract baseline data for ALL electrodes
            baseline_filtered = trial_filtered[:, baseline_start_idx:baseline_end_idx]
            
            # Compute power features from filtered data
            trial_features = self.compute_power_from_filtered(
                trial_filtered, 
                baseline_filtered,
                window_samples, 
                step_samples,
                fs
            )
            
            # Add trial metadata to each window
            for feature_dict in trial_features:
                feature_dict.update({
                    'trial_idx': trial['trial_idx'],
                    'sentence_type': trial['sentence_type'],
                    'modality': trial['modality'],
                    'trial_start_sample': trial_start,
                })
            
            all_trial_features.extend(trial_features)
            
            # ✅ Free memory after each trial
            del trial_data, trial_filtered, baseline_filtered
        
        self.features_df = pd.DataFrame(all_trial_features)
        print(f"✅ Extracted {len(self.features_df)} time windows across {len(self.aligned_trials_df)} trials")
        print(f"   Features per window: {len([c for c in self.features_df.columns if 'electrode' in c])} electrode features")
        
        return self.features_df
        
        def compute_multitaper_power(self, trial_data, baseline_data, 
                                    window_samples, step_samples, fs, freq_range):
            """
            Compute multitaper power spectrograms with baseline normalization
            
            Params: time-bandwidth=3, tapers=4-5 
            """
            n_electrodes = trial_data.shape[0]
            trial_length = trial_data.shape[1]
            
            # Calculate number of windows
            n_windows = (trial_length - window_samples) // step_samples + 1
            
            # Initialize storage
            features_list = []
            
            # Compute baseline power for z-scoring
            baseline_power = np.zeros(n_electrodes)
            baseline_std = np.zeros(n_electrodes)
            
            for elec in range(n_electrodes):
                # Use simple bandpass power for baseline (faster than multitaper)
                sos_gamma = butter(5, [freq_range[0]/(fs/2), freq_range[1]/(fs/2)], 
                                btype='band', output='sos')
                baseline_filtered = sosfiltfilt(sos_gamma, baseline_data[elec, :])
                baseline_power[elec] = np.mean(baseline_filtered**2)
                baseline_std[elec] = np.std(baseline_filtered**2)
            
            # Process each time window
            for win_idx in range(n_windows):
                start_idx = win_idx * step_samples
                end_idx = start_idx + window_samples
                
                if end_idx > trial_length:
                    break
                
                window_data = trial_data[:, start_idx:end_idx]
                
                # Compute gamma power for each electrode
                window_features = {
                    'time_window_idx': win_idx,
                    'time_start_ms': start_idx * 1000 / fs,
                    'time_end_ms': end_idx * 1000 / fs,
                }
                
                for elec in range(n_electrodes):
                    # Bandpass and compute power
                    sos_gamma = butter(5, [freq_range[0]/(fs/2), freq_range[1]/(fs/2)], 
                                    btype='band', output='sos')
                    filtered = sosfiltfilt(sos_gamma, window_data[elec, :])
                    power = np.mean(filtered**2)
                    
                    # Z-score relative to baseline
                    if baseline_std[elec] > 0:
                        z_power = (power - baseline_power[elec]) / baseline_std[elec]
                    else:
                        z_power = 0
                    
                    window_features[f'electrode_{elec}_power'] = power
                    window_features[f'electrode_{elec}_z_power'] = z_power
                
                features_list.append(window_features)
            
            return features_list
        
        def align_triggers_simple(self, triggers_df=None):
            """
            Align sequential trigger info from Var1.<- some of these are duplicates, may end up remove this
            """
            # --- Step 0: Prepare trigger frame ---
            if triggers_df is None:
                triggers_df = self.prepare_trigger_dataframe()
                triggers_df = self.correct_trigger_errors(triggers_df)

            if 'Var1' not in self.trigger_frame.columns:
                print("⚠️ Warning: No Var1 column found in trigger_frame for event alignment")
                return triggers_df

            # --- Step 1: Get events from Var1 (these are direct sample indices) ---
            events_var1 = self.trigger_frame['Var1'].values
            print(f"Found {len(events_var1)} events in Var1 column")
            print(f"Sample range: {np.min(events_var1)} to {np.max(events_var1)}")

            # --- Step 2: Check bounds ---
            data_samples = self.record_filter.shape[1]  # 9,000,001
            print(f"Filtered data has {data_samples} samples")

            # Filter out events outside the available data
            valid_mask = (events_var1 >= 0) & (events_var1 < data_samples)
            num_invalid = (~valid_mask).sum()
            
            if num_invalid > 0:
                print(f"⚠️ WARNING: {num_invalid} events fall outside the recording bounds and will be excluded")
                print(f"Invalid events are at samples: {events_var1[~valid_mask][:10]}...")  # Show first 10
                events_var1 = events_var1[valid_mask]
                triggers_df = triggers_df.iloc[valid_mask].copy()

            # --- Step 3: Write events into triggers_df ---
            triggers_df = triggers_df.reset_index(drop=True)
            triggers_df['event_samples'] = events_var1

            print(f"Aligned {len(triggers_df)} valid triggers with Var1 events")

            return triggers_df
        
    def diagnose_triggers(self):
        """
        Diagnostic function to understand trigger structure<- Debugging purpose
        """
        triggers_df = self.prepare_trigger_dataframe()
        triggers_df = self.correct_trigger_errors(triggers_df)
        
        print("="*60)
        print("TRIGGER DIAGNOSTICS")
        print("="*60)
        
        print(f"\n📊 Total triggers: {len(triggers_df)}")
        print(f"\n📋 Columns: {triggers_df.columns.tolist()}")
        
        print("\n🔢 Type distribution:")
        print(triggers_df['Type'].value_counts())
        
        print("\n🔢 Number distribution:")
        print(triggers_df['number'].value_counts().sort_index())
        
        print("\n📝 Sample rows (first 20):")
        print(triggers_df[['Type', 'number', 'system_timePreOnset', 'system_timePostTrigger']].head(20))
        
        print("\n🔍 Looking for word sequences...")
        # Find potential word triggers
        for type_val in triggers_df['Type'].unique():
            type_df = triggers_df[triggers_df['Type'] == type_val]
            numbers = type_df['number'].dropna().unique()
            if len(numbers) > 0:
                print(f"  Type='{type_val}': has numbers {sorted(numbers)}")
        
        # Check for complete 1-2-3-4 sequences
        word_mask = triggers_df['number'].isin([1, 2, 3, 4])
        word_df = triggers_df[word_mask]
        print(f"\n✅ Found {len(word_df)} triggers with number in [1,2,3,4]")
        
        if len(word_df) > 0:
            print("\n📋 First 20 word triggers:")
            print(word_df[['Type', 'number', 'system_timePreOnset']].head(20))
        
        print("\n" + "="*60)
        
        return triggers_df
    
    def align_triggers_with_word_timing(self, debug=False, diagnose_bounds=False, recording_start_offset=None):
        """
        Align triggers using system_timePreOnset and system_timePostTrigger
        as specified in the instructions (Primary method for decoding)
        
        Parameters
        ----------
        debug : bool
            Print diagnostic information
        diagnose_bounds : bool
            Print detailed boundary checking info for first few trials
        recording_start_offset : float or None
            Time offset to subtract from all trigger times to align with recording start.
            If None, will attempt to auto-detect using experiment start time.
        """
        triggers_df = self.prepare_trigger_dataframe()
        triggers_df = self.correct_trigger_errors(triggers_df)
        
        if debug:
            print("\n=== DEBUG: Trigger DataFrame Info ===")
            print(f"Columns: {triggers_df.columns.tolist()}")
            print(f"\nFirst 20 rows:")
            print(triggers_df[['Type', 'number', 'system_timePreOnset', 'system_timePostTrigger']].head(20))
            if 'Type' in triggers_df.columns:
                print(f"\nUnique Type values: {triggers_df['Type'].unique()}")
        
        fs = int(self.hdr_frame['frequency'][0])
        
        # Check if we need to apply time offset
        if recording_start_offset is None:
            # Auto-detect: use the first trigger time as approximate recording start
            # Or try to get experiment start time
            exp_start = self.get_experiment_start_time()
            
            if exp_start is not None and hasattr(self, 'hdr_raw'):
                # Try to get recording start from hdr
                if 'startdate' in self.hdr_raw['hdr'] and 'starttime' in self.hdr_raw['hdr']:
                    try:
                        startdate = str(self.hdr_raw['hdr']['startdate'])
                        starttime = str(self.hdr_raw['hdr']['starttime'])
                        rec_start = datetime.datetime.strptime(f"{startdate} {starttime}", '%d-%b-%Y %H:%M:%S')
                        
                        # Calculate offset in seconds
                        recording_start_offset = (rec_start - datetime.datetime(1970, 1, 1)).total_seconds()
                        print(f"Using recording start time from hdr: {rec_start}")
                        print(f"   Offset: {recording_start_offset:.2f}s")
                    except Exception as e:
                        print(f"⚠️ Could not parse recording start time: {e}")
            
            # Fallback: Use first START trigger or first trigger as offset
            if recording_start_offset is None:
                start_triggers = triggers_df[triggers_df['Type'] == 'START']
                if len(start_triggers) > 0:
                    recording_start_offset = start_triggers.iloc[0]['system_timePreOnset']
                    print(f"Using first START trigger as recording offset: {recording_start_offset:.2f}s")
                else:
                    recording_start_offset = triggers_df.iloc[0]['system_timePreOnset']
                    print(f"Using first trigger as recording offset: {recording_start_offset:.2f}s")
        
        # Apply offset to all trigger times
        triggers_df['system_timePreOnset_aligned'] = triggers_df['system_timePreOnset'] - recording_start_offset
        triggers_df['system_timePostTrigger_aligned'] = triggers_df['system_timePostTrigger'] - recording_start_offset
        
        if debug:
            print(f"\nAfter alignment:")
            print(f"   Original time range: {triggers_df['system_timePreOnset'].min():.2f}s - {triggers_df['system_timePreOnset'].max():.2f}s")
            print(f"   Aligned time range: {triggers_df['system_timePreOnset_aligned'].min():.2f}s - {triggers_df['system_timePreOnset_aligned'].max():.2f}s")
        
        # The word position is encoded in the Type column itself:
        # WORD1_VIS, WORD2_VIS, WORD3_VIS, WORD4_VIS (visual modality)
        # WORD1_AUD, WORD2_AUD, WORD3_AUD, WORD4_AUD (auditory modality)
        
        # Extract word events and add word_position column
        word_pattern = r'WORD(\d)_(VIS|AUD)'
        triggers_df['word_position'] = triggers_df['Type'].str.extract(word_pattern)[0]
        triggers_df['modality_type'] = triggers_df['Type'].str.extract(word_pattern)[1]
        
        # Filter to only word triggers
        word_triggers = triggers_df[triggers_df['word_position'].notna()].copy()
        word_triggers['word_position'] = word_triggers['word_position'].astype(int)
        
        if len(word_triggers) == 0:
            print("⚠️ No word triggers found!")
            self.aligned_trials_df = pd.DataFrame()
            return self.aligned_trials_df
        
        print(f"Found {len(word_triggers)} word triggers")
        if debug:
            print("\nWord position distribution:")
            print(word_triggers['word_position'].value_counts().sort_index())
            print("\nModality distribution:")
            print(word_triggers['modality_type'].value_counts())
        
        # Group consecutive word events into trials
        # A complete trial has word_position: 1, 2, 3, 4 in sequence
        trials = []
        current_trial = []
        expected_position = 1
        
        for idx, row in word_triggers.iterrows():
            word_pos = row['word_position']
            
            if word_pos == 1:
                # Start new trial
                if len(current_trial) == 4:  # Previous trial was complete
                    trials.append(current_trial)
                current_trial = [row]
                expected_position = 2
            elif word_pos == expected_position and len(current_trial) > 0:
                # Continue current trial
                current_trial.append(row)
                expected_position += 1
                
                # Complete trial when we have all 4 words
                if len(current_trial) == 4:
                    trials.append(current_trial)
                    current_trial = []
                    expected_position = 1
            else:
                # Unexpected sequence - reset
                if debug and word_pos != 1:
                    print(f"Unexpected word sequence at index {idx}: got {word_pos}, expected {expected_position}")
                current_trial = []
                if word_pos == 1:
                    current_trial = [row]
                    expected_position = 2
                else:
                    expected_position = 1
        
        # Add last trial if complete
        if len(current_trial) == 4:
            trials.append(current_trial)
        
        print(f"Identified {len(trials)} complete trials (with all 4 words)")
        
        if len(trials) == 0:
            print("⚠️ No complete trials found.")
            if debug:
                print("\nShowing first 40 word triggers:")
                print(word_triggers[['Type', 'word_position', 'system_timePreOnset_aligned']].head(40))
            self.aligned_trials_df = pd.DataFrame()
            return self.aligned_trials_df
        
        # Build trial dataframe with proper timing
        trial_records = []
        data_samples = self.record_filter.shape[1]
        skipped_trials = 0
        
        # We need to include baseline period BEFORE word1 onset
        baseline_buffer_ms = 400  # 400ms before WORD1 for baseline (-200 to 0 uses first 200ms)
        baseline_buffer_samples = int(baseline_buffer_ms * fs / 1000)
        
        # DIAGNOSTIC: Check recording timing
        if diagnose_bounds:
            print(f"\n=== BOUNDARY DIAGNOSTIC ===")
            print(f"Recording has {data_samples} samples at {fs} Hz = {data_samples/fs:.1f} seconds")
            print(f"First trial WORD1 time (aligned): {trials[0][0]['system_timePreOnset_aligned']:.2f}s")
            print(f"Last trial WORD4 time (aligned): {trials[-1][3]['system_timePostTrigger_aligned']:.2f}s")
            print(f"Baseline buffer: {baseline_buffer_ms}ms = {baseline_buffer_samples} samples")
        
        for trial_idx, trial_words in enumerate(trials):
            word1 = trial_words[0]
            word4 = trial_words[3]
            
            # PRIMARY METHOD: Start baseline_buffer BEFORE WORD1, end at WORD4 + 1000ms
            # USE ALIGNED TIMES
            word1_onset_time = word1['system_timePreOnset_aligned']
            trial_start_time = word1_onset_time - (baseline_buffer_ms / 1000)  # Start 400ms before WORD1
            word1_to_consolidation_end = 4.5  # 4500ms after Word1
            trial_end_time = word1_onset_time + word1_to_consolidation_end + 0.5 
            
            # Convert to samples
            trial_start_sample = int(trial_start_time * fs)
            trial_end_sample = int(trial_end_time * fs)
            word1_onset_sample = int(word1_onset_time * fs)
            
            # DIAGNOSTIC: Show first few trials
            if diagnose_bounds and trial_idx < 5:
                print(f"\nTrial {trial_idx}:")
                print(f"  WORD1 time: {word1_onset_time:.2f}s (sample {word1_onset_sample})")
                print(f"  Trial start time: {trial_start_time:.2f}s (sample {trial_start_sample})")
                print(f"  Trial end time: {trial_end_time:.2f}s (sample {trial_end_sample})")
                print(f"  Recording range: 0 to {data_samples}")
                print(f"  Start valid: {trial_start_sample >= 0}")
                print(f"  End valid: {trial_end_sample < data_samples}")
            
            # Validate bounds
            if trial_start_sample < 0 or trial_end_sample >= data_samples:
                if debug and trial_idx < 5:
                    print(f"ERROR: Trial {trial_idx} out of bounds: {trial_start_sample}-{trial_end_sample}, max: {data_samples}")
                skipped_trials += 1
                continue
            
            # Extract sentence info from word1 (if available)
            sentence_type = 'Unknown'
            modality = word1['modality_type']  # VIS or AUD
            
            if 'sentence' in word1 and isinstance(word1['sentence'], dict):
                if 'sen_field' in word1['sentence']:
                    sen_field = word1['sentence']['sen_field']
                    if isinstance(sen_field, dict):
                        sentence_type = sen_field.get('sentenceType', 'Unknown')
                        # modality from sen_field might be more detailed than VIS/AUD
                        if 'modality' in sen_field:
                            modality = sen_field['modality']
            
            # Store trial info
            trial_record = {
                'trial_idx': trial_idx,
                'trial_start_sample': trial_start_sample,
                'trial_end_sample': trial_end_sample,
                'trial_duration_ms': (trial_end_time - trial_start_time) * 1000,
                'word1_onset_sample': word1_onset_sample,  # Absolute sample index
                'word1_onset_in_trial': word1_onset_sample - trial_start_sample,  # Relative to trial start
                'word4_offset_sample': int(word4['system_timePostTrigger_aligned'] * fs),
                'sentence_type': sentence_type,
                'modality': modality,
            }
            
            # Add word timings
            for i, word in enumerate(trial_words, 1):
                trial_record[f'word{i}_onset'] = int(word['system_timePreOnset_aligned'] * fs)
                trial_record[f'word{i}_offset'] = int(word['system_timePostTrigger_aligned'] * fs)
            
            trial_records.append(trial_record)
        
        self.aligned_trials_df = pd.DataFrame(trial_records)
        print(f"✅ Successfully aligned {len(self.aligned_trials_df)} valid trials")
        if skipped_trials > 0:
            print(f"⚠️ Skipped {skipped_trials} trials due to boundary issues")
        
        # Store the offset for reference
        self.recording_start_offset = recording_start_offset
        
        return self.aligned_trials_df
        """
        Align triggers using system_timePreOnset and system_timePostTrigger
        as specified in the instructions (Primary method for decoding)
        
        Parameters
        ----------
        debug : bool
            Print diagnostic information
        diagnose_bounds : bool
            Print detailed boundary checking info for first few trials
        """
        triggers_df = self.prepare_trigger_dataframe()
        triggers_df = self.correct_trigger_errors(triggers_df)
        
        if debug:
            print("\n=== DEBUG: Trigger DataFrame Info ===")
            print(f"Columns: {triggers_df.columns.tolist()}")
            print(f"\nFirst 20 rows:")
            print(triggers_df[['Type', 'number', 'system_timePreOnset', 'system_timePostTrigger']].head(20))
            if 'Type' in triggers_df.columns:
                print(f"\nUnique Type values: {triggers_df['Type'].unique()}")
        
        fs = int(self.hdr_frame['frequency'][0])
        
        # The word position is encoded in the Type column itself:
        # WORD1_VIS, WORD2_VIS, WORD3_VIS, WORD4_VIS (visual modality)
        # WORD1_AUD, WORD2_AUD, WORD3_AUD, WORD4_AUD (auditory modality)
        
        # Extract word events and add word_position column
        word_pattern = r'WORD(\d)_(VIS|AUD)'
        triggers_df['word_position'] = triggers_df['Type'].str.extract(word_pattern)[0]
        triggers_df['modality_type'] = triggers_df['Type'].str.extract(word_pattern)[1]
        
        # Filter to only word triggers
        word_triggers = triggers_df[triggers_df['word_position'].notna()].copy()
        word_triggers['word_position'] = word_triggers['word_position'].astype(int)
        
        if len(word_triggers) == 0:
            print("⚠️ No word triggers found!")
            self.aligned_trials_df = pd.DataFrame()
            return self.aligned_trials_df
        
        print(f"Found {len(word_triggers)} word triggers")
        if debug:
            print("\nWord position distribution:")
            print(word_triggers['word_position'].value_counts().sort_index())
            print("\nModality distribution:")
            print(word_triggers['modality_type'].value_counts())
        
        # Group consecutive word events into trials
        # A complete trial has word_position: 1, 2, 3, 4 in sequence
        trials = []
        current_trial = []
        expected_position = 1
        
        for idx, row in word_triggers.iterrows():
            word_pos = row['word_position']
            
            if word_pos == 1:
                # Start new trial
                if len(current_trial) == 4:  # Previous trial was complete
                    trials.append(current_trial)
                current_trial = [row]
                expected_position = 2
            elif word_pos == expected_position and len(current_trial) > 0:
                # Continue current trial
                current_trial.append(row)
                expected_position += 1
                
                # Complete trial when we have all 4 words
                if len(current_trial) == 4:
                    trials.append(current_trial)
                    current_trial = []
                    expected_position = 1
            else:
                # Unexpected sequence - reset
                if debug and word_pos != 1:
                    print(f"Unexpected word sequence at index {idx}: got {word_pos}, expected {expected_position}")
                current_trial = []
                if word_pos == 1:
                    current_trial = [row]
                    expected_position = 2
                else:
                    expected_position = 1
        
        # Add last trial if complete
        if len(current_trial) == 4:
            trials.append(current_trial)
        
        print(f"Identified {len(trials)} complete trials (with all 4 words)")
        
        if len(trials) == 0:
            print("⚠️ No complete trials found.")
            if debug:
                print("\nShowing first 40 word triggers:")
                print(word_triggers[['Type', 'word_position', 'system_timePreOnset']].head(40))
            self.aligned_trials_df = pd.DataFrame()
            return self.aligned_trials_df
        
        # Build trial dataframe with proper timing
        trial_records = []
        data_samples = self.record_filter.shape[1]
        skipped_trials = 0
        
        # We need to include baseline period BEFORE word1 onset
        baseline_buffer_ms = 400  # 400ms before WORD1 for baseline (-200 to 0 uses first 200ms)
        baseline_buffer_samples = int(baseline_buffer_ms * fs / 1000)
        
        # DIAGNOSTIC: Check recording timing
        if diagnose_bounds:
            print(f"\n=== BOUNDARY DIAGNOSTIC ===")
            print(f"Recording has {data_samples} samples at {fs} Hz = {data_samples/fs:.1f} seconds")
            print(f"First trial WORD1 time: {trials[0][0]['system_timePreOnset']:.2f}s")
            print(f"Last trial WORD4 time: {trials[-1][3]['system_timePostTrigger']:.2f}s")
            print(f"Baseline buffer: {baseline_buffer_ms}ms = {baseline_buffer_samples} samples")
        
        for trial_idx, trial_words in enumerate(trials):
            word1 = trial_words[0]
            word4 = trial_words[3]
            
            # PRIMARY METHOD: Start baseline_buffer BEFORE WORD1, end at WORD4 + 1000ms
            word1_onset_time = word1['system_timePreOnset']
            trial_start_time = word1_onset_time - (baseline_buffer_ms / 1000)  # Start 400ms before WORD1
            print("I reach here")
            word1_to_consolidation_end = 4.5  # 4500ms after Word1
            trial_end_time = word1_onset_time + word1_to_consolidation_end + 0.5 
            
            # Convert to samples
            trial_start_sample = int(trial_start_time * fs)
            trial_end_sample = int(trial_end_time * fs)
            word1_onset_sample = int(word1_onset_time * fs)
            
            # DIAGNOSTIC: Show first few trials
            if diagnose_bounds and trial_idx < 5:
                print(f"\nTrial {trial_idx}:")
                print(f"  WORD1 time: {word1_onset_time:.2f}s (sample {word1_onset_sample})")
                print(f"  Trial start time: {trial_start_time:.2f}s (sample {trial_start_sample})")
                print(f"  Trial end time: {trial_end_time:.2f}s (sample {trial_end_sample})")
                print(f"  Recording range: 0 to {data_samples}")
                print(f"  Start valid: {trial_start_sample >= 0}")
                print(f"  End valid: {trial_end_sample < data_samples}")
            
            # Validate bounds
            if trial_start_sample < 0 or trial_end_sample >= data_samples:
                if debug and trial_idx < 5:
                    print(f"❌ Trial {trial_idx} out of bounds: {trial_start_sample}-{trial_end_sample}, max: {data_samples}")
                skipped_trials += 1
                continue
            
            # Extract sentence info from word1 (if available)
            sentence_type = 'Unknown'
            modality = word1['modality_type']  # VIS or AUD
            
            if 'sentence' in word1 and isinstance(word1['sentence'], dict):
                if 'sen_field' in word1['sentence']:
                    sen_field = word1['sentence']['sen_field']
                    if isinstance(sen_field, dict):
                        sentence_type = sen_field.get('sentenceType', 'Unknown')
                        # modality from sen_field might be more detailed than VIS/AUD
                        if 'modality' in sen_field:
                            modality = sen_field['modality']
            
            # Store trial info
            trial_record = {
                'trial_idx': trial_idx,
                'trial_start_sample': trial_start_sample,
                'trial_end_sample': trial_end_sample,
                'trial_duration_ms': (trial_end_time - trial_start_time) * 1000,
                'word1_onset_sample': word1_onset_sample,  # Absolute sample index
                'word1_onset_in_trial': word1_onset_sample - trial_start_sample,  # Relative to trial start
                'word4_offset_sample': int(word4['system_timePostTrigger'] * fs),
                'sentence_type': sentence_type,
                'modality': modality,
            }
            
            # Add word timings
            for i, word in enumerate(trial_words, 1):
                trial_record[f'word{i}_onset'] = int(word['system_timePreOnset'] * fs)
                trial_record[f'word{i}_offset'] = int(word['system_timePostTrigger'] * fs)
            
            trial_records.append(trial_record)
        
        self.aligned_trials_df = pd.DataFrame(trial_records)
        print(f"✅ Successfully aligned {len(self.aligned_trials_df)} valid trials")
        if skipped_trials > 0:
            print(f"⚠️ Skipped {skipped_trials} trials due to boundary issues")
        
        return self.aligned_trials_df
        """
        Align triggers using system_timePreOnset and system_timePostTrigger
        as specified in the instructions (Primary method for decoding)
        
        Parameters
        ----------
        debug : bool
            Print diagnostic information
        """
        triggers_df = self.prepare_trigger_dataframe()
        triggers_df = self.correct_trigger_errors(triggers_df)
        
        if debug:
            print("\n=== DEBUG: Trigger DataFrame Info ===")
            print(f"Columns: {triggers_df.columns.tolist()}")
            print(f"\nFirst 20 rows:")
            print(triggers_df[['Type', 'number', 'system_timePreOnset', 'system_timePostTrigger']].head(20))
            if 'Type' in triggers_df.columns:
                print(f"\nUnique Type values: {triggers_df['Type'].unique()}")
        
        fs = int(self.hdr_frame['frequency'][0])
        
        # The word position is encoded in the Type column itself:
        # WORD1_VIS, WORD2_VIS, WORD3_VIS, WORD4_VIS (visual modality)
        # WORD1_AUD, WORD2_AUD, WORD3_AUD, WORD4_AUD (auditory modality)
        
        # Extract word events and add word_position column
        word_pattern = r'WORD(\d)_(VIS|AUD)'
        triggers_df['word_position'] = triggers_df['Type'].str.extract(word_pattern)[0]
        triggers_df['modality_type'] = triggers_df['Type'].str.extract(word_pattern)[1]
        
        # Filter to only word triggers
        word_triggers = triggers_df[triggers_df['word_position'].notna()].copy()
        word_triggers['word_position'] = word_triggers['word_position'].astype(int)
        
        if len(word_triggers) == 0:
            print("⚠️ No word triggers found!")
            self.aligned_trials_df = pd.DataFrame()
            return self.aligned_trials_df
        
        print(f"Found {len(word_triggers)} word triggers")
        if debug:
            print("\nWord position distribution:")
            print(word_triggers['word_position'].value_counts().sort_index())
            print("\nModality distribution:")
            print(word_triggers['modality_type'].value_counts())
        
        # Group consecutive word events into trials
        # A complete trial has word_position: 1, 2, 3, 4 in sequence
        trials = []
        current_trial = []
        expected_position = 1
        
        for idx, row in word_triggers.iterrows():
            word_pos = row['word_position']
            
            if word_pos == 1:
                # Start new trial
                if len(current_trial) == 4:  # Previous trial was complete
                    trials.append(current_trial)
                current_trial = [row]
                expected_position = 2
            elif word_pos == expected_position and len(current_trial) > 0:
                # Continue current trial
                current_trial.append(row)
                expected_position += 1
                
                # Complete trial when we have all 4 words
                if len(current_trial) == 4:
                    trials.append(current_trial)
                    current_trial = []
                    expected_position = 1
            else:
                # Unexpected sequence - reset
                if debug and word_pos != 1:
                    print(f"Unexpected word sequence at index {idx}: got {word_pos}, expected {expected_position}")
                current_trial = []
                if word_pos == 1:
                    current_trial = [row]
                    expected_position = 2
                else:
                    expected_position = 1
        
        # Add last trial if complete
        if len(current_trial) == 4:
            trials.append(current_trial)
        
        print(f"Identified {len(trials)} complete trials (with all 4 words)")
        
        if len(trials) == 0:
            print("⚠️ No complete trials found.")
            if debug:
                print("\nShowing first 40 word triggers:")
                print(word_triggers[['Type', 'word_position', 'system_timePreOnset']].head(40))
            self.aligned_trials_df = pd.DataFrame()
            return self.aligned_trials_df
        
        # Build trial dataframe with proper timing
        trial_records = []
        data_samples = self.record_filter.shape[1]
        skipped_trials = 0
        
        # We need to include baseline period BEFORE word1 onset
        baseline_buffer_ms = 400  # 400ms before WORD1 for baseline (-200 to 0 uses first 200ms)
        baseline_buffer_samples = int(baseline_buffer_ms * fs / 1000)
        
        for trial_idx, trial_words in enumerate(trials):
            word1 = trial_words[0]
            word4 = trial_words[3]
            
            # PRIMARY METHOD: Start baseline_buffer BEFORE WORD1, end at WORD4 + 1000ms
            word1_onset_time = word1['system_timePreOnset']
            trial_start_time = word1_onset_time - (baseline_buffer_ms / 1000)  # Start 400ms before WORD1
            trial_end_time = word4['system_timePostTrigger'] + 1.0  # +1000ms after WORD4
            
            # Convert to samples
            trial_start_sample = int(trial_start_time * fs)
            trial_end_sample = int(trial_end_time * fs)
            word1_onset_sample = int(word1_onset_time * fs)
            
            # Validate bounds
            if trial_start_sample < 0 or trial_end_sample >= data_samples:
                if debug:
                    print(f"Trial {trial_idx} out of bounds: {trial_start_sample}-{trial_end_sample}, max: {data_samples}")
                skipped_trials += 1
                continue
            
            # Extract sentence info from word1 (if available)
            sentence_type = 'Unknown'
            modality = word1['modality_type']  # VIS or AUD
            
            if 'sentence' in word1 and isinstance(word1['sentence'], dict):
                if 'sen_field' in word1['sentence']:
                    sen_field = word1['sentence']['sen_field']
                    if isinstance(sen_field, dict):
                        sentence_type = sen_field.get('sentenceType', 'Unknown')
                        # modality from sen_field might be more detailed than VIS/AUD
                        if 'modality' in sen_field:
                            modality = sen_field['modality']
            
            # Store trial info
            trial_record = {
                'trial_idx': trial_idx,
                'trial_start_sample': trial_start_sample,
                'trial_end_sample': trial_end_sample,
                'trial_duration_ms': (trial_end_time - trial_start_time) * 1000,
                'word1_onset_sample': word1_onset_sample,  # Absolute sample index
                'word1_onset_in_trial': word1_onset_sample - trial_start_sample,  # Relative to trial start
                'word4_offset_sample': int(word4['system_timePostTrigger'] * fs),
                'sentence_type': sentence_type,
                'modality': modality,
            }
            
            # Add word timings
            for i, word in enumerate(trial_words, 1):
                trial_record[f'word{i}_onset'] = int(word['system_timePreOnset'] * fs)
                trial_record[f'word{i}_offset'] = int(word['system_timePostTrigger'] * fs)
            
            trial_records.append(trial_record)
        
        self.aligned_trials_df = pd.DataFrame(trial_records)
        print(f"✅ Successfully aligned {len(self.aligned_trials_df)} valid trials")
        if skipped_trials > 0:
            print(f"⚠️ Skipped {skipped_trials} trials due to boundary issues")
        
        return self.aligned_trials_df
        
        # Build trial dataframe with proper timing
        trial_records = []
        data_samples = self.record_filter.shape[1]
        
        for trial_idx, trial_words in enumerate(trials):
            word1 = trial_words[0]
            word4 = trial_words[-1]
            
            # PRIMARY METHOD: WORD1 start to WORD4 end + 1000ms
            trial_start_time = word1['system_timePreOnset']
            trial_end_time = word4['system_timePostTrigger'] + 1.0  # +1000ms
            
            # Convert to samples
            trial_start_sample = int(trial_start_time * fs)
            trial_end_sample = int(trial_end_time * fs)
            
            # Validate bounds
            if trial_start_sample < 0 or trial_end_sample >= data_samples:
                print(f"Trial {trial_idx} out of bounds: {trial_start_sample}-{trial_end_sample}")
                continue
            
            # Store trial info
            trial_record = {
                'trial_idx': trial_idx,
                'trial_start_sample': trial_start_sample,
                'trial_end_sample': trial_end_sample,
                'trial_duration_ms': (trial_end_time - trial_start_time) * 1000,
                'word1_onset_sample': int(word1['system_timePreOnset'] * fs),
                'word4_offset_sample': int(word4['system_timePostTrigger'] * fs),
                'sentence_type': word1.get('sentenceType', 'Unknown'),
                'modality': word1.get('modality', 'Unknown'),
            }
            
            # Add word timings
            for i, word in enumerate(trial_words[:4], 1):
                trial_record[f'word{i}_onset'] = int(word['system_timePreOnset'] * fs)
                trial_record[f'word{i}_offset'] = int(word['system_timePostTrigger'] * fs)
            
            trial_records.append(trial_record)
        
        self.aligned_trials_df = pd.DataFrame(trial_records)
        print(f"Aligned {len(self.aligned_trials_df)} valid trials")
        return self.aligned_trials_df
        
        def compute_multitaper_power(self, trial_data, baseline_data, 
                                window_samples, step_samples, fs, freq_range):
            
            n_electrodes = trial_data.shape[0]
            trial_length = trial_data.shape[1]
            
            n_windows = (trial_length - window_samples) // step_samples + 1
            features_list = []
            
            # Design bandpass filter for gamma band
            sos_gamma = butter(5, [freq_range[0]/(fs/2), freq_range[1]/(fs/2)], 
                            btype='band', output='sos')
        
            trial_filtered = np.zeros_like(trial_data)
            baseline_filtered = np.zeros_like(baseline_data)
            
            for elec in range(n_electrodes):
                trial_filtered[elec, :] = sosfiltfilt(sos_gamma, trial_data[elec, :])
                baseline_filtered[elec, :] = sosfiltfilt(sos_gamma, baseline_data[elec, :])
            
            # Compute baseline statistics
            baseline_power = np.mean(baseline_filtered**2, axis=1)  # (n_electrodes,)
            baseline_std = np.std(baseline_filtered**2, axis=1)     # (n_electrodes,)
            
            # ===== Now just compute power on pre-filtered data =====
            for win_idx in range(n_windows):
                start_idx = win_idx * step_samples
                end_idx = start_idx + window_samples
                
                if end_idx > trial_length:
                    break
                
                window_filtered = trial_filtered[:, start_idx:end_idx]
                
                # Vectorized power computation across all electrodes at once
                power = np.mean(window_filtered**2, axis=1)  # (n_electrodes,)
                
                # Vectorized z-scoring
                z_power = np.where(baseline_std > 0, 
                                (power - baseline_power) / baseline_std, 
                                0)
                
                window_features = {
                    'time_window_idx': win_idx,
                    'time_start_ms': start_idx * 1000 / fs,
                    'time_end_ms': end_idx * 1000 / fs,
                }
                
                # Add electrode features
                for elec in range(n_electrodes):
                    window_features[f'electrode_{elec}_power'] = power[elec]
                    window_features[f'electrode_{elec}_z_power'] = z_power[elec]
                
                features_list.append(window_features)
        
                return features_list
            
    def extract_epochs(self, lock_to="word1", pre_time=0.4, post_time=4.9):
        """
        Extract epochs locked to a specific word onset per trial
        lock_to: "word1", "word2", "word3", or "word4"
        """
        if not hasattr(self, "aligned_trials_df"):
            raise ValueError("Run align_triggers_with_word_timing() first")

        fs = int(self.hdr_frame['frequency'][0])
        raw = self.record_filter

        epochs = []
        times = np.linspace(
            -pre_time,
            post_time,
            int((pre_time + post_time) * fs),
            endpoint=False
        )

        onset_col = f"{lock_to}_onset"

        for _, trial in self.aligned_trials_df.iterrows():
            onset = int(trial[onset_col])

            start = onset - int(pre_time * fs)
            end   = onset + int(post_time * fs)

            if start < 0 or end > raw.shape[1]:
                continue

            epochs.append(raw[:, start:end])
            print("This is epochs", epochs)
        self.epochs = np.stack(epochs, axis=0)  # (trials, channels, time)
        self.epoch_times = times
        print(f"Extracted {self.epochs.shape[0]} epochs locked to {lock_to}")
   
    
    def align_trials_smart(self, debug=False):
        """Try system_time first, fall back to Var1 if it fails<- WRAPPER function for alignment methods"""
        # Try system_time method
        aligned = self.align_triggers_with_word_timing(debug=debug)
        
        # If no trials found, try Var1 method
        if len(aligned) == 0:
            print("\n system_time alignment failed, trying Var1 method...")
            aligned = self.align_triggers_with_word_timing_var1(debug=debug)
        
        return aligned
    
    def extract_alternative_method_features(self):
        """
        Alternative method from instructions:
        -400ms to +1100ms around word onset
        Used for spectrograms
        
        Extracts features for EACH word (4 epochs per trial)
        """
        if not hasattr(self, 'aligned_trials_df'):
            raise ValueError("Trials not aligned. Call align_triggers_with_word_timing() first")
        
        fs = int(self.hdr_frame['frequency'][0])
        record_data = self.record_filter
        
        pre_onset_ms = -400
        post_onset_ms = 1100
        pre_samples = int(pre_onset_ms * fs / 1000)
        post_samples = int(post_onset_ms * fs / 1000)
        
        word_features = []
        
        for _, trial in self.aligned_trials_df.iterrows():
            # Extract epoch for each of the 4 words
            for word_num in range(1, 5):
                word_onset_col = f'word{word_num}_onset'
                
                if word_onset_col not in trial:
                    continue
                
                word_onset = trial[word_onset_col]
                
                epoch_start = word_onset + pre_samples
                epoch_end = word_onset + post_samples
                
                if epoch_start < 0 or epoch_end >= record_data.shape[1]:
                    continue
                
                epoch_data = record_data[:, epoch_start:epoch_end]
                
                # Compute mean power per electrode
                feature_dict = {
                    'trial_idx': trial['trial_idx'],
                    'word_position': word_num,
                    'word_onset_sample': word_onset,
                    'sentence_type': trial['sentence_type'],
                    'modality': trial['modality'],
                    'method': 'alternative_word_locked',
                    'epoch_start_ms': pre_onset_ms,
                    'epoch_end_ms': post_onset_ms,
                }
                
                for elec in range(record_data.shape[0]):
                    feature_dict[f'electrode_{elec}'] = np.mean(epoch_data[elec, :])
                
                word_features.append(feature_dict)
        
        print(f"Extracted alternative method features for {len(word_features)} word epochs")
        return pd.DataFrame(word_features)
    