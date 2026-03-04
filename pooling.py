import numpy as np
from tqdm import tqdm
import os
import torch
from helpers.helpers import *
from helpers.constants import *
from helpers.Subject import Subject
from helpers.ridge import RidgePerElectrode
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer, AutoModel, T5Model
import torch
import numpy as np
import pandas as pd
import time
import pickle
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split, KFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GroupShuffleSplit
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm
import matplotlib as plt
import warnings
warnings.filterwarnings('ignore')


#pooling.py is a bad name for this file <- consider changing it
#Globals
EMPTY_DICT = {'w1': '', 'w2': '', 'w3': '', 'w4': '', 'EN_translation': '', 
                'imageFile': '', 'falseImageFile': '', 'relatedImage': '', 
                'sentenceType': '', 'w1Type': '', 'w2Type': '', 'w3Type': '', 
                'w4Type': '', 'w1SoundFile': '', 'w2SoundFile': '', 'w3SoundFile': '', 
                'w4SoundFile': '', 'modality': ''}
    
WORD_KEYS = {
    'w1': ['w1', 'word1', 'W1', 'Word1'],
    'w2': ['w2', 'word2', 'W2', 'Word2'],
    'w3': ['w3', 'word3', 'W3', 'Word3'],
    'w4': ['w4', 'word4', 'W4', 'Word4']
}

TYPE_KEYS = {
    'w1Type': ['w1Type', 'word1type', 'w1type', 'W1Type', 'Word1Type'],
    'w2Type': ['w2Type', 'word2type', 'w2type', 'W2Type', 'Word2Type'],
    'w3Type': ['w3Type', 'word3type', 'w3type', 'W3Type', 'Word3Type'],
    'w4Type': ['w4Type', 'word4type', 'w4type', 'W4Type', 'Word4Type']
}

OTHER_KEYS = {
    'falseImageFile': ['falseImageFile', 'falseimageFile', 'FalseImageFile']
}

#Used for prepare sliding window function
OFFSET_MS = 600 #Relative to fixation period, prestimulus period

# Define windows relative to Word1 onset
WINDOWS_RELATIVE_TO_WORD1 = {
    # Individual word presentation windows (875ms each):
    'W1': (0, 875),          # Word1: 0-875ms after Word1 onset
    'W2': (875, 1750),       # Word2: 875-1750ms after Word1 onset
    'W3': (1750, 2625),      # Word3: 1750-2625ms after Word1 onset
    'W4': (2625, 3500),      # Word4: 2625-3500ms after Word1 onset
    'full_consolidation': (3500, 4500),     # All 1000ms
}
global sentence_look
sentence_look = {}

#Regression Parameters
#'W1', 'W2', 'W3', 'W4',
#Reduce # of variables <- get rid of modality and word 1<- unneccesary
TIME_WINDOWS_TO_TEST = ["W2","W3","W4","full_consolidation"] 
CONDITIONS = ["overall", "GS", "NGNS","GNS"]
LAYERS_TO_TEST = ['last', 'late', 'early', 'middle']

def ridge_per_electrode_optimized(
    X, y, groups=None, alpha_range=None, n_folds=5, 
    random_state=42, use_pca=False, n_components=None,
    pca_variance_threshold=None, adaptive_alpha=True,
    n_permutations=100  # Default to 100 permutations
):
    """
    Ridge regression per electrode with optional PCA and adaptive regularization.
    Always computes p-values and stores train/test data for later shuffle tests.
    """
    
    n_samples, n_features = X.shape
    
    # Adaptive alpha range based on dimensionality
    if alpha_range is None:
        if adaptive_alpha:
            ratio = n_features / n_samples
            
            if ratio > 5:
                alpha_range = np.logspace(0, 4, 20)
                print(f"  High dim ratio ({ratio:.1f}): using strong regularization")
            elif ratio > 2:
                alpha_range = np.logspace(-1, 3.5, 20)
                print(f"  Moderate dim ratio ({ratio:.1f}): using medium-strong regularization")
            elif ratio > 1:
                alpha_range = np.logspace(-2, 3, 20)
                print(f"  Medium dim ratio ({ratio:.1f}): using medium regularization")
            else:
                alpha_range = np.logspace(-3, 2, 20)
                print(f"  Low dim ratio ({ratio:.1f}): using standard regularization")
        else:
            alpha_range = np.logspace(-3, 1, 5)
    
    # Train/test split
    if groups is not None:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
        train_idx, test_idx = next(splitter.split(X, y, groups=groups))
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        print(f"  Train: {len(train_idx)} samples ({len(train_idx)/len(X)*100:.1f}%)")
        print(f"  Test:  {len(test_idx)} samples ({len(test_idx)/len(X)*100:.1f}%)")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state
        )
    
    # Store original (unscaled, pre-PCA) data
    X_train_original = X_train.copy()
    X_test_original = X_test.copy()
    y_train_original = y_train.copy()
    y_test_original = y_test.copy()
    
    pca = None
    if use_pca:
        if n_components is None and pca_variance_threshold is None:
            n_components = min(100, len(X_train) // 3, n_features // 2)
            print(f"  Auto-setting n_components to {n_components}")
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        if pca_variance_threshold is not None:
            pca = PCA(n_components=pca_variance_threshold, random_state=random_state)
            X_train = pca.fit_transform(X_train_scaled)
            X_test = pca.transform(X_test_scaled)
            print(f"  Applying PCA (variance threshold={pca_variance_threshold})")
            print(f"    Components: {pca.n_components_}, Variance: {pca.explained_variance_ratio_.sum():.3f}")
        else:
            pca = PCA(n_components=n_components, random_state=random_state)
            X_train = pca.fit_transform(X_train_scaled)
            X_test = pca.transform(X_test_scaled)
            print(f"  Applying PCA ({n_components} components)")
            print(f"    Variance: {pca.explained_variance_ratio_.sum():.3f}")
        
        print(f"    Dimensions: {n_features} → {X_train.shape[1]}")
    
    # Fit model (always with p-values enabled)
    model = RidgePerElectrode(
        alpha_range=alpha_range,
        n_folds=n_folds,
        random_state=random_state,
        compute_pvalues=True,  # Always compute p-values
        n_permutations=n_permutations
    )
    
    # Scale before to improve performance 
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model.fit(X_train_scaled, y_train)

    train_metrics = model.evaluate(X_train_scaled, y_train)
    test_metrics = model.evaluate(X_test_scaled, y_test)
    
    cv_summary = model.get_cv_summary()
    
    # Check for overfitting
    train_r2_mean = train_metrics['r2'].mean()
    test_r2_mean = test_metrics['r2'].mean()
    gap = train_r2_mean - test_r2_mean
    
    if gap > 0.3:
        print(f"\n  ⚠️  WARNING: Large train/test gap!")
        print(f"     Train R²: {train_r2_mean:.3f}, Test R²: {test_r2_mean:.3f}, Gap: {gap:.3f}")
    
    if test_r2_mean < 0:
        print(f"\n  ⚠️  WARNING: Negative test R² ({test_r2_mean:.3f})")
    
    # Results
    n_electrodes = y.shape[1]
    results_df = pd.DataFrame({
        "electrode": [f"electrode_{i}" for i in range(n_electrodes)],
        "best_alpha": model.best_alphas_,
        "cv_R_2_mean": cv_summary['cv_r2_per_electrode'], 
        "train_MSE": train_metrics['mse'],
        "test_MSE": test_metrics['mse'],
        "train_R": train_metrics['pearson_r'],  
        "test_R": test_metrics['pearson_r'],   
        "train_R_2": train_metrics['r2'],       
        "test_R_2": test_metrics['r2'],
        "p_value": cv_summary['pvalues_per_electrode'],  # Always include p-values
    })
    
    # Return comprehensive dictionary
    return_dict = {
        'results_df': results_df,
        'model': model,
        'scaler': scaler,
        'pca': pca,
        # Store train/test data (both original and processed)
        'X_train_original': X_train_original,
        'X_test_original': X_test_original,
        'y_train_original': y_train_original,
        'y_test_original': y_test_original,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        # Also store p-values separately for easy access
        'p_values': cv_summary['pvalues_per_electrode'],
        'n_significant_05': cv_summary.get('n_significant_05', 0),
        'n_significant_01': cv_summary.get('n_significant_01', 0),
    }
    
    return return_dict
def prepare_selective_time_window_X_y(features_df, embeddings_dict, trial_df,
                                      time_window='full_consolidation', use_z_power=True):
    
    power_cols = [c for c in features_df.columns if 'electrode' in c and 'power' in c]
    if use_z_power:
        power_cols = [c for c in power_cols if 'z_power' in c]
    
    if time_window not in WINDOWS_RELATIVE_TO_WORD1:
        raise ValueError(f"Unknown time_window: {time_window}")
    
    # Get window in Word1 coordinates
    word1_start, word1_end = WINDOWS_RELATIVE_TO_WORD1[time_window]
    
    # Convert to trial_start coordinates (what features_df uses)
    start_ms = word1_start + OFFSET_MS
    end_ms = word1_end + OFFSET_MS
    
    print(f"\n  Window '{time_window}':")
    print(f"    In Word1 coords:  {word1_start}-{word1_end}ms")
    print(f"    In trial coords:  {start_ms}-{end_ms}ms (used for filtering)")
    
    # NOW filter features_df using trial_start coordinates
    mask = (features_df['time_start_ms'] >= start_ms) & \
           (features_df['time_end_ms'] <= end_ms)
    
    windowed_df = features_df[mask].copy()
    print(f"    Found {len(windowed_df)} windows")
    
    # Rest stays the same...
    windowed_df = windowed_df.drop_duplicates(subset=['trial_idx', 'time_start_ms'], keep='first')
    
    agg_dict = {col: 'mean' for col in power_cols}
    agg_dict['sentence'] = 'first'
    
    trial_features = windowed_df.groupby('trial_idx')[power_cols + ['sentence']].agg(agg_dict).reset_index()
    
    y = trial_features[power_cols].values  
    sentences = trial_features['sentence'].values
    X = np.array([embeddings_dict[sent] for sent in sentences])
    groups = trial_features['trial_idx'].values
    
    print(f"    Aggregated to {len(trial_features)} trials")
    
    return X, y, groups


def run_ridge_regression_selective_windows(features_df, bert_embeddings, gpt_embeddings, t5_embeddings,
                                           trial_df, time_window='N400',
                                           use_z_power=True, alpha_range=None, n_folds=5):
    """
    Run ridge regression for ONE selective time window.
    
    This is called multiple times in the main loop for different windows.
    """
    
    if alpha_range is None:
        alpha_range = np.logspace(-4, 1, 10)
    
    # Prepare data for this specific window
    X_bert, y, groups = prepare_selective_time_window_X_y(
        features_df, bert_embeddings, trial_df, time_window, use_z_power
    )
    X_gpt, _, _ = prepare_selective_time_window_X_y(
        features_df, gpt_embeddings, trial_df, time_window, use_z_power
    )
    X_t5, _, _ = prepare_selective_time_window_X_y(
        features_df, t5_embeddings, trial_df, time_window, use_z_power
    )
    
    n_samples = X_bert.shape[0]
    n_features_bert = X_bert.shape[1]
    
    use_pca = False
    
    print(f"\nData dimensions: {n_samples} trials × {n_features_bert} BERT features")
    
        ## In your run_ridge_regression_selective_windows function:
    print("\n=== Running BERT ===")
    bert_output = ridge_per_electrode_optimized(
        X_bert, y, groups=groups, alpha_range=alpha_range, n_folds=n_folds,
        use_pca=use_pca, adaptive_alpha=True,
        n_permutations=100  
    )

    print("\n=== Running GPT ===")
    gpt_output = ridge_per_electrode_optimized(
        X_gpt, y, groups=groups, alpha_range=alpha_range, n_folds=n_folds,
        use_pca=use_pca, adaptive_alpha=True,
        n_permutations=100
    )

    print("\n=== Running T5 ===")
    t5_output = ridge_per_electrode_optimized(
        X_t5, y, groups=groups, alpha_range=alpha_range, n_folds=n_folds,
        use_pca=use_pca, adaptive_alpha=True,
        n_permutations=100
    )

    return {
        'bert_results': bert_output['results_df'],
        'gpt_results': gpt_output['results_df'],
        't5_results': t5_output['results_df'],
        'bert_model': bert_output['model'],
        'gpt_model': gpt_output['model'],
        't5_model': t5_output['model'],
        # Store all train/test data and p-values
        'bert_data': bert_output,
        'gpt_data': gpt_output,
        't5_data': t5_output,
        'y': y,
        'groups': groups,
}


# [Keep all your other helper functions: get_layer_indices, get_embeddings_multilayer, 
#  clean_sentences, create_full_sentence, filter_sentences - they're unchanged]

def get_layer_indices(model, layer_spec):
    """Get layer index based on specification (returns single layer)."""
    model_type = type(model).__name__
    
    if hasattr(model, 'h'):  
        n_layers = len(model.h)
        print(f"  Detected GPT-2 model with {n_layers} layers")
    elif hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):  
        n_layers = len(model.encoder.layer)
        print(f"  Detected BERT model with {n_layers} layers")
    elif hasattr(model, 'encoder') and hasattr(model.encoder, 'block'):  
        n_layers = len(model.encoder.block)
        print(f"  Detected T5 model with {n_layers} layers")
    else:
        raise ValueError(f"Unknown model architecture. Model type: {model_type}")
    
    if isinstance(layer_spec, int):
        return [layer_spec]
    elif layer_spec == 'early':
        return [2]
    elif layer_spec == 'middle':
        return [n_layers // 2]
    elif layer_spec == 'late':
        return [n_layers - 3]
    elif layer_spec == 'last':
        return [n_layers - 1]
    else:
        raise ValueError(f"Unknown layer specification: {layer_spec}")


def get_embeddings_multilayer(sentences, model_name, sub_num, layer='last', batch_size=384, max_words=4, pooling='mean'):
    """Get embeddings from specified layer(s) of transformer models."""
    from helpers.constants import ENGLISH_SUBS
    
    if sub_num in ENGLISH_SUBS:
        sentences = [sentence.split()[1:] if sentence.split()[0].lower() == 'the' 
                    else sentence.split() for sentence in sentences]
        sentences = [' '.join(s) for s in sentences]
    
    print(f"Getting {model_name} embeddings from layer(s): {layer}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    is_gpt = "gpt" in model_name.lower()
    is_t5 = "t5" in model_name.lower()
    is_bert = "bert" in model_name.lower()
    
    if is_t5:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if is_t5:
        model = T5Model.from_pretrained(model_name, output_hidden_states=True).to(device)
    else:
        model = AutoModel.from_pretrained(model_name, output_hidden_states=True).to(device)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    
    layer_indices = get_layer_indices(model, layer)
    print(f"  Extracting from layer indices: {layer_indices}")
    
    if is_t5:
        n_layers = len(model.encoder.block)
    elif is_gpt:
        n_layers = len(model.h)  
    elif is_bert:
        n_layers = len(model.encoder.layer)
    else:
        n_layers = 12
    
    print(f"  Model has {n_layers} layers total")
    
    for idx in layer_indices:
        if idx >= n_layers or idx < 0:
            raise ValueError(f"Layer index {idx} out of range [0, {n_layers-1}]")
    
    embeddings_list = []
    num_samples = len(sentences)
    effective_max_words = max_words - 1 if is_gpt else max_words
    
    for i in tqdm(range(0, num_samples, batch_size), desc=f"Processing batches"):
        batch = sentences[i:i + batch_size]
        
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=50).to(device)
        
        with torch.no_grad():
            if is_t5:
                outputs = model.encoder(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    output_hidden_states=True
                )
                hidden_states = outputs.hidden_states
            else:
                outputs = model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states
            
            selected_layers = [hidden_states[idx + 1] for idx in layer_indices]
            
            if pooling == 'mean':
                hidden = torch.stack(selected_layers, dim=0).mean(dim=0)
            elif pooling == 'concat':
                hidden = torch.cat(selected_layers, dim=-1)
            elif pooling == 'first':
                hidden = selected_layers[0]
            elif pooling == 'last':
                hidden = selected_layers[-1]
            else:
                raise ValueError(f"Unknown pooling method: {pooling}")
            
            if is_t5:
                mask = inputs["attention_mask"].unsqueeze(-1)
                sent_emb = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
            elif is_gpt:
                n_words = min(hidden.shape[1], effective_max_words)
                word_emb = hidden[:, :n_words, :]
                
                if n_words < effective_max_words:
                    pad = torch.zeros(hidden.shape[0], effective_max_words - n_words, hidden.shape[2], device=device)
                    word_emb = torch.cat([word_emb, pad], dim=1)
                
                sent_emb = word_emb.reshape(word_emb.shape[0], -1)
            else:  # BERT
                sent_emb = hidden[:, 0, :]
            
            embeddings_list.append(sent_emb.cpu().numpy())
    
    embeddings = np.vstack(embeddings_list)
    print(f"  Final embedding shape: {embeddings.shape}")
    
    return embeddings


def clean_sentences(sentences_dicts):
    dataframe_rows = []
    for key, value in sentences_dicts.items():  
        if not isinstance(value, dict):
            dataframe_rows.append(EMPTY_DICT)
            continue
        if "sen_field" not in value:
            dataframe_rows.append(EMPTY_DICT)
            continue
        
        sen_field_data = value['sen_field']
        
        if type(sen_field_data).__name__ == 'MatlabOpaque':
            dataframe_rows.append(EMPTY_DICT)
            continue
        
        if not isinstance(sen_field_data, dict):
            try:
                if hasattr(sen_field_data, 'item'):
                    curr_row = dict(sen_field_data.item())
                else:
                    dataframe_rows.append(EMPTY_DICT)
                    continue
            except:
                dataframe_rows.append(EMPTY_DICT)
                continue
        else:
            curr_row = sen_field_data.copy()
        
        for correct_key, variants in WORD_KEYS.items():
            for variant in variants:
                if variant in curr_row:
                    if variant != correct_key:
                        curr_row[correct_key] = curr_row.pop(variant)
                    break
        
        for correct_key, variants in TYPE_KEYS.items():
            for variant in variants:
                if variant in curr_row:
                    if variant != correct_key:
                        curr_row[correct_key] = curr_row.pop(variant)
                    break
        
        for correct_key, variants in OTHER_KEYS.items():
            for variant in variants:
                if variant in curr_row:
                    if variant != correct_key:
                        curr_row[correct_key] = curr_row.pop(variant)
                    break
    
        dataframe_rows.append(curr_row)
    return dataframe_rows


def create_full_sentence(entry):
    return " ".join([entry["w1"], entry["w2"], entry["w3"], entry["w4"]]).strip()


def filter_sentences(sentences_list, condition):
    filter_sen= []
    for sentence in sentences_list:
        if sentence in  sentence_look[condition]:
            filter_sen.append(sentence)
    return filter_sen


step = 20
def main():
    #4,5,6,7,8,9,10,11,12,13,14,15
    for num, _ in DIRECTORYS.items():
        if num in [1,2,3,4,5,6,8,9,10,11,12,13,14,15,16,17]: #redo 7 layer<- small data issue
            continue
        print(f"\n{'='*70}")
        print(f"PROCESSING SUBJECT {num}")
        print(f"{'='*70}") 

        global sentence_look
        sentence_look = {}
        
        print("\n1) Loading data from pickle file")
        trial_df = pd.read_pickle(f"S{num}_{step}_trial_df.pkl")
        print(f"   trial_df shape: {trial_df.shape}")
        
        print("\n2) Processing feature_df")
        features_df = pd.read_pickle(f"pickle_features/S{num}_{step}_feature_df.pkl")
        features_df = features_df.apply(
            lambda col: pd.to_numeric(col, errors='coerce') 
            if col.name not in ['sentence_type', 'modality'] 
            else col
        )
        features_df = features_df.fillna(0)
        print(f"   features_df shape: {features_df.shape}")
        
        print("\n3) Cleaning and creating sentence lookup dict")
        sentences_list = trial_df['sentence'].dropna().astype(str).to_list()
        sentences_iter = np.load(f"sentences_subs/S{num}_sentences_iter.npy", allow_pickle=True).item()
        sentence_frame = pd.DataFrame(clean_sentences(sentences_iter))
        sentence_lookup = sentence_frame.dropna(subset=["w1", "w2", "w3", "w4", "sentenceType"])
        sentence_lookup["full_sentence"] = sentence_lookup.apply(create_full_sentence, axis=1)

        sentence_lookup['modality'] = sentence_lookup['modality'].astype(str).str.strip()
        sentence_lookup['modality'] = sentence_lookup['modality'].replace(['NA', 'nan', ''], pd.NA)

        # Build condition-specific sentence sets
        sentence_look["overall"] = set(sentence_lookup["full_sentence"])
        sentence_look["NGNS"] = set(sentence_lookup[sentence_lookup['sentenceType'] == "NGNS"]["full_sentence"])
        sentence_look["GNS"] = set(sentence_lookup[sentence_lookup['sentenceType'] == "GNS"]["full_sentence"]) 
        sentence_look["GS"] = set(sentence_lookup[sentence_lookup['sentenceType'] == "GS"]["full_sentence"])
        sentence_look["a"] = set(sentence_lookup[sentence_lookup['modality'] == "a"]["full_sentence"]) 
        sentence_look["v"] = set(sentence_lookup[sentence_lookup['modality'] == "v"]["full_sentence"])
        
        print(f"   Sentence counts by condition:")
        for cond, sents in sentence_look.items():
            print(f"     {cond}: {len(sents)} sentences")
        
        all_unique_sentences = list(set(sentences_list))
        print(f"\n   Total unique sentences: {len(all_unique_sentences)}")
        
        trial_to_sentence = dict(zip(trial_df['trial_idx'], trial_df['sentence']))
        print(f"   Created trial→sentence mapping for {len(trial_to_sentence)} trials")
        
        # ========== LAYER LOOP ==========
        for layer in LAYERS_TO_TEST:
            print(f"\n{'#'*80}")
            print(f"# PROCESSING LAYER: {layer.upper()}")
            print(f"{'#'*80}")
            
            print(f"\n4) Generating {layer} layer embeddings for all unique sentences...")
            
            start_time = time.time()
            bert_emb = get_embeddings_multilayer(all_unique_sentences, "google-bert/bert-base-multilingual-cased", num, layer=layer)
            print(f"   BERT ({layer}) done in {time.time() - start_time:.1f}s")
            
            start_time = time.time()
            gpt_emb = get_embeddings_multilayer(all_unique_sentences, "gpt2-xl", num, layer=layer)
            print(f"   GPT ({layer}) done in {time.time() - start_time:.1f}s")
            
            start_time = time.time()
            t5_emb = get_embeddings_multilayer(all_unique_sentences, "google/mt5-xl", num, layer=layer)
            print(f"   T5 ({layer}) done in {time.time() - start_time:.1f}s")
            
            bert_embeddings_dict = {sent: bert_emb[i] for i, sent in enumerate(all_unique_sentences)}
            gpt_embeddings_dict = {sent: gpt_emb[i] for i, sent in enumerate(all_unique_sentences)}
            t5_embeddings_dict = {sent: t5_emb[i] for i, sent in enumerate(all_unique_sentences)}

            # ========== CONDITION LOOP ==========
            print(f"\n5) Running regressions for each condition with {layer} layer")
            for condition in CONDITIONS:
                
                filtered_sentences = filter_sentences(sentences_list, condition)
                print(f"\n{'='*60}")
                print(f"CONDITION: {condition} | LAYER: {layer}")
                print(f"  Sentences: {len(filtered_sentences)}")
                print(f"{'='*60}")
                
                if len(filtered_sentences) == 0:
                    print(f"   ⚠️  No sentences, skipping...")
                    continue
                
                # Filter features_df
                if condition == "a" or condition == "v":
                    features_filter = features_df[features_df["modality"] == condition].copy()
                elif condition == "NGNS" or condition == "GNS" or condition == "GS":
                    features_filter = features_df[features_df["sentence_type"] == condition].copy()
                elif condition.lower() == "overall":
                    features_filter = features_df.copy()
                else:
                    print(f"ERROR: Invalid condition, skipping...")
                    continue
                
                features_filter['sentence'] = features_filter['trial_idx'].map(trial_to_sentence)
                features_filter = features_filter.dropna(subset=['sentence'])
                features_filter = features_filter[features_filter['sentence'].isin(filtered_sentences)]
                
                if len(features_filter) == 0:
                    print(f"   No data after filtering, skipping...")
                    continue

                print(f"\n  Diagnostic: Available time ranges in features_filter:")
                print(f"    Min time_start_ms: {features_filter['time_start_ms'].min()}")
                print(f"    Max time_end_ms: {features_filter['time_end_ms'].max()}")
                print(f"    Number of unique trials: {features_filter['trial_idx'].nunique()}")
                for time_window in TIME_WINDOWS_TO_TEST:
                    print(f"\n  --- Time Window: {time_window} ---")
                    
                    try:
                        print(f"\n{'='*60}")
                        print(f"Running S{num}")
                        results = run_ridge_regression_selective_windows(
                            features_df=features_filter,
                            bert_embeddings=bert_embeddings_dict,
                            gpt_embeddings=gpt_embeddings_dict,
                            t5_embeddings=t5_embeddings_dict,
                            trial_df=trial_df,
                            time_window=time_window,
                            use_z_power=True,
                            alpha_range=np.logspace(1, 6, 8), #Shift up alpha range
                            n_folds=5,
                        )
                        
                        # Save results
                        output_dir = f"results_selective_windows/{condition}/{layer}/{time_window}"
                        os.makedirs(output_dir, exist_ok=True)
                        output_path = f"{output_dir}/S{num}_{condition}_{layer}_{time_window}.pkl"
                        
                        with open(output_path, 'wb') as f:
                            pickle.dump(results, f)
                        print(f"   ✓ Results saved to {output_path}")
                        
                        # Print top results
                        print(f"\n   __________________RESULTS: {condition} - {layer}__________________") #only show high performing electrodes
                        print("\n   BERT Top 10 Electrodes:")
                        print(results['bert_results'].nlargest(10, 'test_R_2')[['electrode', 'train_R_2','test_R_2','train_R','test_R', 'cv_R_2_mean', 'best_alpha','p_value']].to_string(index=False))
                        print("\n   GPT Top 10 Electrodes:")
                        print(results['gpt_results'].nlargest(10, 'test_R_2')[['electrode','train_R_2', 'test_R_2', 'train_R','test_R', 'cv_R_2_mean', 'best_alpha', 'p_value']].to_string(index=False))
                        print("\n   T5 Top 10 Electrodes:")
                        print(results['t5_results'].nlargest(10, 'test_R_2')[['electrode', 'train_R_2','test_R_2', 'train_R','test_R', 'cv_R_2_mean', 'best_alpha', 'p_value']].to_string(index=False))

                        print(f"\n   Significance Summary (p < 0.05):")
                        print(f"     BERT: {results['bert_data']['n_significant_05']}/{len(results['bert_results'])} electrodes")
                        print(f"     GPT:  {results['gpt_data']['n_significant_05']}/{len(results['gpt_results'])} electrodes")
                        print(f"     T5:   {results['t5_data']['n_significant_05']}/{len(results['t5_results'])} electrodes")
                        
                    except Exception as e:
                        print(f"   ❌ Error: {str(e)}")
                        import traceback
                        traceback.print_exc()
                        continue
        
        print(f"\n{'='*70}")
        print(f"COMPLETED SUBJECT {num}")
        print(f"{'='*70}\n")
        #Subject 7 has a values issue <- look into this


if __name__ == "__main__":
    main()