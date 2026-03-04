def is_undesired_format(sentence_dict):
    if not isinstance(sentence_dict, dict):
        return True  # or return False, depending on your requirements
    
    # Define the desired keys
    desired_keys = ['imageFile', 'falseImageFile', 'relatedImage', 'sentenceType', 'w1Type', 'w2Type', 'w3Type', 'w4Type', 'modality']
    word_keys = {
        'w1': ['w1', 'word1', 'W1', 'Word1'],
        'w2': ['w2', 'word2', 'W2', 'Word2'],
        'w3': ['w3', 'word3', 'W3', 'Word3'],
        'w4': ['w4', 'word4', 'W4', 'Word4']
    }
    
    # Check if all desired keys are present
    if not all(key in sentence_dict for key in desired_keys):
        return True
    
    # Check if word keys are present
    for key in word_keys:
        if not any(word_key in sentence_dict for word_key in word_keys[key]):
            return True
    
    return False