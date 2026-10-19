import json
from pathlib import Path

import cv2
import numpy as np
import torch
from opencc import OpenCC
from sklearn.preprocessing import LabelEncoder
from torch import softmax

from hanzi_ocr.HMNIST import HMNISTModel
from hanzi_ocr.pipelines import normalization_pipeline


def load_model(model_dir, device):
    """Creates model, loads weights and obtains LabelEncoder
    
    Parameters
    -----------
    model_dir: str or Path
        Path where the model's data was downloaded in the current device
    device: str
        Device model will use for inference, most likely the CPU
    
    Returns
    -----------
    model: torch.nn.Module
        Model with trained weights
    le: sklearn.preprocessing.LabelEncoder
        LabelEncoder with encoders created during training
    """
    if isinstance(model_dir, str):
        model_dir = Path.get(model_dir)
        
    state_dict = torch.load(model_dir / "checkpoint.pt", map_location=device, weights_only=True)
    with open(model_dir / "le.json") as jsonfile:
        classes = json.load(jsonfile)
        
    le = LabelEncoder()
    le.classes_ = np.array(classes)
    
    model = HMNISTModel(len(classes))
    model.load_state_dict(state_dict)
    model.eval()
    model = model.to(device)
    
    return model, le

def create_character_transformers():
    """Creates OpenCC character transformers
    
    Returns
    --------
    simplified_converter: opencc.OpenCC
        initialized traditional to simplified character converter
    traditional_converter: opencc.OpenCC
        initialized simplified to traditinal character converter
    """
    simplified_converter = OpenCC("t2s")
    traditional_converter = OpenCC("s2t")
    
    return simplified_converter, traditional_converter
    

def preprocess_image(image_bytes):
    """Receives an image's raw bytes and loads it into a 
    format Torch can work with
    
    Parameters
    -----------
    image_bytes:
        Image's raw bytes
    
    Returns
    ----------
    torch.Tensor
        One batch tensor with the image representation
        
    Excepts
    ----------
    ValueError
        If bytes received are not an image, or cannot be parsed as such
    """
    
    array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Image invalid, try uploading it again")
    

    tensor = normalization_pipeline(image=image)["image"]
    return tensor.unsqueeze(0)

def find_character_pairs(chars, simplified_converter, traditional_converter):
    """Given a hanzi returns a duple with simplified traditional pairs
    
    Parameters
    -----------
    chars: list of str
        Characters to be transformed into pairs
        
    Returns
    ----------
    list of (str, str)
        The first value is the simplified character, the second traditional
    """
    
    is_traditional = [(ord(char) == ord(traditional_converter.convert(char)[0])) for char in chars]
    is_simplified = [(ord(char) == ord(simplified_converter.convert(char)[0])) for char in chars]
    
    character_pairs = []
    for char, simplified, traditional in zip(chars, is_simplified, is_traditional):
        if simplified and traditional:
            character_pairs.append((char, char))
        elif traditional:
            character_pairs.append((simplified_converter.convert(char)[0], char))
        else:
            character_pairs.append((char, traditional_converter.convert(char)[0]))
        
    return character_pairs

def inference(model, le, batch, top_k):
    """Given an image with a hanzi character, returns its best predictions, 
    along with the character predicted
    
    Parameters
    ------------
    model: torch.nn.Module
        Model with the trained weights
    le: sklearn.preprocessing.LabelEncoder
        LabelEncoder with encodings obtained during training
    batch: torch.utils.data.DataLoader
        Batch with image samples to be processed
    
    Returns
    -----------
    list of (float, (str, str))
        Where the float is the probability is the confidence the classifier 
        in classifying the class
        And the tuple is the pair of predicted characters, simplified on 
        the left, traditional on the right
    
    """
    with torch.inference_mode():
        logits = model(batch)
    probs = softmax(logits, dim=-1)
    probs = torch.squeeze(probs)
    
    batch_values, batch_indices = torch.topk(probs, top_k, dim=-1)
    batch_values = batch_values.cpu().numpy()
    
    original_characters = le.inverse_transform(batch_indices)
    original_characters = [chr(char) for char in original_characters]
    top_predictions = find_character_pairs(original_characters)
    
    return list(zip(batch_values, top_predictions)) 