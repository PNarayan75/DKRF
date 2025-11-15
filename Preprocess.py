import sys
# sys.path.append('..')
# from utils.tools import *
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer, RobertaModel, Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model, Wav2Vec2PreTrainedModel
import torchaudio
import numpy as np
import os
import random
import pickle
from config.hyper_params_config import dataset_root, pretrained_root

class IEMOCAPDataset(Dataset):
    def __init__(self, feature_save_path):
        # Directly load the data from the saved pkl file
        with open(feature_save_path, 'rb') as f:
            self.data = pickle.load(f)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        return self.data[index]

def process_and_save_dataset(data_list_path, iemocap_aug_datapath, feature_save_path, max_duration=25):
    # Check if the saved pkl file already exists
    if os.path.exists(feature_save_path):
        print(f"{feature_save_path} already exists, skipping data processing.")
        return
    augmented_audio = os.listdir(iemocap_aug_datapath)
    augmented_audio_dictionary = {}
    with open(data_list_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for item in augmented_audio:
        gt_audio = "Ses" + item.split("Ses")[-1]
        if gt_audio in augmented_audio_dictionary:
            augmented_audio_dictionary[gt_audio].append(iemocap_aug_datapath + item)
        else:
            augmented_audio_dictionary[gt_audio] = [iemocap_aug_datapath + item]
    dataset_features = []
    # Traverse the entire dataset and process features
    for index in range(len(lines)):
        raw_audio_path, raw_text, label, augmented_text = lines[index].replace('\n', '').split('\t')
        raw_audio_name = os.path.basename(raw_audio_path)
        augmented_wav_index = random.randint(0, len(augmented_audio_dictionary[raw_audio_name]) - 1)
        augmented_wav_path = augmented_audio_dictionary[raw_audio_name][augmented_wav_index]
        # Load original and augmented audio
        raw_waveform, raw_sample_rate = torchaudio.load(raw_audio_path)
        aug_waveform, aug_sample_rate = torchaudio.load(augmented_wav_path)
        # Calculate audio duration and filter out audio exceeding 25 seconds
        raw_duration = raw_waveform.shape[1] / raw_sample_rate
        aug_duration = aug_waveform.shape[1] / aug_sample_rate
        # If the audio duration is greater than max_duration, skip this audio and its corresponding text and label
        if raw_duration > max_duration or aug_duration > max_duration:
            print(f"Skipping sample {raw_audio_name}, because the audio duration exceeded {max_duration} seconds.")
            continue
        # Save audio features
        raw_waveform = raw_waveform.numpy()
        aug_waveform = aug_waveform.numpy()
        raw_audio_seq_input, raw_audio_cls_input, raw_audio_dimension = process_func(raw_waveform, raw_sample_rate)
        aug_audio_seq_input, aug_audio_cls_input, aug_audio_dimension = process_func(aug_waveform, aug_sample_rate)
        # Store features in a dictionary
        features = {
            'raw_waveform': raw_waveform,
            'aug_waveform': aug_waveform,
            'raw_sample_rate': raw_sample_rate,
            'aug_sample_rate': aug_sample_rate,
            'raw_sentimental_density': raw_audio_dimension,
            'aug_sentimental_density': aug_audio_dimension,
            'raw_text_input': raw_text,
            'aug_text_input': augmented_text,
            'label': int(label)
        }
        # Add to the list
        dataset_features.append(features)
    # Save all processed features to a pkl file
    with open(feature_save_path, 'wb') as f:
        pickle.dump(dataset_features, f)
    print(f"Features have been saved to {feature_save_path}")

def collate_fn(batch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_waveforms = [sample['raw_waveform'].squeeze(0) for sample in batch]
    aug_waveforms = [sample['aug_waveform'].squeeze(0) for sample in batch]
    raw_sample_rates = [sample['raw_sample_rate'] for sample in batch]
    aug_sample_rates = [sample['aug_sample_rate'] for sample in batch]
    # Use processor to process audio sequences and get input_ids and attention_masks
    raw_audio_processed = processor(
        raw_waveforms, sampling_rate=raw_sample_rates[0], return_tensors="pt", padding=True, return_attention_mask=True
    )
    aug_audio_processed = processor(
        aug_waveforms, sampling_rate=aug_sample_rates[0], return_tensors="pt", padding=True, return_attention_mask=True
    )
    # Get input_ids and attention_masks from processed data
    raw_audio_input_ids = raw_audio_processed['input_values'].to(device)
    raw_attention_masks = raw_audio_processed['attention_mask'].to(device)
    aug_audio_input_ids = aug_audio_processed['input_values'].to(device)
    aug_attention_masks = aug_audio_processed['attention_mask'].to(device)
    # Get text inputs and labels
    raw_texts = [sample['raw_text_input'] for sample in batch]
    aug_texts = [sample['aug_text_input'] for sample in batch]
    labels = torch.tensor([sample['label'] for sample in batch], dtype=torch.long).to(device)
    # Get audio dimension features and move them to GPU
    raw_audio_dimensions = torch.stack([torch.tensor(sample['raw_sentimental_density']) for sample in batch]).to(device)
    aug_audio_dimensions = torch.stack([torch.tensor(sample['aug_sentimental_density']) for sample in batch]).to(device)
    return {
        'raw_audio_input_ids': raw_audio_input_ids,
        'raw_attention_masks': raw_attention_masks,
        'aug_audio_input_ids': aug_audio_input_ids,
        'aug_attention_masks': aug_attention_masks,
        'raw_audio_dimensions': raw_audio_dimensions,
        'aug_audio_dimensions': aug_audio_dimensions,
        'raw_text': raw_texts,
        'aug_text': aug_texts,
        'label': labels
    }

class RegressionHead(nn.Module):
    r"""Classification head."""
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

### Load pre-trained Wav2vec2 model, where hidden_states1 is the result of average pooling along the time sequence of the last layer's hidden states, and logits is the result of mapping the pooled vector to a three-dimensional continuous emotional space.
### If you want to extract the non-pooled hidden_states, i.e., dimension [batch, seq_len, 1024], just extract hidden_states0 = outputs[0]
class EmotionModel(Wav2Vec2PreTrainedModel):
    r"""Speech emotion classifier."""
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.init_weights()
    def forward(
            self,input_values,attention_mask=None):
        outputs = self.wav2vec2(input_values,attention_mask=attention_mask)
        hidden_states0 = outputs[0]
        hidden_states1 = torch.mean(hidden_states0, dim=1)
        logits = self.classifier(hidden_states1)
        return hidden_states0, hidden_states1, logits

### Encapsulate the entire wav2vec computation process, where the processor encodes the speech, similar to the tokenizer in BERT.
def process_func(x: np.ndarray, sampling_rate: int) -> np.ndarray:
    y = processor(x, sampling_rate=sampling_rate)
    y = y['input_values'][0]
    y = y.reshape(1, -1)
    y = torch.from_numpy(y).to(device)
    ### Here, if embeddings is true, the returned y is actually the pooled hidden_states, otherwise it is logits
    with torch.no_grad():
        y = audio_model(y)
    return y[0],y[1],y[2]

### Use processor to get the speech tokenizer
def audio_processor(x: np.ndarray, sampling_rate: int) -> np.ndarray:
    inputs = processor(x, sampling_rate=sampling_rate,return_attention_mask=True, padding=True)
    input_values = inputs['input_values'][0] # Input audio features (input_ids)
    attention_mask = inputs['attention_mask'][0] # attention mask
    return input_values,attention_mask

# if __name__ =='__main__':
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = pretrained_root/'wav2vec2-large-uncased'
processor = Wav2Vec2Processor.from_pretrained(model_name)
audio_model = EmotionModel.from_pretrained(model_name).to(device)
# Process and save training and validation data
session_id=1
train_data_list_path = dataset_root/f'dataset_session{session_id}/train_data.txt'
val_data_list_path = dataset_root/f'dataset_session{session_id}/test_data.txt'
# iemocap_aug_datapath = dataset_root/'MMER-main/data/iemocap_aug/out/'# it's no use in main experiment
# uncomment if you want to process audio file
# Set maximum audio length to 25 seconds
# process_and_save_dataset(train_data_list_path, iemocap_aug_datapath, f'train_features_session{session_id}.pkl', max_duration=25)
# process_and_save_dataset(val_data_list_path, iemocap_aug_datapath, f'val_features_session{session_id}.pkl', max_duration=25)
print("pre_processor finished")
# # # Load saved features
# train_dataset = IEMOCAPDataset('train_features_session2.pkl')
# val_dataset = IEMOCAPDataset('val_features_session2.pkl')
# # # Create dataloader
# train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
# val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
#
# for batch in train_dataloader:
# raw_audio_input_ids, raw_attention_masks, raw_text, labels = batch['raw_audio_input_ids'], batch['raw_attention_masks'], \
# batch['raw_text'], batch['label']
# print("raw_audio_input_ids:",raw_audio_input_ids.shape)
# print("raw_attention_masks:", raw_attention_masks.shape)
# print("raw_text:", raw_text)
# print("labels:", labels)
