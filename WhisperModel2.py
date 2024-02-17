from huggingface_hub import notebook_login
notebook_login()

import datasets
import librosa

from datasets import load_dataset, DatasetDict

common_voice = DatasetDict()

common_voice["train"] = load_dataset("mozilla-foundation/common_voice_16_0", "lg", split="train", trust_remote_code=True)
common_voice["validation"] = load_dataset("mozilla-foundation/common_voice_16_0", "lg", split="validation", trust_remote_code=True)
common_voice["test"] = load_dataset("mozilla-foundation/common_voice_16_0", "lg", split="test", trust_remote_code=True)

common_voice = common_voice.remove_columns(["accent", "age", "client_id", "down_votes", "gender","path", "locale", "segment", "up_votes"])

from transformers import WhisperFeatureExtractor
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")

from transformers import WhisperTokenizer
tokenizer_general = WhisperTokenizer.from_pretrained("openai/whisper-small", task="transcribe")
tokenizer_swahili = WhisperTokenizer.from_pretrained("openai/whisper-small", language="Swahili", task="transcribe")

input_str = common_voice["train"][0]["sentence"]
labels = tokenizer_general(input_str).input_ids
decoded_with_special = tokenizer_general.decode(labels, skip_special_tokens=False)
decoded_str = tokenizer_general.decode(labels, skip_special_tokens=True)
print(f"Input:                 {input_str}")
print(f"Decoded w/ special:    {decoded_with_special}")
print(f"Decoded w/out special: {decoded_str}")
print(f"Are equal:             {input_str == decoded_str}")

input_str = common_voice["train"][0]["sentence"]
labels = tokenizer_swahili(input_str).input_ids
decoded_with_special = tokenizer_swahili.decode(labels, skip_special_tokens=False)
decoded_str = tokenizer_swahili.decode(labels, skip_special_tokens=True)
print(f"Input:                 {input_str}")
print(f"Decoded w/ special:    {decoded_with_special}")
print(f"Decoded w/out special: {decoded_str}")
print(f"Are equal:             {input_str == decoded_str}")

from transformers import WhisperProcessor
processor_general = WhisperProcessor.from_pretrained("openai/whisper-small", task="transcribe")
processor_swahili = WhisperProcessor.from_pretrained("openai/whisper-small", language="Swahili", task="transcribe")

#downsample to match whisper sampling rate
from datasets import Audio
common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))



from pydub import AudioSegment
import librosa
import numpy as np
import os

# Function to get the local path of a datapoint
def get_local_path(example, key):
    return os.path.abspath(example[key].source)

# Apply the function to each element in the dataset
dataset_with_local_paths = common_voice.map(lambda examples: {"local_path": get_local_path(examples, "audio")}, batched=False, batch_size=-1)

# Example of printing the local paths for the first three datapoints
for idx, example in enumerate(dataset_with_local_paths["train"][:3]):
    print(f"Local Path {idx}: {example['local_path']}")
In this example, the get_local_path function takes an example and the key of the feature (i.e., "audio"), returning the absolute path of the local audio file. The map function applies the get_local_path function to each element in the dataset, generating a new column called "local_path". Lastly, the example prints the local paths for the first three datapoints in the dataset.




#def convert_and_resample(batch):
#    audio_path = batch["audio"]["path"]
 #   audio = AudioSegment.from_mp3(audio_path)
  #  # Export as WAV
   # wav_path = audio_path.replace(".mp3", ".wav")
    #audio.export(wav_path, format="wav")
    
    ## Optionally resample and load with librosa
#    y, sr = librosa.load(wav_path, sr=16000)  # Resampling to 16 kHz
 #   os.remove(wav_path)  # Optionally remove the WAV file after processing
    
  #  # Update the batch (this part depends on how you want to structure your data)
   # batch["audio"] = {
    #    "array": np.array(y, dtype=np.float32),
     #   "sampling_rate": sr
   # }
    #return batch

#common_voice = common_voice.map(convert_and_resample)

problematic_files = []

def safe_process(file_paths):
    for file_path in file_paths:
        try:
            # Assuming file_paths contains MP3 files you want to convert
            wav_path = file_path.replace(".mp3", ".wav")
            audio = AudioSegment.from_mp3(file_path)
            audio.export(wav_path, format="wav")
            # Further processing here
        except Exception as e:
            problematic_files.append(file_path)
            print(f"Problem processing file {file_path}: {e}")
            return None  # or appropriate failure indication

for item in common_voice:
    safe_process(item["path"])

# Now, filter out the problematic files from your dataset
common_voice = [item for item in common_voice if item["file_path"] not in problematic_files]


def prepare_dataset_general(batch):

    # load and resample audio data from 48 to 16kHz
    audio, sr = librosa.load(batch["audio"]['path'], sr=16000)

    # compute log-Mel input features from input audio array 
    batch["input_features"] = feature_extractor(audio, sampling_rate=sr).input_features[0]
    
    # encode target text to label ids 
    batch["labels"] = tokenizer_general(batch["sentence"]).input_ids
    return batch
    
common_voice = common_voice.map(prepare_dataset_general, remove_columns=common_voice.column_names["train"])
    

def prepare_dataset_swahili(batch):
    # load and resample audio data from 48 to 16kHz
    audio, sr = librosa.load(batch["audio"]['path'], sr=16000)
    
    # Your existing preprocessing logic

    batch["input_features"] = feature_extractor(audio, sampling_rate=sr).input_features[0]
    batch["labels"] = tokenizer_swahili(batch["sentence"]).input_ids
    return batch

common_voice_swahili = common_voice.map(prepare_dataset_swahili, remove_columns=common_voice.column_names["train"])

import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union

#define a data collator
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

data_collator_general = DataCollatorSpeechSeq2SeqWithPadding(processor=processor_general)
data_collator_swahili = DataCollatorSpeechSeq2SeqWithPadding(processor=processor_swahili)
#evalution metricswer and wil
import evaluate
metric = evaluate.load("wer")
from torchmetrics.functional.text import word_information_lost


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    wil(pred_str, label_str)

    return {"wer": wer, "wil": wil}

#set checkpoints so that training progress is properly made
from transformers import WhisperForConditionalGeneration

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

#define training arguments
from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-small-hi",  # change to a repo name of your choice
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=4000,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=True,
)

from transformers import Seq2SeqTrainer

#train model without specific language
trainer_general = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=common_voice_general["train"],
    eval_dataset=common_voice_general["test"],
    data_collator=data_collator_general,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

trainer_general.train()

#train model with swahili specified
trainer_swahili = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=common_voice_swahili["train"],
    eval_dataset=common_voice_swahili["test"],
    data_collator=data_collator_swahili,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)
trainer_swahili.train()
