from huggingface_hub import login, logout
import librosa
import numpy as np
import os
import datasets
import torch

from datasets import load_dataset, DatasetDict

login("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")


common_voice = DatasetDict()

common_voice["train"] = load_dataset("mozilla-foundation/common_voice_15_0", "lg", split="train", use_auth_token=True)
common_voice["test"] = load_dataset("mozilla-foundation/common_voice_15_0", "lg", split="validation", use_auth_token=True)

common_voice = common_voice.remove_columns(["accent", "age", "client_id", "down_votes", "gender","path","variant" ,"locale", "segment", "up_votes"])

from transformers import WhisperFeatureExtractor
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")

from transformers import WhisperTokenizer
tokenizer_general = WhisperTokenizer.from_pretrained("openai/whisper-small", language="English", task="transcribe")
tokenizer_swahili = WhisperTokenizer.from_pretrained("openai/whisper-small", language="Swahili", task="transcribe")

from transformers import WhisperProcessor
processor_general = WhisperProcessor.from_pretrained("openai/whisper-small", language="English", task="transcribe")
processor_swahili = WhisperProcessor.from_pretrained("openai/whisper-small", language="Swahili", task="transcribe")

#downsample to match whisper sampling rate
from datasets import Audio
common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

from pydub import AudioSegment


def prepare_dataset_general(batch):

    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array 
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    
    # encode target text to label ids 
    batch["labels"] = tokenizer_general(batch["sentence"]).input_ids
    return batch
    
common_voice_general = common_voice.map(prepare_dataset_general, remove_columns=common_voice.column_names["train"])
    

def prepare_dataset_swahili(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array 
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    
    batch["labels"] = tokenizer_swahili(batch["sentence"]).input_ids
    return batch

common_voice_swahili = common_voice.map(prepare_dataset_swahili, remove_columns=common_voice.column_names["train"])


from dataclasses import dataclass
from typing import Any, Dict, List, Union

#define a data collator
@dataclass
class DataCollatorSpeechSeq2SeqWithPaddingGeneral:
    processor_general: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor_general.feature_extractor_genreral.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor_general.tokenizer_general.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor_general.tokenizer_general.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch
    
@dataclass
class DataCollatorSpeechSeq2SeqWithPaddingSwahili:
    processor_swahili: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor_swahili.feature_extractor_swahili.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor_swahili.tokenizer_swahili.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor_swahili.tokenizer_swahili.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

data_collator_general = DataCollatorSpeechSeq2SeqWithPaddingGeneral(processor_general=processor_general)
data_collator_swahili = DataCollatorSpeechSeq2SeqWithPaddingSwahili(processor_swahili=processor_swahili)

#evalution metricswer and wil
import evaluate
metric = evaluate.load("wer")
from torchmetrics.functional.text import word_information_lost


def compute_metrics(pred, tokenizer):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

#set checkpoints so that training progress is properly made
from transformers import WhisperForConditionalGeneration

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
model.generation_config.language ='sw'

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
    save_steps=500,
    eval_steps=500,
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
    compute_metrics=compute_metrics(tokenizer=tokenizer_general),
    tokenizer=processor_general.feature_extractor,
)

trainer_general.train()

#train model with swahili specified
trainer_swahili = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=common_voice_swahili["train"],
    eval_dataset=common_voice_swahili["test"],
    data_collator=data_collator_swahili,
    compute_metrics=compute_metrics(tokenizer=tokenizer_swahili),
    tokenizer=processor_swahili.feature_extractor,
)
trainer_swahili.train()

logout()