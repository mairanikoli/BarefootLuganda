from huggingface_hub import notebook_login
notebook_login()

import datasets
from datasets import load_dataset, DatasetDict

common_voice = DatasetDict()

common_voice["train"] = load_dataset("mozilla-foundation/common_voice_16_0", "lg", split="train", trust_remote_code=True)
common_voice["validation"] = load_dataset("mozilla-foundation/common_voice_16_0", "lg", split="validation", trust_remote_code=True)
common_voice["test"] = load_dataset("mozilla-foundation/common_voice_16_0", "lg", split="test", trust_remote_code=True)

#from mp3 to wav
from pydub import AudioSegment

def convert_mp3_to_wav(mp3_file_path, wav_file_path):
    audio = AudioSegment.from_mp3(mp3_file_path)
    audio.export(wav_file_path, format="wav")


#downsample to match whisper sampling rate
#from datasets import Audio
#common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))
import librosa
import os


def resample_audio(batch):
    # load and resample audio data from 48 to 16kHz
    audio_path = batch["audio"]["path"]
    wav_path = audio_path.replace(".mp3", ".wav").replace("/mp3/", "/wav/")

    if not os.path.exists(wav_path):
        # Ensure the target directory exists
        os.makedirs(os.path.dirname(wav_path), exist_ok=True)
        # Convert MP3 to WAV
        convert_mp3_to_wav(audio_path, wav_path)
    
    # Update the script to load the WAV file instead of the MP3 file
    audio_data, sample_rate = soundfile.read(wav_path)
    batch["audio"]["array"] = audio_data
    batch["audio"]["sampling_rate"] = sample_rate
    
    # Assuming record['audio'] is a path to the audio file
    audio_data, _ = librosa.load(record['audio']['path'], sr=16000)  # Resample to 16000 Hz
    record['audio']['array'] = audio_data
    record['audio']['sampling_rate'] = 16000
    return record

common_voice = common_voice.map(resample_audio)

def prepare_dataset(batch):
    # compute log-Mel input features from input audio array 
    from transformers import WhisperFeatureExtractor
    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")

    from transformers import WhisperTokenizer
    #I have chosen Swahili in the same family as Luganda
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", task="transcribe")

    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids 
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch
    
common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=4)

#downsample to match whisper sampling rate
from datasets import Audio
common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))
    

def prepare_dataset_swahili(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array 
    from transformers import WhisperFeatureExtractor
    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")

    from transformers import WhisperTokenizer
    #tokenizer in swahili
    tokenizer_swahili = WhisperTokenizer.from_pretrained("openai/whisper-small", language="Swahili", task="transcribe")

    # Your existing preprocessing logic
    audio = batch["audio"]
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["labels"] = tokenizer_swahili(batch["sentence"]).input_ids
    return batch

common_voice_general = common_voice.map(prepare_dataset_general, remove_columns=common_voice.column_names["train"])
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

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

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
    data_collator=data_collator,
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
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)
trainer_swahili.train()
