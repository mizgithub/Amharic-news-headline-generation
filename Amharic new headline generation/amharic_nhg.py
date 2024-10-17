import os
import logging
import nltk
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pyarrow as pa
import pyarrow.dataset as ds
import pandas as pd
from datasets import Dataset
from transformers import TFAutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
import keras_nlp
from transformers.keras_callbacks import KerasMetricCallback
import pickle
from transformers import pipeline
from transformers import PreTrainedTokenizerFast


MAX_INPUT_LENGTH = 300  # Maximum length of the input to the model
MAX_TARGET_LENGTH = 31  # Maximum length of the output by the model

# This notebook is built on the t5-small checkpoint from the Hugging Face Model Hub
MODEL_CHECKPOINT = "t5-small"

def ANHG():
    def train(self,dataset):
        #loading tokenizer
        tokenizer = PreTrainedTokenizerFast(tokenizer_file="./tokenizer/Amharic_tokenizer.json")
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        #organizing the dataset for training

        headline = dataset["0"]
        text = dataset["1"]
        dataset_dict = []
        for i in range(len(headline)):
            data = {}
            data['document'] = text[i]
            data['summary'] = headline[i]
            dataset_dict.append(data)
        df = pd.DataFrame(dataset_dict)
        raw_datasets = Dataset(pa.Table.from_pandas(df))
        raw_datasets = raw_datasets.train_test_split(
        test_size=0.2
        )
        if MODEL_CHECKPOINT in ["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"]:
            prefix = "summarize: "
        else:
            prefix = ""
        ##tokenizing the dataset using the custom tokenizer built
        def preprocess_function(examples):
            inputs = [prefix + doc for doc in examples["document"]]
            model_inputs = tokenizer(inputs, max_length=MAX_INPUT_LENGTH, truncation=True)

            # Setup the tokenizer for targets
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(
                    examples["summary"], max_length=MAX_TARGET_LENGTH, truncation=True
                )

            model_inputs["labels"] = labels["input_ids"]

            return model_inputs
        
        tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)

        ##loading the t5-small checkpoint as a base model
        model = TFAutoModelForSeq2SeqLM.from_pretrained("t5-small") ##hugging face check point
        
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="tf")
        ## training and testing split
        train_dataset = tokenized_datasets["train"].to_tf_dataset(
            batch_size=8,
            columns=["input_ids", "attention_mask", "labels"],
            shuffle=False,
            collate_fn=data_collator,
        )
        test_dataset = tokenized_datasets["test"].to_tf_dataset(
            batch_size=8,
            columns=["input_ids", "attention_mask", "labels"],
            shuffle=False,
            collate_fn=data_collator,
        )
        generation_dataset = (
            tokenized_datasets["test"]
            .select(list(range(200)))
            .to_tf_dataset(
                batch_size=8,
                columns=["input_ids", "attention_mask", "labels"],
                shuffle=False,
                collate_fn=data_collator,
            )
        )
        # defining optimizer and learning rates
        optimizer = keras.optimizers.Adam(learning_rate=1e-3)
        model.compile(optimizer=optimizer)

        model.summary()
        #preparing evaluation metrics
        rouge_l = keras_nlp.metrics.RougeL()
        def compute_metrics(eval_predictions):
            predictions, labels = eval_predictions
            decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
            for label in labels:
                label[label < 0] = tokenizer.pad_token_id  # Replace masked label tokens
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            result = rouge_l(decoded_labels, decoded_predictions)
            # We will print only the F1 score, you can use other aggregation metrics as well
            result = {"RougeL": result["f1_score"]}

            return result
        metric_callback = KerasMetricCallback(
                metric_fn=compute_metrics,
                eval_dataset=test_dataset,
                predict_with_generate=True,
                    )

        
        # history = model.fit(
        #     train_dataset, validation_data=test_dataset, epochs=20,callbacks=metric_callback
        # )
        ## the training will be step by step, output model of each training step will be saved
        
        def join_hist(hist):
            with open('./models/history_last_chkpt', 'rb') as file_pi:
                prev_hist = pickle.load(file_pi)
            prev_hist['loss'].append(hist['loss'][0])
            prev_hist['val_loss'].append(hist['val_loss'][0])
            with open('./models/history_last_chkpt', 'wb') as file_pi:
                pickle.dump(prev_hist,file_pi)

        for i in range(20):
            print("Epoch "+str(i+1)+" of 20\n")
            history = model.fit(
                train_dataset, validation_data=test_dataset, epochs=1
            )
            model.save_pretrained("./models/t5small_last_chkpt", from_pt=True)
            join_hist(history.history)
            print("Check point saved in models folder.")
        def summarize(self, text):

            ##loading the trained model  
            model = TFAutoModelForSeq2SeqLM.from_pretrained("./models/t5small_last_chkpt")
            # loading tokenizer
            tokenizer = PreTrainedTokenizerFast(tokenizer_file="./tokenizer/Amharic_tokenizer.json")
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, framework="tf")

            predicted = summarizer(
                text,
                min_length=0,
                max_length=31,
            )

            return predicted
