import pandas as pd
from tokenizers import Tokenizer
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

def Custom_tokenizer():
    def tokenize(self,dataset):
        dataset = pd.read_csv("./data/processed_data.csv")
        text = dataset['1']
        headline = dataset['0']
        ##Creating a bulk file for the tokenizer
        text_data=""
        for i in range(len(dataset)):
            text_data+=" "+dataset.iloc[i]['1']
            text_data+=" "+dataset.iloc[i]['0']
        #preparing a build data (combination of the headline and body of news article)
        with open("./data/bulk_data.txt",'w', encoding='utf=8') as file:
            file.write(text_data)

        trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
        tokenizer.pre_tokenizer = Whitespace()
        files = ["./data/bulk_data.raw"]
        tokenizer.train(files, trainer)
        tokenizer.save("./tokenizer/Amharic_tokenizer.json")
        print("Tokenizer successfully created and saved in ./tokenizer/Amharic_tokenizer.json");
    def encode(self,text):
        tokenizer = Tokenizer.from_file("Amharic_tokenizer.json")
        output = tokenizer.encode(text)
        return output
