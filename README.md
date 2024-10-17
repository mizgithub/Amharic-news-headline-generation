# Fine-Tuned Pretrained Transformer for Amharic News Headline Generation

This repository contains the code and resources for the paper **"Fine-Tuned Pretrained Transformer for Amharic News Headline Generation"**. The project aims to generate news headlines for Amharic articles using a **fine-tuned T5v1.1 model (t5-small)**. The model was trained on **over 70k Amharic news articles** and evaluated using **ROUGE-L, BLEU, and METEOR metrics**, achieving competitive performance compared to non-fine-tuned models and previous studies.

## 🔗 Paper Publication  
**Fine-Tuned Pretrained Transformer for Amharic News Headline Generation**.  
[Link to the Paper ](https://doi.org/10.1002/ail2.98)

## 📄 Abstract  
Amharic is an under-resourced language, which makes automatic news headline generation a challenging task due to the limited availability of high-quality datasets. This project fine-tunes the **T5v1.1 model (t5-small)** to generate Amharic news headlines using **TF-IDF optimization** and **Byte Pair Encoding (BPE) tokenization**.  

The system achieved:  
- **ROUGE-L**: 0.72  
- **BLEU**: 0.52  
- **METEOR**: 0.81  

These results significantly outperform the baseline **non-fine-tuned T5 model**, which achieved ROUGE-L: 0.1, BLEU: 0.03, and METEOR: 0.14.  
The contributions of this study provide insights for further improvements, such as increasing dataset size, exploring other transformer models, and developing **adaptive post-processing techniques**.
![image](https://github.com/user-attachments/assets/70ee2e45-235d-4668-8179-f4459689da4b)

---

## 🚀 How to Use the Code  

### Step 1: Clone the Repository  
```bash
git clone https://github.com/mizgithub/Amharic-news-headline-generation.git
cd Amharic-news-headline-generation
pip install -r requirements.txt
```

---
### Step 2: Creating tokenizer, Optional. You can also use already created tokenizer
#### Create tokenizer

```
from tokenization import custom_tokenizer
tokenizer = custom_tokenizer(dataset)
```

### Step3: Train or use the model
```
from amharic_nhg import ANHG

#Initialize the model
model = ANHG()

# Train the model
dataset = "path/to/dataset.csv"
model.train(dataset)

# Predict a headline
text = "Sample Amharic news article"
headline = model.predict(text)
print("Generated Headline:", headline)
```
```
Links to dataset: https://drive.google.com/drive/folders/1hK8s8Tk99lCoikCNoBHCGdvTr3rGOG9q?usp=sharing
```
```
Link to the finetuned model: https://drive.google.com/drive/folders/1BFBLTRZExBtqghG7pk5lha5emhQkNOgl?usp=sharing
```

