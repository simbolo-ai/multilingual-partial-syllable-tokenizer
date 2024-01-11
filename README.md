# multilingual-partial-syllable-tokenizer
We would like to introduce Multilingual Partial is Tokenization—a novel rule-based tokenization method that avoids breaking into complete syllables. Through experimentation, its utility has been uncovered in keyword detection, effectively minimizing False Positive errors and helping a lot in Burmese's rules-based+machine learning name recognition. Notably, this tokenization method is designed to align with the linguistic nuances of languages, but without requiring an exhaustive understanding of each specific language. Now now it is integreated with frequencey based approach to generate tokens.

###  Related Work
Numerous researchers have undertaken extensive investigations into syllable tokenization. This exposition aims to delineate various tokenization methodologies, with particular emphasis on selected examples. Dr. Ye Kyaw's sylbreak tokenizer, as detailed in the associated repository (https://github.com/ye-kyaw-thu/sylbreak), employs regular expressions [1] to accomplish syllable tokenization. The research article titled "sylbreak4all: Regular Expressions for Syllable Breaking of Nine Major Ethnic Languages of Myanmar" authored by Dr. Ye Kyaw Thaw and other researchers introduces a syllable tokenization approach applicable to nine languages, employing Regular Expression [2].

Furthermore, a syllable tokenizer designed for four languages, accessible at https://github.com/kaunghtetsan275/pyidaungsu, also utilizes a combination of Regular Expression and Rule-Based techniques [3]. Additionally, Maung, Zin, and Mikami, Yoshiki, in their publication "A Rule-based Syllable Tokenization of Myanmar Text" [4], present a rules-based approach for syllable tokenization.

### Multilingual Partial-syllable Tokenization

Partial Syllable RE Pattern of Tokenizer: [Maybe Preceded By][Maybe Followed By]{0 or more repetition}
**Partial-syllable-level Tokenization for specified languages**
1. burmese, 2. paoh, 3. shan, 4. mon, 5. rakhine, 6. pali
7. Sgaw-karen, 8. pwo-karen, 9. pa'o, 10. karenni (also known as Kayah or Red Karen), 11. kayan (also known as Padaung)            
12. devangari, 13. gurmukhi, 14. gujarati, 15. oriya, 16. tamil, 17. telugu, 18. kannada, 
19. malayalam, 20. sinhala, 21. thai, 22. lao, 23. tibetan, 24. khmer,25. aiton, 26. phake

Word-level Tokenization for English languages
Character-level Tokenization for other languages

### How to use
```
#pip install simmpst==0.1.1
import simmpst
from simmpst.tokenization import MultilingualPartialSyllableTokenization

# First Initialize the class, and provide vocab size to build the tokenizer
tokenizer = MultilingualPartialSyllableTokenization(vocab_size=500)

# Please provide your training data in .txt file, then the tokenizer model file will be saved in your folder
tokenizer.train(train_data_path='/train_small.txt')

# Load Your Model
tokenizer_path = '/content/partial_syllable.model.pkl'


# Using with Dataloader that can be used with transoformer model
import re
import torch
import json
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def collate_fn(batch, maxlen=100, truncating='post', padding='post'):
    return torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0)

class CustomDataset(Dataset):
    def __init__(self, texts, tokenizer_path):
        self.texts = texts

        # Load the tokenizer from the saved file
        with open(tokenizer_path, 'rb') as f:
            self.tokenizer = pickle.load(f)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx, maxlen=100, truncating='post', padding='post'):
        print("Idx", idx)
        text = tokenizer.tokenize_text(self.texts[idx])

        # Tokenize the text using your custom tokenizer
        testing_sequences = self.tokenizer.texts_to_sequences([text])[0]
        print(testing_sequences)
        tokens = pad_sequences([testing_sequences], maxlen=maxlen, truncating=truncating, padding=padding)[0]

        # Convert NumPy array to PyTorch tensor
        tokens = torch.tensor(tokens)

        # Print information for debugging
        print(f"Text: {text}")
        print(f"Tokens: {tokens}")

        return tokens

# Replace `texts` with your actual data
texts = ["ဝီကီပီးဒီးယားသည် သုံးစွဲသူများက ပူးပေါင်း၍ ရေးသားတည်းဖြတ်သော စွယ်စုံကျမ်းဖြစ်ပါသည်။", "ဝီကီဟု ခေါ်သော ဝက်ဘ်ဆိုက် ပုံစံတစ်မျိုးကို အသုံးပြု၍ ပူးပေါင်းရေးသားခြင်းကို အဆင်ပြေစေရန် စီမံထားခြင်း ဖြစ်ပါသည်။", "သုံးစွဲသူများမှ နာရီအလိုက် ပြင်ဆင်မှုပေါင်း များစွာကို ပြုလုပ်၍ ဝီကီပီးဒီးယားကို ပို၍ကောင်းမွန်အောင် ဆောင်ရွက်နေကြပါသည်။","မြန်မာဝီကီပီးဒီးယားတွင် ယူနီကုဒ် ၅.၂ စံနှုန်းကို လိုက်နာသော မည်သည့်ဖောင့်အမျိုးအစား နှင့်မဆို ဖတ်ရှုခြင်း၊ တည်းဖြတ်ခြင်းများ ပြုလုပ်နိုင်ပါသည်။"]


# Create your dataset
dataset = CustomDataset(texts, tokenizer_path)

# Set your desired parameters
maxlen = 120  # Replace with your desired value
truncating = 'post'  # Replace with 'pre' or 'post' based on your preference
padding = 'post'  # Replace with 'pre' or 'post' based on your preference

# Create a DataLoader with the collate function for padding
batch_size = 2  # Replace with your desired batch size
dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda b: collate_fn(b, maxlen, truncating, padding))

# Iterate through the dataloader and print batches
for batch in dataloader:
    print(batch)

# Sample Outputs:
[455, 434, 1, 1, 262, 3, 5, 147, 359, 69, 22, 21, 391, 105, 7, 58, 49, 81, 9, 37, 19, 16, 31, 226, 25, 281, 78, 39, 19, 8, 54, 3, 5, 1]
Text: ဝီ ကီ ပီး ဒီး ယား သ ည်   သုံး စွဲ သူ များ က   ပူး ပေါ င်း ၍   ရေး သား တ ည်း ဖြ တ် သော   စွ ယ် စုံ ကျ မ်း ဖြ စ် ပါ သ ည် ။ 
Tokens: tensor([455, 434,   1,   1, 262,   3,   5, 147, 359,  69,  22,  21, 391, 105,
          7,  58,  49,  81,   9,  37,  19,  16,  31, 226,  25, 281,  78,  39,
         19,   8,  54,   3,   5,   1,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0], dtype=torch.int32)
```

### Some Research on our tokenizer (version 1)
Distinguishing Burmese Male and Female Names: https://github.com/simbolo-ai/multilingual-partial-syllable-tokenizer/blob/main/Binary%20Brigade-%20Distinguishing%20Burmese%20Male%20and%20Female%20Names.pdf

### Bibtex
```
@article{SaPhyoThuHtet,
  title={multilingual-partial-syllable-tokenizer},
  author={Sa Phyo Thu Htet},
  journal={https://github.com/SaPhyoThuHtet/multilingual-partial-syllable-tokenizer},
  year={2019-2024}
}
```

### Acknowledgment Statement from Sa Phyo Thu Htet
I woul like thank Dr. Ye Kyaw Thu, Dr. Hnin Aye Thant, Ma Aye Hninn Khine, ​and Ma Yi Yi Chan Myae Win Shein for their guidance, support, and suggestions. The skills acquired from Dr. Ye Kyaw Thu's NLP Class helped me a lot to develop new ideas in the NLP Field and this repo. And a shoutout to the creators of Rabbit Converter and jrgraphix.net's Unicode Character Table. These tools were super helpful in developing concepts, especially for the Burmese Language. Thanks.

### Acknowledgment 
We would like to thank everyone who contributed to the field of NLP and Myanmar NLP. And would like to thank Simbolo Servicio, a branch of Simbolo, for the financial support.

### References
References: 
[1] Ye Kyaw Thu, sylbreak, https://github.com/ye-kyaw-thu/sylbreak
[2] Y. K. Thu et al., "sylbreak4all: Regular Expressions for Syllable Breaking of Nine Major Ethnic Languages of Myanmar," 2021 16th International Joint Symposium on Artificial Intelligence and Natural Language Processing (iSAI-NLP), 2021, pp. 1-6, doi: 10.1109/iSAI-NLP54397.2021.9678188.
[3] Kaung Htet San, Pyidaungsu, https://github.com/kaunghtetsan275/pyidaungsu
[4] Maung, Zin & Mikami, Yoshiki. (2008). A Rule-based Syllable Segmentation of Myanmar Text. 
[5] Ye Kyaw Thu, NLP Class UTYCC, https://github.com/ye-kyaw-thu/NLP-Class
[6] Unicode Character Table, https://jrgraphix.net/r/Unicode/1000-109F
[7] Rabbit Converter, http://www.rabbit-converter.org/
