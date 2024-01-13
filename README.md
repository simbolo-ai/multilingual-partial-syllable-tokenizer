# multilingual-partial-syllable-tokenizer
![stokenizer](https://github.com/simbolo-ai/multilingual-partial-syllable-tokenizer/assets/71957964/9d1e2684-9fde-4ae9-8521-8364d536aeea)

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

### How to use (Getting Started)
```
#pip install simmpst==0.1.1
import simmpst
from simmpst.tokenization import MultilingualPartialSyllableTokenization

# First Initialize the class, and provide vocab size to build the tokenizer
tokenizer = MultilingualPartialSyllableTokenization(vocab_size=500)

# Replace `texts` with your actual data
sample_text = "ဝီကီပီးဒီးယားသည် သုံးစွဲသူများက ပူးပေါင်း၍ ရေးသားတည်းဖြတ်သော စွယ်စုံကျမ်းဖြစ်ပါသည်။"
tokenize_text = tokenizer.tokenize_text(sample_text)

print(tokenize_text)
#ဝီ ကီ ပီး ဒီး ယား သ ည်   သုံး စွဲ သူ များ က   ပူး ပေါ င်း ၍   ရေး သား တ ည်း ဖြ တ် သော   စွ ယ် စုံ ကျ မ်း ဖြ စ် ပါ သ ည် ။ 
```

### How to train and load with DataLoader
```
import simmpst
from simmpst.tokenization import MultilingualPartialSyllableTokenization

tokenizer = MultilingualPartialSyllableTokenization(vocab_size=500)

# Please provide your training data in .txt file, then the tokenizer model file will be saved in your folder
tokenizer.train(train_data_path='/train_small.txt')

# Using with Dataloader that can be used with transoformer model
import re
import torch
import json
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader

def collate_fn(batch, maxlen=100, truncating='post', padding='post'):
    return torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0)

class CustomDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx, maxlen=100, truncating='post', padding='post'):
        print("Idx", idx)
        text = tokenizer.tokenize_text(self.texts[idx])
        tokens = tokenizer.encode(text)
        
        # Convert NumPy array to PyTorch tensor
        tokens_tensors = torch.tensor(tokens)

        # Print information for debugging
        print(f"Text: {text}")
        print(f"Tokens_numpy: {tokens}")
        print(f"Tokens_tensor: {tokens_tensors}")

        return tokens_tensors

# Replace `texts` with your actual data
texts = ["ဝီကီပီးဒီးယားသည် သုံးစွဲသူများက ပူးပေါင်း၍ ရေးသားတည်းဖြတ်သော စွယ်စုံကျမ်းဖြစ်ပါသည်။", "ဝီကီဟု ခေါ်သော ဝက်ဘ်ဆိုက် ပုံစံတစ်မျိုးကို အသုံးပြု၍ ပူးပေါင်းရေးသားခြင်းကို အဆင်ပြေစေရန် စီမံထားခြင်း ဖြစ်ပါသည်။", "သုံးစွဲသူများမှ နာရီအလိုက် ပြင်ဆင်မှုပေါင်း များစွာကို ပြုလုပ်၍ ဝီကီပီးဒီးယားကို ပို၍ကောင်းမွန်အောင် ဆောင်ရွက်နေကြပါသည်။","မြန်မာဝီကီပီးဒီးယားတွင် ယူနီကုဒ် ၅.၂ စံနှုန်းကို လိုက်နာသော မည်သည့်ဖောင့်အမျိုးအစား နှင့်မဆို ဖတ်ရှုခြင်း၊ တည်းဖြတ်ခြင်းများ ပြုလုပ်နိုင်ပါသည်။"]


# Create your dataset
dataset = CustomDataset(texts)

# Set your desired parameters
maxlen = 120  # Replace with your desired value
truncating = 'post'  # Replace with 'pre' or 'post' based on your preference
padding = 'post'  # Replace with 'pre' or 'post' based on your preference

# Create a DataLoader with the collate function for padding
batch_size = 1  # Replace with your desired batch size
dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda b: collate_fn(b, maxlen, truncating, padding))

# Iterate through the dataloader and print batches
for batch in dataloader:
    print(batch)
    print("========================================")

# Sample Output
Idx 0
Text: ဝီ ကီ ပီး ဒီး ယား သ ည်   သုံး စွဲ သူ များ က   ပူး ပေါ င်း ၍   ရေး သား တ ည်း ဖြ တ် သော   စွ ယ် စုံ ကျ မ်း ဖြ စ် ပါ သ ည် ။ 
Tokens_numpy: [455, 434, 1, 1, 262, 3, 5, 147, 359, 69, 22, 21, 391, 105, 7, 58, 49, 81, 9, 37, 19, 16, 31, 226, 25, 281, 78, 39, 19, 8, 54, 3, 5, 1]
Tokens_tensor: tensor([455, 434,   1,   1, 262,   3,   5, 147, 359,  69,  22,  21, 391, 105,
          7,  58,  49,  81,   9,  37,  19,  16,  31, 226,  25, 281,  78,  39,
         19,   8,  54,   3,   5,   1])
tensor([[455, 434,   1,   1, 262,   3,   5, 147, 359,  69,  22,  21, 391, 105,
           7,  58,  49,  81,   9,  37,  19,  16,  31, 226,  25, 281,  78,  39,
          19,   8,  54,   3,   5,   1]])
========================================
Idx 1
Text: ဝီ ကီ ဟု   ခေါ် သော   ဝ က် ဘ် ဆို က်   ပုံ စံ တ စ် မျိုး ကို   အ သုံး ပြု ၍   ပူး ပေါ င်း ရေး သား ခြ င်း ကို   အ ဆ င် ပြေ စေ ရ န်   စီ မံ ထား ခြ င်း   ဖြ စ် ပါ သ ည် ။ 
Tokens_numpy: [455, 434, 82, 207, 31, 29, 6, 1, 74, 6, 159, 277, 9, 8, 92, 18, 4, 147, 97, 58, 391, 105, 7, 49, 81, 66, 7, 18, 4, 38, 2, 1, 148, 11, 10, 181, 354, 106, 66, 7, 19, 8, 54, 3, 5, 1]
Tokens_tensor: tensor([455, 434,  82, 207,  31,  29,   6,   1,  74,   6, 159, 277,   9,   8,
         92,  18,   4, 147,  97,  58, 391, 105,   7,  49,  81,  66,   7,  18,
          4,  38,   2,   1, 148,  11,  10, 181, 354, 106,  66,   7,  19,   8,
         54,   3,   5,   1])
tensor([[455, 434,  82, 207,  31,  29,   6,   1,  74,   6, 159, 277,   9,   8,
          92,  18,   4, 147,  97,  58, 391, 105,   7,  49,  81,  66,   7,  18,
           4,  38,   2,   1, 148,  11,  10, 181, 354, 106,  66,   7,  19,   8,
          54,   3,   5,   1]])
========================================
Idx 2
Text: သုံး စွဲ သူ များ မှ   နာ ရီ အ လို က်   ပြ င် ဆ င် မှု ပေါ င်း   များ စွာ ကို   ပြု လု ပ် ၍   ဝီ ကီ ပီး ဒီး ယား ကို   ပို ၍ ကော င်း မွ န် အော င်   ဆော င် ရွ က် နေ ကြ ပါ သ ည် ။ 
Tokens_numpy: [147, 359, 69, 22, 41, 127, 188, 4, 109, 6, 34, 2, 38, 2, 73, 105, 7, 22, 177, 18, 97, 101, 14, 58, 455, 434, 1, 1, 262, 18, 112, 58, 94, 7, 299, 10, 133, 2, 88, 2, 170, 6, 57, 50, 54, 3, 5, 1]
Tokens_tensor: tensor([147, 359,  69,  22,  41, 127, 188,   4, 109,   6,  34,   2,  38,   2,
         73, 105,   7,  22, 177,  18,  97, 101,  14,  58, 455, 434,   1,   1,
        262,  18, 112,  58,  94,   7, 299,  10, 133,   2,  88,   2, 170,   6,
         57,  50,  54,   3,   5,   1])
tensor([[147, 359,  69,  22,  41, 127, 188,   4, 109,   6,  34,   2,  38,   2,
          73, 105,   7,  22, 177,  18,  97, 101,  14,  58, 455, 434,   1,   1,
         262,  18, 112,  58,  94,   7, 299,  10, 133,   2,  88,   2, 170,   6,
          57,  50,  54,   3,   5,   1]])
========================================
Idx 3
Text: မြ န် မာ ဝီ ကီ ပီး ဒီး ယား တွ င်   ယူ နီ ကု ဒ်   ၅ . ၂   စံ နှု န်း ကို   လို က် နာ သော   မ ည် သ ည့် ဖော င့် အ မျိုး အ စား   နှ င့် မ ဆို   ဖ တ် ရှု ခြ င်း ၊   တ ည်း ဖြ တ် ခြ င်း များ   ပြု လု ပ် နို င် ပါ သ ည် ။ 
Tokens_numpy: [48, 10, 86, 455, 434, 1, 1, 262, 20, 2, 155, 197, 107, 305, 87, 45, 277, 348, 27, 18, 109, 6, 127, 31, 12, 5, 3, 52, 407, 15, 4, 92, 4, 132, 24, 15, 12, 74, 114, 16, 360, 66, 7, 13, 9, 37, 19, 16, 66, 7, 22, 97, 101, 14, 42, 2, 54, 3, 5, 1]
Tokens_tensor: tensor([ 48,  10,  86, 455, 434,   1,   1, 262,  20,   2, 155, 197, 107, 305,
         87,  45, 277, 348,  27,  18, 109,   6, 127,  31,  12,   5,   3,  52,
        407,  15,   4,  92,   4, 132,  24,  15,  12,  74, 114,  16, 360,  66,
          7,  13,   9,  37,  19,  16,  66,   7,  22,  97, 101,  14,  42,   2,
         54,   3,   5,   1])
tensor([[ 48,  10,  86, 455, 434,   1,   1, 262,  20,   2, 155, 197, 107, 305,
          87,  45, 277, 348,  27,  18, 109,   6, 127,  31,  12,   5,   3,  52,
         407,  15,   4,  92,   4, 132,  24,  15,  12,  74, 114,  16, 360,  66,
           7,  13,   9,  37,  19,  16,  66,   7,  22,  97, 101,  14,  42,   2,
          54,   3,   5,   1]])
========================================
```

### Some Research on our tokenizer (version 1)
[Distinguishing Burmese Male and Female Names:](https://github.com/simbolo-ai/multilingual-partial-syllable-tokenizer/blob/main/Binary%20Brigade-%20Distinguishing%20Burmese%20Male%20and%20Female%20Names.pdf) 

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

[4] Maung, Zin & Mikami, Yoshiki. (2008). A Rule-based Syllable Segmentation of Myanmar Text

[5] Ye Kyaw Thu, NLP Class UTYCC, https://github.com/ye-kyaw-thu/NLP-Class

[6] Unicode Character Table, https://jrgraphix.net/r/Unicode/1000-109F

[7] Rabbit Converter, http://www.rabbit-converter.org/
