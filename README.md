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
I would like to thank Dr. Ye Kyaw Thu, Dr. Hnin Aye Thant, Ma Aye Hninn Khine, ​and Ma Yi Yi Chan Myae Win Shein for their guidance, support, and suggestions. The skills acquired from Dr. Ye Kyaw Thu's NLP Class helped me a lot in order to develop new ideas in NLP Field and this repo. And a shoutout to the creators of Rabbit Converter and jrgraphix.net's Unicode Character Table. These tools were super helpful to develop nlp-concepts especially for Burmese Language. Thanks.

### Acknowledgment 
We would like to thank everyone who contributed in the field of NLP and Myanmar NLP. And would like to thank Simbolo Servicio which is a branch of Simbolo for the financial support.

### References
References: 
[1] Ye Kyaw Thu, sylbreak, https://github.com/ye-kyaw-thu/sylbreak
[2] Y. K. Thu et al., "sylbreak4all: Regular Expressions for Syllable Breaking of Nine Major Ethnic Languages of Myanmar," 2021 16th International Joint Symposium on Artificial Intelligence and Natural Language Processing (iSAI-NLP), 2021, pp. 1-6, doi: 10.1109/iSAI-NLP54397.2021.9678188.
[3] Kaung Htet San, Pyidaungsu, https://github.com/kaunghtetsan275/pyidaungsu
[4] Maung, Zin & Mikami, Yoshiki. (2008). A Rule-based Syllable Segmentation of Myanmar Text. 
[5] Ye Kyaw Thu, NLP Class UTYCC, https://github.com/ye-kyaw-thu/NLP-Class
[6] Unicode Character Table, https://jrgraphix.net/r/Unicode/1000-109F
[7] Rabbit Converter, http://www.rabbit-converter.org/