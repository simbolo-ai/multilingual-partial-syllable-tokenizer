import re
import json
import pickle
import tensorflow
from typing import List
from tensorflow.keras.preprocessing.text import Tokenizer
from transformers import PreTrainedTokenizerFast

class MultilingualPartialSyllableTokenization:
    """
    MultilingualPartialSyllableTokenization class for tokenizing text using a Multilingual Partial Syllable Tokenizer.
    This class can be used for training, tokenizing, and saving the tokenizer model.

    Parameters:
    - vocab_size: The size of the vocabulary.
    - max_length: The maximum length of the sequences.
    - trunc_type: The truncation type for sequences ('pre' or 'post').
    - oov_tok: The out-of-vocabulary token.
    - model_path: The path to save or load the tokenizer model.
    """

    def __init__(self, vocab_size: int = 500, max_length: int = 500, trunc_type: str = 'post', oov_tok: str = 'OOV',
                 model_path: str = 'partial_syllable.model'):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.trunc_type = trunc_type
        self.oov_tok = oov_tok
        self.model_path = model_path
        self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token=self.oov_tok)

    def tokenize_text(self, input) -> str:
        """
        Introducing Multilingual Partial Tokenization for Baharmic Script—a novel rule-based tokenization method that avoids breaking into complete syllables.
        Through experimentation, its utility has been uncovered in keyword detection, effectively minimizing False Positive errors.
        Notably, this tokenization method is designed to align with the linguistic nuances of Baharmic-scripted languages, offering a powerful tool without requiring an exhaustive understanding of each specific language.

        Partial Syllable RE Pattern of Tokenizer: [Maybe Preceeded By][Maybe Followed By]{0 or more repetition}

        Partial-syllable-level Tokenization for specified languages
        1. burmese, 2. paoh, 3. shan, 4. mon, 5. rakhine, 6. pali
        7. Sgaw-karen, 8. pwo-karen, 9. pa'o, 10. karenni (also known as Kayah or Red Karen), 11. kayan (also known as Padaung)
        12. devangari, 13. gurmukhi, 14. gujarati, 15. oriya, 16. tamil, 17. telugu, 18. kannada,
        19. malayalam, 20. sinhala, 21. thai, 22. lao, 23. tibetan, 24. khmer,25. aiton, 26. phake


        Word-level Tokenization for English languages
        Character-level Tokenization for other languages
        """

        try:
            burmese_paoh_shan_karen_mon_rakhine_pali = '[က-ဪဿ၌-၏ၐ-ၕၚ-ၝၡၥၦၮ-ၰၵ-ႁႎ႐-႙႟][ါ-ှၖ-ၙၞ-ၠၢ-ၤၧ-ၭၱ-ၴႂ-ႍႏႚ-႞ꩻ]{0,}|'
            devangari = '[ऄ-हॐक़-ॡ।-ॿ][ऀ-ःऺ-ॏ॑-ॗॢ-ॣ]{0,}|'
            bengali   = '[ঀঅ-ঌএ-ঐও-নপ-রলশ-হঽৎড়-ঢ়য়-ৡ০-৽][ঁ-ঁ়া-ৄে-ৈো-্ৗৢ-ৣ৾]{0,}|'
            gurmukhi  = '[ਅ-ਊਏ-ਐਓ-ਨਪ-ਰਲ-ਲ਼ਵ-ਸ਼ਸ-ਹਖ਼-ੜਫ਼੦-੯ੲ-ੴ੶][ਁ-ਃ਼ਾ-ੂੇ-੍ੑੰ-ੱੵ]{0,}|'
            gujarati  = '[અ-ઍએ-ઑઓ-નપ-રલ-ળવ-હઽૐૠ-ૡ૦-૱ૹ][ઁ-ઃ઼ા-ૅે-ૉો-્ૢ-ૣૺ-૿]{0,}|'
            oriya     = '[ଅ-ଌଏ-ଐଓ-ନପ-ରଲ-ଳଵ-ହଽଡ଼-ଢ଼ୟ-ୡ୦-୷][ଁ-ଃ଼ା-ୄେ-ୈୋ-୍୕-ୗୢ-ୣ]{0,}|'
            tamil     = '[ஃஅ-ஊஎ-ஐஒ-கங-சஜஞ-டண-தந-பம-ஹௐ௦-௺][ஂா-ூெ-ைொ-்ௗ]{0,}|'
            telugu    = '[అ-ఌఎ-ఐఒ-నప-హఽౘ-ౚౠ-ౡ౦-౯౸-౿][ఀ-ఄా-ౄె-ైొ-్ౕ-ౖౢ-ౣ]{0,}|'
            kannada   = '[ಀ಄-ಌಎ-ಐಒ-ನಪ-ಳವ-ಹಽೞೠ-ೡ೦-೯ೱ-ೲ][ಁ-ಃ಼ಾ-ೄೆ-್ೕ-ೖೢ-ೣ]{0,}|'
            malayalam = '[അ-ഌഎ-ഐഒ-ഺ൏ൔ-ൖ൘-ൡ൦-ൿ][ഀ-ഃ഻-ൄെ-ൈൊ-ൎൗൢ-ൣ]{0,}|'
            sinhala   = '[අ-ඖක-නඳ-රලව-ෆෘ-ෟ෦-෯෴][ං-ඃ්ා-ුූෲ-ෳ]{0,}|'
            thai      = '[ก-ฯา-ำเ-ๆ๏-๛][ะ-ั|ิ-ฺ|็-๎]{0,}|'
            lao       = '[ກ-ຂຄງ-ຈຊຍດ-ທນ-ຟມ-ຣລວສ-ຫອ-ຯາ-ຳຽເ-ໄໆ໐-໙ໜ-ໟ][ະ-ັ|ິ-ູ|ົ-ຼ|່-ໍ]{0,}|'
            tibetan   = '[ༀ-༗༚-༴༶༸༺-༽ཀ-ཇཉ-ཬ྅ྈ-ྌ྾-࿔࿙-࿚][༘-༹༙༵༷༾-༿ཱ-྄྆-྇ྍ-ྗྙ-ྼ]{0,}|'
            khmer     = '[ក-ឳ។-ៜ០-៩៰-៹][ា-៓៝]{0,}|'
            english   = '[a-zA-Z]+|'
            other_lan = '.'
            pattern   = r'('+english+burmese_paoh_shan_karen_mon_rakhine_pali+devangari+bengali+gurmukhi+gujarati+oriya+tamil+telugu+kannada+malayalam+sinhala+thai+lao+tibetan+khmer+other_lan+')'
            return re.sub(pattern,r"\1 ", input)
        except Exception as e:
          print(e)

    def train(self, train_data_path: str = 'train.txt') -> None:
        with open(train_data_path, 'r') as file:
            training_sentences = [self.tokenize_text(i) for i in file]

        self.tokenizer.fit_on_texts(training_sentences)

        tokenizer_model_path = f"{self.model_path}.pkl"
        with open(tokenizer_model_path, 'wb') as pickle_file:
            pickle.dump(self.tokenizer, pickle_file)

        vocab_json_path = f"{self.model_path}_vocab.json"
        with open(vocab_json_path, 'w') as json_file:
            json.dump(self.tokenizer.word_index, json_file)

    def load_tokenizer(self, model_path: str) -> None:
        with open(model_path, 'rb') as pickle_file:
            self.tokenizer = pickle.load(pickle_file)

    def encode(self, text: str) -> List[int]:
        """
        Encodes a text using the loaded tokenizer.

        Parameters:
        - text: The input text to be encoded.

        Returns:
        - encoded_sequence: The encoded sequence of integers.
        """
        # Tokenize the text using the loaded tokenizer
        tokenized_text = self.tokenizer.texts_to_sequences([self.tokenize_text(text)])

        # Flatten the list of lists to get a single list of integers
        encoded_sequence = [item for sublist in tokenized_text for item in sublist]

        return encoded_sequence

    def decode(self, encoded_sequence: List[int]) -> str:
        """
        Decodes an encoded sequence using the loaded tokenizer.

        Parameters:
        - encoded_sequence: The encoded sequence of integers.

        Returns:
        - decoded_text: The decoded text.
        """
        # Convert the encoded sequence back to text using the loaded tokenizer
        decoded_text = self.tokenizer.sequences_to_texts([encoded_sequence])

        return decoded_text[0]
    def get_vocab(self) -> dict:
        # Check if tokenizer is initialized
        if not hasattr(self, 'tokenizer') or self.tokenizer is None:
            raise ValueError("Tokenizer not initialized. Load the tokenizer using load_tokenizer method.")

        # Get the word index directly from the tokenizer
        return self.tokenizer.word_index