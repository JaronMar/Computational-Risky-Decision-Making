from utils import *
from tensorflow.keras import backend as K
from tensorflow.keras.utils import Sequence
import random
import tensorflow as tf

class DataGenerator(Sequence):
    def __init__(self, document_texts, vocab_dictionary, categories, category_dict, max_len, batch=1):
        self.words = [self.generate_input_words_vector(text, vocab_dictionary) for text in document_texts]
        self.categories = [category_dict[category] for category in categories]
        self.number_categories = len(category_dict)
        self.indexes = [i for i in range(len(self.words))]
        self.batch_size = batch
        self.max_len = max_len

    @staticmethod
    def generate_input_words_vector(document_word_list, vocab_dict, limit=0, mask_percentage=0.0):
        document_word_list = list(filter(lambda x: x in vocab_dict, document_word_list))
        if limit == 0:
            return [vocab_dict[word] if random.random() >= mask_percentage else 0 for word in document_word_list]
        else:
            output = [vocab_dict[word] for word in document_word_list]
            if len(output) >= limit:
                return output[:limit]
            else:
                return output + ([0] * (limit - len(output)))

    def __len__(self):
        return len(self.words)

    def __getitem__(self, item):
        pass

class PredictionGenerator(DataGenerator):

    def __getitem__(self, idx):
        input_words = []
        input_random_category = []
        output_category = []
        output_true_category = []

        random_category = random.randint(0, self.number_categories - 1)

        document_output = [0] * self.number_categories
        document_output[self.categories[idx]] = 1

        input_words += [self.words[idx][:self.max_len]]
        input_random_category += [[random_category]]
        output_true_category += [[1] if self.categories[idx] == random_category else [0]]
        output_category += [document_output]

        padded_words = tf.keras.preprocessing.sequence.pad_sequences(input_words,
                                                                      padding='post', maxlen=self.max_len)

        return [padded_words, np.array(input_random_category)], [np.array(output_true_category),
                                                                          np.array(output_category)]

class Cat2VecPredictor:

    def __init__(self, version, max_len=50):
        self.model_name = "./cat2vec/model/cat2vec_{0}".format(version)
        self.model = load_model(self.model_name)
        self.category_list = open("./cat2vec/model/categories_{0}.txt".format(version), "r",
                             encoding="utf8").read().strip().split("\n")
        self.category_dict = {c.split(",")[0]: int(c.split(",")[1]) for c in self.category_list}
        self.category_dict_index = {int(c.split(",")[1]):c.split(",")[0] for c in self.category_list}
        #
        vocab = open("./cat2vec/model/vocab_{0}.txt".format(version), "r", encoding="utf8").read().strip().split(
            "\n")
        self.vocab_dict = {v.split(",")[0]: int(v.split(",")[1]) for v in vocab}

        self.category_embeddings = self.model.get_layer('cat_embedding').get_weights()[0]
        self.max_len = max_len


    def predict(self, text_list):
        document_texts = [preprocess_text(text) for text in text_list]

        get_all_layer_outputs = K.function([self.model.layers[0].input],
                                           [self.model.get_layer('document_output').input])

        categories = ["life" for i in range(len(text_list))]

        generator = PredictionGenerator(document_texts, self.vocab_dict, categories, self.category_dict, self.max_len)
        predicted = []
        for i in range(len(text_list)):
            data = generator.__getitem__(i)
            layer_output = get_all_layer_outputs([data[0][0]])[0][0]
            distances = squared_euclidean_distance(np.asarray([layer_output]), self.category_embeddings).tolist()[0]
            predicted += [self.category_dict_index[distances.index(min(distances))]]
        return predicted


