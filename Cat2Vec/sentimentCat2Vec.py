
from utils import *
from tensorflow.keras import backend as K
from tensorflow.keras.utils import Sequence
import random
import tensorflow as tf
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Dense

class DataGenerator(Sequence):
    def __init__(self, document_texts, vocab_dictionary, categories, category_dict, batch=1):
        self.words = [self.generate_input_words_vector(text, vocab_dictionary) for text in document_texts]
        self.categories = [category_dict[category] for category in categories]
        self.number_categories = len(category_dict)
        self.indexes = [i for i in range(len(self.words))]
        self.batch_size = batch

    @staticmethod
    def generate_input_words_vector(document_word_list, vocab_dict):
        document_word_list = list(filter(lambda x: x in vocab_dict, document_word_list))
        return [vocab_dict[word] for word in document_word_list]

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

        input_words += [self.words[idx][:75]]
        input_random_category += [[random_category]]
        output_true_category += [[1] if self.categories[idx] == random_category else [0]]
        output_category += [document_output]

        padded_words = tf.keras.preprocessing.sequence.pad_sequences(input_words,
                                                                      padding='post', maxlen=50)

        return [padded_words, np.array(input_random_category)], [np.array(output_true_category),
                                                                          np.array(output_category)]

class SentimentCat2VecPredictor:

    def __init__(self, version):
        self.model_name = "./model/cat2vec_{0}".format(version)
        self.model = load_model(self.model_name)

        self.category_list = open("./model/categories_{0}.txt".format(version), "r",
                                 encoding="utf8").read().strip().split("\n")

        self.category_dict = {c.split(",")[0]: int(c.split(",")[1]) for c in self.category_list}
        # self.category_dict_index = {int(c.split(",")[1]):c.split(",")[0] for c in self.category_list}


        vocab = open("./model/vocab_{0}.txt".format(version), "r", encoding="utf8").read().strip().split(
                "\n")

        self.vocab_dict =  {v.split(",")[0]: int(v.split(",")[1]) for v in vocab}
        self.category_embeddings = self.model.get_layer('cat_embedding').get_weights()[0]

    def get_category_embeddings(self):
        return self.model.get_layer('cat_embedding').get_weights()[0]

    def predict(self, text_list):
        document_texts = [preprocess_text(text) for text in text_list]

        model_layer_outputs = K.function([self.model.layers[0].input],
                                            [self.model.get_layer('document_output').input])

        categories = ["life" for i in range(len(text_list))]

        generator = PredictionGenerator(document_texts, self.vocab_dict, categories, self.category_dict)
        predicted = []
        for i in range(len(text_list)):
            data = generator.__getitem__(i)
            layer_output = model_layer_outputs([data[0]])[0][0]
            distances = squared_euclidean_distance(np.asarray([layer_output]), self.category_embeddings).tolist()[0]
            predicted += [self.category_dict_index[distances.index(min(distances))]]
        return predicted

    def get_category_sentiments(self):

        category_embeddings = self.category_embeddings

        input = Input(shape=(300,))
        d = Dense(1, input_dim=300, activation="sigmoid", name="document_output")(input)
        partial_model = Model(input, d)
        partial_model.load_weights(self.model_name + ".h5", by_name=True)
        partial_model.summary()

        categorical_sentiments = {}
        categorical_sentiments2 = {}
        for i in range(len(category_embeddings)):
            layer_output = partial_model.predict(np.asarray([category_embeddings[i]]))[0]
            categorical_sentiments[list(self.category_dict.keys())[i]] = {"pos": layer_output[0], "neg": 1 - layer_output[0]}
            categorical_sentiments2[list(self.category_dict.keys())[i]] = {"pos": 1 - layer_output[0],
                                                                          "neg": layer_output[0]}
            print(list(self.category_dict.keys())[i], layer_output)
        print(categorical_sentiments)
        print(categorical_sentiments2)

# cat2vec = SentimentCat2VecPredictor("sentiment")
# cat2vec.get_category_sentiments()