import random

from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Reshape, Bidirectional, Lambda, Flatten
from tensorflow.keras.utils import plot_model, Sequence
from gensim.models import Word2Vec, KeyedVectors

# from Cat2Vec.Attention import Attention
from tensorflow.python.keras.layers import Attention, GlobalAveragePooling1D, Subtract

from Cat2Vec.cat2vecUtils import *
from Cat2Vec.sentimentCat2Vec import SentimentCat2VecPredictor
from utils import *
from Cat2Vec.sqllite_adapter import *
import tensorflow as tf


random.seed(0)

def pretrain_gensim_word2vec(text, model_name="word2vec.model"):
    model = Word2Vec(text, size=300, window=4, min_count=1, workers=8, iter=100)
    model.save(model_name)
    return model

def vocab_dictionary_from_pretrained(model):
    word2index = {}
    for index, word in enumerate(model.wv.index2word):
        word2index[word] = index
    return word2index

def embedding_layer_from_pretrain(model):
    embedding_layer = model.wv.get_keras_embedding(train_embeddings=False)
    embedding_layer.input_length=None
    # embedding_layer.name = "word_embedding"
    embedding_layer.mask_zero = True
    return embedding_layer

def create_vocab(document_texts):
    vocab = list(set(flatten(document_texts)))
    vocab.sort()
    return vocab

def vocab_dictionary(vocab):
    return {vocab[i]: i for i in range(1,len(vocab))}

def category_dictionary(categories):
    category_set = list(set(categories))
    category_set.sort()
    return {category_set[i]: i for i in range(len(category_set))}

def generate_input_words_vector(document_word_list, vocab_dict):
    document_word_list = list(filter(lambda x: x in vocab_dict, document_word_list))
    return [vocab_dict[word] for word in document_word_list ]


def generate_input_category_vector(category, category_dict):
    return np.array([category_dict[category]])

def euclidean_distance(vects):
    x, y = vects
    sum_square = tf.math.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True)
    return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))

def contrastive_loss(y_true, y_pred, margin=1.0):
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.dtypes.cast(y_true, y_pred.dtype)
    return K.mean(y_true * tf.math.square(y_pred) + (1.0 - y_true) * tf.math.square(
        tf.math.maximum(margin - y_pred, 0.0)
    ))

def create_model(vocab, categories, embedding_length, word_embedding_layer=None, sentiments=None, max_len=50):

    input_words = tf.keras.Input(shape=(max_len,), name="text_input")
    if word_embedding_layer is None:
        word_embeddings = Embedding(len(vocab), embedding_length, name='word_embedding', input_length=None, trainable=True, mask_zero=True)
    else:
        word_embeddings = word_embedding_layer

    document_word_embeddings = word_embeddings(input_words)

    bilstm = Bidirectional(LSTM(embedding_length//2, dropout=0.5, return_sequences=True, input_shape=(None, embedding_length)))
    lstm_embedding = bilstm(document_word_embeddings)
    lstm_attention = GlobalAveragePooling1D()(Attention()([lstm_embedding, lstm_embedding]))

    if sentiments is not None:
        doc_output = Dense(1, input_dim=embedding_length, activation="sigmoid", name="document_output")(lstm_attention)

    document_embedding = Reshape((embedding_length, 1))(lstm_attention)

    input_categories = tf.keras.Input(shape=(1,), name="cat_input")
    category_embeddings = Embedding(len(categories), embedding_length, name='cat_embedding', trainable=True)
    document_category = category_embeddings(input_categories)
    document_category = Reshape((embedding_length, 1))(document_category)

    embedded_distance = Subtract(name='subtract_embeddings')([document_embedding, document_category])
    dot_product = Lambda(
        lambda x: K.sum(K.abs(x), axis=-1, keepdims=True),
        name='euclidean_distance')(embedded_distance)

    dot_output = Dense(1, activation='sigmoid', name="category_output")(dot_product)

    losses = {
        "document_output": "categorical_crossentropy" if sentiments is None else "binary_crossentropy",
        "category_output": contrastive_loss,
    }

    if sentiments is not None:
        model = Model(inputs=[input_words, input_categories], outputs=[dot_output, doc_output])
    else:
        model = Model(inputs=[input_words, input_categories], outputs=[dot_output])

    model.compile(loss=losses, optimizer='adam', metrics=[tf.keras.metrics.BinaryAccuracy()])
    model.summary()

    return model

class TrainingDataGenerator(Sequence):

    def __init__(self, document_texts, vocab_dictionary, categories, category_dict, sentiments=None, sentiments_dict=None, batch=256, max_len=100):
        self.words = [generate_input_words_vector(text, vocab_dictionary) for text in document_texts]
        self.categories = [category_dict[category] for category in categories]
        self.number_categories = len(category_dict)
        self.indexes = [i for i in range(len(self.words))]
        self.batch_size = batch
        self.max_len = max_len
        self.sentiments = sentiments
        if sentiments is not None:
            self.sentiments = [sentiments_dict[sentiment] for sentiment in sentiments]
        print()

        self.indices = np.arange(len(self.words))
        np.random.shuffle(self.indices)

    def __len__(self):
        number_of_batches = math.ceil(np.floor(len(self.words) / self.batch_size))
        return number_of_batches

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

    def __getitem__(self, index):
        # idx_range = range(index * self.batch_size, (index + 1) * self.batch_size)
        idx_range = self.indices[index * self.batch_size: (index + 1) * self.batch_size]
        input_words = []
        input_random_category = []
        output_category = []
        output_true_category = []
        output_sentiments = []

        for idx in idx_range:
            document_output = [0] * self.number_categories
            document_output[self.categories[idx]] = 1

            input_words += [self.words[idx][:self.max_len]]
            input_random_category += [[self.categories[idx]]]
            output_true_category += [[1]]
            output_category += [document_output]

            if self.sentiments is not None:
                output_sentiments += [[self.sentiments[idx]]]

            for i in range(1):
                random_category = random.randint(0, self.number_categories-1)

                if self.sentiments is not None:
                    output_sentiments += [[self.sentiments[idx]]]

                document_output = [0] * self.number_categories
                document_output[self.categories[idx]] = 1

                input_words += [self.words[idx][:self.max_len]]
                input_random_category += [[random_category]]
                output_true_category += [[1] if self.categories[idx] == random_category else [0]]
                output_category += [document_output]

        padded_words = tf.keras.preprocessing.sequence.pad_sequences(input_words,
                                                                      padding='post', maxlen=self.max_len)
        if self.sentiments is not None:
            return [padded_words, np.array(input_random_category)], [np.array(output_true_category), np.array(output_sentiments)]

        return [padded_words, np.array(input_random_category)], [np.array(output_true_category), np.array(output_category)]

def save_dictionary(save_dictionary, filename):
    out = open(filename, "w", encoding="utf8")
    for data in save_dictionary:
        out.write("{},{}\n".format(data, save_dictionary[data]))
    out.close()

def train_pretrained(version="v5"):

    db = create_connection("../Datasets/CategoricalNews.db")
    articles = get_articles(db)
    db.close()

    print("Load Documents")
    document_texts = [preprocess_text(article[-1]) for article in articles]
    categories = [article[-2] for article in articles]
    category_dict = category_dictionary(categories)
    print("Process Documents")
    pretrained = KeyedVectors.load(r"enwiki_20180420_300d_no_entities.wv", mmap='r')

    vocab_dict = vocab_dictionary_from_pretrained(pretrained)
    print("Train Word2Vec Model")
    model = create_model(vocab_dict, category_dict, 300, word_embedding_layer=embedding_layer_from_pretrain(pretrained), max_len=50)
    loss = model.fit_generator(TrainingDataGenerator(document_texts, vocab_dict, categories, category_dict, max_len=50), epochs=20)

    save_dictionary(vocab_dict, "model/vocab_{}.txt".format(version))
    save_dictionary(category_dict, "model/categories_{}.txt".format(version))
    save_model(model, "model/cat2vec_{}".format(version))



def train_sentiment(version="w100v1"):
    db = create_connection("../Datasets/CategoricalNews.db")
    articles = get_articles(db)
    db.close()
    print("Load Documents")
    document_texts = [preprocess_text(article[-1]) for article in articles]
    sentiments = [article[-3] for article in articles]
    sentiments_dict = {"negative": 0, "positive": 1}
    categories = [article[-2] for article in articles]
    category_dict = category_dictionary(categories)
    print("Process Documents")
    pretrained = KeyedVectors.load(r"enwiki_20180420_300d_no_entities.wv", mmap='r')
    vocab_dict = vocab_dictionary_from_pretrained(pretrained)
    print("Train Word2Vec Model")
    model = create_model(vocab_dict, category_dict, 300, word_embedding_layer=embedding_layer_from_pretrain(pretrained), sentiments=sentiments_dict, max_len=100)

    loss = model.fit_generator(TrainingDataGenerator(document_texts, vocab_dict, categories, category_dict, sentiments=sentiments, sentiments_dict=sentiments_dict, max_len=100), epochs=50)

    save_dictionary(category_dict, "model/categories_{}.txt".format(version))
    save_model(model, "model/cat2vec_{}".format(version))

    cat2vec = SentimentCat2VecPredictor("sentiment")
    cat2vec.get_category_sentiments()


if __name__ == '__main__':
    train_sentiment("sentiment")

