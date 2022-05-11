import math
import random
import numpy as np
from tensorflow.keras.utils import Sequence
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Dropout, Dense, Dot, Attention, Subtract, Lambda, GlobalAveragePooling1D
from tensorflow.python.keras.metrics import BinaryAccuracy
from tensorflow.python.keras.optimizer_v2.adam import Adam

from transformers import TFAutoModel, AutoTokenizer, DistilBertConfig, T5Config, TFDistilBertModel, AutoConfig
import tensorflow as tf
from tensorflow.keras import backend as K
import tensorflow_addons as tfa

from Cat2Vec.cat2vec import category_dictionary, save_dictionary, save_model
from Cat2Vec.sqllite_adapter import create_connection, get_articles
from utils import preprocess_text, contrastive_loss


def tokenise(tokenizer, document_list, max_len=50):
    word_tokens, mask_tokens = np.zeros((len(document_list), max_len)),  np.zeros((len(document_list), max_len))
    for i, document in enumerate(document_list):
        tokens = tokenizer.encode_plus(" ".join(document), max_length=max_len, truncation=True,
                              pad_to_max_length=True, add_special_tokens=True,
                              return_attention_mask=True, return_token_type_ids=False,
                              return_tensors='tf')
        word_tokens[i, :] = tokens['input_ids']
        mask_tokens[i, :] = tokens['attention_mask']
    return word_tokens, mask_tokens

def build_model(trarnsformer_model, categories, sentiments=None, max_len=50):

    model_config = AutoConfig.from_pretrained(trarnsformer_model, output_hidden_states=False, output_attentions=False)
    embedding_length = 768
    transformer_model = TFAutoModel.from_pretrained(trarnsformer_model, config=model_config, trainable=True, name="transformer")

    input_ids = tf.keras.layers.Input(shape=(max_len,), name='input_ids', dtype='int32')
    mask = tf.keras.layers.Input(shape=(max_len,), name='attention_mask', dtype='int32')

    embeddings = transformer_model(input_ids, attention_mask=mask)[0]
    doc_embedding = GlobalAveragePooling1D()(Attention()([embeddings, embeddings]))


    if sentiments is not None:
        doc_output = Dense(1, input_dim=embedding_length, activation="sigmoid", name="document_output")(doc_embedding)
    else:
        doc_output = Dense(len(categories), input_dim=embedding_length,
                           activation="softmax", name="document_output")(doc_embedding)

    input_categories = tf.keras.Input(shape=(1,), name="input_categories")
    category_embeddings = tf.keras.layers.Embedding(len(categories), embedding_length, name='cat_embedding',
                                                    trainable=True)
    document_category = category_embeddings(input_categories)

    gated_category_embeddings = tf.keras.layers.Embedding(len(categories), embedding_length, name='gated_cat_embedding', trainable=True)

    gate_layer = tf.keras.layers.Dense(embedding_length, activation="sigmoid", name="gate")
    gated_category1 = gate_layer(gated_category_embeddings(input_categories))

    tanh = tf.keras.layers.Dense(embedding_length, activation="tanh", name="tanh1")
    gated_category = tanh(gated_category1)

    gated_document_category = tf.keras.layers.Multiply(name="multiply_category")([gated_category, document_category])
    gated_document_embedding = tf.keras.layers.Multiply(name="multiply_document")([gated_category, doc_embedding])

    document_embedding = tf.keras.layers.Reshape((embedding_length, 1), name="doc_reshape")(gated_document_embedding)
    document_category = tf.keras.layers.Reshape((embedding_length, 1), name="category_reshape")(gated_document_category)


    embedded_distance = Subtract(name='subtract_embeddings')([document_embedding, document_category])
    dot_product = Lambda(
        lambda x: K.sum(K.abs(x), axis=-1, keepdims=True),
        name='euclidean_distance')(embedded_distance)

    dot_output = tf.keras.layers.Dense(1,  activation="sigmoid", name="category_output")(dot_product)

    model = tf.keras.Model(inputs=[input_ids, mask, input_categories], outputs=[dot_output, doc_output])

    layers_without_transformer = model.layers
    layers_without_transformer.remove(model.get_layer("transformer"))
    layers_without_transformer.remove(model.get_layer("input_ids"))
    layers_without_transformer.remove(model.get_layer("attention_mask"))

    optimisers_and_layers = [(Adam(2e-5), model.get_layer("transformer")),
                (Adam(1e-3), layers_without_transformer)]
    optimiser = tfa.optimizers.MultiOptimizer(optimisers_and_layers)

    losses = {
        "document_output": "categorical_crossentropy" if sentiments is None else "binary_crossentropy",
        "category_output": contrastive_loss,
    }

    model.compile(optimiser, loss=losses, metrics=tf.keras.metrics.BinaryAccuracy())
    # model.summary()
    return model

class BertCat2vecTrainingDataGenerator(Sequence):

    def __init__(self, document_texts, categories, category_dict, transformer_model, max_len=50, batch=32, sentiments=None, sentiments_dict=None):

        tokenizer = AutoTokenizer.from_pretrained(transformer_model)
        tokenizer.pad_token = 0

        self.words, self.mask = tokenise(tokenizer, document_texts, max_len)
        self.categories = [category_dict[category] for category in categories]
        self.number_categories = len(category_dict)
        self.indexes = [i for i in range(len(self.words))]
        self.batch_size = batch
        self.max_len = max_len
        self.sentiments = sentiments
        if sentiments is not None:
            self.sentiments = [sentiments_dict[sentiment] for sentiment in sentiments]
        self.indices = np.arange(len(self.words))
        np.random.shuffle(self.indices)

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

    def __len__(self):
        number_of_batches = math.ceil(np.floor(len(self.words) / self.batch_size))
        return number_of_batches

    def __getitem__(self, index):

        # idx_range = range(index * self.batch_size, (index + 1) * self.batch_size)
        idx_range = self.indices[index * self.batch_size: (index + 1) * self.batch_size]

        input_words = []
        input_masks = []
        input_random_category = []
        output_true_category = []
        output_category = []
        output_sentiments = []

        for idx in idx_range:
            input_words += [self.words[idx]]
            input_masks += [self.mask[idx]]
            input_random_category += [[self.categories[idx]]]
            output_true_category += [[1]]

            document_output = [0] * self.number_categories
            document_output[self.categories[idx]] = 1
            output_category += [document_output]

            if self.sentiments is not None:
                output_sentiments += [[self.sentiments[idx]]]

            for i in range(1):
                # random_idx = random.randint(0, len(self.words) - 1)
                random_category = random.randint(0, self.number_categories - 1)

                if self.sentiments is not None:
                    output_sentiments += [[self.sentiments[idx]]]

                document_output = [0] * self.number_categories
                document_output[self.categories[idx]] = 1
                output_category += [document_output]

                input_words += [self.words[idx]]
                input_masks += [self.mask[idx]]

                input_random_category += [[random_category]]
                output_true_category += [[1] if self.categories[idx] == random_category else [0]]

        if self.sentiments is not None:
            return [np.asarray(input_words), np.asarray(input_masks), np.asarray(input_random_category)], [np.asarray(output_true_category), np.asarray(output_sentiments)]

        return [np.asarray(input_words), np.asarray(input_masks), np.asarray(input_random_category)], [np.asarray(output_true_category), np.asarray(output_category)]

def train_transformer_cat2vec(transformer_model, version):
    db = create_connection("../Datasets/CategoricalNews.db")
    articles = get_articles(db)
    db.close()
    print("Load Documents")
    document_texts = [preprocess_text(article[-1]) for article in articles]
    categories = [article[-2] for article in articles]
    category_dict = category_dictionary(categories)
    print("Process Documents")

    model = build_model(transformer_model, category_dict, max_len=50)

    loss = model.fit_generator(BertCat2vecTrainingDataGenerator(document_texts, categories, category_dict, transformer_model, max_len=50), epochs=10)

    save_dictionary(category_dict, "model/categories_{}.txt".format(version))
    model.save_weights("model/cat2vec_{}.h5".format(version))

def train_sentiment_transformer_cat2vec(transformer_model, version):
    db = create_connection("../Datasets/CategoricalNews.db")
    articles = get_articles(db)
    db.close()
    print("Load Documents")
    document_texts = [preprocess_text(article[-1]) for article in articles]
    categories = [article[-2] for article in articles]
    category_dict = category_dictionary(categories)
    sentiments = [article[-3] for article in articles]
    sentiments_dict = {"negative": 0, "positive": 1}
    print("Process Documents")

    model = build_model(transformer_model, category_dict, sentiments=sentiments_dict, max_len=50)

    loss = model.fit_generator(BertCat2vecTrainingDataGenerator(document_texts, categories, category_dict, transformer_model, sentiments=sentiments, sentiments_dict=sentiments_dict, max_len=50), epochs=10)

    save_dictionary(category_dict, "model/categories_{}.txt".format(version))
    model.save_weights("model/cat2vec_{}.h5".format(version))
    cat2vec = BertCat2VecPredictor(transformer_model, version)
    cat2vec.get_category_sentiments()


class BertCat2VecPredictor:

    def __init__(self, transformer_model, model_name, max_len=50):
        self.transformer_model = transformer_model
        self.model_name = model_name

        self.category_list = open("./model/categories_{}.txt".format(model_name), "r",
                             encoding="utf8").read().strip().split("\n")

        self.category_dict = {",".join(c.split(",")[:-1]): int(c.split(",")[-1]) for c in self.category_list}
        self.category_dict_index = {v:k for k,v in self.category_dict.items()}

        self.model = build_model(transformer_model, self.category_list, sentiments={0:1,2:0}, max_len=max_len)
        self.max_len = max_len
        self.model.load_weights("./model/cat2vec_{}.h5".format(model_name))
        self.category_embeddings = self.model.get_layer('cat_embedding').get_weights()[0]
        self.model_name = "./model/cat2vec_{}".format(model_name)

    def get_category_sentiments(self):

        category_embeddings = self.category_embeddings

        input = Input(shape=(768,))
        d = Dense(1, input_dim=768, activation="sigmoid", name="document_output")(input)
        partial_model = Model(input, d)
        partial_model.load_weights(self.model_name + ".h5", by_name=True)

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

if __name__ == '__main__':
    for transformer_model in ["distilbert-base-uncased", "distilgpt2", "distilroberta-base"]: #
        # train_transformer_cat2vec(transformer_model, transformer_model + "50")
        train_sentiment_transformer_cat2vec(transformer_model, transformer_model + "50-sentiment")