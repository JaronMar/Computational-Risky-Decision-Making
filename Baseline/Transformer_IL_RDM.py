import math
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from tensorflow.python.keras.layers import Dense
from tensorflow.keras.utils import Sequence
from tensorflow.python.keras.optimizer_v2.adam import Adam
from transformers import TFAutoModel, AutoTokenizer, AutoConfig
import tensorflow as tf
from tensorflow.keras import backend as K
from Cat2Vec.Attention import Attention
from IndividualExp import group_gain_loss_pairs
from utils import flatten, chunk, average, standard_error
import json

class TransformerConfig:

    def __init__(self):
        self.config = {
            "transformer_model": "distilbert-base-uncased",
            "lr_transformer": 2e-5,
            "SEQ_LEN": 100,
            "lower": True,
            "epochs": 3,
        }

    def load_config_from_file(self, config_file):
        self.config = json.load(open(config_file, "r", encoding="utf-8"))

    def save_config_to_file(self, config_file):
        with open(config_file, "w", encoding="utf-8") as config:
            config.write(json.dumps(self.config))

def tokenise(tokenizer, document_list, max_len=200):
    word_tokens, mask_tokens = np.zeros((len(document_list), max_len)),  np.zeros((len(document_list), max_len))
    for i, document in enumerate(document_list):
        tokens = tokenizer.encode_plus(" ".join(document), max_length=max_len, truncation=True,
                              pad_to_max_length=True, add_special_tokens=True,
                              return_attention_mask=True, return_token_type_ids=False,
                              return_tensors='tf')
        word_tokens[i, :] = tokens['input_ids']
        mask_tokens[i, :] = tokens['attention_mask']
    return word_tokens, mask_tokens

def build_model(config, n_categories):

    model_config = AutoConfig.from_pretrained(config["transformer_model"], output_hidden_states=False, output_attentions=False)
    transformer_model = TFAutoModel.from_pretrained(config["transformer_model"], config=model_config, trainable=True, name="transformer")

    input_ids = tf.keras.layers.Input(shape=(config["SEQ_LEN"],), name='input_ids', dtype='int32')
    mask = tf.keras.layers.Input(shape=(config["SEQ_LEN"],), name='attention_mask', dtype='int32')

    embeddings = transformer_model(input_ids, attention_mask=mask)[0]
    doc_embedding = Attention()(embeddings)

    doc_output = Dense(n_categories,  activation="softmax", name="output")(doc_embedding)

    model = tf.keras.Model(inputs=[input_ids, mask], outputs=doc_output)

    model.compile(Adam(config["lr_transformer"]), loss="sparse_categorical_crossentropy", metrics=tf.keras.metrics.SparseCategoricalAccuracy())
    return model

class TransformerDataGenerator(Sequence):

    def __init__(self, data, transformer_model, max_len=50, batch=4, mode="train"):

        document_texts, categories = data
        tokenizer = AutoTokenizer.from_pretrained(transformer_model)
        tokenizer.pad_token = 0

        self.words, self.mask = tokenise(tokenizer, document_texts, max_len)
        self.categories = categories
        self.mode = mode
        self.batch_size = batch if mode == "train" else 4
        self.max_len = max_len
        self.indices = np.arange(len(self.words))

    def __len__(self):
        number_of_batches = math.ceil(np.floor(len(self.words) / self.batch_size)) \
            if self.mode == "train" else math.ceil(len(self.words) / self.batch_size)
        return number_of_batches

    def __getitem__(self, index):
        idx_range = self.indices[index * self.batch_size: (index + 1) * self.batch_size]

        input_words = []
        input_masks = []
        output_category = []

        for idx in idx_range:
            if self.mode == "test" and index >= len(self.words):
                break
            input_words += [self.words[idx]]
            input_masks += [self.mask[idx]]
            output_category += [[self.categories[idx]]]


        return [np.asarray(input_words), np.asarray(input_masks)], [np.asarray(output_category)]


def train_transformer_baseline(config, train_data):
    model = build_model(config, 2)
    model.fit(TransformerDataGenerator(train_data, config["transformer_model"], max_len=config["SEQ_LEN"], batch=len(train_data[0])), epochs=config["epochs"])
    return model

def predict_transformer(transformer_model, model, test_data, max_len=200):
    get_all_layer_outputs = K.function([model.get_layer("input_ids").input, model.get_layer("attention_mask").input],
                                       [model.get_layer('output').output])

    generator = TransformerDataGenerator(test_data, transformer_model, max_len, batch=len(test_data[0]), mode="test")

    predicted = []
    n_classes = 2
    for i in range(math.ceil(len(test_data[0]) / generator.batch_size)):
        data = generator.__getitem__(i)
        network_output = get_all_layer_outputs([data[0]])
        softmax_output = flatten(network_output[-1])
        softmax_output_chunks = chunk(softmax_output, n_classes)
        for softmax_chunk in softmax_output_chunks:
            predicted += [softmax_chunk.index(max(softmax_chunk))]

    return predicted

def test_transformer_baseline_model(model, config, test_data):
    true_labels = test_data[1]
    predicted = predict_transformer(config["transformer_model"], model, test_data, config["SEQ_LEN"])
    accuracy = accuracy_score(true_labels, predicted)
    return accuracy


def train_transformer(transformer_model, training_data, test_data):
    config_class = TransformerConfig()
    config = config_class.config
    config["transformer_model"] = transformer_model

    model = train_transformer_baseline(config, training_data)
    return test_transformer_baseline_model(model, config, test_data)

def ilrdm_baseline(transformer_model, folds=5):
    dataset = open("../Datasets/individual.csv", "r", encoding="utf8").read().split("\n")
    dataset_questions = json.load(open('../Datasets/individual-questions.json', "r", encoding="utf8"))
    keys = list(dataset_questions.keys())

    groups, processed = group_gain_loss_pairs(dataset_questions)
    kf = KFold(n_splits=folds, shuffle=False)

    cross_validation_splits = kf.split([i for i in range(len(groups))])

    cross_validation_accuracy = []
    cross_validation_count = 1
    person_accuracy = {i:[] for i in range(len(dataset[1:]))}
    kfold_question_accuracy = {i:0 for i in range(len(dataset_questions))}

    for train_index, test_index in cross_validation_splits:
        training_choices, training_keys, test_choices = [], [], []
        for i in range(len(groups)):
            if i in train_index:
                training_choices += [groups[i]]
                training_keys += processed[i]
            else:
                test_choices += [groups[i]]

        training_choices = ["{} {}".format(question["choice 1"], question["choice 2"]) for question in flatten(training_choices)]
        test_choices = ["{} {}".format(question["choice 1"], question["choice 2"]) for question in flatten(test_choices)]
        training_split_indexes = [keys.index(x) for x in training_keys]

        accuracy = []
        all_question_accuracy = np.asarray([0 for i in range(len(test_choices))])

        for idx, person in enumerate(dataset[1:]):
            persons_training_decisions, persons_test_decisions = [], []
            persons_data = person.split(",")[3:]
            for i in range(len(flatten(groups))):
                if i in training_split_indexes:
                    persons_training_decisions += [int(persons_data[i])]
                else:
                    persons_test_decisions += [int(persons_data[i])]

            accuracy += [train_transformer(transformer_model, (training_choices, persons_training_decisions), (test_choices, persons_test_decisions))]

            person_accuracy[idx] += [accuracy[-1]]
            print("Person {} Accuracy:{}".format(idx, accuracy[-1]))

        info = "Mean Accuracy:{} SE: {} \n".format(average(accuracy), standard_error(accuracy))
        print("Fold {}: ".format(cross_validation_count) + info)
        with open("{}-final.txt".format(transformer_model), "a", encoding="utf-8") as f:
            f.write("{}\n".format(info))

        cross_validation_count += 1
        cross_validation_accuracy += [average(accuracy)]

        for index, test_idx in enumerate(test_index):
            kfold_question_accuracy[test_idx] = all_question_accuracy[index]

        print(kfold_question_accuracy)
    print(cross_validation_accuracy)

    with open("{}-final.txt".format(transformer_model), "a", encoding="utf-8") as f:
        f.write("{}\n{}".format(json.dumps(cross_validation_accuracy), average(cross_validation_accuracy)))

    return average(cross_validation_accuracy)

if __name__ == "__main__":
    for transformer_model in ["distilgpt2", "distilroberta-base", "distilbert-base-uncased"]:
        ilrdm_baseline(transformer_model)
