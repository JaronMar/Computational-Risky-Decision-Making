from transformers import AutoTokenizer

from Cat2Vec.transformer_cat2vec import tokenise, build_model
from utils import *
from tensorflow.keras import backend as K
from tensorflow.keras.utils import Sequence

class BertPredictionGenerator(Sequence):

    def __init__(self, transformer_model, document_texts, max_len=50):
        tokenizer = AutoTokenizer.from_pretrained(transformer_model)
        tokenizer.pad_token = 0
        self.words, self.mask = tokenise(tokenizer, document_texts, max_len)
        self.batch_size = 1

    def __len__(self):
        return math.ceil(len(self.words)/self.batch)

    def __getitem__(self, idx):
        input_words = []
        input_masks = []
        input_random_category = []
        output_true_category = []
        idx_range = range(idx * self.batch_size, (idx + 1) * self.batch_size)
        for index in idx_range:
            input_words += [self.words[index]]
            input_masks += [self.mask[index]]
            input_random_category += [[0]]
            output_true_category += [[1]]

        return [np.asarray(input_words), np.asarray(input_masks), np.array(input_random_category)], [np.array(output_true_category)]


class BertCat2VecPredictor:

    def __init__(self, transformer_model, model_name):

        self.transformer_model = transformer_model
        self.model_name = model_name
        self.category_list = open("./cat2vec/model/categories_{}.txt".format(model_name), "r",
                             encoding="utf8").read().strip().split("\n")

        self.category_dict = {",".join(c.split(",")[:-1]): int(c.split(",")[-1]) for c in self.category_list}
        self.category_dict_index = {v:k for k,v in self.category_dict.items()}

        max_len = 50
        self.model = build_model(transformer_model, self.category_list, max_len=max_len)
        self.max_len = max_len
        self.model.load_weights("./cat2vec/model/cat2vec_{}.h5".format(model_name))
        self.category_embeddings = self.model.get_layer('cat_embedding').get_weights()[0]
        self.model_name = model_name


    def predict(self, text_list):
        get_all_layer_outputs = K.function([self.model.get_layer("input_ids").input, self.model.get_layer("attention_mask").input, self.model.get_layer("input_categories").input],
                                           [self.model.get_layer('document_output').input])

        generator = BertPredictionGenerator(self.transformer_model, text_list, max_len=self.max_len)
        predicted = []
        for i in range(len(text_list)):
            data = generator.__getitem__(i)
            layer_output = get_all_layer_outputs([data[0]])[0][0]
            distances = squared_euclidean_distance(np.asarray([layer_output]), self.category_embeddings).tolist()[0]
            predicted += [self.category_dict_index[distances.index(min(distances))]]
        return predicted




