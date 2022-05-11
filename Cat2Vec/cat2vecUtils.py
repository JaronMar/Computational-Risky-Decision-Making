from itertools import *
import numpy as np
from tensorflow.keras.models import model_from_json
from Cat2Vec.Attention import create_custom_objects
import re
# from tensorboardX import SummaryWriter

def flatten(listOfLists):
    return list(chain.from_iterable(listOfLists))

def save_embeddings(output_filename, words, embeddings):
    # embeddings_out = open(join(self.temp_path, self.embedding_type + "_" + output_filename), "w", encoding="utf8")
    file_type = output_filename.split(".")[-1]
    words_out = open("words_" + output_filename, "w", encoding="utf8")

    words_out.write(",".join(words))
    npy_out = re.sub("." + file_type, ".npy", output_filename)
    np.save(npy_out, embeddings)

    words_out.close()

def embeddings_from_file(filepath):
    file_type = filepath.split(".")[-1]
    return np.load(re.sub("." + file_type, ".npy", filepath), allow_pickle=True)

def save_model(model, output_name):
    print(output_name)
    # serialize model to JSON
    model_json = model.to_json()
    with open(output_name + ".json", "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    model.save_weights(output_name + ".h5")

def load_model(model_file):
    json_file = open(model_file + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json, custom_objects=create_custom_objects())
    # loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    # loaded_model.load_weights(model_file + ".h5")
    loaded_model.load_weights(model_file + ".h5")
    return loaded_model

def preprocess_text(text, length_limit=3):
    from gensim.parsing.preprocessing import remove_stopwords

    special_words = re.compile(r"([a-z0-9_\-.@]+\.(com|edu|co\.uk))", flags=re.IGNORECASE)
    text = special_words.sub('', text)
    special_pattern = re.compile("[<!,'\"“@.,”?#…():‘’\[\]\|_]")
    text = special_pattern.sub('', text)
    special_pattern2 = re.compile("(-{2,}|\.{2,})")
    text = special_pattern2.sub('', text)
    special_with_space = re.compile("[\\]|[/>*]")
    text = special_with_space.sub(' ', text)
    text = remove_stopwords(text.lower())

    return [word for word in text.split() if len(word) >= length_limit]

def embedding_projector(model_name, vectors, labels):
    writer = SummaryWriter("logs/" + model_name)
    writer.add_embedding(vectors, labels)
    writer.close()

def count_word_occurences(text_list):
    from collections import Counter
    return Counter(flatten(text_list))

def filter_texts_by_count(doc_list, count_dict, count):
    exclude_list = ["nntp-posting-host"]
    filtered_doc_list = []
    for doc in doc_list:
        filtered_doc = []
        for word in doc:
            if count_dict[word] >= count and word not in exclude_list and len(word) > 2:
                filtered_doc += [word]
        if "subject" in filtered_doc and filtered_doc.index("subject") <= 10:
            filtered_doc = filtered_doc[filtered_doc.index("subject") + 1:]
        filtered_doc_list += [filtered_doc]
    return filtered_doc_list

