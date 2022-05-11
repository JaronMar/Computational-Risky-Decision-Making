import os

from cat2vec_predict import Cat2VecPredictor
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
sys.stderr = stderr

from utils import *
from word2number import w2n
import random
cat2vec = Cat2VecPredictor("w50v1", 50)
random.seed(0)
cache = {}
class GistDecisionMaker:

    cat2vec_sentiment = {'acq': {'pos': 0.99988186, 'neg': 0.00011813640594482422}, 'alt.atheism': {'pos': 0.98571116, 'neg': 0.014288842678070068}, 'animals': {'pos': 0.99998605, 'neg': 1.3947486877441406e-05}, 'art': {'pos': 1.0, 'neg': 0.0}, 'business': {'pos': 0.99962175, 'neg': 0.0003782510757446289}, 'cats': {'pos': 0.9999838, 'neg': 1.621246337890625e-05}, 'comp.graphics': {'pos': 0.9999689, 'neg': 3.1113624572753906e-05}, 'comp.os.ms-windows.misc': {'pos': 0.9999839, 'neg': 1.609325408935547e-05}, 'comp.sys.ibm.pc.hardware': {'pos': 0.9999083, 'neg': 9.167194366455078e-05}, 'comp.sys.mac.hardware': {'pos': 0.69537884, 'neg': 0.304621160030365}, 'comp.windows.x': {'pos': 0.9988747, 'neg': 0.0011252760887145996}, 'crude': {'pos': 1.3835386e-05, 'neg': 0.9999861646138015}, 'currency': {'pos': 0.74145424, 'neg': 0.25854575634002686}, 'death': {'pos': 7.1834834e-06, 'neg': 0.9999928165166239}, 'dogs': {'pos': 1.0, 'neg': 0.0}, 'earn': {'pos': 0.6690957, 'neg': 0.3309043049812317}, 'economy': {'pos': 0.0024026062, 'neg': 0.9975973938126117}, 'entertainment': {'pos': 0.99999964, 'neg': 3.5762786865234375e-07}, 'finance': {'pos': 0.998965, 'neg': 0.0010349750518798828}, 'grain': {'pos': 0.0002153714, 'neg': 0.999784628598718}, 'interest': {'pos': 0.09013966, 'neg': 0.9098603427410126}, 'life': {'pos': 0.99991107, 'neg': 8.893013000488281e-05}, 'losing': {'pos': 0.10093925, 'neg': 0.8990607485175133}, 'misc.forsale': {'pos': 0.99903655, 'neg': 0.0009634494781494141}, 'money': {'pos': 0.98023486, 'neg': 0.019765138626098633}, 'money-fx': {'pos': 0.5967114, 'neg': 0.4032886028289795}, 'property': {'pos': 0.9997576, 'neg': 0.00024241209030151367}, 'rec.autos': {'pos': 0.39146128, 'neg': 0.6085387170314789}, 'rec.motorcycles': {'pos': 0.3122573, 'neg': 0.6877427101135254}, 'rec.sport.baseball': {'pos': 0.33713648, 'neg': 0.6628635227680206}, 'rec.sport.hockey': {'pos': 0.9837834, 'neg': 0.016216576099395752}, 'sci.crypt': {'pos': 0.11769616, 'neg': 0.8823038414120674}, 'sci.electronics': {'pos': 0.99953616, 'neg': 0.00046384334564208984}, 'sci.med': {'pos': 0.22202973, 'neg': 0.7779702693223953}, 'sci.space': {'pos': 0.0027214098, 'neg': 0.997278590220958}, 'science': {'pos': 0.99987924, 'neg': 0.0001207590103149414}, 'ship': {'pos': 1.9206385e-09, 'neg': 0.9999999980793615}, 'soc.religion.christian': {'pos': 0.99708444, 'neg': 0.0029155611991882324}, 'sport': {'pos': 0.9999473, 'neg': 5.269050598144531e-05}, 'talk.politics.guns': {'pos': 2.5198112e-09, 'neg': 0.9999999974801888}, 'talk.politics.mideast': {'pos': 4.2336014e-06, 'neg': 0.9999957663985697}, 'talk.politics.misc': {'pos': 2.2260311e-08, 'neg': 0.9999999777396891}, 'talk.religion.misc': {'pos': 0.12643728, 'neg': 0.8735627233982086}, 'technology': {'pos': 1.0, 'neg': 0.0}, 'trade': {'pos': 0.565118, 'neg': 0.4348819851875305}, 'travel': {'pos': 0.11218566, 'neg': 0.8878143429756165}, 'winning': {'pos': 1.0, 'neg': 0.0}}

    def category_ratio(self):
        ratio = {}
        for key in self.cat2vec_sentiment:
            pos, neg = self.cat2vec_sentiment[key]["pos"], self.cat2vec_sentiment[key]["neg"]
            total = pos + neg
            ratio[key] = {"pos": pos/total, "neg": neg/total}

    def __init__(self, choice_list, need_for_cognition, numeracy, reward_sensitivity, categories=None, verbose=False):

        choice_hash = hash(tuple(choice_list)) if categories is None else hash(tuple(choice_list + categories))
        if choice_hash in cache:
            quantities = cache[choice_hash]["quantities"]
            probabilities = cache[choice_hash]["probabilities"]
            expected_values = cache[choice_hash]["expected values"]
            categories = cache[choice_hash]["categories"]
            categorical_quantifiers = cache[choice_hash]["categorical quantifiers"]
        else:
            quantities = []
            probabilities = []
            for choice in choice_list:
                q, p = self.identify_probability_and_quantities(choice)
                quantities.append(q)
                probabilities.append(p)

            for index, prob in enumerate(probabilities):
                if sum(prob) < 1:
                    probabilities[index] = np.asarray(probabilities[index].tolist() + [1 - sum(probabilities[index])])
                    quantities[index] = np.asarray(quantities[index].tolist() + [0])

            expected_values = [self.calculate_expected_values(quantities[i], probabilities[i]) for i in range(len(choice_list))]

            categories = cat2vec.predict(choice_list) if categories is None else categories

            categorical_quantifiers = [self.extract_categorical_quantifiers(quantity) for quantity in quantities]

            cache[choice_hash] = {"quantities":quantities, "probabilities":probabilities, "expected values": expected_values, "categories": categories, "categorical quantifiers": categorical_quantifiers}

        self.choices = []
        for i in range(len(choice_list)):
            categorical_utility, total_categorical_utility = self.categorical_decision_utility(categories[i], categorical_quantifiers[i], need_for_cognition)
            interval_utility = self.interval_decision_utility(expected_values[i], categorical_utility, numeracy, len(probabilities[i]), sum(probabilities[i]))

            self.choices += [{
                "id":i,
                "expected_value": expected_values[i],
                "probabilities": probabilities[i],
                "categorical_utility":categorical_utility,
                "total_categorical_utility": total_categorical_utility,
                "interval_utility": interval_utility
            }]


        self.decision = self.agent_decision(reward_sensitivity)

    @staticmethod
    def cognition(true_expected_value, cognition_percentage):
        mu, sigma = true_expected_value, abs(true_expected_value) * (10 * abs(cognition_percentage - 1))
        return np.random.logistic(mu, sigma)

    @staticmethod
    def numeracy(true_expected_value, numeracy_percentage):
        mu, sigma = true_expected_value, abs(true_expected_value) * (10 * abs(numeracy_percentage - 1))
        return np.random.logistic(mu, sigma)

    @staticmethod
    def cognition_error(true_expected_value, cognition_percentage):
        # normally distribute the expected value with mean and standard deviation
        mu, sigma = true_expected_value, abs(true_expected_value) * abs(cognition_percentage - 1)
        return np.random.normal(mu, sigma, 1)

    @staticmethod
    def risky_decision_probability(sensitivity):
        return 1/(1+math.exp(-1 * sensitivity))

    def categorical_decision_utility(self, category, categorical_quantifiers, cognition_percentage):
        categorical_utility = self.categorical_utility(category)
        categorical_utility = self.cognition(categorical_utility, cognition_percentage)
        total_utility = sum([categorical_utility if quantifier == "some" else -1 * categorical_utility for quantifier in categorical_quantifiers])
        return categorical_utility, total_utility

    def categorical_utility(self, category):
        return self.cat2vec_sentiment[category]["pos"] - self.cat2vec_sentiment[category]["neg"]

    def get_category_valence(self, general_knowledge):
        pos_valence, neg_valence = [], []
        for category in general_knowledge:
            pos_valence += [self.cat2vec_sentiment[category]["pos"]]
            neg_valence += [self.cat2vec_sentiment[category]["neg"]]
        return np.asarray(pos_valence), np.asarray(neg_valence)

    def interval_decision_utility(self, expected_value, category_utility, numeracy_percentage, numbers_in_calculation, sum_probabilities):
        if numbers_in_calculation > 1 and sum_probabilities != numbers_in_calculation:
            expected_value = self.numeracy(expected_value, numeracy_percentage)
        return expected_value * category_utility

    @staticmethod
    def text_to_number(text):

        def update_text(text, match_iter, mode=None):
            mapping = {"half": 2, "third": 3, "forth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eigth": 8, "ninth": 9, "tenth": 10}
            new_text = text
            for match in match_iter:
                start_index, end_index = match.start(), match.end()
                text_to_update = text[start_index:end_index]
                if mode == "probability":
                    numbers = text_to_update.split("-") if "-" in text_to_update else text_to_update.split()
                    number_probability = "{}/{}".format(w2n.word_to_num(numbers[0]), mapping[numbers[1].strip("s")])
                    new_text = re.sub(text_to_update, number_probability, new_text)
                else:
                    numbers = w2n.word_to_num(text_to_update.strip())
                    new_text = re.sub(text_to_update.strip(), str(numbers), new_text)

            return new_text

        text_probability_regex = re.compile(r"(one|two|three|four|five|six|seven|eight|nine|ten)(\s|-)(half|third|forth|fifth|sixth|seventh|eigth|ninth|tenth)s?")
        updated_text = update_text(text, re.finditer(text_probability_regex, text), mode="probability")

        text_numbers_regex = re.compile(r"((\b(eleven|twelve|thirtheen|fourteen|fiftheen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|(?<!no\s)one|two|three|four|five|six|seven|eight|nine|ten)\s?)+(hundred|thousand|million|billion)*(\sand\s)?)+")
        updated_text = update_text(updated_text, re.finditer(text_numbers_regex, updated_text))

        return updated_text

    def identify_probability_and_quantities(self, text):
        sentences = text.split(",")
        quantities = []
        probabilities = []
        for sentence in sentences:
            sentence_quantities = self.identify_quantities(sentence)
            sentence_probabilities = self.identify_probabilities(sentence)

            if len(sentence_quantities) == 0 and len(sentence_probabilities) == 0:
                continue

            if len(sentence_probabilities) == 0:
                sentence_probabilities += [1] * (len(sentence_quantities))
            elif len(sentence_quantities) == 0:
                sentence_quantities += [1] * (len(sentence_probabilities))

            if len(sentence_probabilities) < len(sentence_quantities):
                sentence_probabilities += [sentence_probabilities[-1]] * (len(sentence_quantities) - len(sentence_probabilities))
            if len(sentence_quantities) < len(sentence_probabilities):
                sentence_quantities += [1] * (len(sentence_probabilities) - len(sentence_quantities))


            probabilities.append(sentence_probabilities)
            quantities.append(sentence_quantities)

        return np.asarray(flatten(quantities)), np.asarray(flatten(probabilities))

    def identify_probabilities(self, text, quantities=None):
        number_text = self.text_to_number(text)
        number_text = re.sub("(?<!all\s)of the \d+", "", number_text)
        # print(number_text, text)
        probability_regex = re.compile(r'(((\d{0,3}%|\d{1,2}\.\d+%|0?\.\d+|\d+\/\d+)\s(chance|probability))|(for sure)|(no chance)|sure gain|sure loss|heads|tails)')
        matches = probability_regex.findall(number_text)

        result = [self.convert_text_to_probability(match[0]) for match in matches]

        if quantities is not None and len(result) < len(quantities):
            result += [1] * (len(quantities) - len(result))

        return result

    @staticmethod
    def convert_text_to_probability(text):
        if text == "for sure" or text == "sure gain" or text == "will":
            return 1

        if text == "no chance" or text == "sure loss":
            return 0

        if text == "heads" or text == "tails":
            return 0.5

        number = text.split()[0]
        if "%" in number:
            return float(number[:number.index("%")]) / 100
        elif "/" in number:
            return float(number.split("/")[0])/float(number.split("/")[1])
        elif "." in number: #already a probability
            return float(number)

    def identify_quantities(self, text):
        processed_text = self.text_to_number(text)
        processed_text = re.sub("(?<!all\s)of the \d+", "", processed_text)
        quantity_regex = re.compile(r'(((d+\.\d+|\d{0,3}%|\d{1,2}\.\d+%|0?\.\d+|\d+\/\d+|\d+)\s(chance|probability))|\d+(?!:)|no money|nothing|no people|no jobs|no plants|no lives|no one|none|not losing any|no additional|nobody)')
        quantities = quantity_regex.findall(processed_text)

        result = [float(self.convert_text_to_quantity(quantity[0])) for quantity in quantities if
                    not ("chance" in quantity[0] or "probability" in quantity[0] or "%" in quantity[0])]

        return result

    @staticmethod
    def convert_text_to_quantity(text):
        if text == "nobody" or text == "no money" or text == "no jobs" or text == "no plants" or text == "nothing" or text == "no lives" or text == "no people" or text == "no one" or text == "none" or text == "not losing any" or text == "no additional":
            return 0
        return text

    @staticmethod
    def calculate_expected_values(quantity, probability):
        try:
            return np.dot(quantity, probability)
        except:
            print(quantity, probability)

    @staticmethod
    def extract_categorical_quantifiers(quantities):
        quantifiers = []
        for quantity in quantities:
            if quantity > 0:
                quantifiers += ["some"]
            elif quantity == 0:
                quantifiers += ["none"]
        return quantifiers

    def agent_decision(self, reward_sensitivity):

        categorical_preference = self.preferred_choice(self.choices, "total_categorical_utility")
        # return self.choices[list(categorical_preference)[0]]
        interval_preference = self.preferred_choice(self.choices, "interval_utility")
        # return self.choices[list(interval_preference)[0]]

        union = categorical_preference.intersection(interval_preference)

        if len(union) == 1:
            return self.choices[list(union)[0]]

        #no preference, decide based on risk
        risk_ordered_choices = self.order_risky_choices(self.choices)

        risk = self.risky_decision_probability(reward_sensitivity)

        if random.random() <= risk:
            return risk_ordered_choices[0]             #choose the risky decision
        else:
            return risk_ordered_choices[-1]

    @staticmethod
    def preferred_choice(choice_list, utility_type):
        sorted_choices = sorted(choice_list, key=lambda x: (x[utility_type], x["id"]), reverse=True)
        preferred_choices = set()
        highest_utility = sorted_choices[0][utility_type]

        if utility_type == "ordinal_utility" and highest_utility == 0: #no ordinally preferred choice
            return set()

        for choice in sorted_choices:
            if choice[utility_type] == highest_utility:
                preferred_choices.add(choice["id"])
            else:
                break
        return preferred_choices

    @staticmethod
    def order_risky_choices(choice_list):
        return sorted(choice_list, key=lambda x: (len(x['probabilities']), x['id']), reverse=True)

    def get_decision_id(self):
        return self.decision["id"]



