import json
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import random
from sklearn.metrics import accuracy_score
# import XLNet

baseline_cache = {}

def test_random(choices, individuals_decisions):
    predicted_decisions = [random.randint(0,1) for i in range(len(choices))]
    return accuracy_score(individuals_decisions, predicted_decisions)

def test_vader(choices, individuals_decisions):
    predicted_decisions = [vader(choices[i]) for i in range(len(choices))]
    return accuracy_score(individuals_decisions, predicted_decisions)

def test_sentiment_cat2vec(choices, individuals_decisions):
    predicted_decisions = []
    for i in range(len(choices)):
        choice_hash = hash(choices[i])
        if choice_hash in baseline_cache:
            predicted_decisions += [baseline_cache[choice_hash]]
        else:
            xl = sentiment_cat2vec(choices[i])
            predicted_decisions += [xl]
            baseline_cache[choice_hash] = xl
    return accuracy_score(individuals_decisions, predicted_decisions)

def test_xlnet(choices, individuals_decisions):
    predicted_decisions = []
    for i in range(len(choices)):
        choice_hash = hash(choices[i])
        if choice_hash in baseline_cache:
            predicted_decisions += [baseline_cache[choice_hash]]
        else:
            xl = xlnet_sentiment(choices[i])
            predicted_decisions += [xl]
            baseline_cache[choice_hash] = xl

    return accuracy_score(individuals_decisions, predicted_decisions)

def sentiment_cat2vec(choices):
    cat2vec_sentiment = {'acq': {'pos': 0.99988186, 'neg': 0.00011813640594482422}, 'alt.atheism': {'pos': 0.98571116, 'neg': 0.014288842678070068}, 'animals': {'pos': 0.99998605, 'neg': 1.3947486877441406e-05}, 'art': {'pos': 1.0, 'neg': 0.0}, 'business': {'pos': 0.99962175, 'neg': 0.0003782510757446289}, 'cats': {'pos': 0.9999838, 'neg': 1.621246337890625e-05}, 'comp.graphics': {'pos': 0.9999689, 'neg': 3.1113624572753906e-05}, 'comp.os.ms-windows.misc': {'pos': 0.9999839, 'neg': 1.609325408935547e-05}, 'comp.sys.ibm.pc.hardware': {'pos': 0.9999083, 'neg': 9.167194366455078e-05}, 'comp.sys.mac.hardware': {'pos': 0.69537884, 'neg': 0.304621160030365}, 'comp.windows.x': {'pos': 0.9988747, 'neg': 0.0011252760887145996}, 'crude': {'pos': 1.3835386e-05, 'neg': 0.9999861646138015}, 'currency': {'pos': 0.74145424, 'neg': 0.25854575634002686}, 'death': {'pos': 7.1834834e-06, 'neg': 0.9999928165166239}, 'dogs': {'pos': 1.0, 'neg': 0.0}, 'earn': {'pos': 0.6690957, 'neg': 0.3309043049812317}, 'economy': {'pos': 0.0024026062, 'neg': 0.9975973938126117}, 'entertainment': {'pos': 0.99999964, 'neg': 3.5762786865234375e-07}, 'finance': {'pos': 0.998965, 'neg': 0.0010349750518798828}, 'grain': {'pos': 0.0002153714, 'neg': 0.999784628598718}, 'interest': {'pos': 0.09013966, 'neg': 0.9098603427410126}, 'life': {'pos': 0.99991107, 'neg': 8.893013000488281e-05}, 'losing': {'pos': 0.10093925, 'neg': 0.8990607485175133}, 'misc.forsale': {'pos': 0.99903655, 'neg': 0.0009634494781494141}, 'money': {'pos': 0.98023486, 'neg': 0.019765138626098633}, 'money-fx': {'pos': 0.5967114, 'neg': 0.4032886028289795}, 'property': {'pos': 0.9997576, 'neg': 0.00024241209030151367}, 'rec.autos': {'pos': 0.39146128, 'neg': 0.6085387170314789}, 'rec.motorcycles': {'pos': 0.3122573, 'neg': 0.6877427101135254}, 'rec.sport.baseball': {'pos': 0.33713648, 'neg': 0.6628635227680206}, 'rec.sport.hockey': {'pos': 0.9837834, 'neg': 0.016216576099395752}, 'sci.crypt': {'pos': 0.11769616, 'neg': 0.8823038414120674}, 'sci.electronics': {'pos': 0.99953616, 'neg': 0.00046384334564208984}, 'sci.med': {'pos': 0.22202973, 'neg': 0.7779702693223953}, 'sci.space': {'pos': 0.0027214098, 'neg': 0.997278590220958}, 'science': {'pos': 0.99987924, 'neg': 0.0001207590103149414}, 'ship': {'pos': 1.9206385e-09, 'neg': 0.9999999980793615}, 'soc.religion.christian': {'pos': 0.99708444, 'neg': 0.0029155611991882324}, 'sport': {'pos': 0.9999473, 'neg': 5.269050598144531e-05}, 'talk.politics.guns': {'pos': 2.5198112e-09, 'neg': 0.9999999974801888}, 'talk.politics.mideast': {'pos': 4.2336014e-06, 'neg': 0.9999957663985697}, 'talk.politics.misc': {'pos': 2.2260311e-08, 'neg': 0.9999999777396891}, 'talk.religion.misc': {'pos': 0.12643728, 'neg': 0.8735627233982086}, 'technology': {'pos': 1.0, 'neg': 0.0}, 'trade': {'pos': 0.565118, 'neg': 0.4348819851875305}, 'travel': {'pos': 0.11218566, 'neg': 0.8878143429756165}, 'winning': {'pos': 1.0, 'neg': 0.0}}
    sentiments = []
    for i,choice in enumerate(choices):
        category = json.loads(
            requests.post('http://127.0.0.1:8000/categoryPredictor', json=json.dumps([choice])).content)[0]

        sentiments += [(cat2vec_sentiment[category]["pos"], -1 * cat2vec_sentiment[category]["neg"], choice, i)]
    sentiments.sort(key=lambda x: (x[0], x[1]))
    return choices.index(sentiments[-1][-2])

def vader(choices):
    analyzer = SentimentIntensityAnalyzer()
    sentiments = []
    index = 0
    for choice in choices:
        vs = analyzer.polarity_scores(choice)
        sentiments += [(vs["pos"], vs["neg"], choice, index)]
        index += 1
    sentiments.sort(key=lambda x: (x[0], x[1]))
    return choices.index(sentiments[-1][-2])

def xlnet_sentiment(choices):
    sentiments = []
    index = 0
    for choice in choices:
        print(choice)
        vs = XLNet.predict_sentiment(choice)
        sentiments += [(vs["pos"], -1 * vs["neg"], choice, index)]
        index += 1
    sentiments.sort(key=lambda x: (x[0], x[1]))
    return choices.index(sentiments[-1][-2])
