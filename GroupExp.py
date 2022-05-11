from evolutionary_optimiser import *
from decision_making import *
import math
import json

def log_odds_ratio(p_gain, p_loss):
    try:
        return math.log((p_gain * (1 - p_loss))/(p_loss * (1 - p_gain)))
    except:
        return 100

def wald_statistic(true_lor, predicted_lor, se):
    return ((true_lor - predicted_lor)/se) ** 2

def euclidean_distance(x, y):
    return sum([(a - b) ** 2 for a, b in zip(x, y)])

def group_objective(experimental_log_odds_array, n_training_set, gain_choices, loss_choices, gain_categories=None, loss_categories=None, steps=10):

    def objective(fpd):
        cognition, numeracy, reward_sensitivity = fpd.value

        predicted_log_odds = []
        for i in range(len(n_training_set)):
            n_gain, n_loss = n_training_set[i]
            predicted_p_gain, predicted_p_loss = [], []
            for j in range(steps):
                gain_decisions = [GistDecisionMaker(gain_choices[i], cognition, numeracy, reward_sensitivity, gain_categories).get_decision_id() for n in range(n_gain)]
                predicted_p_gain.append(sum(gain_decisions) / n_gain)
                loss_decisions = [GistDecisionMaker(loss_choices[i], cognition, numeracy, reward_sensitivity, loss_categories).get_decision_id() for n in range(n_loss)]
                predicted_p_loss.append(sum(loss_decisions)/n_loss)
            predicted_log_odds.append([average(predicted_p_gain), average(predicted_p_loss)])

        return -1 * average([abs(experimental_log_odds_array[i][0] - predicted_log_odds[i][0]) + abs(experimental_log_odds_array[i][1] - predicted_log_odds[i][1]) for i in range(len(experimental_log_odds_array))])

    return objective


def run_group_optimiser(training_set, n_training_set, n_test, true, gain_choices, loss_choices, n_epochs=50, log_name=""):

    evo = Evolution(
        pool_size=100, fitness=group_objective(training_set, n_training_set, gain_choices, loss_choices, steps=3), individual_class=FixedParameterDecision, n_offsprings=25,
        pair_params={'alpha': 0.5},
        mutate_params={'rate': 0.1, 'dim': 1, "lower bounds": [0, 0, -3], "upper bounds": [1, 1, 3]},
        init_params={'agents per population': 200}
    )

    best = []
    with open(log_name + "-log.txt", "a", encoding="utf8") as log:
        for i in range(n_epochs):
            evo.step()
            cognition, numeracy, reward_sensitivity = evo.pool.individuals[-1].value
            p_loss, p_gain, true_lor, predicted_lor, se, wald = test(cognition, numeracy, reward_sensitivity, n_test, true, gain_choices[0], loss_choices[0], sample_size=100)

            best.append([evo.pool.individuals[-1].fitness, evo.pool.individuals[-1].value])

            info = "epoch: {} Fitness: {} Best Individual: {} PLoss: {}, PGain: {}, True Log Odds: {}, Predicted Log Odds: {}, SE: {}, Wald Statistic: {}".format(i, evo.pool.individuals[-1].fitness,
                                                                     evo.pool.individuals[-1].value, p_loss, p_gain, true_lor, predicted_lor, se, wald)
            log.write(info + "\n")
            print(info)

    best.sort()
    with open(log_name + ".txt", "a", encoding="utf8") as file:
        file.write(",".join([str(x) for x in best[-1][1]]) + "\n")


def test(cognition, numeracy, reward_sensitivity, n_test, true, gain_choices, loss_choices, sample_size=100, gain_categories=None, loss_categories=None):
        avg_pgain, avg_ploss, avg_se = [], [], []
        true_p_gain, true_p_loss = true
        n_gain, n_loss = n_test
        for i in range(sample_size):
            try:
                gain_decisions = [GistDecisionMaker(gain_choices, cognition, numeracy, reward_sensitivity, gain_categories).get_decision_id() for i in range(n_gain)]
                loss_decisions = [GistDecisionMaker(loss_choices, cognition, numeracy, reward_sensitivity, loss_categories).get_decision_id() for i in range(n_loss)]

                n_certain_loss = sum(gain_decisions) if sum(gain_decisions) > 0 else 1
                n_certain_gain = n_gain - n_certain_loss if n_gain - n_certain_loss > 0 else 1
                n_gamble_loss = sum(loss_decisions) if sum(loss_decisions) > 0 else 1
                n_gamble_gain = n_loss - n_gamble_loss if n_loss - n_gamble_loss > 0 else 1

                avg_pgain += [sum(gain_decisions) / n_gain]
                avg_ploss += [sum(loss_decisions) / n_loss]
                avg_se += [math.sqrt(1 / n_certain_gain + 1 / n_certain_loss + 1 / n_gamble_gain + 1 / n_gamble_loss)]
            except:
                print("Error")
        avg_pgain, avg_ploss, avg_se = average(avg_pgain), average(avg_ploss), average(avg_se)


        wald = wald_statistic(log_odds_ratio(true_p_loss, true_p_gain), log_odds_ratio(avg_ploss, avg_pgain), avg_se)
        return avg_ploss, avg_pgain, log_odds_ratio(true_p_loss, true_p_gain), log_odds_ratio(avg_ploss, avg_pgain), avg_se, wald

def run_experiment1():

    dataset = json.loads(open("Datasets/group.json", "r", encoding="utf8").read())
    categories = set([dataset[key]["category"] for key in dataset])

    def run_group(data, n_people, gain_choices, loss_choices, log, start=0):
        if len(data) == 1:
            run_group_optimiser(data, n_people, n_people[0], data[0], gain_choices, loss_choices, n_epochs=20, log_name=log)
        else:
            for i in range(start, len(data)):
                run_group_optimiser(data[:i] + data[i + 1:], n_people[:i] + n_people[i + 1:], n_people[i], data[i], gain_choices[:i] + gain_choices[i + 1:], loss_choices[:i] + loss_choices[i + 1:], n_epochs=20, log_name=log)

    for category in categories:
            category_data = list(filter(lambda key: dataset[key]["category"] == category, dataset))
            decision_distributions = list(map(lambda key: dataset[key]["risky choice probability"], category_data))
            gain_choices = list(map(lambda key: dataset[key]["gain frame"], category_data))
            loss_choices = list(map(lambda key: dataset[key]["loss frame"], category_data))
            n_people = list(map(lambda key: dataset[key]["participants"], category_data))
            run_group(decision_distributions, n_people, gain_choices, loss_choices, "./Results/group-category" + str(category))


if __name__ == "__main__":
    ray.init()
    run_experiment1()
    ray.shutdown()






