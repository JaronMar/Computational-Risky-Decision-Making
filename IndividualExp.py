import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from IndividualBaselines import test_random
from decision_making import *
from evolutionary_optimiser import *
import ray
import numpy as np
import json

dataset = open("Datasets/individual.csv", "r", encoding="utf8").read().split("\n")

def test_fdip(model, choices, individuals_decisions, n=100):

    expected_decisions = []
    cognition, numeracy, reward_sensitivity = model
    for i in range(len(choices)):
        avg_choice_0, avg_choice_1 = [], []
        for j in range(3):
            decision = [GistDecisionMaker([choices[i][0], choices[i][1]], cognition, numeracy, reward_sensitivity).get_decision_id() for j in range(n)]
            avg_choice_0 += [(n - sum(decision))/n]
            avg_choice_1 += [sum(decision)/n]

        expected_decisions += [1] if average(avg_choice_1) > average(avg_choice_0) else [0]

    return accuracy_score(individuals_decisions, expected_decisions), np.asarray([1 if individuals_decisions[i] == expected_decisions[i] else 0 for i in range(len(individuals_decisions))])

def individual_objective(choices, individuals_decisions, n=100):
    def objective(fpd):
        predicted_decisions, predicted_decisions_probability = [], []
        cognition, numeracy, reward_sensitivity = fpd.value
        for i in range(len(choices)):
                avg_p_0, avg_p_1 = [], []
                for j in range(3):
                    decision = [GistDecisionMaker([choices[i][0], choices[i][1]], cognition, numeracy, reward_sensitivity).get_decision_id() for x in range(n)]
                    avg_p_0 += [(n - sum(decision)) / len(decision)]
                    avg_p_1 += [(sum(decision)) / len(decision)]

                predicted_decisions_probability += [[average(avg_p_0), average(avg_p_1)]]

                predicted_decisions += [1] if predicted_decisions_probability[-1][1] > predicted_decisions_probability[-1][0] else [0]

        a = accuracy_score(individuals_decisions, predicted_decisions)
        b = average([predicted_decisions_probability[i][predicted_decisions[i]] if predicted_decisions[i] == individuals_decisions[i] else -1 * predicted_decisions_probability[i][predicted_decisions[i]] for i in range(len(choices))])
        return a + b
    return objective


def run_individual_optimiser(choices, individuals_decisions, n=100, n_epochs=100, log_file="test"):

    evo = Evolution(
        pool_size=100, fitness=individual_objective(choices, individuals_decisions, n), individual_class=FixedParameterDecision, n_offsprings=25,
        pair_params={'alpha': 0.5},
        mutate_params={'rate': 0.1, 'dim': 1, "lower bounds": [0, 0, -3], "upper bounds": [1, 1, 3]},
        init_params={'agents per population': 100}
    )

    best = []
    with open(log_file + "-log.txt", "a", encoding="utf8") as logger:
        for i in range(n_epochs):
            evo.step()
            best.append([evo.pool.individuals[-1].fitness, evo.pool.individuals[-1].value])
            info = "epoch: {} Fitness: {} Best Individual: {}".format(i, evo.pool.individuals[-1].fitness, evo.pool.individuals[-1].value)
            logger.write(info + "\n")

    best.sort()
    with open(log_file +"-models.txt", "a", encoding="utf8") as logger:
        logger.write(",".join([str(x) for x in best[-1][1]]) + "\n")

    return best[-1][1]

def group_gain_loss_pairs(dataset):
    groups = []
    processed = []

    for key in dataset:
        if key in flatten(processed):
            continue
        opposite = dataset[key]["opposite frame"]
        # if opposite == "null":
        groups += [[dataset[key]]]
        processed += [[key]]
        # else:
        #     groups += [[dataset[key], dataset[opposite]]]
        #     processed += [[key, opposite]]
    return groups, processed

def train_group(log=None, baseline=None, folds=5):
    dataset_questions = json.load(open('Datasets/individual-questions.json', "r", encoding="utf8"))
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

        training_choices = [(question["choice 1"], question["choice 2"], question["frame"]) for question in flatten(training_choices)]
        test_choices = [(question["choice 1"], question["choice 2"], question["frame"]) for question in flatten(test_choices)]
        training_split_indexes = [keys.index(x) for x in training_keys]

        accuracy = []
        all_question_accuracy = np.asarray([0 for i in range(len(test_choices))])
        ray.init()
        for count, person in enumerate(dataset[1:]):
            persons_training_decisions, persons_test_decisions = [], []
            persons_data = person.split(",")[3:]
            for i in range(len(flatten(groups))):
                if i in training_split_indexes:
                    persons_training_decisions += [int(persons_data[i])]
                else:
                    persons_test_decisions += [int(persons_data[i])]

            if baseline is None:
                model = run_individual_optimiser(training_choices, persons_training_decisions, 100, 20, log + str(cross_validation_count))
                acc, question_accuracy = test_fdip(model, test_choices, persons_test_decisions)

                accuracy += [acc]
                all_question_accuracy = all_question_accuracy + question_accuracy
            else:
                accuracy += [baseline(test_choices, persons_test_decisions)]

            person_accuracy[count] += [accuracy[-1]]
            print("Person {} Accuracy:{}".format(count, accuracy[-1]))

        info = "Mean Accuracy:{} SE: {} \n".format(average(accuracy), standard_error(accuracy))
        if log is not None:
            with open(log + "{}-results.txt".format(cross_validation_count), "a", encoding="utf8") as logger:
                logger.write(json.dumps(accuracy) + "\n")
        plt.hist(accuracy, color="#93A6E7")
        plt.show()
        if log is not None:
            with open(log + "{}-results.txt".format(cross_validation_count), "a", encoding="utf8") as logger:
                logger.write(info)
        print("Fold {}: ".format(cross_validation_count) + info)
        cross_validation_count += 1
        cross_validation_accuracy += [average(accuracy)]
        # ray.shutdown()

        for index, test_idx in enumerate(test_index):
            kfold_question_accuracy[test_idx] = all_question_accuracy[index]

        print(kfold_question_accuracy)
    print(cross_validation_accuracy)
    return average(cross_validation_accuracy)


if __name__ == "__main__":
    # print(train_group("./Results/BaselineResults-", folds=5, baseline=test_random))

    # print(train_group("./Results/BaselineResults-", folds=5, baseline=test_vader))

    print(train_group("./Results/IndividualResults-", folds=5))


