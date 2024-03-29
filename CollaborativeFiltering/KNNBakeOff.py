"""
Implements User-based KNN,  Item-based KNN & Random algorithms and
evaluates performance (RMSE/MAE) and prints top-N recommendations
"""

from MovieLens import MovieLens
from surprise import KNNBasic
from surprise import NormalPredictor
from Evaluator import Evaluator

import random
import numpy as np


def LoadMovieLensData():
    ml1 = MovieLens()
    print("Loading movie ratings...")
    data1 = ml1.loadMovieLensLatestSmall()
    print("\nComputing movie popularity ranks so we can measure novelty later...")
    rankings1 = ml1.getPopularityRanks()
    return ml1, data1, rankings1


np.random.seed(0)
random.seed(0)

# Load up common data set for the recommender algorithms
(ml, evaluationData, rankings) = LoadMovieLensData()

# Construct an Evaluator to evaluate them
evaluator = Evaluator(evaluationData, rankings)

# User-based KNN
UserKNN = KNNBasic(sim_options={'name': 'cosine', 'user_based': True})
evaluator.AddAlgorithm(UserKNN, "User KNN")

# Item-based KNN
ItemKNN = KNNBasic(sim_options={'name': 'cosine', 'user_based': False})
evaluator.AddAlgorithm(ItemKNN, "Item KNN")

# Just make random recommendations
Random = NormalPredictor()
evaluator.AddAlgorithm(Random, "Random")

evaluator.Evaluate(False)

# Print sample top-N recommendations
evaluator.SampleTopNRecs(ml)
