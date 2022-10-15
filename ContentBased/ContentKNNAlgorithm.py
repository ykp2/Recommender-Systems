from surprise import AlgoBase
from surprise import PredictionImpossible
from MovieLens import MovieLens
import math
import numpy as np
import heapq


class ContentKNNAlgorithm(AlgoBase):

    def __init__(self, k=40):
        """Create an instance of ContentKNNAlgorithm that inherits from AlgoBase class from Surprise library"""
        AlgoBase.__init__(self)
        self.k = k
        # Compute genre distance for every movie combination as a 2x2 matrix
        self.similarities = np.zeros((self.trainset.n_items, self.trainset.n_items))

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        # Compute item similarity matrix based on content attributes

        # Load up genre vectors for every movie
        ml = MovieLens()
        genres = ml.getGenres()
        years = ml.getYears()

        print("Computing content-based similarity matrix...")
        
        for thisRating in range(self.trainset.n_items):
            if thisRating % 100 == 0:
                print(thisRating, " of ", self.trainset.n_items)
            for otherRating in range(thisRating+1, self.trainset.n_items):
                thisMovieID = int(self.trainset.to_raw_iid(thisRating))
                otherMovieID = int(self.trainset.to_raw_iid(otherRating))
                genreSimilarity = self.computeGenreSimilarity(thisMovieID, otherMovieID, genres)
                yearSimilarity = self.computeYearSimilarity(thisMovieID, otherMovieID, years)
                self.similarities[thisRating, otherRating] = genreSimilarity * yearSimilarity
                self.similarities[otherRating, thisRating] = self.similarities[thisRating, otherRating]
        print("...done.")
                
        return self

    @staticmethod
    def computeGenreSimilarity(movie1, movie2, genres):
        """
        Computes the similarity score between 2 movies based on their genre
        :param movie1: movie 1 ID
        :param movie2: movie 2 ID
        :param dict genres: Dictionary of genre of movies
        :return float: value of similarity score of movies
        """
        genres1 = genres[movie1]
        genres2 = genres[movie2]
        sumxx, sumxy, sumyy = 0, 0, 0
        for i in range(len(genres1)):
            x = genres1[i]
            y = genres2[i]
            sumxx += x * x
            sumyy += y * y
            sumxy += x * y
        
        return sumxy/math.sqrt(sumxx*sumyy)

    @staticmethod
    def computeYearSimilarity(movie1, movie2, years):
        """
        Computes the similarity score between 2 movies based on their release year
        :param movie1: movie 1 ID
        :param movie2: movie 2 ID
        :param dict years: Dictionary of years of movie release
        :return float: value of similarity score of movies
        """
        diff = abs(years[movie1] - years[movie2])
        sim = math.exp(-diff / 10.0)
        return sim

    def estimate(self, u, i):
        """
        Estimates the rating for any item-user pair
        :param u: User ID
        :param i: Item ID
        :return: Predicted rating for given item for given user
        """

        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unknown.')
        
        # Build up similarity scores between this item and everything the user rated
        neighbors = []
        for rating in self.trainset.ur[u]:
            genreSimilarity = self.similarities[i, rating[0]]
            neighbors.append((genreSimilarity, rating[1]))
        
        # Extract the top-K most-similar ratings
        k_neighbors = heapq.nlargest(self.k, neighbors, key=lambda t: t[0])
        
        # Compute average sim score of K neighbors weighted by user ratings
        simTotal = weightedSum = 0
        for (simScore, rating) in k_neighbors:
            if simScore > 0:
                simTotal += simScore
                weightedSum += simScore * rating
            
        if simTotal == 0:
            raise PredictionImpossible('No neighbors')

        predictedRating = weightedSum / simTotal

        return predictedRating
