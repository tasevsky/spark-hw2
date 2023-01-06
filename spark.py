from dataclasses import dataclass, field
import pyspark
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
from pyspark.sql.types import IntegerType

import os
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("Domasna2").getOrCreate()

sc = spark.sparkContext

df = spark.read.text("ml-100k/u.data")

df = df.selectExpr("split(value, '\t') as userID_itemID_rating_timestamp")

# print(df.head(3))

# add columns for each element of the array to seperate and cast
df = df.withColumn('user_id', df['userID_itemID_rating_timestamp'][0].cast(IntegerType()))
df = df.withColumn('item_id', df['userID_itemID_rating_timestamp'][1].cast(IntegerType()))
df = df.withColumn('rating', df['userID_itemID_rating_timestamp'][2].cast(IntegerType()))
df = df.withColumn('timestamp', df['userID_itemID_rating_timestamp'][3].cast(IntegerType()))

df = df.drop('userID_itemID_rating_timestamp')

# show the dataframe
df.show()
# df.summary().show()
# print(df.head(3))

# map each row from the df into an objet of type Rating
ratings = df.rdd.map(lambda l: Rating(user=l['user_id'],product=l['item_id'],rating=l['rating']))


# a class for representing a mean squared error for the model with give parameters
@dataclass(order=True)
class MSE:
    sort_index: int = field(init=False)
    mse: float
    rank: int
    iterations: int
    l: int

    def __post_init__(self):
        self.sort_index = int (self.mse * 10000)


# train an ALS model with the prepared data
ranks = [i for i in range(10,18,2)]
numIterations = [i for i in range (10,18,2)]
lambdas = [0.001, 0.01, 0.1]

mean_squared_errors = []    

# iterate for the given parameters, train and save a model based on them, save the parameters with the mse
for rank in ranks:
    for iterations in numIterations:
        for l in lambdas:
            model = ALS.train(ratings, rank, iterations, l)
            testdata = ratings.map(lambda p: (p[0], p[1]))
            predictions = model.predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2]))        
            ratesAndPreds = ratings.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
            mse = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
            print('{0} {1} {2}'.format(rank, iterations, l))
            print("Mean Squared Error = " + str(mse))
            mse_obj = MSE(rank=rank,iterations=iterations,l=l,mse=mse)
            mean_squared_errors.append(mse_obj)
            model.save(sc, "models/model-"+str(rank)+"-"+str(iterations)+"-"+str(l))


# print(mean_squared_errors)
sorted_list = sorted(mean_squared_errors, key=lambda x: x.sort_index)
for element in sorted_list:
    print(element)

# load the best model
best_model_mse = sorted_list[0]
print(best_model_mse)
best_model = MatrixFactorizationModel.load(sc, "models/model-"+str(best_model_mse.rank)+"-"+str(best_model_mse.iterations)+"-"+str(best_model_mse.l))


# predictions from the loaded model
predictions = best_model.predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2]))
ratesAndPreds = ratings.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
print("Mean Squared Error For The Best Model = " + str(MSE))

