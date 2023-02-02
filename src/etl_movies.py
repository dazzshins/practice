# importing packages
import findspark
import os

findspark.init()

import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.types import StructField, StructType, StringType, LongType, IntegerType, TimestampType, DoubleType
from pyspark.sql.functions import col, row_number, concat_ws, collect_list

# initialising spark
spark = SparkSession.builder.appName('movie_prac').getOrCreate()

current_dir = os.getcwd()
print(current_dir)

# to create movie dataframe from read and initialise its schema 
custom_schema_mov = StructType([
    StructField("MovieID", IntegerType(), True),
    StructField("Title", StringType(), True),
    StructField("Genres", StringType(), True),
])

movies_df = spark.read.format("csv") \
  .schema(custom_schema_mov) \
  .option("header", False) \
  .option("quote", "") \
  .option("delimiter", "::") \
  .option("ignoreTrailingWhiteSpace", True) \
  .load('{0}/data/movies.dat'.format(current_dir))

movies_df.printSchema()
movies_df.show(5)

# to create users dataframe from read and initialise its schema 
custom_schema_users = StructType([
    StructField("UserID", IntegerType(), True),
    StructField("Gender", StringType(), True),
    StructField("Age", IntegerType(), True),
    StructField("Occupation", StringType(), True),
    StructField("Zip_code", StringType(), True),
])

users_df = spark.read.format("csv") \
  .schema(custom_schema_users) \
  .option("header", False) \
  .option("quote", "") \
  .option("delimiter", "::") \
  .option("ignoreTrailingWhiteSpace", True) \
  .load('{0}/data/users.dat'.format(current_dir))

users_df.printSchema()
users_df.show(5)

# to create ratings dataframe from read and initialise its schema 
custom_schema_ratings = StructType([
    StructField("UserID", IntegerType(), True),
    StructField("MovieID", IntegerType(), True),
    StructField("Rating", DoubleType(), True),
    StructField("Timestamp", LongType(), True),
])

ratings_df = spark.read.format("csv") \
  .schema(custom_schema_ratings) \
  .option("header", False) \
  .option("quote", "") \
  .option("delimiter", "::") \
  .option("ignoreTrailingWhiteSpace", True) \
  .load('{0}/data/ratings.dat'.format(current_dir))

ratings_df.printSchema()
ratings_df.show(5)

# to find the min, max and average rating of movies
n_df = movies_df.join(ratings_df, 
                      on="movieId", 
                      how='inner')
n_df.printSchema()
n_df.take(5)

n_df.createOrReplaceTempView('new_df')
res1_df = spark.sql('SELECT MovieID, Title, Genres,'
                    +' min(Rating) as min, max(Rating) as max, avg(Rating) as avg FROM new_df'
                    +' group by MovieID, Title, Genres')
res1_df.show()

# to find the top 3 movies of each userId
w1 = Window.partitionBy("UserID") \
           .orderBy(col("Rating").desc())

ratings_df.withColumn("row",row_number().over(w1)) \
          .filter(col("row") <= 3) \
          .drop("row") \
          .select('UserID', 'MovieId','Rating') \
          .groupby('UserID') \
          .agg(concat_ws( ';', 
                          collect_list('MovieId')
                          )
                ).alias('top3 movies') \
          .show()

ratings_df.write.parquet('output/top3movies.parquet')
res1_df.write.parquet('output/minmax_data.parquet')

# cmd: spark-submit --deploy-mode client --master local ./src/etl_movies.py



