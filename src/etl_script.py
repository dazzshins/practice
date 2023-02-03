# ETL script to read input .dat files , apply transformations 
# and write output in parquet files.
# Run using: spark-submit --deploy-mode client --master local ./src/etl_script.py

# importing packages
import findspark
import os
import logging
from typing import Optional

findspark.init()

import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.types import StructField, StructType, StringType, LongType, IntegerType, DoubleType
from pyspark.sql.functions import col, row_number, concat_ws, collect_list

# defining spark as global var
spark = SparkSession.builder.appName('movie_prac').getOrCreate()

class LoggerProvider:
    def get_logger(self, spark: SparkSession, custom_prefix: Optional[str] = ""):
        log4j_logger = spark._jvm.org.apache.log4j
        return log4j_logger.LogManager.getLogger(custom_prefix + self.__full_name__())

    def __full_name__(self):
        klass = self.__class__
        module = klass.__module__
        if module == "__builtin__":
            return klass.__name__
        return module + "." + klass.__name__

class SampleApp(LoggerProvider):
    '''Class to contain the functionality that is implemented in Pyspark.'''

    def __init__(self):
        current_dir = os.getcwd()
        print(current_dir)
        os.environ["SPARK_CONF_DIR"] = "{}/conf".format(current_dir)
        self.logger = self.get_logger(spark)
  
    def remove_duplicates_df(df):
        return df.distinct()

    def get_agg_rating(table_df):
        df = spark.sql('SELECT MovieID, Title, Genres,'
                            +' min(Rating) as min, max(Rating) as max,'
                            +' avg(Rating) as avg FROM '+table_df
                            +' group by MovieID, Title, Genres')
        return df
        
    def get_top3_movies(mv_df, rate_df):
        '''to find the top 3 movies of each userId.'''

        w1 = Window.partitionBy("UserID") \
                  .orderBy(col("Rating").desc())

        # group by UserId and order the movies in Descending order by ratings value, 
        # using row_number and deleting row_number column 
        ordered_ratings_df = (
          rate_df
          .withColumn("row",row_number().over(w1))
          .filter(col("row") <= 3)
          .drop("row")
        )

        # join to get movie title from cleaned movie dataset
        movie_ord_rating_df = ordered_ratings_df.join(
          mv_df,
          on="MovieId", 
          how='inner'
        )

        # selecting the main columns, group by UserId to collect Movie Titles in a list
        df = (
          movie_ord_rating_df
          .select('UserID', 'Title','Rating')
          .groupby('UserID')
          .agg(concat_ws( ';', 
                        collect_list('Title')
                    ).alias('top3 movies')
        ))
        return df

    def main(self):
        current_dir = os.getcwd()

        # to create movie dataframe from read and initialise its schema 
        custom_schema_mov = StructType([
        StructField("MovieID", IntegerType(), True),
        StructField("Title", StringType(), True),
        StructField("Genres", StringType(), True),
        ])

        movies_df = self.spark.read.format("csv") \
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

        self.logger.info('INFO log_msg = {}'.format("Unedited count: "+str(movies_df.count())))

        # remove rows that have MovieId as null
        movies_df.na.drop(subset=["MovieId"]).show(truncate=False)
        self.logger.info('INFO log_msg = {}'.format("Post Removing Null rows count: "+str(movies_df.count())))

        # call function to remove duplicates in movies dataframe
        distinctMoviesDF = self.remove_duplicates_df(movies_df)
        self.logger.info('INFO log_msg = {}'.format("Distinct count: "+str(distinctMoviesDF.count())))

        # to find the min, max and average rating of movies
        n_df = distinctMoviesDF.join(ratings_df, 
                              on="MovieId", 
                              how='inner')
        n_df.printSchema()
        self.logger.info('INFO log_msg = {}'.format("Joined table count: "+str(n_df.count())))
        n_df.take(5)

        # select only columns that are needed
        n_df = n_df.select('MovieID', 'Title', 'Genres', 'Rating')
        n_df.createOrReplaceTempView('new_df')

        # call function to find aggregate values of Rating (min,max,avg)
        res1_df = self.get_agg_rating('new_df')
        res1_df.show()

        # call function to find the top 3 movies of each userId
        agg_ratings_df = self.get_top3_movies(distinctMoviesDF, ratings_df)
        agg_ratings_df.show() 

        # write dataframes as parquet files. This is stored in output/
        agg_ratings_df.write.parquet('output/top3movies.parquet')
        res1_df.write.parquet('output/minmax_data.parquet')

if __name__ == '__main__':
  app_obj = SampleApp()
  app_obj.main()

