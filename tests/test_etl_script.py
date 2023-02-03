# Pytests to test etl_script functions. 
# Run using: pytest tests/test_etl_script.py 

import src
from src import SampleApp
import pytest
import findspark
import os
import logging
from typing import Optional

findspark.init()

import pyspark
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('movie_prac').getOrCreate()

def test_remove_duplicates_df():
    '''Pytest to check if duplicates are removed from dataframe.'''
    data = [
        ("1", "Titanic", "Action"),
        ("1", "Titanic", "Action"),
        ("1", "Titanic", "Action"),
        ("1", "Titanic", "Action")
    ]
    df = spark.createDataFrame(data, ["MovieID", "Title", "Genres"])

    data_exp = [
        ("1", "Titanic", "Action")
    ]
    expected_df = spark.createDataFrame(data_exp, ["MovieID", "Title", "Genres"])

    res_df = SampleApp.remove_duplicates_df(df)
    assert res_df.collect() == expected_df.collect()

def test_agg_rating():
    '''Pytest to check if aggregation on ratings are performed correctly.'''
    data = [
        (1, 'Copper Chimney', 'Horror', 5),
        (1, 'Copper Chimney', 'Horror', 5),
        (1, 'Copper Chimney', 'Horror', 5),
        (1, 'Copper Chimney', 'Horror', 5),
        (1, 'Copper Chimney', 'Horror', 5),
        (1, 'Copper Chimney', 'Horror', 5),       
    ]
    df = spark.createDataFrame(data, ['MovieID', 'Title', 'Genres', 'Rating'])
    df.createOrReplaceTempView('df')
    res_df = SampleApp.get_agg_rating(table_df='df')

    data_exp = [
        (1, "Copper Chimney", "Horror", 5, 5, 5.0)
    ]
    expected_df = spark.createDataFrame(data_exp, ['MovieID', 'Title', 'Genres', 'min', 'max', 'avg'])
    assert res_df.collect() == expected_df.collect()

    data_2 = [
        (1, 'Copper Chimney', 'Horror', 1),
        (1, 'Copper Chimney', 'Horror', 3),
        (1, 'Copper Chimney', 'Horror', 5),
        (1, 'Copper Chimney', 'Horror', 4),
        (1, 'Copper Chimney', 'Horror', 5),
        (1, 'Copper Chimney', 'Horror', 5),       
    ]
    df_2 = spark.createDataFrame(data_2, ['MovieID', 'Title', 'Genres', 'Rating'])
    df_2.createOrReplaceTempView('df_2')
    res_df_2 = SampleApp.get_agg_rating(table_df='df_2')

    data_exp_2 = [
        (1, "Copper Chimney", "Horror", 1, 5, 3.8333333333333335)
    ]
    expected_df_2 = spark.createDataFrame(data_exp_2, ['MovieID', 'Title', 'Genres', 'min', 'max', 'avg'])
    assert res_df_2.collect() == expected_df_2.collect()

def test_top3_movies():
    '''Pytest to check if Top 3 movies of each user are calculated correctly.'''
    data = [
        (1, "Copper Chimney", "Horror"),
        (2, 'Titanic', "Romance"),
        (3, 'Joda', "Action"),
        (4, 'Treasure Hunt', "Adventure"),
        (5, 'Friends', "Comedy"),
        (6, 'Marley & me', "Comedy"),       
    ]
    df = spark.createDataFrame(data, ['MovieID', 'Title', 'Genres'])

    data2 = [
        (1, 1, 5, 12375),
        (1, 2, 2, 12375),
        (1, 3, 4, 12375),
        (1, 4, 5, 12375),
        (1, 5, 2, 12375),
        (1, 6, 2, 12375),
        (2, 5, 2, 12375),
        (2, 6, 2, 12375),
    ]
    ratings_df = spark.createDataFrame(data2, ['UserID', 'MovieID', 'Rating', 'Timestamp'])
    res_df = SampleApp.get_top3_movies(df, ratings_df)

    data_exp = [
        (1, "Copper Chimney;Joda;Treasure Hunt"),
        (2, "Friends;Marley & me")
    ]
    expected_df = spark.createDataFrame(data_exp, ['UserID', 'top3 movies'])
    assert res_df.collect() == expected_df.collect()