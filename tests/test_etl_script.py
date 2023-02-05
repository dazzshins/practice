"""
Pytests to test etl_script functions. 
Run using: pytest tests/test_etl_script.py 
"""

import os
import logging

import pytest
import findspark
import pyspark
from pyspark.sql import SparkSession

import src
from src import MAnalyseApp


findspark.init()
spark = SparkSession.builder.appName('movie_prac').getOrCreate()
ob = MAnalyseApp()

def test_remove_duplicates_df():
    """Pytest to check if duplicates are removed from dataframe."""

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

    res_df = ob.remove_duplicates_df(df)
    assert res_df.collect() == expected_df.collect()

def test_agg_rating_simple():
    """Pytest to check if aggregation on ratings are performed correctly
    when all data corresponds to 1 movie only and has 1 rating throughout."""
    
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
    res_df = ob.get_agg_rating(table_df='df')

    data_exp = [
        (1, "Copper Chimney", "Horror", 5, 5, 5.0)
    ]
    expected_df = spark.createDataFrame(data_exp, ['MovieID', 'Title', 'Genres', 'min', 'max', 'avg'])
    assert res_df.collect() == expected_df.collect()

def test_agg_rating_2():
    """Pytest to check if aggregation on ratings are performed correctly
    when all data corresponds to 1 movie only and has a range of ratings."""

    data = [
        (1, 'Copper Chimney', 'Horror', 1),
        (1, 'Copper Chimney', 'Horror', 3),
        (1, 'Copper Chimney', 'Horror', 5),
        (1, 'Copper Chimney', 'Horror', 4),
        (1, 'Copper Chimney', 'Horror', 5),
        (1, 'Copper Chimney', 'Horror', 5),       
    ]
    df = spark.createDataFrame(data, ['MovieID', 'Title', 'Genres', 'Rating'])
    df.createOrReplaceTempView('df')
    res_df = ob.get_agg_rating(table_df='df')

    data_exp = [
        (1, "Copper Chimney", "Horror", 1, 5, 3.8333333333333335)
    ]
    expected_df = spark.createDataFrame(data_exp, ['MovieID', 'Title', 'Genres', 'min', 'max', 'avg'])
    assert res_df.collect() == expected_df.collect()

def test_agg_rating_3():
    """Pytest to check if aggregation on ratings are performed correctly
    when all data corresponds to different movies ."""

    data = [
        (1, 'Copper Chimney', 'Horror', 1),
        (1, 'Copper Chimney', 'Horror', 3),
        (2, 'Titanic', 'Romance', 5),
        (3, 'Woof!', 'Action', 0),
        (6, '', 'Horror', 5),
        (7, 'Copper Chimney', 'Fiction', 2),       
    ]
    df = spark.createDataFrame(data, ['MovieID', 'Title', 'Genres', 'Rating'])
    df.createOrReplaceTempView('df')
    res_df = ob.get_agg_rating(table_df='df')

    data_exp = [
        (1, "Copper Chimney", "Horror", 1, 3, 2),
        (2, "Titanic", "Romance", 5, 5, 5),
        (3, 'Woof!', 'Action', 0, 0, 0),
        (6, "", "Horror", 5, 5, 5),
        (7, "Copper Chimney", "Fiction", 2, 2, 2),
    ]
    expected_df = spark.createDataFrame(data_exp, ['MovieID', 'Title', 'Genres', 'min', 'max', 'avg'])
    assert res_df.collect() == expected_df.collect()


def test_top3_movies():
    """Pytest to check if Top 3 movies of each user are calculated correctly
    and returned in the correct order."""

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
    res_df = ob.get_top3_movies(df, ratings_df)

    data_exp = [
        (1, "Copper Chimney;Treasure Hunt;Joda"),
        (2, "Friends;Marley & me")
    ]
    expected_df = spark.createDataFrame(data_exp, ['UserID', 'top3 movies'])
    assert res_df.collect() == expected_df.collect()