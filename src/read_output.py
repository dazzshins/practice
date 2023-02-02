import findspark
import os

findspark.init()

import pyspark
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('read_op').getOrCreate()

current_dir = os.getcwd()
print(current_dir)

op1 = spark.read.parquet('{0}/output/minmax_data.parquet'.format(current_dir))

op1.show(5)

# spark-submit --deploy-mode client --master local ./src/read_output.py