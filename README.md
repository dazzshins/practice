# practice

The goal of this repository is to find:
1. the minimum, maximum and average ratings of movies from Movie dataset and Ratings dataset
2. the top 3 movies liked by each user from Users dataset based on its rating

## File Structure
data: this folder contains the following .dat files: 
*movies*, *ratings*, *users*. 
It also has a README which has details about each file.

src: this folder contains **etl_movies.py**: the PySpark code for this analysis. 
In order to run it, *pip, pyspark, findspark* has to be installed in the Python virtual environment. 
There is a **read_output.py** file to read output generated by minmax logic in etl_movies.py

output: this folder will contain the output files generated by src/etl_movies.py

## Pre-requisites
Python3 should be installed in your machine. Install Visual Code Studio as IDE.

## Commands
Open new Terminal in Visual Code Studio.
Change your working directory to a new one that will contain your Python Virtual environments.
To install new python virtual environment in Visual Code Studio, go to Terminal and type:
```python3 -m venv <new_venv_name>```

To activate this venv, type:
```source ./bin/activate```

Go to *View menu* on Visual Code Studio, select *Command Palette* and search for *Python:Select Interpreter* and ensure you select the above **venv**.

now, on the open Visual Studio Code Terminal, you will see: 
```(new_venv_name) username@your_machine folder_name %```

Now change directory to where the source code lies using 
```cd..``` and ```cd <path>```

Your present working directory should be: 
.../practice

Now, to install pip, 
Use Homebrew which will install python, pip and setuptools: 
```brew install python```

To upgrade pip, use: 
```pip3 install --upgrade pip```
To install findspark, use: 
```pip install findspark```
To install pyspark, use: 
```pip install pyspark```

To run etl_movies.py spark script, use: 
```spark-submit --deploy-mode client --master local ./src/etl_movies.py```
To run read_output.py spark script, use: 
```spark-submit --deploy-mode client --master local ./src/read_output.py```

Note: the spark-submit option **--master** is set to local as we are running the Spark application locally with a one worker thread.
the spark-submit option **--deploy-mode** is set to client as we are running in client mode for interactive and debugging purposes.

