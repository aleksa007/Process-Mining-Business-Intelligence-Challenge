# Business Process Intelligence Challenge

This is a project by Data Science students from Technical University of Eindhoven (TU/e) for their course
Process Mining. Its goal is to implement different ways of predicting events of event log traces, as well as
the time it will take for them to occur. Our code implements 3 different techniques, and works on 4 different datasets 
of various complexities and sizes. 

## Instructions

To run the tool do the following:

-  Change your working directory into the repository directory and in your terminal run: 

```pip install -r requirements.txt```
-  Next, run: 

```python ./src/main.py ./data/road-train.csv ./data/road-test.csv ./predict.csv```
-  Wait until the tool is done running.
-  The new predict.csv file will show in the /build directory that includes predictions for:
    - Decision tree algorith,
    - Combinations algorithm,
    - Baseline
- In addition, the report.txt file inside /build will contain the performance scores. 

## Datasets

Our tool can be run on 4 different BPI datasets. Only one of them is inside the /data directory. 
Others can be found at these URL's:
- BPI Challenge 2012: https://data.4tu.nl/repository/uuid:3926db30-f712-4394-aebc-75976070e91f
- BPI Challenge 2017: https://data.4tu.nl/repository/uuid:5f3067df-f10b-45da-b98b-86ae4c7a310b
- BPI Challenge 2018: https://data.4tu.nl/repository/uuid:3301445f-95e8-4ff0-98a4-901f1f204972 

> Before you can run the tool on these datasets, you'll have to convert them from XES to CSV files. For now, it is only possible to do the following from Windows. 
1. Download any of the datasets inside the /data directory. 
2. Unzip the file in the same directory.
3. In your Terminal, navigate to /data dir and run the following command, depending on the Year of your chosen dataset.

```java -jar -Xmx8G xes2csv.jar BPI_Challenge_<YEAR>.xes <YEAR>-train.csv <YEAR>-test.csv```

PS Change <YEAR> with the Year number (YYYY). 

Data that's already in the repo is from this link:
https://data.4tu.nl/repository/uuid:270fd440-1057-4fb9-89a9-b699b47990f5

### Run on other datasets

```python ./src/main.py ./data/<YEAR>-train.csv ./data/<YEAR>-test.csv ./predict-<YEAR>.csv```

* Keep in mind that this takes approximately 40 minutes.

Tool can be run on 2018 data as well, by running the following command:
- python gucci_mane.py ./data/2018-train.csv ./data/2018-test.csv ./predict-2018.csv

Keep in mind that 2018 data is extremely large, and running the tool is likely impossible on a normal PC.
We've tested the tool in Google Colab with GPU enabled.

The tool is not intended to be run on any data other than these four datasets mentioned. 

## Process
This tool predicts the next event based on the combinations of previous two events, and predicts the most frequent event
after that specific combination.

Then, taking that prediction, it predicts the time it will take to get to it based on the 
average time it takes for that specific combination. 

For more detailed info about the tool, take a look at the Poster_Final.pdf

We are open to all feedback and remarks, hoping to learn from our mistakes!