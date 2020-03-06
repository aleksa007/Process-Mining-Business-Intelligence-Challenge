To run the tool do the following:

1. Change the working directory into the /main directory and run pip install -r requirements.txt
2. python main.py ./data/road-train.csv ./data/road-test.csv ./predict.csv
3. Wait for a few moments
4. The new predict.csv file will show in the /main directory


At the moment, the tool has not been tested with other data. However, if you try to run it, then the script
preprocessing is not going to run, because it was partly hard coded.

Preprocessing of road data combines the data back again, and splits it in a way such that:
- cases in train data finish by a certain 'now' moment
- cases in test data start only after the 'now' moment
- the 'now' moment is the hard-coded part, which is why this script is not run on other data

This naive model predicts the following event based on the most frequent event on that position in the training data.
It also takes the average amount of days it takes for the following event to occur at that position.

Only one case in test data is longer than the longest case in the training data, thus the events can not be predicted and
are filled with 'Unknown' and 31 as the time difference.