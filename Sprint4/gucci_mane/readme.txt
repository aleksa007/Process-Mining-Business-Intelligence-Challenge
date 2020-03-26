To run the tool do the following:

1. Change the working directory into the /main directory and run pip install -r requirements.txt
2. python gucci_mane.py ./data/road-train.csv ./data/road-test.csv ./predict.csv
3. Wait for a few moments
4. The new predict.csv file will show in the /main directory that includes predictions for both the new Decision tree algorith, 
	the 'combinations' algorithm,
	as well as for the baseline
5. # TODO: Fix this -> When running the demo, after installing dependencies as in Step 1, run:
	python main.py ./data/demo_train.csv ./data/demo_test.csv ./predict.csv --demo
	-! In this case, the data is halved(both train and test) and output csv file only contains
	predictions for the 'combinations' algorithm.

Tool can be run on 2017 data as well, by running the following command:
- python gucci_mane.py ./data/2017-train.csv ./data/2017-test.csv ./predict-2017.csv
* Keep in mind that this takes approximately 40 minutes. 

Preprocessing of the data combines the data back again, and splits it in a way such that:
- cases in train data finish by a certain 'now' moment
- cases in test data start only after the 'now' moment
- the 'now' moment is the hard-coded part, which is why this script is not run on other data

This tool predicts the next event based on the combinations of previous two events, and predicts the most frequent event
after that specific combination. Then, taking that prediction, it predicts the time it will take to get to it based on the 
average time it takes for that specific combination. 

Also, a decision tree is implemented as the third iteration of the model. 

For a more detailed description inspect the Poster attached. 