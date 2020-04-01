To run the tool do the following:

1. Change the working directory into the /main directory and run pip install -r requirements.txt
2. python gucci_mane.py ./data/road-train.csv ./data/road-test.csv ./predict.csv
3. Wait for a few moments
4. The new predict.csv file will show in the /main directory that includes predictions for both the new Decision tree algorith, 
	the 'combinations' algorithm,
	as well as for the baseline
5. To run the demo, only difference is the data you will be giving to the tool. Run:
	python main.py ./data/demo_data/road_train_0.4.csv ./data/demo_data/road_test_0.4.csv ./build/predict_0.4.csv


Tool can be run on 2017 data as well, by running the following command:
- python gucci_mane.py ./data/2017-train.csv ./data/2017-test.csv ./predict-2017.csv
* Keep in mind that this takes approximately 40 minutes.
Tool can be run on 2018 data as well, by running the following command:
- python gucci_mane.py ./data/2018-train.csv ./data/2018-test.csv ./predict-2018.csv
* Keep in mind that this data is extremely large, and running the tool is likely impossible on a normal PC.
We've tested the tool in Google Colab with GPU enabled.
A jupyter notebook that can be uploaded to Google Colab and run in it is provided.

Preprocessing of the data combines the data back again,
and splits it in a way such that we are not training on future data.

- the 'now' moment is the hard-coded part(for each dataset),
 which is why this script is not run on unknown data

This tool predicts the next event based on the combinations of previous two events, and predicts the most frequent event
after that specific combination. Then, taking that prediction, it predicts the time it will take to get to it based on the 
average time it takes for that specific combination. 

Also, a decision tree is implemented as the third iteration of the model. 

Thank you,
Group 13 - Process Mining