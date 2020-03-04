#import prediction as p
import os
import argparse
import road_preprocess
import time

def main():
    start = time.time()
    parser = argparse.ArgumentParser("PM_tool")
    parser.add_argument("train_file", help="Process mining model will be trained on this", type=str)
    parser.add_argument("test_file", help="Process mining model will be tested on this", type=str)
    parser.add_argument("output_file", help="Predictions will be saved into a new CSV with this name", type=str)

    args = parser.parse_args()

    train_path = args.train_file
    test_path = args.test_file

    out_path = args.output_file

    if os.path.exists(out_path):
        os.remove(out_path)

    if (train_path == './data/road-train.csv') and (test_path == './data/road-test.csv'):
        road_preprocess.pre(train_path, test_path)
        train_path = './data/road-train-pre.csv'
        test_path = './data/road-test-pre.csv'
        print('Done with preprocessing')
    else:
        print('oops, skipped preprocessing of road data')


    print('Prediction started')
    os.system('python prediction.py {} {} {}'.format(train_path, test_path, out_path))
    print('Prediction ended \n Evaluating...')

    os.system('python evaluate.py {}'.format(out_path))

    os.remove(train_path)
    os.remove(test_path)

    print("Tool running time --- %s seconds ---" % (time.time() - start))

if __name__ == "__main__":
    main()