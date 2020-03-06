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
    parser.add_argument("--demo", help="Demo version, skipping preprocessing", action ='store_true')
    args = parser.parse_args()

    if args.demo:
        train_path = './demo/road-train-pre.csv'
        test_path = './demo/road-test-pre.csv'
    else:
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
        print('Oops, skipped preprocessing of road data for the sake of the demo version.')


    print('Prediction started')
    os.system('python prediction.py {} {} {}'.format(train_path, test_path, out_path))
    print('Prediction ended \n Evaluating...')

    os.system('python evaluate.py {}'.format(out_path))
    if args.demo:
       pass
    else:
        os.remove(train_path)
        os.remove(test_path)

    print("Tool running time --- %s seconds ---" % (time.time() - start))

if __name__ == "__main__":
    main()