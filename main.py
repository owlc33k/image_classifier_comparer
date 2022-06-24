import argparse
import pprint

from image_classifier_comparer.comparer import ImageClassifierComparer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('img_path', type=str)
    args = parser.parse_args()
    icc = ImageClassifierComparer()
    result_dict = icc.compare(args.img_path)

    pprint.pprint(result_dict)
