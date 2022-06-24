import argparse
import pprint

from image_classifier_comparer.comparer import ImageClassifierComparer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('img_path', type=str)
    args = parser.parse_args()
    icc = ImageClassifierComparer()
    icc.compare(args.img_path)

    pprint.pprint(icc.post_processed_preds_dict)
