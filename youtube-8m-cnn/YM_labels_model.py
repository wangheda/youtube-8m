from __future__ import print_function

import tensorflow as tf
from tensorflow import flags
import numpy as np
import csv

flags.DEFINE_string("src_path_1", "predictions_best.csv", "")

def main():
    labels_uni = np.zeros([4716,1])
    with open(flags.FLAGS.src_path_1, "rt", encoding='utf-8') as csvfile:
        spamreader = csv.reader(csvfile)
        line_num = 0
        for row in spamreader:
            line_num += 1
            print('the '+str(line_num)+'th file is processing')
            if line_num==1:
                continue
            lbs = row[1].split()
            for i in range(0,len(lbs),2):
                labels_uni[int(lbs[i])] += 1
    np.savetxt('labels_model.out', labels_uni, delimiter=',')

if __name__=='__main__':
    main()
