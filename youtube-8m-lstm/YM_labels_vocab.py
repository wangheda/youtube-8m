from __future__ import print_function

from tensorflow import flags
import numpy as np
import random
import csv

flags.DEFINE_string("src_path_1", "vocabulary.csv", "")

def main():
    rootclass = {}
    with open(flags.FLAGS.src_path_1, "rt", encoding='utf-8') as csvfile:
        spamreader = csv.reader(csvfile)
        line_num = 0
        for row in spamreader:
            line_num += 1
            print('the '+str(line_num)+'th file is processing')
            if line_num==1:
                continue
            if row[5] in rootclass:
                rootclass[row[5]].append(line_num-2)
            else:
                rootclass[row[5]] = [line_num-2]
    labels_ordered = []
    for x in rootclass:
        labels_ordered.extend(rootclass[x])
    labels_ordered = [int(l) for l in labels_ordered]
    reverse_ordered = np.zeros([4716,1])
    for i in range(len(labels_ordered)):
        reverse_ordered[labels_ordered[i]] = i
    print(len(rootclass))
    print(labels_ordered)
    np.savetxt('labels_ordered.out', reverse_ordered, delimiter=',')
    random.shuffle(labels_ordered)
    reverse_unordered = np.zeros([4716,1])
    for i in range(len(labels_ordered)):
        reverse_unordered[labels_ordered[i]] = i
    print(labels_ordered)
    np.savetxt('labels_unordered.out', reverse_unordered, delimiter=',')
    labels_class = np.zeros([len(rootclass),4716])
    flag = 0
    for x in rootclass:
        for i in rootclass[x]:
            labels_class[flag,i] = 1
        flag +=1

    np.savetxt('labels_class.out', labels_class)

if __name__=='__main__':
    main()
