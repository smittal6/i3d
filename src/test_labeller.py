import os
import sys
import numpy as np


class_file = '/mnt/data1/UCF-101/classInd.txt'
test_file = '../list/testlist01.txt'

def main():

    dict_name = {}
    with open(class_file) as w:
        tmp = [line.rstrip('\n') for line in w]
        seq = ' '
        categories = [line.split(seq,1)[0] for line in tmp]
        name_cat = [line.split(seq,1)[-1] for line in tmp]
        for m,n in zip(name_cat,categories):
            # print("M: %s, N: %d"%(m,int(n)))
            dict_name[m] = int(n)

    # category = []
    # pass

    prepend = '/mnt/data1/UCF-101/'
    with open(test_file) as w:
        tmp = [line.rstrip('\n') for line in w]
        classnames = [line.split('/')[0] for line in tmp]
        f_names = [line.split('/')[1].split('.')[0] for line in tmp]
        for m,n in zip(classnames,f_names):
            print("%s %d"%(prepend+m+'/'+ n,dict_name[m]))

if __name__ == '__main__':
    main()
