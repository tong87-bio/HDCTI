from random import random
from io import FileIO
class DataSplit(object):

    def __init__(self):
        pass
    @staticmethod
    def crossValidation(data, k, output=True, path='./dataset/tcmsuite', order=1):
        if k <= 1 or k > 10:
            k = 3
        for i in range(k):
            trainingSet = []
            testSet = []
            for ind, line in enumerate(data):
                if ind % k == i:
                    testSet.append(line[:])
                else:
                    trainingSet.append(line[:])

            # 如果开启了保存功能
            if output:
                save_path = f"{path}/test_fold_{i}.txt"
                with open(save_path, 'w', encoding='utf-8') as f:
                    for item in testSet:
                        # 假设每一项是一个可转为字符串的列表
                        if isinstance(item, list):
                            f.write('\t'.join(map(str, item)) + '\n')
                        else:
                            f.write(str(item) + '\n')

            yield trainingSet, testSet



