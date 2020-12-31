#import sys,os
# curPath = os.path.abspath(os.path.dirname(__file__))
# print(sys.path)
# # print(os.path.dirname(__file__))
# # print(curPath)
# # rootPath = os.path.split(curPath)[0]
# # print(os.path.split(curPath))
# # print(rootPath)
# # sys.path.append(rootPath)
#sys.path.append('E:\\Anaconda\\lib\\site-packages\\')
# # sys.path = ['C:\\Users\\73416\\PycharmProjects\\HSIproject', 'E:\\Python37\\python37.zip', 'E:\\Python37\\DLLs', 'E:\\Python37\\lib', 'E:\\Python37', 'E:\\Python37\\lib\\site-packages', 'E:\\Python37\\lib\\site-packages\\win32', 'E:\\Python37\\lib\\site-packages\\win32\\lib', 'E:\\Python37\\lib\\site-packages\\Pythonwin', 'E:\\Anaconda\\lib\\site-packages\\']

#print(sys.path)
# from utils import open_file
import sklearn
import numpy
import scipy
import joblib
import seaborn

def main():

    print('sklearn:',sklearn.__version__)
    print('numpy:',numpy.__version__)
    print('scipy:',scipy.__version__)
    print('joblib:',joblib.__version__)

    #print('done!')
    pass

if __name__ == '__main__':
    main()