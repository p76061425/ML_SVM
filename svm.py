import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn import metrics
import numpy as np
import os
import sys
import argparse
import pickle

def warn(*args, **kwargs):    
    pass

def read_data(path):
    x_list = []
    y_list = []
    
    #i = 1
    for dirpath,dirnames,files in os.walk(path):
        #size = len(files)
        
        for filename in files:
            #if(i>1000):break
            #i+=1
            img = plt.imread(path+filename).reshape(-1)
            x_list += [img]
            y_list += [filename[0]]
                
    x_data = np.array(x_list)
    y_data = np.array(y_list)
    return x_data,y_data
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s',
                        default='true',
                        dest='SCALING',
                        help='Features scaling, default:true')
    parser.add_argument('-k',
                        default='rbf',
                        dest='KERNEL',
                        help='svm kernel,default=rbf')    
    parser.add_argument('-n',
                        default='2',
                        dest='N_COMPONENTS',
                        help='PCA n_components,default=2')    
    parser.add_argument('-c',
                        default='1',
                        dest='PENALTY_C',
                        help='SVC penalty parameter C of the error term,default=1')        
    parser.add_argument('-cv',
                        default='10',
                        dest='CV',
                        help='cross_validate,default=10')                 
    parser.add_argument('-pp',
                        default='true',
                        dest='PP',
                        help='print progress,default = true')                             
                        
    args = parser.parse_args()

    import warnings
    warnings.warn = warn
        
    if(args.SCALING == "true"):     
        SCALING = True
    elif(args.SCALING == "false"):
        SCALING = False
    else:
        print("-s(scaling) must be true or false")
        sys.exit()
   
    if(args.PP == "true"):     
        PP = True
    elif(args.PP == "false"):
        PP = False
    else:
        print("-pp(print progress) must be true or false")
        sys.exit()

    KERNEL = args.KERNEL
    N_COMPONENTS = int(args.N_COMPONENTS)
    PENALTY_C = float(args.PENALTY_C)
    CV = int(args.CV)

    print("scaling:",SCALING)
    print("kernel:",KERNEL)
    print("n_components:",N_COMPONENTS)
    print("penalty C:",PENALTY_C)
    print("cross_validate:",CV,'\n')
    
    trainX_path = "CSL/training/"
    x_train,y_train = read_data(trainX_path)

    if SCALING:
        if(PP):
            print("scaling pca...")
        pca = make_pipeline( StandardScaler(), PCA(n_components = N_COMPONENTS) )
    else:
        if(PP):
            print("non scaling pca...")
        pca = PCA( n_components = N_COMPONENTS)
    
    pca_result = pca.fit_transform(x_train)
    #print("pca_result.shape:",pca_result.shape)
    
    if(PP):
        print("training...")
    svm_model = svm.SVC(gamma='scale', kernel=KERNEL, C = PENALTY_C, max_iter = 10000)
    svm_model.fit(pca_result, y_train)
    
    if(PP):
        print("cross validate...")
    scores = cross_validate(svm_model, pca_result, y_train, cv=CV,
                        scoring=('precision_macro', 'recall_macro', 'accuracy'))

    cv_num = 0
    avg_train_acc = 0
    avg_train_pre = 0
    avg_train_rec = 0
    avg_test_acc  = 0
    avg_test_pre  = 0
    avg_test_rec  = 0
    print("cross validation result:")
    print('|CV No.| Accuracy  | Percision | Recall   | Accuracy  | Percision | Recall    |')
    print('|---   |---        |---        |---       |---        |---        |---        |')
    print('|Type  | Train     | Train     | Train    | Test      | Test      | Test      |')
    for train_accuracy, train_precision, train_recall, \
        test_accuracy,  test_precision,  test_recall in \
        zip(scores['train_accuracy'], scores['train_precision_macro'], scores['train_recall_macro'], \
            scores['test_accuracy'],  scores['test_precision_macro'],  scores['test_recall_macro']):
        print('| cv{}  |  {:0.4f}   |  {:0.4f}   |  {:0.4f}  |  {:0.4f}   |  {:0.4f}   |  {:0.4f}   |'.format(cv_num, \
              train_accuracy, train_precision, train_recall, \
              test_accuracy, test_precision, test_recall))    
              
        avg_train_acc += train_accuracy
        avg_train_pre += train_precision
        avg_train_rec += train_recall
        avg_test_acc  += test_accuracy
        avg_test_pre  += test_precision
        avg_test_rec  += test_recall
        cv_num+=1
    print('| avg  |  {:0.4f}   |  {:0.4f}   |  {:0.4f}  |  {:0.4f}   |  {:0.4f}   |  {:0.4f}   |'.format(\
            avg_train_acc/CV, avg_train_pre/CV, avg_train_rec/CV, \
            avg_test_acc/CV, avg_test_pre/CV, avg_test_rec/CV ))         
    
    testX_path = "CSL/test/"
    x_test, y_test = read_data(testX_path)
    #print(x_test, y_test)
    
    if(PP):
        print("testing...")
    y_predict = svm_model.predict(pca.transform(x_test))
    accuracy = metrics.accuracy_score(y_test, y_predict)
    precision = metrics.precision_score(y_test, y_predict, average='macro')
    recall = metrics.recall_score(y_test, y_predict, average='macro')
    
    print()
    print("testing:\t\t\t")
    print("accuracy:",accuracy)
    print("precision:",precision)
    print("recall:",recall)
    
    print()
    print("---")

   

   
   