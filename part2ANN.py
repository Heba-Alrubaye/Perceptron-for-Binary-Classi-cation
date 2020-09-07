import sys
import csv
import numpy as np
from sklearn.neural_network import MLPClassifier

def getdata(file):
    array = []
    with open(file,'r') as f:
        next(f)
        reader = csv.reader(f)
        for row in reader:
            array.append([])#2D array
            array[-1]= list(np.fromstring(row[0], dtype=float, sep=' '))
    return array

def getnormalize(a): #The scaling is carried out to improve accuracy of subsequent numeric computation and obtain better output
    maxvalue=a.max(axis=0)
    minvalue=a.min(axis=0)
    dif=maxvalue-minvalue
    normlized= (a-minvalue)/dif
    return normlized

def main():
    args = {'winetran': sys.argv[1],'winetest': sys.argv[2]}
    view = getdata(args['winetran'])
    view_testset= getdata(args['winetest'])
    view = np.array(view)
    X = view[:,:-1]
    y = view[:,-1]
    sum = 0
    sum_tran=0

    #start experiment loop here
    random_seeds =[90,91,92,93,95,97, 98, 106, 115, 110]
    for experiment in range(1,11):
        seed = random_seeds[experiment-1]
        clf = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(2,),max_iter=400, random_state=seed)
        clf.fit(getnormalize(X), y)
        view_testset = np.array(view_testset)
        X_testset = view_testset[:,:-1]
        y_testset = view_testset[:,-1]

        arrange_test = getnormalize(np.array(X_testset))
        arrange_training = getnormalize(np.array(X))
        #After fitting (training), the model can predict labels for new samples:
        test_predict=clf.predict(arrange_test)#y
        # correct_test(d)/total_testset
        training_predict=clf.predict(arrange_training)
        count =0
        for i in range(len(view_testset)):
            if test_predict[i] == y_testset[i] :
                count+=1
        accuracy = count/len(view_testset)*100
        sum+=accuracy
        count1 = 0
        for i in range(len(view)):
            if training_predict[i] == y[i]:
                count1 += 1
        accuracy = count1 / len(view) * 100
        sum_tran += accuracy
    acc_tr = (sum_tran/ 10)
    acc = (sum / 10)
    print("accuracy average for test set after 10 experiment: %s" % acc)
    print("accuracy average for training set after 10 experiment: %s" % acc_tr)
if __name__ == "__main__":
    main ()


