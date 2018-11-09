# Snehith Raj Bada

import pandas as pd
import numpy as np
import math
import operator


#Calculate neighbours ans sort based on euclidean distance
def getNeighbors(training, testing, k):
    distances = []
    length = testing.shape[0] - 1
    # Find distance between each object of training dataset with every object of testing dataset
    for x in range(len(training)):
        dist = euclideanDistance(testing, training[x], length)
        distances.append((training[x], dist))

    #sorting the distances
    distances.sort(key=operator.itemgetter(1))
    neighbors = []

    #based on the k value selecting k number of nearest neighbours
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

# Calculate Euclidean Distance between Testing and Training data sets
def euclideanDistance(coordinate1,coordinate2 , length):
    distance = 0
    for x in range(length):
        distance += pow((coordinate1[x ] - coordinate2[x ]), 2)
    return math.sqrt(distance)


#Predicting the position based on the neighbours based on highest number of occurences of postion among the neighbours
def prediction(neighbors):
    number = {}
    for x in range(len(neighbors)):
        position = neighbors[x][-1]
        if position in number:
            number[position] += 1
        else:
            number[position] = 1
    sortedNumber = sorted(number.items(), key=operator.itemgetter(1), reverse=True)
    print("Sorted Number",sortedNumber)
    return sortedNumber[0][0]

#Calculating the accuracy
def getAccuracy(testing, predictions):
    correct = 0
    for x in range(testing.shape[0]):
        if testing[x][-1] == predictions[x]:
            correct += 1
    return (correct / float(testing.shape[0])) * 100.0

#Function class = myknn(X, test, k) that performs k-nearest neighbor (k-NN) clas- sification
# where X ∈ Rn×p (n number of objects and p number of attributes) is training data, test is testing data,
# and k is a user parameter.
def myknn(X, test, k):
    predictions = []
    testing = test.values
    training = X.values
    #print(training, testing)
    print("size of testing:",testing.shape[0])
    for x in range(testing.shape[0]):
        neighbors = getNeighbors(training, testing[x], k)
        result = prediction(neighbors)
        predictions.append(result)
        print('> predicted=' + repr(result) + ', actual=' + repr(testing[x][-1]))
    accuracy = getAccuracy(testing, predictions)
    print('Accuracy: ' + repr(accuracy) + '%')

#Standardize the data(zero-mean and standard deviation = 1)
def standardization(df):
    for column in df:
        x = pd.DataFrame(data=df[column])
        print(np.mean(x))
        x -= np.mean(x)
        print(x)
        x /= np.std(x)
        print(x)
        df[column] = x
    return df

if __name__ == "__main__":
    col_names = ['Player', 'Pos', 'Age', 'Tm', 'G', 'GS', 'MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', '2P', '2PA',
                 '2P%',
                 'eFG%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PS/G']
    dataframe = pd.read_csv('NBAstats.csv', names=col_names, skiprows=1) # Store the csv data into dataframe
    df = dataframe

# Data selection and creating datasets(Training,Testing)
    def datacleaning(df):
        df = standardization(df)
        #print(dataframe[['Pos']])
        df = df.join(dataframe[['Pos']])
        print(df)
        df_train = df.iloc[:375] #Creating Training dataset
        df_test = df.iloc[375:] #Creating Testing dataset
        print("df-train", df_train)
        print("df-test", df_test)
        k = int(input("Enter k value\n"))
        myknn(df_train, df_test, k)


    c = int(input(
        "Select one of the following for KNN\n2 : Use all features except team\nor\n3 : Use the following set of attributes {2P%, 3P%, FT%, TRB, AST, STL, BLK}\n"))

    # 2  Use all features except team. Use your k-NN code to perform classification.
    if(c==2):
        new_col=['Age', 'G', 'GS', 'MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', '2P', '2PA','2P%',
                 'eFG%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PS/G']
        dfk = dataframe[new_col]
        datacleaning(dfk)

    # 3 Use the following set of attributes {2P%, 3P%, FT%, TRB, AST, STL, BLK}
    # to perform k-NN classification
    if(c==3):
        new_col = ['2P%', '3P%', 'FT%', 'TRB', 'AST', 'STL', 'BLK']
        dfk = dataframe[new_col]
        datacleaning(dfk)