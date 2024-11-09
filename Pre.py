import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import shuffle


def signumActivationFunction(f_net):
    return np.where(f_net > 0, 1, np.where(f_net < 0, -1, 0))


def linearActivationFunction(f_net):
    return f_net


class SingleLayerPerceptron:
    def __init__(self, epochsNum, addBias, learningRate, misThreshold):
        self.weights = None
        self.bias = None
        self.epochsNum = int(epochsNum)
        self.learningRate = int(learningRate)
        self.addBias = bool(addBias)
        self.misThreshold = bool(misThreshold)

    def train(self, X, Y):
        Y = np.where(Y == 0, -1, 1)
        self.weights = np.ones(X.shape[1]) * 0.001
        self.bias = 0.001 if self.addBias else 0

        for epoch in range(int(self.epochsNum)):
            misclassify = 0
            for (data, target) in zip(X, Y):
                F_net = np.dot(data, self.weights)
                if self.addBias:
                    F_net += self.bias

                y_predict = signumActivationFunction(F_net)

                error = target - y_predict

                if error != 0:
                    misclassify += 1

                self.weights += self.learningRate * error * data
                if self.addBias:
                    self.bias += self.learningRate * error

            if misclassify <= int(self.misThreshold):
                break


class AdalineAlgorithm:
    def __init__(self, epochsNum, addBias, learningRate, mseThreshold):
        self.weights = None
        self.bias = None
        self.epochsNum = epochsNum
        self.learningRate = learningRate
        self.addBias = addBias
        self.mseThreshold = mseThreshold

    def train(self, X, Y):
        self.weights = np.ones(X.shape[1]) * 0.001
        self.bias = 0.001 if self.addBias else 0

        for epoch in range(self.epochsNum):
            total_error = 0
            for (data, target) in zip(X, Y):
                F_net = np.dot(data, self.weights)
                if self.addBias:
                    F_net += self.bias

                y_predict = linearActivationFunction(F_net)

                error = target - y_predict

                self.weights += self.learningRate * error * data
                if self.addBias:
                    self.bias += self.learningRate * error

            for (data, target) in zip(X, Y):
                F_net = np.dot(data, self.weights)
                if self.addBias:
                    F_net += self.bias

                y_predict = linearActivationFunction(F_net)
                error = target - y_predict
                total_error += np.round(error, 4) ** 2

            # print(total_error)
            mse = total_error / len(X)
            if mse <= self.mseThreshold:
                break


class Preprocess:
    def __init__(self, classes_input, features_input):
        self.classes_input = classes_input
        self.features_input = features_input
        self.encoder = LabelEncoder()
        self.scaler = StandardScaler()

    def gender_mapping(self, mydata):
        mydata['gender'] = self.encoder.fit_transform(mydata['gender'])
        mydata.iloc[:, 0:5] = mydata.iloc[:, 0:5].astype(float)
        return mydata

    def category_mapping(self, y_train, y_test):
        y_train = self.encoder.fit_transform(y_train)
        y_test = self.encoder.transform(y_test)
        return y_train, y_test

    def filtering(self):
        if self.classes_input == "C1 & C2":
            return ['A', 'B']
        elif self.classes_input == "C2 & C3":
            return ['B', 'C']
        else:
            return ['A', 'C']

    def is_binary(self, data):
        unique_values = np.unique(data)
        return len(unique_values) == 2

    def gender_first_col_check(self, x_train, x_test):
        if self.is_binary(x_train[:, 0]):
            x_train[:, -1] = self.scaler.fit_transform(x_train[:, -1].reshape(-1, 1)).ravel()
            x_test[:, -1] = self.scaler.transform(x_test[:, -1].reshape(-1, 1)).ravel()
        else:
            x_train[:, :] = self.scaler.fit_transform(x_train[:, :])
            x_test[:, :] = self.scaler.transform(x_test[:, :])
        x_train[:, :] = x_train[:, :].astype(float)
        x_test[:, :] = x_test[:, :].astype(float)
        return x_train, x_test

    def extract_features(self):
        features_index = {"Gender": '0',
                          "Body Mass": '1',
                          "Beak Length": '2',
                          "Beak Depth": '3',
                          "Fin Length": '4'
                          }
        feature = self.features_input.split(" & ")

        features_col_number = []
        for f in feature:
            if f in features_index:
                features_col_number.append(int(features_index[f]))

        features_col_number.append(int(5))
        return features_col_number

    def train_test_split(self, x, y):
        AXTrain = x[0:30, :]
        BXTrain = x[50:80, :]

        AXTest = x[30:50, :]
        BXTest = x[80:100, :]

        AYTrain = y[0:30]
        BYTrain = y[50:80]

        AYTest = y[30:50]
        BYTest = y[80:100]

        x_train = np.concatenate([AXTrain, BXTrain], axis=0)
        x_test = np.concatenate([AXTest, BXTest], axis=0)
        y_train = np.concatenate([AYTrain, BYTrain], axis=0)
        y_test = np.concatenate([AYTest, BYTest], axis=0)

        x_train, y_train = shuffle(x_train, y_train, random_state=42)
        x_test, y_test = shuffle(x_test, y_test, random_state=42)
        return x_train, x_test, y_train, y_test

    def preprocessing(self):

        mydata = pd.read_csv('birds.csv')
        # print(mydata.isnull().sum())
        mydata.fillna(mydata.ffill(), inplace=True)
        # print(mydata.isnull().sum())

        mydata = self.gender_mapping(mydata)
        print("Mapped data:", mydata)
        specific_columns = self.extract_features()
        print("specific_columns : ", specific_columns)

        criteria = mydata['bird category'].isin(self.filtering())
        print("Classes selected Before Filter:", self.classes_input)
        print("Classes selected After Filter:", self.filtering())

        filtered_birds = mydata.loc[criteria, mydata.columns[specific_columns]]
        print("filtered_birds : \n", filtered_birds)
        x = filtered_birds.iloc[:, :-1].values
        y = filtered_birds.iloc[:, -1].values

        x_train, x_test, y_train, y_test = self.train_test_split(x, y)

        x_train, x_test = self.gender_first_col_check(x_train, x_test)

        y_train, y_test = self.category_mapping(y_train, y_test)

        return x_train, x_test, y_train, y_test


class Test:
    def __init__(self, x_test, y_test, weights, bias, model, addBias):
        self.addBias = addBias
        self.x_test = x_test
        self.y_test = y_test
        self.weights = weights
        self.bias = bias
        self.f_net = None
        self.model = model

    def choose_model(self):
        if self.model == 1:
            return signumActivationFunction(self.f_net)
        elif self.model == 2:
            return linearActivationFunction(self.f_net)
        else:
            print("Testing tell : Model Error\n")

    def test(self):
        self.f_net = np.dot(self.x_test, self.weights)
        if self.addBias:
            self.f_net += self.bias

        y_pred = self.choose_model()

        if self.model == 1:
            y_pred_binary = np.where(y_pred > 0, 1, 0)
            y_test_binary = self.y_test
        else:
            y_pred_binary = np.where(y_pred > 0, 1, 0)
            y_test_binary = self.y_test

        cm = confusion_matrix(y_test_binary, y_pred_binary)

        tn, fp, fn, tp = cm.ravel()
        accuracy = accuracy_score(y_test_binary, y_pred_binary)

        print(f"\nResults:")
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print("\nConfusion Matrix:")
        print(f"True Positives (TP): {tp}")
        print(f"False Positives (FP): {fp}")
        print(f"True Negatives (TN): {tn}")
        print(f"False Negatives (FN): {fn}")
        return tp, fp, tn, fn, accuracy
