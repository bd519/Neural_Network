import json
import math
import pickle

import sklearn.metrics
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.optim as optim
import sklearn.model_selection
from sklearn import preprocessing

class Regressor(nn.Module):

    def __init__(self, x=None, nb_epoch = 1000, batch_size = 128, learning_rate = 0.01, nb_overfitting_epoch_threshold =30, layer_sizes = [60,40,50], fromJSON=None):
        # You can add any input parameters you need
        # Remember to set them with a default value for LabTS tests
        """ 
        Initialise the model.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape 
                (batch_size, input_size), used to compute the size 
                of the network.
            - nb_epoch {int} -- number of epoch to train the network.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Replace this code with your own
        super(Regressor, self).__init__()
        print("train")
        if fromJSON is not None:
            with open(fromJSON, 'rb') as f:
                data = json.load(f)
                self.input_size = data["input_size"]
                self.nb_epoch = data["nb_epoch"]
                self.layer_sizes = data["layer_sizes"]
                structure = list()
                structure.append(nn.Linear(self.input_size, self.layer_sizes[0]))
                structure.append(nn.Tanh())
                for i in range(1, len(self.layer_sizes)):
                    structure.append(nn.Linear(self.layer_sizes[i - 1], self.layer_sizes[i]))
                    structure.append(nn.Tanh())
                structure.append(nn.Linear(self.layer_sizes[-1], 1))
                self.stack = nn.Sequential(*structure)
                self.nb_overfitting_epoch_threshold = data["nb_overfitting_epoch_threshold"]
                if data["scaler_x"] is not None:
                    scaler_filename = data["scaler_x"]
                    self.scaler_x = pickle.load(open(scaler_filename,'rb'))
                else:
                    self.scaler_x = None
                if data["scaler_y"] is not None:
                    scaler_filename = data["scaler_y"]
                    self.scaler_y = pickle.load(open(scaler_filename,'rb'))
                else:
                    self.scaler_y = None
                self.batch_size = data["batch_size"]
                self.learning_rate = data["learning_rate"]
                return

        X, _ = self._preprocessor(x, training = True)
        self.input_size = X.size(dim=1)
        self.output_size = 1
        self.nb_epoch = nb_epoch
        self.layer_sizes = layer_sizes
        structure = list()
        structure.append(nn.Linear(self.input_size, layer_sizes[0]))
        structure.append(nn.Tanh())
        for i in range(1, len(layer_sizes)):
                structure.append(nn.Linear(layer_sizes[i-1], layer_sizes[i]))
                structure.append(nn.Tanh())
        structure.append(nn.Linear(layer_sizes[-1], 1))

        self.stack = nn.Sequential(*structure)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.nb_overfitting_epoch_threshold = nb_overfitting_epoch_threshold
        self.scaler_x = None
        self.scaler_y = None
        return

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def saveConfiguration(self, name):
        scaler_x_filename  = None
        if self.scaler_x is not None:
            scaler_x_filename = "scaler_x.save"
            with open(scaler_x_filename, "wb") as f:
                pickle.dump(self.scaler_x, f)

        scaler_y_filename= None
        if self.scaler_y is not None:
            scaler_y_filename = "scaler_y.save"
            with open(scaler_y_filename, "wb") as f:
                pickle.dump(self.scaler_y, f)

        config = {
            "input_size": self.input_size,
            "nb_epoch" : self.nb_epoch,
            "layer_sizes" :self.layer_sizes,
            "nb_overfitting_epoch_threshold" : self.nb_overfitting_epoch_threshold,
            "scaler_x": scaler_x_filename,
            "scaler_y": scaler_y_filename,
            "learning_rate" : self.learning_rate,
            "batch_size" : self.batch_size
        }
        with open(name, 'w') as f:
            json.dump(config, f)

    def _preprocessor(self, x, y = None, training = False):
        """ 
        Preprocess input of the network.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or 
                testing the model.

        Returns:
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed input array of
              size (batch_size, input_size).
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed target array of
              size (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Replace this code with your own
        # Return preprocessed x and y, return None for y if it was None
        print("preprocess")
        x = x.replace(to_replace="<1H OCEAN", value="0.0,0.0,0.0,0.0,1.0")
        x = x.replace(to_replace="INLAND", value="0.0,0.0,0.0,1.0,0.0")
        x = x.replace(to_replace="NEAR BAY", value="0.0,0.0,1.0,0.0,0.0")
        x = x.replace(to_replace="NEAR OCEAN", value="0.0,1.0,0.0,0.0,0.0")
        x = x.replace(to_replace="ISLAND", value="1.0,0.0,0.0,0.0,0.0")

        print(x)
        x[['<1H OCEAN','INLAND','NEAR BAY','NEAR OCEAN', "ISLAND"]] = x["ocean_proximity"].str.split(",", expand=True)
        pd.set_option('display.max_columns', None)
        x = x.drop(columns=["ocean_proximity"])


        x[['<1H OCEAN','INLAND','NEAR BAY','NEAR OCEAN','ISLAND']] = x[['<1H OCEAN','INLAND','NEAR BAY','NEAR OCEAN','ISLAND']].apply(pd.to_numeric)

        if y is not None:
            XY = pd.concat([x,y], axis=1)
            XY = XY.dropna()
            x = XY.iloc[:,:-1]
            y = XY.iloc[:,-1:]
        else:
            x = x.fillna(x.mean())

        x = x.to_numpy()
        if y is not None:
            y = y.to_numpy()

        if training:
            self.scaler_x = preprocessing.StandardScaler()
            self.scaler_x.fit(x)
            if y is not None:
                self.scaler_y = preprocessing.StandardScaler()
                self.scaler_y.fit(y)
                self.mean_y = np.mean(y)

        x = torch.Tensor(self.scaler_x.transform(x))
        y = torch.Tensor(self.scaler_y.transform(y)) if y is not None else None


        return x, y

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


    def fit(self, x, y, output_label="median_house_value"):
        """
        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # create your optimizer
        print("fit")
        optimizer = optim.SGD(self.parameters(), lr= self.learning_rate)
        criterion = nn.MSELoss()

        training_data = pd.concat([x,y], axis=1)
        train, validate = sklearn.model_selection.train_test_split(training_data, test_size=(1.0 / 3.0))
        x_train = train.loc[:, train.columns != output_label]
        y_train = train.loc[:, [output_label]]
        x_validate = validate.loc[:, validate.columns != output_label]
        y_validate = validate.loc[:, [output_label]]
        X_train, Y_train = self._preprocessor(x_train, y=y_train, training=True)  # Do not forget
        X_validate, Y_validate = self._preprocessor(x_validate, y_validate, training=False)

        train_dataset = torch.utils.data.TensorDataset(X_train,Y_train)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size)

        min_validation_loss = 0
        nb_epoch_overfitting = 0
        epoch_num = 0
        for i in range(self.nb_epoch):
            for x,y in train_dataloader:
                optimizer.zero_grad()  # zero the gradient buffers
                output = self(x)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
            validation_loss = criterion(self(X_validate), Y_validate)
            if min_validation_loss == 0:
                min_validation_loss = validation_loss
                nb_epoch_overfitting = 0
            elif validation_loss < min_validation_loss:
                min_validation_loss = validation_loss
                nb_epoch_overfitting = 0
            elif validation_loss >= min_validation_loss:
                nb_epoch_overfitting += 1
                if nb_epoch_overfitting == self.nb_overfitting_epoch_threshold:
                    break;
            epoch_num += 1
        return self

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def forward(self, x):
        return self.stack(x)

    def predict(self, x):
        """
        Ouput the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).

        Returns:
            {np.darray} -- Predicted value for the given input (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        print("predict")
        X, _ = self._preprocessor(x, training = False) # Do not forget
        return self.scaler_y.inverse_transform(self(X).detach().numpy())


        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def score(self, x, y):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw ouput array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        print("score")
        X, Y = self._preprocessor(x, y = y, training = False) # Do not forget
        criterion = nn.MSELoss()
        output = self(X)
        print(output.size())
        print(Y.size())
        return criterion(output, Y)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def non_normalised_score(self, x, y):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw ouput array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, Y = self._preprocessor(x, y = y, training = False) # Do not forget
        criterion = nn.MSELoss()
        output = self(X)
        scaled_output = self.scaler_y.inverse_transform(output.detach().numpy())
        scaled_expected_output = self.scaler_y.inverse_transform(Y.detach().numpy())

        return sklearn.metrics.mean_squared_error(scaled_output, scaled_expected_output)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


def save_regressor(trained_model): 
    """ 
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    trained_model.saveConfiguration('reg_struct_part_2_model.json')
    torch.save(trained_model.state_dict(), 'part2_model.torch')
    print("\nSaved model in part2_model.pickle\n")


def load_regressor():
    """
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    model = Regressor(fromJSON='reg_struct_part_2_model.json')
    model.load_state_dict(torch.load('part2_model.torch'))
    print("\nLoaded model in part2_model.torch\n")
    return model

def find_best_batch_and_learning(x_train, y_train, x_test, y_test, layer_sizes):
    batch_size_arr = [32, 64, 128]
    learning_rate_arr = [0.1, 0.075, 0.05, 0.025, 0.01]
    best_regressor_score = 100
    best_batch_size = 0
    best_learning_rate = 0
    for batch_size in batch_size_arr:
        for learning_rate in learning_rate_arr:
            r = Regressor(x = x_train,batch_size=batch_size, learning_rate=learning_rate, layer_sizes=layer_sizes)
            r.fit(x_train, y_train)
            score = r.score(x_test, y_test)

            if best_regressor_score > score:
                best_regressor_score = score
                best_learning_rate = learning_rate
                best_batch_size = batch_size
            print(f"Score, {score},Layer Sizes, {layer_sizes},Learning Rate, {learning_rate}, Batch Size, {batch_size}")

    return best_batch_size, best_learning_rate

def SimpleSearch():

    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas Dataframe as inputs
    data = pd.read_csv("housing.csv")
    train, test = sklearn.model_selection.train_test_split(data, test_size=0.2)
    # Spliting input and output
    x_train = train.loc[:, train.columns != output_label]
    y_train = train.loc[:, [output_label]]
    r = Regressor(x=x_train, batch_size=10, learning_rate=0.075, layer_sizes=[10,20])
    r.fit(x_train,y_train)
    return r

def RegressorHyperParameterSearch(file_path):
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented 
    in the Regressor class.

    Arguments:
        Add whatever inputs you need.
        
    Returns:
        The function should return your optimised hyper-parameters. 

    """


    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################

    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas Dataframe as inputs
    data = pd.read_csv(file_path)
    train, test = sklearn.model_selection.train_test_split(data, test_size=0.2)
    # Spliting input and output
    x_train = train.loc[:, train.columns != output_label]
    y_train = train.loc[:, [output_label]]
    x_test = test.loc[:, test.columns != output_label]
    y_test = test.loc[:, [output_label]]

    output_neurons = range(10, 110, 10)

    best_layer_sizes = list()
    best_score = list()
    best_batch, best_learn = find_best_batch_and_learning(x_train, y_train, x_test, y_test, [13])

    for i in range(3):
        best_layer_regressor_score = 100
        best_layer_size = 0
        for l in output_neurons:
            r = Regressor(x=x_train, batch_size=best_batch, learning_rate=best_learn,
                          layer_sizes=best_layer_sizes + [l])
            r.fit(x_train, y_train)
            score = r.score(x_test, y_test)
            print(f"Score, {score},Layer Sizes, {best_layer_sizes+[l]},Learning Rate, {best_learn}, Batch Size, {best_batch}")

            if best_layer_regressor_score > score:
                best_layer_regressor_score = score
                best_layer_size = l
        best_layer_sizes.append(best_layer_size)
        best_score.append(best_layer_regressor_score)


    best_number_of_layers = best_score.index(min(best_score))
    best_layer_config = best_layer_sizes[0:best_number_of_layers+1]

    best_batch, best_learn = find_best_batch_and_learning(x_train, y_train, x_test, y_test, best_layer_config)


    # Return the chosen hyper parameters

    optimal_model = Regressor(x=x_train, batch_size=best_batch, learning_rate=best_learn, layer_sizes=best_layer_config)
    optimal_model.fit(x_train, y_train)
    print("\nRegressor error: {}\n".format(optimal_model.scaler_y.inverse_transform([[math.sqrt(optimal_model.score(x_test, y_test))]]) - optimal_model.mean_y))
    print("\nRegressor error: {}\n".format(math.sqrt(optimal_model.non_normalised_score(x_test, y_test))))
    print("\nRegressor error: {}\n".format(optimal_model.score(x_test, y_test)))
    return optimal_model


    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################



def example_main():
    # result = RegressorHyperParameterSearch()
    # save_regressor(result)

    solution = load_regressor()
    data = pd.read_csv("housing.csv")
    train, test = sklearn.model_selection.train_test_split(data, test_size=0.2)

    output_label = "median_house_value"
    print(solution.score( train.loc[:, train.columns != output_label], train.loc[:, [output_label]] ))

if __name__ == "__main__":
    example_main()

