# NALAI THIRAN IBM
# DOMAIN:APPLIED DATASCIENCE
# PROJECT TITLE: STOCK PRICE PREDICTION 
# SOURCE DATASET:MSFT.CSV
# REQUIRED LIBRARIES FOR STOCK PRICE PREDICTION
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    from sklearn.metrics import mean_squared_error
    from math import sqrt
# STEPS FOR PREDICTING THE STOCK PRICE 
    1.Use Python and import necessary libraries.
    2.Load historical stock price data into a Pandas DataFrame.
    3.Explore the dataset to understand its structure and characteristics.
    4.Handle missing values and preprocess the data and create relevant features for stock price prediction.
    5.Divide the dataset into training and testing sets.
    6.Standardize or normalize the features for consistent model training.
    7.Choose a model and train it on the training data.
    8.Use the trained model to make predictions on the test data.
    9.Evaluate the model's performance using metrics like Mean Squared Error.
    10.Plot actual vs. predicted values for a visual assessment.
    11.Fine-tune the model parameters and iterate on feature engineering to improve predictions.

# PROGRAM 

    # Load the dataset (replace with your data loading code)
    data = pd.read_csv('MSFT.csv')

    # Extract the 'Close' prices
    data = data[['Close']]
    dataset = data.values
    dataset = dataset.astype('float32')

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    # Split the dataset into training and testing sets
    train_size = int(len(dataset) * 0.67)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

    # Create datasets with look-back time steps
    def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

    look_back = 1
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    # Reshape input to be [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

    # Make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    # Inverse transform the predictions
    trainPredict =  scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])

    # Calculate root mean squared error
    trainScore = sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
    print(f"Train RMSE: {trainScore}")
    testScore = sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
    print(f"Test RMSE: {testScore}")

    # Plot the results
     trainPredictPlot = np.empty_like(dataset)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict

    testPredictPlot = np.empty_like(dataset)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(dataset) - 1, :] = testPredict

    plt.plot(scaler.inverse_transform(dataset), label='Original Data')
    plt.plot(trainPredictPlot, label='Training Predictions')
    plt.plot(testPredictPlot, label='Testing Predictions')
    plt.legend()
    plt.show()
