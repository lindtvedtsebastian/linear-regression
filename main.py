import linreg.LinearRegression as lg


ln = lg.LinearRegression('data/boligOslo.csv')

#Train the model with 500 iterations at a learning rate of 0.000001
ln.train(iterations=500, learning_rate=0.000001)

#Plots the data of the model
ln.plot_model_data()

#Visualizes the loss
ln.plot_loss()

#Predicts a value for y given input of X = 50
ln.predict(50)


