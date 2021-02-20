import linreg.LinearRegression as lg

boligOslo = lg.LinearRegression('data/boligOslo.csv')

boligOslo.train(iterations=300)
#boligOslo.plot_model_data()
boligOslo.plot_loss()


