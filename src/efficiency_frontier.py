import pandas as pd 
import numpy as np
import math
import matplotlib.pyplot as plt
from os import listdir 
from scipy.optimize import minimize

def calculate_efficient_portfolios():
	my_path = "../firms"
	number_of_days = 124
	number_of_objective_profits = 100

	files = listdir(my_path)

	#print(files)

	company_names = []

	for file in files:
		company_names.append(file[0:-4])

	#print(company_names)

	company_close_price = pd.DataFrame(columns=company_names)

	#print(company_close_price)

	for file in files: 
		data = pd.read_csv(my_path+'/'+file)
		company_close_price[file[0:-4]] = data['Close'].values[-number_of_days:]

	#print(company_close_price)

	for i in range (number_of_days):
		company_close_price.iloc[number_of_days-1-i] = (company_close_price.iloc[number_of_days-1-i]-company_close_price.iloc[number_of_days-1-i-1])/company_close_price.iloc[number_of_days-1-i-1]

	company_close_price.drop([0], inplace=True)
	company_close_price.reset_index(drop=True, inplace=True)
	daily_profit_rate = company_close_price
	#print(daily_profit_rate)

	average_day_profit = daily_profit_rate.mean(axis=0)
	#print(average_day_profit)

	profit_standard_deviation = daily_profit_rate.std(axis=0)
	#print(profit_standard_deviation)

	profit_variance = daily_profit_rate.var(axis=0)
	#print(profit_variance)

	analysis_chart = pd.DataFrame(columns=['ratio', 'average_profit','standard_deviation','variance'])
	analysis_chart['average_profit'] = average_day_profit
	analysis_chart['standard_deviation'] = profit_standard_deviation
	analysis_chart['variance'] = profit_variance
	analysis_chart['ratio'] = 1/len(company_names)

	#print(analysis_chart)

	covariance_matrix = daily_profit_rate.cov()
	#print(covariance_matrix)

	positive_profits = analysis_chart.query('average_profit > 0')
	min_positive_profits = positive_profits['average_profit'].min()
	max_positive_profits = positive_profits['average_profit'].max()
	#print(min_positive_profits)
	#print(max_positive_profits)

	objective_profit_array = np.linspace(min_positive_profits, max_positive_profits, number_of_objective_profits)
	#print(objective_profit_array)

	portfolio_chart = pd.DataFrame(columns=np.append(['objective_profit', 'portfolio_standard_deviation', ], company_names))
	portfolio_chart['objective_profit'] = objective_profit_array
	#print(portfolio_chart)
	#print(covariance_matrix.iloc[0,0])

	def objective(x):
		total_variance = 0
		for i in range(len(company_names)):
			for j in range(len(company_names)):
				total_variance += covariance_matrix.iloc[i,j]*x[i]*x[j]
		return total_variance

	def constraint1(x):
		portfolio_profit = portfolio_chart.iloc[profit_number,0]
		for i in range(len(company_names)):
			portfolio_profit -= analysis_chart.iloc[i,1]*x[i]
		return portfolio_profit

	def constraint2(x):
		weight_sum = 1
		for i in range(len(company_names)):
			weight_sum -= x[i]
		return weight_sum

	x0 = np.zeros(len(company_names))
	for i in range(len(company_names)):
		if analysis_chart.iloc[i,1] < 0:
			x0[i] = 0
		else:
			x0[i] = 1/len(positive_profits.values.tolist())
	#print(x0)
	profit_number = 0

	for i in range(number_of_objective_profits):
		#print('Initial SSE Objective: ' + str(objective(x0)))

		b = (0,1.0)
		bnds = (b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b)
		con1 = {'type': 'eq', 'fun': constraint1}
		con2 = {'type': 'eq', 'fun': constraint2}
		cons = ([con1, con2])
		solution = minimize(objective,x0,method='SLSQP',bounds=bnds,constraints=cons)

		x = solution.x
		x = np.round(x,6)

		portfolio_chart.iloc[i,1] = round(math.sqrt(objective(x)),6)

		#print('Final SSE Objective: ' + str(objective(x)))

		#print('Solution')
		#for i in range(len(company_names)):
			#print('x'+str(i)+' = '+ str(x[i]))
		for j in range(len(company_names)):
			portfolio_chart.iloc[i,j+2] = x[j]

		#print(x.sum())

		profit_number += 1

	#print(portfolio_chart)

	portfolio_chart.to_csv(r'./results.csv', index=False)


	#plt.scatter(portfolio_chart['portfolio_standard_deviation'].to_numpy(), portfolio_chart['objective_profit'].to_numpy())
	#plt.show()

calculate_efficient_portfolios()






