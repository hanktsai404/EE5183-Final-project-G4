import pandas as pd 

data1 = pd.read_csv('./2020-03-10.csv')
data2 = pd.read_csv('./results.csv')

chosen_portfolio = None
expected_return = 0

all_firms_list = data1['Firm'].tolist()

risk_preference = input("Enter risk preference(low, medium, high):")

objective_firm_list = data1.nlargest(5,['Increase'])['Firm'].tolist()
data2['Sum'] = 0

for firm in objective_firm_list:
	data2['Sum'] += data2[firm]

if risk_preference == 'low':
	low_risk_data = data2.iloc[0:34]
	chosen_portfolio_index = int(low_risk_data[low_risk_data['Sum']==low_risk_data['Sum'].max()].index.values)
	chosen_portfolio = data2.iloc[chosen_portfolio_index]
	for firm in all_firms_list:
		expected_return += data1['Increase'][int(data1[data1['Firm']==firm].index.values)] * low_risk_data[firm][chosen_portfolio_index]
elif risk_preference == 'medium':
	medium_risk_data = data2.iloc[34:67]
	chosen_portfolio_index = int(medium_risk_data[medium_risk_data['Sum']==medium_risk_data['Sum'].max()].index.values)
	chosen_portfolio = data2.iloc[chosen_portfolio_index]
	for firm in all_firms_list:
		expected_return += data1['Increase'][int(data1[data1['Firm']==firm].index.values)] * medium_risk_data[firm][chosen_portfolio_index]
elif risk_preference == 'high':
	high_risk_data = data2.iloc[67:100]
	chosen_portfolio_index = int(high_risk_data[high_risk_data['Sum']==high_risk_data['Sum'].max()].index.values)
	chosen_portfolio = data2.iloc[chosen_portfolio_index]
	for firm in all_firms_list:
		expected_return += data1['Increase'][int(data1[data1['Firm']==firm].index.values)] * high_risk_data[firm][chosen_portfolio_index]

chosen_portfolio.to_csv(r'./chosen_porfolio.csv')
print(expected_return-1)

