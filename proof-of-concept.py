import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

'''
We consider for this example a retailer who wants to predict daily sales using a Poison distribution as sales are count data. 
The retailer recorded sales data for the past few months and used the Maximum Likelihood Estimation 
(MLE) method to estimate the parameter, λ (lambda), of the Poisson distribution.
(Here we asume that the underlying distribution of the daily sales follows a Poisson distribution). 
'''

# These are simulated daily sales data for the past 30 days as sales_data.
sales_data = np.array([50,60,55,48,52,47,59,55,51,52,53,50,56,57,58,56,54,52,51,50,53,54,55,56,55,54,57,58,56,57])

# Estimation of Lambda using Maximum Likelihood Estimation i.e Average of data
lambda_estimate = np.mean(sales_data)

print(f'Lambda estimate: {lambda_estimate}')

# we then plot the Poisson distribution for the estimated lambda.
x = np.arange(40, 70)
plt.figure(figsize=(10,6))
plt.plot(x, stats.poisson.pmf(x, lambda_estimate), 'bo', ms=8, label='poisson pmf')
plt.vlines(x, 0, stats.poisson.pmf(x, lambda_estimate), colors='b', lw=5, alpha=0.5)
plt.title('Poisson Distribution of Sales with lambda={0:.2f}'.format(lambda_estimate))
plt.ylabel('Probability')
plt.xlabel('Number of Sales')
plt.show()

# We then get an estimate of 54.03 and some grafics

'''
In reality, we'd need more data (like historical sales information across several years) 
to make reliable predictions. This is simply a "proof of concept" scenario
'''