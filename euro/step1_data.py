import pandas as pd
import numpy as np
options = pd.read_csv('ndx.csv')
options['strike_price'] = options['strike_price']/1000
options['date'] = pd.to_datetime(options['date'])
options['exdate'] = pd.to_datetime(options['exdate'])
options['tte'] = ((options['exdate'] - options['date']).dt.days - options['am_settlement'])/365
options['price'] = (options['best_bid']+options['best_offer'])/2
options = options[['date', 'exdate', 'strike_price', 'price', 'impl_volatility', 'contract_size', 'tte']]

sec = pd.read_csv('ndx_security.csv')
sec = sec[['date', 'close']]
sec = sec.rename(columns={'close': 'sec_price'})
sec['date'] = pd.to_datetime(sec['date'])
# Merge datasets on 'date'
merged_data = options.merge(sec, on='date', how='left')
rate = pd.read_csv('1_year_rate.csv')
rate = rate.rename(columns={'THREEFY1': 'risk_free'})
rate = rate.rename(columns={'observation_date': 'date'})
rate['risk_free'] = rate['risk_free']/100
rate['date'] = pd.to_datetime(rate['date'])
merged_data = merged_data.merge(rate, on='date', how='left')
# Define the date range
start_date = "2021-06-01"
end_date = "2021-08-31"

# Filter the dataset
merged_data = merged_data[(merged_data['date'] >= start_date) & (merged_data['date'] <= end_date)]

merged_data = merged_data[['strike_price', 'price', 'impl_volatility', 'tte', 'sec_price', 'risk_free']]
merged_data.to_csv('option_data.csv', index=False)