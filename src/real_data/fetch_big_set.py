import pandas as pd
import polars as pl
import numpy as np
import time
import logging
import wrds
import os
import sys
import sqlite3
from pathlib import Path
from dotenv import load_dotenv

# Set up logging
logging_file = os.path.join(Path(os.getcwd())) + '/logs/data_fetching.log'
logger = logging.getLogger(__name__)
logging.basicConfig(filename=logging_file, encoding='utf-8', level=logging.INFO)
load_dotenv()

#############################
# Connect to databases
#################################

path_to_external_hd_db = '/media/miroslav/Miroslav Backup/cpop_data/option_prices_raw.db'  # path to external hard drive
path_to_old_db = os.path.join(Path(os.getcwd())) + "/data/db/option_prices_backup.db"

sqlite_conn = sqlite3.connect(path_to_external_hd_db)
old_db = sqlite3.connect(path_to_old_db)
wrds_conn = wrds.Connection(wrds_username=os.getenv("WRDS_USERNAME"), wrds_password=os.getenv("WRDS_PASSWORD"))

#################################
# Setup params
#################################

option_table = 'opprcd'
forward_table = 'fwdprd'
price_table = 'secprd'
ir_table = 'zerocd'
index_secid=108105
years_range = range(1996, 2024)

start_hist = pd.Timestamp('1996-01-01')

#################################
# Make constituents lists 
#################################

sp500_const_permno = pd.read_sql("select * from sp500_constituents", old_db)
secid_permno_link  = pd.read_sql("select * from crsp_opm_link", old_db)
sp500_const_permno['start'] = pd.to_datetime(sp500_const_permno['start'])
sp500_const_permno['ending'] = pd.to_datetime(sp500_const_permno['ending'])
unique_sdates = sp500_const_permno.start.unique()
unique_edates = sp500_const_permno.ending.unique()
change_dates = (pd.concat([pd.Series(unique_sdates), pd.Series(unique_edates)])
                .sort_values()
                .reset_index(drop=True)
                .drop_duplicates()
                # remove dates before 
                .loc[lambda x: x >= start_hist]
                .reset_index(drop=True))
constituents = pd.DataFrame(columns=['secid_constituents_list'], index=change_dates, dtype='object')
secid_permno_link.edate = secid_permno_link.edate.astype('datetime64[ns]')
secid_permno_link.sdate = secid_permno_link.sdate.astype('datetime64[ns]')
for date in change_dates[:-1]:
    constituents_permnos = sp500_const_permno[(sp500_const_permno['start'] <= date) & (sp500_const_permno['ending'] > date)]
    sp500_const_secids = secid_permno_link[(secid_permno_link.permno.isin(constituents_permnos.permno.values)) & (secid_permno_link.sdate <= date) & (secid_permno_link.edate > date)].secid.values
    constituents.loc[date] = ','.join(sp500_const_secids.astype(str))
constituents = constituents.dropna()
constituents.index = constituents.index.map(lambda x: x.date())

#################################
# Fetch option data
#################################

for year in years_range:
    start_year = time.time()
    logging.info(f'Fetching dates for year {year}')
    unique_dates = np.sort(wrds_conn.raw_sql(f"""select distinct date from optionm.{option_table}{year}""").values)
    logging.info(f'Unique dates for year {year} fetched in {time.time()-start_year} s')
    for current_date in unique_dates:
        print(current_date)
        # unpack array in the array
        temp_date = current_date[0]
        date_str = temp_date.strftime('%Y-%m-%d')

        last_available_constituents = (
            constituents
            .loc[constituents.index[constituents.index <= temp_date][-1]]
            .values[0]
        )
        
        start_inner = time.time()
        logging.info(f'Fetching data for date {date_str}')

        option_table_df = wrds_conn.raw_sql(f"""
        SELECT secid, date, exdate, last_date, best_bid, best_offer, volume, open_interest, cp_flag, impl_volatility, strike_price 
        FROM optionm.opprcd{year} 
        WHERE date='{date_str}'::DATE 
        AND secid in ({last_available_constituents}) 
        AND secid IS NOT NULL 
        AND date IS NOT NULL 
        AND exdate IS NOT NULL 
        AND last_date IS NOT NULL 
        AND best_bid IS NOT NULL 
        AND best_offer IS NOT NULL 
        AND volume IS NOT NULL 
        AND open_interest IS NOT NULL 
        AND cp_flag IS NOT NULL 
        AND impl_volatility IS NOT NULL 
        AND strike_price IS NOT NULL 
        AND volume > 0 
        AND open_interest > 0 
        AND best_bid > 0  
        AND (exdate::DATE - date::DATE) > 9 
        AND (exdate::DATE - date::DATE) < 730 
        AND (date::DATE - last_date::DATE) < 5 
        """)

        logging.info(f'Option data for {date_str} fetched in {(time.time()-start_inner):.2f} s')

        start_inner_stocks = time.time()

        stock_table_df = wrds_conn.raw_sql(f"""select * from optionm.secprd{year} 
        where date='{date_str}'::DATE 
        and secid in ({last_available_constituents})""")

        logging.info(f'Stock data for {date_str} fetched in {(time.time()-start_inner_stocks)/60:.2f} minutes')

        start_writing = time.time()
        option_table_df.to_sql(name='option_price_table', con=sqlite_conn, if_exists='append', index=False)
        stock_table_df.to_sql(name='stock_price_table', con=sqlite_conn, if_exists='append', index=False)
        logging.info(f'Data for {date_str} written in {(time.time()-start_writing):.2f} s')
    logging.info(f'Year {year} done in {time.time()-start_year} s')


##### Add interest rates

ir = wrds_conn.get_table(library='optionm', table='zerocd')
ir.to_sql('zerocd', con=sqlite_conn, if_exists='replace', index=False)