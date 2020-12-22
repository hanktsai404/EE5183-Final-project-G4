'''
EE5183 Fin-tech Final Project G4
Created on 16th Dec. 2020

@author: hanktsai404

This module is intended to get the daily stock price data of Taiwanese firms and exchange rate data from yahoo finance.
'''
from datetime import datetime, date, timedelta
from io import StringIO
import pandas as pd
import os
import requests
from bs4 import BeautifulSoup
import yfinance as yf
import pandas_datareader.data as web

def years_ago(year: int) -> datetime:
    now = datetime.now()
    this_year = now.year
    result_date = date.today()
    if not (now.month > 2 or (now.month == 2 and now.date == 29)):
        this_year = this_year - 1
    for i in range(year):
        if this_year % 4 == 0:
            result_date = result_date - timedelta(days = 366)
        else:
            result_date = result_date - timedelta(days = 365)
        this_year = this_year - 1
    return result_date
    
class crawler():
    '''Crawling stock price data from yahoo finance'''

    def __init__(self):
        pass
    
    def crawl_firm(self, str_index: str) -> pd.DataFrame:
        '''Crawl daily historical stock price with the given stock index'''
        stock = yf.Ticker(str(str_index)+".TW")
        price_data = stock.history("10y")
        price_data = price_data.drop(columns = ["Dividends", "Stock Splits"])
        price_data.reset_index(inplace = True)
        price_data["Date"] = [int(d.strftime("%Y%m%d")) for d in price_data["Date"]]
        price_data = price_data.sort_values(["Date"], ascending = False)
        price_data["Date"] = [datetime.strptime(str(d), "%Y%m%d") for d in price_data["Date"]]
        price_data = price_data.sort_index()
        print(price_data)
        return price_data
    
    def compute_WMA_and_momentum(self, price_data: pd.DataFrame) -> pd.DataFrame:
        close = price_data[["Close"]].values.tolist()
        WMA = [0]*len(close)
        momentum = [0]*len(close)
        for i in range(len(WMA)):
            if i-9 < 0:
                WMA[i] = None
                momentum[i] = None
                continue
            sum = 0
            n = 0
            for j in range(10):
                sum += (10 - j)*(close[i-j][0])
                n += (10 - j)
            WMA[i] = sum / n
            momentum[i] = close[i][0] - close[i-9][0]
        
        price_data["WMA"] = WMA
        price_data["Momentum"] = momentum
        return price_data
    
    def LL_and_HH(self, price_data: pd.DataFrame) -> pd.DataFrame:
        low = price_data["Low"].to_list()
        high = price_data["High"].to_list()
        LL = [0]*len(low)
        HH = [0]*len(high)
        
        for i in range(len(LL)):
            if i-9 < 0:
                LL[i] = None
                HH[i] = None
                continue
            ten_day_low = low[i-9:i+1]
            ten_day_high = high[i-9:i+1]
            LL[i] = min(ten_day_low)
            HH[i] = max(ten_day_high)
        
        price_data["LL"] = LL
        price_data["HH"] = HH
        return price_data

    def sto_K_and_D(self, price_data: pd.DataFrame) -> pd.DataFrame:
        close = price_data["Close"].to_list()
        LL = price_data["LL"].to_list()
        HH = price_data["HH"].to_list()
        sto_K = [0]*len(HH)
        sto_D = [0]*len(HH)

        for i in range(len(sto_K)):
            if i-9 < 0:
                sto_K[i] = None
                continue
            if LL[i-9] == None:
                sto_K[i] = None
                continue
            sto_K[i] = (close[i] - LL[i-9])/(HH[i-9]-LL[i-9])
        
        for i in range(len(sto_D)):
            if i-9 < 0:
                sto_D[i] = None
                continue
            if sto_K[i-9] == None:
                sto_D[i] = None
                continue
            K_sum = 0
            for j in range(10):
                K_sum += sto_K[i-j]
            sto_D[i] = K_sum / 10
        
        price_data["Sto_K"] = sto_K
        price_data["Sto_D"] = sto_D
        return price_data
    
    def compute_LW_r(self, price_data: pd.DataFrame) -> pd.DataFrame:
        close = price_data["Close"].to_list()
        HH = price_data["HH"].to_list()
        LL = price_data["LL"].to_list()
        LW_R = [0]*len(close)
        for i in range(len(LW_R)):
            if HH[i] == None:
                LW_R[i] = None
                continue
            LW_R[i] = (HH[i] - close[i])/(HH[i] - LL[i])
        price_data["LW_R"] = LW_R
        return price_data
    
    def compute_AD_osc(self, price_data: pd.DataFrame) -> pd.DataFrame:
        close = price_data["Close"].to_list()
        high = price_data["High"].to_list()
        low = price_data["Low"].to_list()
        AD_osc = [0]*len(close)
        for i in range(len(AD_osc)):
            if i-1 < 0:
                AD_osc[i] = None
                continue
            if high[i] == low[i]:
                continue
            AD_osc[i] = (high[i] - close[i-1])/(high[i] - low[i])
        price_data["AD_osc"] = AD_osc
        return price_data

    def compute_CCI(self, price_data: pd.DataFrame) -> pd.DataFrame:
        close = price_data["Close"].to_list()
        high = price_data["High"].to_list()
        low = price_data["Low"].to_list()
        mean = [(c + h + l)/3 for c, h, l in zip(close, high, low)]
        s_mean = [0]*len(close)
        abs_d = [0]*len(close)
        CCI = [0]*len(close)

        for i in range(len(s_mean)):
            if i-9 < 0:
                s_mean[i] = None
                continue
            m_sum = 0
            for j in range(10):
                m_sum += mean[i-j]
            s_mean[i] = m_sum/10
        
        for i in range(len(abs_d)):
            if i-9 < 0:
                abs_d[i] = None
                CCI[i] = None
                continue
            if mean[i-9] == None:
                abs_d[i] = None
                CCI[i] = None
                continue
            abs_sum = 0
            for j in range(10):
                abs_sum += abs(mean[i-j] - s_mean[i])
            abs_d[i] = abs_sum/10
            CCI[i] = (mean[i] - s_mean[i])/(0.015*abs_d[i])
        
        price_data["CCI"] = CCI
        return price_data

    def compute_price_indicators(self, price_data: pd.DataFrame) -> pd.DataFrame:
        '''Compute price indicator that may help prediction, see Jigar Patel, Sahil Shah, Priyank Thakkar, K Kotecha (2014)'''
        '''The prediction target is Close'''
        # Moving average = (Close_t+...+Close_(t-9))/9
        price_data["MA"] = price_data[["Close"]].rolling(window = 10).mean()
        
        # Weighted moving average and momentum
        price_data = self.compute_WMA_and_momentum(price_data)

        # Lowest low (LL) and highest high (HH) in last 10 days
        price_data = self.LL_and_HH(price_data)

        # Stochastic K and D
        price_data = self.sto_K_and_D(price_data)

        # Larry Williams R%
        price_data = self.compute_LW_r(price_data)

        # A/D Oscillator
        price_data = self.compute_AD_osc(price_data)
         
        # CCI
        price_data = self.compute_CCI(price_data)

        price_data = price_data.drop(columns = ["Open", "High", "Low", "LL", "HH"])
        # print(price_data.iloc[:50,])
        return price_data

class market_value_table():
    '''
    Member variables:
        mv_table: The market value table in dataframe type, including rank, id, name and proportion
    '''
    def __init__(self):
        pass

    def crawl_market_value_table(self):
        url = "https://www.taifex.com.tw/cht/9/futuresQADetail"
        mv_web = requests.get(url)
        print("check")
        soup = BeautifulSoup(mv_web.text, "html.parser")
        bs_table = soup.find_all('table')[0]
        l_bs_table = bs_table.text.replace(" ", "").replace("\r", "").split("\n")[10:]
        d_table = {"Rank":[], "Stock_id":[], "Company_name":[], "Proportion":[]}
        for i in range(len(l_bs_table)):
            if not i % 5 == 0:
                pass
            else:
                if l_bs_table[i] == "":
                    break
                d_table["Rank"].append(int(l_bs_table[i]))
                d_table["Stock_id"].append(l_bs_table[i+1])
                d_table["Company_name"].append(l_bs_table[i+2])
                d_table["Proportion"].append(float(l_bs_table[i+3].replace("%", "")) / 100)
        self.mv_table = pd.DataFrame(d_table).sort_values(["Rank"]).reset_index(drop = True)
        print(self.mv_table)
        mv_web.close()
    
    def write_mv_table_to_csv(self):
        now_path = os.getcwd()
        target_csv = open(now_path + "\\" +"../data/Market_Value_Table.csv", "w", newline="", encoding="UTF-8")
        self.mv_table.to_csv(target_csv)
        target_csv.close()

class exchange_rate():
    '''Crawling historical data of exchange rate USD/Target currency from yahoo finance'''
    
    def __init__(self):
        pass
    
    def crawl_exchange_rate(self, target: str) -> pd.DataFrame:
        today = date.today()
        rates_df = web.DataReader(target + "=x", "yahoo", start = years_ago(10), end = today)
        rates_df.reset_index(inplace = True)
        rates_df["Date"] = [int(d.strftime("%Y%m%d")) for d in  rates_df["Date"]]
        rates_df =  rates_df.sort_values(["Date"], ascending = False)
        rates_df["Date"] = [datetime.strptime(str(d), "%Y%m%d") for d in  rates_df["Date"]]
        rates_df =  rates_df.sort_index()
        print(rates_df)
        return rates_df

if __name__ == "__main__":
    '''Testing section'''
    mv_table = pd.read_csv("../data/Market_Value_Table.csv")
    firm_list = mv_table["Stock_id"].to_list()[:1]
    crawler = crawler()
    datas_list = []
    for firm_id in firm_list:
        print(firm_id)
        price_df = crawler.crawl_firm(firm_id)
        price_df = crawler.compute_price_indicators(price_df)
        print(price_df.iloc[:50,])
        datas_list.append(price_df)
    ex_rate = exchange_rate()
    TWD_rates = ex_rate.crawl_exchange_rate("TWD")
    CNY_rates = ex_rate.crawl_exchange_rate("CNY")