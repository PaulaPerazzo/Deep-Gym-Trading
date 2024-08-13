import yfinance as yf
import investpy as inv
import pandas as pd

def get_br_stocks():
    br_stocks = inv.stocks.get_stocks(country='brazil')
    br_stocks.to_csv(f'../../data/br_stocks.csv')

    wallet = []

    for stock in br_stocks["symbol"]:
        if len(stock) <= 5:
            wallet.append(stock + ".SA")
    
    return wallet


def get_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date, group_by='ticker')
    stock_data.to_csv(f'../../data/ibovesta_stock.csv')

    return stock_data


def get_adj_close(stock_data):
    df_temp = []
    df_temp.append(stock_data["Ticker"])

    print(stock_data.columns.tolist)

    for column in stock_data:
        if stock_data[column][0] == "Adj Close" or stock_data[column][0] == "^BVSP":
            df_adj_close = stock_data[column].copy()
            df_temp.append(df_adj_close)
    
    df_adj_close = pd.concat(df_temp, axis=1)
    df_adj_close = df_adj_close[2:]
    df_adj_close.to_csv(f'../../data/adj_close_stocks.csv')

    return df_adj_close


def split_data(data):
    train_data = data.iloc[:int(len(data)*0.7)]
    validation_data = data.iloc[int(len(data)*0.7):int(len(data)*0.85)]
    test_data = data.iloc[int(len(data)*0.85):]

    train_data.to_csv(f'../../data/train_data.csv')
    validation_data.to_csv(f'../../data/validation_data.csv')
    test_data.to_csv(f'../../data/test_data.csv')

    return train_data, validation_data, test_data
