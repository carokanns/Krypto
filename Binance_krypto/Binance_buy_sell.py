# %% [markdown]
# # Buy and sell coins based on Binance_live_crypto.ipynb
# ### Thanks to Algovibes YouTube channel

# %%
import asyncio
from binance import AsyncClient, BinanceSocketManager
import pandas as pd
import datetime as dt
from binance.client import Client
from sqlalchemy import create_engine

api_key = '0RKDJcPdqgKzMy4I2j2dXusji6RPSkfU8hJMeP8ssaQTDv64qGCLmvnCEiejw09u'
api_secret = 'nJIIKm9e77fJ0EqbFYDgwgbfUyglywMMjbuhDOCbVd1DCrlwdReBmtNm6PgqnCxK'
engine = create_engine('sqlite:///CryptoDB.db')
client = Client(api_key, api_secret)


# %%
symbols = pd.read_sql('SELECT name FROM sqlite_master WHERE type="table"', engine).name.tolist()

def qry(symbol, lookback:int):
    now = dt.datetime.now() - dt.timedelta(hours=1)	# binance time
    before = now - dt.timedelta(minutes=lookback)
    qry_str = f"""SELECT * FROM '{symbol}' WHERE time >= '{before}'"""
    return pd.read_sql(qry_str, engine)


# %%
rets=[]
for symbol in symbols:
    prices = qry(symbol,3).Price
    cumret = (prices.pct_change()+1).prod()-1
    rets.append(cumret)


# %%
if len(rets)>0:
    top_coin = symbols[rets.index(max(rets))]
    print(top_coin)
else:
    top_coin=None
    print('No coins to buy')


# %%
investment_amount = float(300)
MY_BALANCE = float(500)  ##### This is a temporary variable before we go sharp
 
def get_my_balance():
    global MY_BALANCE
    if False:
        free_usd = [i for i in client.get_account()['balances'] if i['asset']=='USDT'][0]['free']
    
    # temporary override free_usd
    free_usd =  MY_BALANCE
    
    return free_usd


# %%
def create_buy_order(symbol, quantity,price):
    global MY_BALANCE
    if False: 
        order = client.create_order(symbol=symbol, side='BUY',type='MARKET', quantity=quantity)
    
    # temporary until we go sharp
    priset=quantity*price
    order = {'fills':[{'price':price},{'price':price} ]}
    MY_BALANCE += priset
    return order


# %%

info=client.get_symbol_info(symbol=top_coin)
LotSize = float([i for i in info['filters'] if i['filterType']=='LOT_SIZE'][0]['minQty'])
price = float(client.get_symbol_ticker(symbol=top_coin)['price'])
decimals = len(str(LotSize).split('.')[1])
buy_quantity = round(investment_amount/price,decimals)

free_usd = get_my_balance()
if float(free_usd) > investment_amount:
    order=create_buy_order(top_coin,buy_quantity,price)
    buyprice = float(order['fills'][0]['price'])
    print(f'Köpte {buy_quantity} {top_coin} för {buyprice}. Totalt: {buyprice*buy_quantity}')
else:
    buyprice=None
    print('You have not enough USDT to buy the coin', top_coin, free_usd, investment_amount)
    quit()


# %%
def createframe(msg):
    df = pd.DataFrame([msg])
    df = df.loc[:,['s','E','p']]
    df.columns=['symbol', 'Time', 'Price']
    df.Price = df.Price.astype(float)
    df.Time = pd.to_datetime(df.Time, unit='ms')
    return df

def create_sell_order(symbol, quantity,price):
    global MY_BALANCE
    if False:
        order = client.create_order(symbol=symbol, side='SELL', type='MARKET', quantity=quantity)
    
    # temporary until we go sharp
    priset=price*quantity
    order = {'fills':[{'price':price},{'price':price} ]}
    MY_BALANCE -= priset
    return order


# %%
async def main(coin):
    # client = await AsyncClient.create()
    # print('start main')
    min_percent = 0.995
    max_percent = 1.005
    start_time = dt.datetime.now()
    bm = BinanceSocketManager(client)
    ts = bm.trade_socket(coin)
    async with ts as tscm:
        # print('start trade loop')
        while True:
            res = await tscm.recv()
            if res:
                # set elapsed time since start
                elapsed_time = dt.datetime.now() - start_time
                frame = createframe(res)
                if elapsed_time.seconds > 300: # 5 minutes
                    start_time = dt.datetime.now()
                    print(frame.Price[0],'min:',round(buyprice*min_percent,3), 'max:', round(buyprice*max_percent,3))
                if frame.Price[0] < buyprice*min_percent or frame.Price[0] > buyprice*max_percent:
                    order = create_sell_order(coin, buy_quantity,frame.Price[0])
                    print(f'Sålde {buy_quantity} {top_coin} för {frame.Price[0]} Totalt: {buy_quantity*frame.Price[0]}' )
                    print('\n',order)
                    # loop.stop()
                    break
    
    # await client.close_connection()
    
if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main(top_coin))


