# Buy and sell coins based on Binance_live_crypto.ipynb

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
top_coin = symbols[rets.index(max(rets))]
print(top_coin, max(rets))


# %%
investment_amount = 300
info=client.get_symbol_info(symbol=top_coin)
LotSize = float([i for i in info['filters'] if i['filterType']=='LOT_SIZE'][0]['stepSize'])
prize = float(client.get_symbol_ticker(symbol=top_coin)['price'])
decimals = len(str(LotSize).split('.')[1])
buy_quantity = round(investment_amount/prize/LotSize,decimals)

free_usd = [i for i in client.get_account()['balances'] if i['asset']=='USDT'][0]['free']
if float(free_usd) < investment_amount:
    order = client.order_market_buy(symbol=top_coin, side='BUY',type='MARKET', quantity=buy_quantity)
    buyprice = float(order['fills'][0]['price'])
    print(top_coin,buy_quantity,buyprice)
else:
    print('You have enough USDT to buy the coin')
    quit()


# %%
investment_amount/prize/LotSize, buy_quantity*prize


# %%
def createframe(msg):
    df = pd.DataFrame([msg['data']])
    df['symbol'] = msg['stream']
    df = df.loc[:,['s','E','p']]
    df.columns=['symbol', 'Time', 'Price']
    df.Price = df.Price.astype(float)
    df.Time = pd.to_datetime(df.Time, unit='ms')
    return df

def create_sell_order(coin, quantity):
    order = client.create_order(symbol=coin, side='SELL', type='MARKET', quantity=quantity)
    return order


# %%
async def main(coin):
    # client = await AsyncClient.create()
    bm = BinanceSocketManager(client)
    ts = bm.trade_socket(coin)
    async with ts as tscm:
        while True:
            res = await tscm.recv()
            if res:
                frame = createframe(res)
                if frame.Price[0] < buyprice*0.97 or frame.Price[0] > buyprice*1.05:
                    order = create_sell_order(coin, buy_quantity)
                    buyprice = float(order['fills'][0]['price'])
                    print(top_coin,buy_quantity, buyprice )
                    loop.stop()
    
    await client.close_connection()
if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main(top_coin))


