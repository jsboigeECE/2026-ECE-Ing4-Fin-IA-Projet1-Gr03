class MarketMaker:
    
        def __init__(self, inventory_max=10):
         self.inventory = 0
         self.cash = 0.0
         self.inventory_max = inventory_max

        def quote_prices(self, price, spread):
         bid = price - spread / 2
         ask = price + spread / 2
         return bid,ask
        
        def execute_order(self, side, price):
                   if side == "buy" and self.inventory < self.inventory_max:
                      self.inventory += 1
                      self.cash -= price
                   elif side == "sell" and self.inventory > -self.inventory_max:
                     self.inventory -= 1
                     self.cash += price

        def pnl(self, price):
         return self.cash + self.inventory * price






