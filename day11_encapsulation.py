# DAY 11: OOP - Encapsulation (Wallet Theme 👛)

class Wallet:
    def __init__(self, owner, cash, coins):
        self.owner = owner                # public
        self._wallet = "Armani Exchange"  # protected
        self.__cash = cash                # private
        self.__coins = coins              # private

    # Getters
    def get_cash(self):
        return self.__cash

    def get_coins(self):
        return self.__coins

    def get_total(self):
        return self.__cash + (self.__coins / 100)  # coins in paise

    # Deposit cash
    def deposit_cash(self, amount):
        if amount > 0:
            self.__cash += amount
            print(f"Deposited ₹{amount} cash")
        else:
            print("Invalid amount!")

    # Deposit coins
    def deposit_coins(self, amount):
        if amount > 0:
            self.__coins += amount
            print(f"Deposited {amount} coins")
        else:
            print("Invalid coins!")

    # Withdraw cash
    def withdraw_cash(self, amount):
        if amount <= self.__cash:
            self.__cash -= amount
            print(f"Withdrawn ₹{amount} cash")
        else:
            print("Insufficient cash!")

    # Wallet info
    def wallet_info(self):
        print(f"Owner: {self.owner}")
        print(f"Wallet Brand: {self._wallet}")
        print(f"Cash: ₹{self.__cash}")
        print(f"Coins: {self.__coins}")
        print(f"Total: ₹{self.get_total()}")

# Creating object
MyWallet = Wallet("Kunal", 5000, 50)

# Wallet info
MyWallet.wallet_info()

# Deposit
MyWallet.deposit_cash(1000)
MyWallet.deposit_coins(25)

# Withdraw
MyWallet.withdraw_cash(2000)

# Final balance
print("\n--- Final Wallet ---")
MyWallet.wallet_info()