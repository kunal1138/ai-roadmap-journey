# DAY 11: OOP - Encapsulation (Wallet Theme 👛)

class Wallet:
    def __init__(self, owner, money):
        self.owner = owner          # public
        self._wallet = "Armani Exchange"  # protected
        self._money = money         # private (hidden!)

    # Getter - to access private money
    def get_money(self):
        return self._money

    # Setter - deposit money
    def deposit(self, amount):
        if amount > 0:
            self._money += amount
            print(f"Deposited {amount}")

    # Setter - withdraw money
    def withdraw(self, amount):
        if amount <= self._money:
            self._money -= amount
            print(f"Withdrawn {amount}")
        else:
            print("Insufficient Balance!")

# Creating object
MyWallet = Wallet("Kunal", 5000)

# Accessing variables
print(MyWallet.owner)         # public ✅
print(MyWallet._wallet)       # protected ✅
print(MyWallet.get_money())   # private via getter ✅

# Testing deposit and withdraw
MyWallet.deposit(1000)
print(MyWallet.get_money())   # 6000

MyWallet.withdraw(2000)
print(MyWallet.get_money())   # 4000

MyWallet.withdraw(10000)      # Insufficient Balance!