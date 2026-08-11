# DAY 10: OOP - Polymorphism (Spiderman Multiverse Theme 🕷️)

# Parent class
class Spiderman:
    def __init__(self, name, power):
        self.name = name        # hero name
        self.power = power      # hero power

    # Default attack method
    def attack(self):
        print(f"{self.name} attack..!")

    # Info method
    def info(self):
        print(f"Hero: {self.name}, Power is: {self.power}")

# Child class 1 - Tobey Maguire Spiderman
class TobeySpiderman(Spiderman):
    def attack(self):   # overriding parent attack
        print(f"{self.name} throws organic web, THWIP...!")

# Child class 2 - Andrew Garfield Spiderman
class AndrewSpiderman(Spiderman):
    def attack(self):   # overriding parent attack
        print(f"{self.name} throws artificial web, Thwip..!")

# Child class 3 - Tom Holland Spiderman
class TomSpiderman(Spiderman):
    def attack(self):   # overriding parent attack
        print(f"{self.name} throws artificial technology web, Thwip!")

# Polymorphism in action - Multiverse of Spiderman!
MultiverseOfSpiderman = [
    TobeySpiderman("Tobey Spiderman", "Organic Web"),
    AndrewSpiderman("Andrew Spiderman", "Artificial Web"),
    TomSpiderman("Tom Spiderman", "Technology Web")
]

# Same method, different behavior = Polymorphism!
for spiderman in MultiverseOfSpiderman:
    spiderman.attack()