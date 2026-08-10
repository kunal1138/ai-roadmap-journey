# DAY 9: OOP - Inheritance (Ben10 Theme 🦸)

# Parent class
class Ben10:
    def __init__(self, name, alien):
        self.name = name    # character name
        self.alien = alien  # alien form

    # Method to transform
    def transform(self):
        print(f"It's hero time {self.alien}...!")

    # Method to introduce
    def intro(self):
        print(f"I'm {self.name} Tennyson")

# Child class 1 - HeatBlast inherits from Ben10
class HeatBlast(Ben10):
    def __init__(self, name, alien, typ):
        super().__init__(name, alien)  # calls Ben10's __init__
        self.typ = typ  # extra attribute

    # Overriding method
    def say(self):
        print(f"{self.name} HeatBlast..!")

# Child class 2 - FourArms inherits from Ben10
class FourArms(Ben10):
    def __init__(self, name, alien):
        super().__init__(name, alien)  # calls Ben10's __init__
        print(f"{self.name} FourArms..!")

# Creating objects
heatblast = HeatBlast("Ben10", "HeatBlast", "Pyronite")
fourarms = FourArms("Ben10", "FourArms")

# Calling methods
heatblast.intro()
heatblast.transform()
heatblast.say()

fourarms.intro()
fourarms.transform()