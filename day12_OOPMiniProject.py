# ---------------------------------------------------------
# Family Conversation - Python OOP Practice
# Concepts Used:
# 1. Classes and Objects
# 2. Constructor (__init__)
# 3. Inheritance
# 4. Encapsulation (private attribute)
# 5. Method Overriding
# 6. f-strings
# ---------------------------------------------------------


# Parent Class
class Conversation:

    # Constructor
    def __init__(self, dad, mom, son):
        self.dad = dad
        self.son = son

        # Private attribute
        self.__mom = mom

    # Method to get the dad's name
    def dad_son(self):
        return self.dad

    # Method to get the mom's name
    def mom_son(self):
        return self.__mom

    # Display basic family information
    def info(self):
        print(f"I am {self.son}")
        print(f"dad of {self.son}")
        print(f"mom of {self.son}")


# Child Class: Son
# Inherits from Conversation
class Son(Conversation):

    # Son's story
    def story(self):
        print(
            f"{self.son} broke his premium brand-new iPhone "
            f"because he became exhausted while playing games!"
        )


# Child Class: Mom
# Demonstrates method overriding
class Mom(Conversation):

    def mom_son(self):
        print(
            f"Mom: {self.son}, what did you do? "
            f"I broke my new phone. Please don't tell Dad; "
            f"he will beat me aggressively."
        )


# Child Class: Dad
# Demonstrates method overriding
class Dad(Conversation):

    def dad_mom(self):
        print(
            f"Dad: Hey dear, why is our {self.son} so silent "
            f"and not causing any trouble?"
        )
        print("Mom: Nothing!")


# Child Class: DadMom
# Another example of method overriding
class DadMom(Conversation):

    def dad_mom(self):
        print(
            f"Mom tells Dad: Our {self.son} broke his new phone. "
            f"Don't beat him. We can explain that he should not "
            f"do it again, play fewer games, and focus on his studies."
        )


# ---------------------------------------------------------
# Creating Objects
# ---------------------------------------------------------

# Parent class object
conversation = Conversation("Dad", "Mom", "Son")

# Child class objects
s = Son("Dad", "Mom", "Kunal")
m = Mom("Dad", "Mom", "Kunal")
d = Dad("Dad", "Mom", "Kunal")
dm = DadMom("Dad", "Mom", "Kunal")


# ---------------------------------------------------------
# Family Information
# ---------------------------------------------------------

s.info()


# ---------------------------------------------------------
# Family Conversation
# ---------------------------------------------------------

print("\n________ One Day ________")
print("Family Conversation.....!")

# Son tells his story
s.story()

# Mom talks to the son
m.mom_son()

# Dad talks to Mom
d.dad_mom()


# ---------------------------------------------------------
# Few Days Later
# ---------------------------------------------------------

print("\n________ Few Days Later ________")

# Mom explains everything to Dad
dm.dad_mom()


# ---------------------------------------------------------
# Program Completed Successfully
# ---------------------------------------------------------