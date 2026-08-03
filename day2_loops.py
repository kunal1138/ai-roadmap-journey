# DAY 2: Conditionals, Loops, Data Structures

# --- Conditionals ---
age = 18
if age >= 18:
    print("Adult")
elif age >= 13:
    print("Teenager")
else:
    print("Child")

# --- Loops ---
# FizzBuzz
for i in range(1, 101):
    if i % 15 == 0:
        print("FizzBuzz")
    elif i % 3 == 0:
        print("Fizz")
    elif i % 5 == 0:
        print("Buzz")
    else:
        print(i)

# While loop
count = 0
while count < 5:
    print(count)
    count += 1

# --- Data Structures ---
# List
fruits = ["apple", "banana", "mango"]
print(fruits[0])

# Tuple
coordinates = (10, 20)
print(coordinates)

# Dictionary
student = {"name": "Raj", "age": 20}
print(student["name"])

# Set
unique = {1, 2, 3, 2, 1}
print(unique)