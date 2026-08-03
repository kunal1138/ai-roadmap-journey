# DAY 1: Variables, Functions, Operators, Modules

# --- Variables ---
name = "Your Name"
age = 20
gpa = 8.5
is_student = True

print(name, age, gpa, is_student)

# --- Operators ---
a = 10
b = 3
print(a + b)   # Addition
print(a - b)   # Subtraction
print(a * b)   # Multiplication
print(a / b)   # Division
print(a % b)   # Modulus
print(a ** b)  # Power

# --- Functions ---
def greet(name):
    return "Hello " + name

print(greet("Aakash"))

def add(a, b):
    return a + b

print(add(5, 3))

# --- Modules ---
import math
print(math.sqrt(16))
print(math.pi)

import random
print(random.randint(1, 10))