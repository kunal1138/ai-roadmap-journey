# DAY 4: Functions Deep Dive

# *args
def function_1(*args):
    print("the name of student is", args[0], 
          "and the age is", args[1], 
          "and the rollno is", args[2])

function_1("kunal", 20, 40)

# **kwargs
def printMarks(**kwargs):
    for key, value in kwargs.items():
        print(key, value)

marklist = {"kunal": 100, "abhinav": 90, "dimpesh": 80}
printMarks(**marklist)

# Lambda
Ok = lambda k: k * 2      # multiply
cube = lambda k: k ** 3   # cube

print(Ok(5))
print(cube(3))

# Global and Local variable
x = 4  # Global variable
print(x)

def Im():
    y = 5  # Local variable
    print(y)

Im()

# Nested loop
for x in range(3):
    for y in range(1, 9):
        print(y, end=" ")
    print()