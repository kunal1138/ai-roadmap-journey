# DAY 8: OOP - Classes and Objects

class Student:
    # Constructor - runs when object is created
    def __init__(self, name, age, grade):
        self.name = name    # instance variable
        self.age = age      # instance variable
        self.grade = grade  # instance variable

    # Method to introduce student
    def introduce(self):
        print(f"Hi, My name is {self.name} and my age is {self.age}, I got {self.grade}")

    # Method to check passing status
    def passing(self):
        if self.grade >= 90:
            print(f"{self.name} you are Topper..!")
        elif self.grade >= 30:
            print(f"{self.name} you are Passed..!")
        elif self.grade >= 0:
            print(f"{self.name} you are Failed..!")
        else:
            print("Put only grade values")

# Creating 3 student objects
student1 = Student("Kunal", 20, 91)
student2 = Student("Abhinav", 20, 85)
student3 = Student("Sonu", 20, 29)

# Calling introduce method
student1.introduce()
student2.introduce()
student3.introduce()

# Calling passing method
student1.passing()
student2.passing()
student3.passing()