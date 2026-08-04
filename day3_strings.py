Name = "kunal"
vowels = "aeiouAEIOU"

# Length of the Name
print(Name)
print(len(Name))
print(Name[::-1])

# Number of Vowels
count = 0
for text in Name:
    if text in vowels:
        count += 1
print("Number of Vowels:", count)

# Comprehension with squares and even numbers
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
squares = [n**2 for n in numbers]
print(squares)

even_num = [num for num in numbers if num % 2 == 0]
print(even_num)

# f-string with your name and age
age = 20
name = "kunal"
print(f"My name is {name} and my age is {age} years old, but after 5 years will be {age+5} years old")