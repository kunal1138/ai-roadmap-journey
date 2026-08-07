# DAY 6: Exception Handling

# try and except
try:
    a = int(input("Enter the Number: "))
    print(f"Multiplication table {a} is:")
    for i in range(1, 11):
        print(f"{a} * {i} = {a*i}")
except ValueError:
    print("Put value only Integer")
    print("Don't put string values")
except Exception as e:
    print("Some error occurred:", e)

# else and finally
try:
    l = [2, 4, 5, 9]
    i = int(input("Enter the index: "))
    print(l[i])
except:
    print("Some error occurred")
finally:
    print("I always executed, Anyhow")

# raise exception
def check_age(age):
    if age < 0:
        raise ValueError("The Age cannot be negative bro")
    print("Valid age:", age)

check_age(-5)