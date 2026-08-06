# DAY 5: File Handling

# --- Write a file ---
with open("student.txt", "w") as f:
    f.write("Name: Kunal\n")
    f.write("Age: 20\n")
    f.write("Course: B.Sc CS\n")

# --- Read the file ---
with open("student.txt", "r") as f:
    print(f.read())

# --- Append to file ---
with open("student.txt", "a") as f:
    f.write("College: Nagpur University\n")

# --- Read line by line ---
with open("student.txt", "r") as f:
    for line in f:
        print(line, end="")

# --- readline and readlines ---
with open("student.txt", "r") as f:
    print(f.readline())   # reads first line only

with open("student.txt", "r") as f:
    lines = f.readlines() # reads all lines as list
    print(lines)

# --- File modes ---
# "r" = read only
# "w" = write (overwrites existing)
# "a" = append (adds to existing)
# "r+" = read and write