# DAY 7: Mini Project - To-Do List App

def to_do_list():
    tasks = []  # Empty list to store all tasks

    while True:  # Keep running until user quits
        # Show menu options
        print("\n===== TO-DO LIST =====")
        print("1. Add task")
        print("2. Remove task")
        print("3. Show tasks")
        print("4. Quit")

        choice = input("Enter the choice: ")  # Take user input

        # Choice 1: Add a new task
        if choice == "1":
            task = input("Enter the task: ")  # Ask user for task name
            tasks.append(task)  # Add task to the list
            print(f"Task '{task}' added!")  # Confirm task added

        # Choice 2: Remove an existing task
        elif choice == "2":
            task = input("Enter task to remove: ")  # Ask which task to remove
            if task in tasks:  # Check if task exists in list
                tasks.remove(task)  # Remove task from list
                print(f"Task '{task}' removed!")  # Confirm removal
            else:
                print("Task not found")  # Task doesn't exist

        # Choice 3: Show all tasks
        elif choice == "3":
            if tasks:  # Check if list is not empty
                print("Tasks:")
                for task in tasks:  # Loop through each task
                    print("- " + task)  # Print task with dash
            else:
                print("No tasks yet!")  # List is empty

        # Choice 4: Quit the app
        elif choice == "4":
            print("Goodbye Kunal!")
            break  # Exit the while loop

        # Any other input
        else:
            print("Invalid Choice")  # Wrong input entered

# Run the app
to_do_list()