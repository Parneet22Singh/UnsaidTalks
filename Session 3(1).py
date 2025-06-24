# Store credentials
credentials = {}

# Check strong password using lambda and filter
def is_strong(password):
    has_length = len(password) > 8
    has_upper = any(c.isupper() for c in password)
    has_lower = any(c.islower() for c in password)
    has_digit = any(c.isdigit() for c in password)
    has_special = any(not c.isalnum() for c in password)

    return has_length and has_upper and has_lower and has_digit and has_special

    passed = list(filter(lambda check: check(password), checks))
    return len(passed) == len(checks)

# Classify password strength using if-else
def classify(password):
    score = 0
    if len(password) > 8:
        score += 1
    if any(c.isupper() for c in password):
        score += 1
    if any(c.islower() for c in password):
        score += 1
    if any(c.isdigit() for c in password):
        score += 1
    if any(not c.isalnum() for c in password):
        score += 1

    if score <= 2:
        return "Weak"
    elif score <= 4:
        return "Medium"
    else:
        return "Strong"

# Menu loop
while True:
    print("\n--- Menu ---")
    print("1. Add New Credential")
    print("2. View Credentials")
    print("3. Delete a Credential")
    print("4. Analyze Password Strength")
    print("5. Exit")

    choice = input("Enter choice: ")

    if choice == "1":
        username = input("Enter username: ")
        password = input("Enter password: ")
        credentials[username] = password
        print("Credential added.")
    elif choice == "2":
      u=input("Enter the username whose password you want to view: ")
      if u in credentials:
        print(f"The password for {u} is {credentials[u]}")
    elif choice == "3":
        username = input("Enter username to delete: ")
        if username in credentials:
            del credentials[username]
            print("Deleted.")
        else:
            print("Username not found.")
    elif choice == "4":
      u=input("Enter the username whose password strength you want to check: ")
      if u in credentials:
        print(f"{u} -> Password: {credentials[u]}-> Strength: {classify(credentials[u])} | Strong: {is_strong(credentials[u])}")
    elif choice == "5":
        break
    else:
        print("Invalid choice.")
