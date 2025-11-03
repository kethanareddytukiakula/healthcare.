"""
Healthcare / Vaccination Portal - Single File Project
Covers Modules:
- Module 1: Python fundamentals (data types, I/O, operators, builtin funcs)
- Module 2: Program flow (conditionals, loops, break/continue/pass)
- Module 3: Data organization & searching/sorting (lists, tuples, dict, set, linear/binary search, sorts)
- Module 4: Modular programming (functions, args, scope, recursion, stacks/queues, menu-driven)
- Module 5: Numpy & Pandas (dataframes, cleaning, filtering, grouping, sorting, aggregation)

File-based persistent storage: CSV (vaccination_data.csv)

Run: python vaccination_portal.py
"""

import os
import sys
import csv
import math
from collections import deque
from getpass import getpass

# Third-party libraries. Ensure installed: pip install pandas numpy
try:
    import pandas as pd
    import numpy as np
except Exception as e:
    print("This script requires pandas and numpy. Install them with: pip install pandas numpy")
    raise e

# ---------------------------
# Problem Analysis (pseudocode)
# ---------------------------
# PSEUDOCODE:
# - Maintain vaccination records in CSV
# - Provide Admin menu (CRUD, stats) and Client menu (view/search/sort)
# - Demonstrate searching/sorting algorithms (Python implementations)
# - Provide numpy/pandas examples for aggregation/filtering
# - Keep menu-driven main loop

# ---------------------------
# Constants / File handling
# ---------------------------
DATA_FILE = "vaccination_data.csv"
VACCINE_INFO_FILE = "vaccine_info.csv"  # example to demonstrate merge/joins

# Columns used for DataFrame
COLUMNS = ["ID", "Name", "Age", "Gender", "Vaccine", "Dose", "City"]

# ---------------------------
# Module 1: Setup & CSV helpers (file I/O, data types)
# ---------------------------

def ensure_files_exist():
    """Create CSV files if they don't exist (basic file handling)."""
    if not os.path.exists(DATA_FILE):
        df = pd.DataFrame(columns=COLUMNS)
        df.to_csv(DATA_FILE, index=False)
    if not os.path.exists(VACCINE_INFO_FILE):
        df_v = pd.DataFrame([
            {"Vaccine": "Covishield", "Manufacturer": "AstraZeneca", "DosesRequired": 2},
            {"Vaccine": "Covaxin", "Manufacturer": "BharatBiotech", "DosesRequired": 2},
            {"Vaccine": "Sputnik", "Manufacturer": "Gamaleya", "DosesRequired": 2},
            {"Vaccine": "Pfizer", "Manufacturer": "Pfizer/BioNTech", "DosesRequired": 2},
            {"Vaccine": "Moderna", "Manufacturer": "Moderna", "DosesRequired": 2},
        ])
        df_v.to_csv(VACCINE_INFO_FILE, index=False)

def load_df():
    """Load vaccination data into a pandas DataFrame."""
    return pd.read_csv(DATA_FILE)

def save_df(df):
    """Save DataFrame to CSV (persist changes)."""
    df.to_csv(DATA_FILE, index=False)

# ---------------------------
# Utilities and ID generation
# ---------------------------

def generate_id(df: pd.DataFrame) -> int:
    """Generate incremental integer ID (demonstrates numeric types)."""
    if df.empty:
        return 1
    else:
        return int(df["ID"].max()) + 1

# ---------------------------
# Module 3: Data structures (lists, tuples, dicts, sets)
# ---------------------------

def demo_collections():
    """Short demo of list, tuple, dict, set and comprehensions."""
    print("\n--- Collections Demo ---")
    lst = [1, 2, 3, 4]                    # list (mutable)
    tpl = (10, 20, 30)                    # tuple (immutable)
    dct = {"a": 1, "b": 2}                # dict (mapping)
    st = {1, 2, 3, 3}                     # set (unique elements)
    comp = [x * x for x in range(5)]      # list comprehension

    print("List:", lst)
    print("Tuple:", tpl)
    print("Dict:", dct)
    print("Set:", st)
    print("Comprehension (squares):", comp)

# ---------------------------
# Module 2: Control flow examples
# ---------------------------

def control_flow_examples(n=5):
    """Demonstrate loops, break, continue, pass."""
    print("\n--- Control Flow Demo ---")
    i = 0
    while i < n:
        i += 1
        if i == 2:
            print("continue at", i)
            continue
        if i == 4:
            print("break at", i)
            break
        print("Loop i:", i)
    # pass example inside conditional
    if n > 0:
        pass  # placeholder for future logic

# ---------------------------
# Module 3: Searching algorithms
# ---------------------------

def linear_search(arr, target):
    """Linear search (O(n)). Returns index or -1."""
    for idx, val in enumerate(arr):
        if val == target:
            return idx
    return -1

def binary_search_recursive(arr, target, left=0, right=None):
    """Recursive binary search on sorted arr. Returns index or -1."""
    if right is None:
        right = len(arr) - 1
    if left > right:
        return -1
    mid = (left + right) // 2
    if arr[mid] == target:
        return mid
    elif arr[mid] > target:
        return binary_search_recursive(arr, target, left, mid - 1)
    else:
        return binary_search_recursive(arr, target, mid + 1, right)

# ---------------------------
# Module 3: Sorting algorithms
# (these operate on Python lists)
# ---------------------------

def bubble_sort(arr):
    """Bubble sort - O(n^2). Returns new sorted list."""
    a = arr[:]
    n = len(a)
    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            if a[j] > a[j + 1]:
                a[j], a[j + 1] = a[j + 1], a[j]
                swapped = True
        if not swapped:
            break
    return a

def selection_sort(arr):
    a = arr[:]
    n = len(a)
    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            if a[j] < a[min_idx]:
                min_idx = j
        a[i], a[min_idx] = a[min_idx], a[i]
    return a

def insertion_sort(arr):
    a = arr[:]
    for i in range(1, len(a)):
        key = a[i]
        j = i - 1
        while j >= 0 and a[j] > key:
            a[j + 1] = a[j]
            j -= 1
        a[j + 1] = key
    return a

def quick_sort(arr):
    """Quick sort using recursion (returns new sorted list)."""
    if len(arr) <= 1:
        return arr[:]
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)

def merge_sort(arr):
    if len(arr) <= 1:
        return arr[:]
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

def merge(left, right):
    res = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            res.append(left[i]); i += 1
        else:
            res.append(right[j]); j += 1
    res.extend(left[i:])
    res.extend(right[j:])
    return res

# ---------------------------
# Module 4: Stacks, Queues (data structures)
# ---------------------------

class SimpleStack:
    """Stack using Python list (LIFO)"""
    def __init__(self):
        self._data = []
    def push(self, x):
        self._data.append(x)
    def pop(self):
        if self._data:
            return self._data.pop()
        return None
    def peek(self):
        return self._data[-1] if self._data else None
    def is_empty(self):
        return len(self._data) == 0
    def __repr__(self):
        return repr(self._data)

class SimpleQueue:
    """Queue using collections.deque (FIFO)"""
    def __init__(self):
        self._q = deque()
    def enqueue(self, x):
        self._q.append(x)
    def dequeue(self):
        if self._q:
            return self._q.popleft()
        return None
    def is_empty(self):
        return len(self._q) == 0
    def __repr__(self):
        return repr(list(self._q))

# ---------------------------
# Module 4: Functions (positional, keyword, default, *args, **kwargs, lambda)
# ---------------------------

def greet(name, msg="Welcome"):
    """Positional and default argument example."""
    return f"{msg}, {name}!"

def summarize(*numbers, method="mean"):
    """
    Arbitrary args example. numbers is a tuple.
    method is a keyword-arg with default value.
    """
    arr = np.array(numbers, dtype=float)
    if arr.size == 0:
        return None
    if method == "mean":
        return float(np.mean(arr))
    elif method == "median":
        return float(np.median(arr))
    elif method == "std":
        return float(np.std(arr))
    else:
        raise ValueError("Unknown method")

# lambda example:
square = lambda x: x * x

# ---------------------------
# Module 4: Recursion example
# ---------------------------

def factorial(n):
    """Recursive factorial (demonstrates recursion)."""
    if n <= 1:
        return 1
    return n * factorial(n - 1)

# ---------------------------
# Module 5: Numpy & Pandas examples & CRUD operations
# ---------------------------

def add_record_interactive():
    """Admin: Add a vaccination record (demonstrates input, type conversion, pandas)."""
    df = load_df()
    print("\n--- Add Vaccination Record ---")
    name = input("Name: ").strip()
    # Input validation for age (int)
    while True:
        try:
            age = int(input("Age: "))
            if age < 0:
                print("Age must be >= 0")
                continue
            break
        except ValueError:
            print("Enter valid integer for age.")
    gender = input("Gender (M/F/O): ").strip()
    vaccine = input("Vaccine name: ").strip()
    while True:
        try:
            dose = int(input("Dose number (1 or 2): "))
            break
        except ValueError:
            print("Enter valid integer for dose.")
    city = input("City: ").strip()
    new_id = generate_id(df)
    new_row = {"ID": new_id, "Name": name, "Age": age, "Gender": gender, "Vaccine": vaccine, "Dose": dose, "City": city}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    save_df(df)
    print(f"Record added with ID: {new_id}")

def view_all_records():
    """Client/Admin: Display all records using pandas (demonstrates DataFrame printing)."""
    df = load_df()
    print("\n--- All Vaccination Records ---")
    if df.empty:
        print("No records found.")
        return
    print(df.to_string(index=False))

def update_record_interactive():
    """Admin: Update specific fields of a record."""
    df = load_df()
    if df.empty:
        print("No records available.")
        return
    view_all_records()
    try:
        record_id = int(input("Enter ID to update: "))
    except ValueError:
        print("Invalid ID")
        return
    if record_id not in df["ID"].values:
        print("ID not found.")
        return
    print("Choose field to update: 1.Name 2.Age 3.Gender 4.Vaccine 5.Dose 6.City")
    choice = input("Choice: ").strip()
    if choice == "1":
        df.loc[df["ID"] == record_id, "Name"] = input("New Name: ")
    elif choice == "2":
        df.loc[df["ID"] == record_id, "Age"] = int(input("New Age: "))
    elif choice == "3":
        df.loc[df["ID"] == record_id, "Gender"] = input("New Gender: ")
    elif choice == "4":
        df.loc[df["ID"] == record_id, "Vaccine"] = input("New Vaccine: ")
    elif choice == "5":
        df.loc[df["ID"] == record_id, "Dose"] = int(input("New Dose: "))
    elif choice == "6":
        df.loc[df["ID"] == record_id, "City"] = input("New City: ")
    else:
        print("Invalid choice.")
        return
    save_df(df)
    print("Record updated.")

def delete_record_interactive():
    df = load_df()
    if df.empty:
        print("No records available.")
        return
    view_all_records()
    try:
        record_id = int(input("Enter ID to delete: "))
    except ValueError:
        print("Invalid ID")
        return
    if record_id not in df["ID"].values:
        print("ID not found.")
        return
    df = df[df["ID"] != record_id]
    save_df(df)
    print("Record deleted.")

# ---------------------------
# Module 3/5: Searching & Sorting with pandas
# ---------------------------

def search_records_pandas():
    """Search by name or city or vaccine using pandas string matching."""
    df = load_df()
    if df.empty:
        print("No data.")
        return
    q = input("Enter a search keyword (name/city/vaccine): ").strip().lower()
    mask = (df["Name"].str.lower().str.contains(q, na=False)) | \
           (df["City"].str.lower().str.contains(q, na=False)) | \
           (df["Vaccine"].str.lower().str.contains(q, na=False))
    result = df[mask]
    if result.empty:
        print("No matching records.")
    else:
        print(result.to_string(index=False))

def sort_records_pandas():
    """Sort records by a chosen column using pandas (demonstrates sorting & ordering)."""
    df = load_df()
    if df.empty:
        print("No data.")
        return
    print("Sort by: 1.Name 2.Age 3.City 4.Vaccine 5.Dose")
    choice = input("Choice: ").strip()
    if choice == "1":
        s = df.sort_values(by="Name")
    elif choice == "2":
        s = df.sort_values(by="Age")
    elif choice == "3":
        s = df.sort_values(by="City")
    elif choice == "4":
        s = df.sort_values(by="Vaccine")
    elif choice == "5":
        s = df.sort_values(by="Dose")
    else:
        print("Invalid choice.")
        return
    print(s.to_string(index=False))

# ---------------------------
# Module 5: Aggregation / Grouping / Stats (numpy + pandas)
# ---------------------------

def vaccine_statistics():
    """Show counts, averages using pandas & numpy."""
    df = load_df()
    if df.empty:
        print("No data.")
        return
    print("\n--- Vaccine Counts ---")
    print(df["Vaccine"].value_counts().to_string())
    ages = df["Age"].dropna().astype(float).values
    if ages.size > 0:
        ages_np = np.array(ages)
        print(f"\nAverage age (numpy): {np.mean(ages_np):.2f}")
        print(f"Median age (numpy): {np.median(ages_np):.2f}")
        print(f"Std dev (numpy): {np.std(ages_np):.2f}")
    else:
        print("No age data.")

def group_by_city():
    """Group records by city and show counts (pandas groupby)."""
    df = load_df()
    if df.empty:
        print("No data.")
        return
    grouped = df.groupby("City").agg({"ID": "count", "Age": "mean"})
    grouped = grouped.rename(columns={"ID": "Count", "Age": "AverageAge"})
    print("\nRecords grouped by City:")
    print(grouped.to_string())

def merge_example():
    """Demonstrate merging/joining datasets (pandas.merge)."""
    df = load_df()
    df_v = pd.read_csv(VACCINE_INFO_FILE)
    merged = pd.merge(df, df_v, on="Vaccine", how="left")
    print("\n--- Merged with Vaccine Info ---")
    print(merged.to_string(index=False))

# ---------------------------
# Module 3: Demonstrate searching & sorting algorithms on sample list
# ---------------------------

def demo_search_and_sort_algorithms():
    print("\n--- Demo Searching & Sorting Algorithms ---")
    sample = [12, 4, 56, 7, 4, 2, 99, 45]
    print("Sample list:", sample)
    print("Bubble sort:", bubble_sort(sample))
    print("Selection sort:", selection_sort(sample))
    print("Insertion sort:", insertion_sort(sample))
    print("Quick sort:", quick_sort(sample))
    print("Merge sort:", merge_sort(sample))
    # linear search for 7
    print("Linear search for 7:", linear_search(sample, 7))
    # For binary search, use sorted list
    sorted_sample = merge_sort(sample)
    print("Sorted sample for binary search:", sorted_sample)
    print("Binary search for 7 (recursive):", binary_search_recursive(sorted_sample, 7))

# ---------------------------
# Authentication (simple)
# ---------------------------

ADMIN_PASSWORD = "admin123"

# For demonstration we store client users in a small CSV (username,password) - plain text for learning only
USERS_FILE = "portal_users.csv"

def ensure_users_file():
    if not os.path.exists(USERS_FILE):
        with open(USERS_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["username", "password"])
            # add a sample client
            writer.writerow(["client", "client123"])

def admin_login():
    pwd = getpass("Enter admin password: ")
    return pwd == ADMIN_PASSWORD

def register_client():
    ensure_users_file()
    uname = input("Choose username: ")
    pwd = getpass("Choose password: ")
    # append to CSV (no encryption - for learning only)
    with open(USERS_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([uname, pwd])
    print("Client registered. Use the client menu to login.")

def client_login():
    ensure_users_file()
    uname = input("Username: ")
    pwd = getpass("Password: ")
    with open(USERS_FILE, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["username"] == uname and row["password"] == pwd:
                print("Login successful.")
                return True
    print("Invalid credentials.")
    return False

# ---------------------------
# Menu-driven CLI (Module 4: modular programming & top-down design)
# ---------------------------

def admin_menu():
    while True:
        print("\n=== ADMIN MENU ===")
        print("1. Add record")
        print("2. View all records")
        print("3. Update record")
        print("4. Delete record")
        print("5. Vaccine statistics (numpy/pandas)")
        print("6. Group by city")
        print("7. Merge with vaccine info")
        print("8. Demo collections/algorithms (learning)")
        print("9. Logout")
        choice = input("Choice: ").strip()
        if choice == "1":
            add_record_interactive()
        elif choice == "2":
            view_all_records()
        elif choice == "3":
            update_record_interactive()
        elif choice == "4":
            delete_record_interactive()
        elif choice == "5":
            vaccine_statistics()
        elif choice == "6":
            group_by_city()
        elif choice == "7":
            merge_example()
        elif choice == "8":
            demo_collections()
            control_flow_examples()
            demo_search_and_sort_algorithms()
        elif choice == "9":
            print("Logging out admin.")
            break
        else:
            print("Invalid choice.")

def client_menu():
    while True:
        print("\n=== CLIENT MENU ===")
        print("1. View all records")
        print("2. Search records")
        print("3. Sort records")
        print("4. Register as client")
        print("5. Login as client (to view extra features)")
        print("6. Back to main")
        choice = input("Choice: ").strip()
        if choice == "1":
            view_all_records()
        elif choice == "2":
            search_records_pandas()
        elif choice == "3":
            sort_records_pandas()
        elif choice == "4":
            register_client()
        elif choice == "5":
            if client_login():
                # after login, show some extra stats using numpy/pandas
                print("Client extra features:")
                vaccine_statistics()
                group_by_city()
        elif choice == "6":
            break
        else:
            print("Invalid choice.")

# ---------------------------
# Extra teaching utilities
# ---------------------------

def demo_numpy_examples():
    print("\n--- Numpy Array Examples ---")
    a = np.array([1, 2, 3, 4, 5])
    print("Array:", a, "Type:", type(a))
    print("Mean:", np.mean(a))
    print("Sum:", np.sum(a))
    print("Std dev:", np.std(a))

def demo_pandas_examples():
    print("\n--- Pandas Examples ---")
    df = load_df()
    print("Head (first 5 rows):")
    print(df.head().to_string(index=False))
    print("\nDescribe numeric columns:")
    print(df.describe().to_string())

# ---------------------------
# Main program (Module 1 & 4: top-level menu)
# ---------------------------

def main_menu():
    ensure_files_exist()
    ensure_users_file()
    print("Welcome to Healthcare / Vaccination Portal (CLI)")
    while True:
        print("\n=== MAIN MENU ===")
        print("1. Admin Login")
        print("2. Client Portal")
        print("3. Demo: Numpy & Pandas examples")
        print("4. Exit")
        choice = input("Choice: ").strip()
        if choice == "1":
            if admin_login():
                admin_menu()
            else:
                print("Incorrect admin password.")
        elif choice == "2":
            client_menu()
        elif choice == "3":
            demo_numpy_examples()
            demo_pandas_examples()
        elif choice == "4":
            print("Exiting. Goodbye!")
            break
        else:
            print("Invalid choice.")

# ---------------------------
# If the script is run directly
# ---------------------------
if __name__ == "__main__":
    # Some small initial data to help demonstration (only if DB is empty)
    ensure_files_exist()
    df = load_df()
    if df.empty:
        sample = [
            {"ID": 1, "Name": "Anita", "Age": 30, "Gender": "F", "Vaccine": "Covishield", "Dose": 1, "City": "Tirupati"},
            {"ID": 2, "Name": "Ravi", "Age": 45, "Gender": "M", "Vaccine": "Covaxin", "Dose": 2, "City": "Vellore"},
            {"ID": 3, "Name": "Sara", "Age": 28, "Gender": "F", "Vaccine": "Pfizer", "Dose": 1, "City": " Chennai"},
        ]
        df = pd.DataFrame(sample, columns=COLUMNS)
        save_df(df)
    main_menu()
