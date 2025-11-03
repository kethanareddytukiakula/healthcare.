import mysql.connector
import pandas as pd
import numpy as np

# ---------- Module 4: Modular Programming ----------
def connect_db():
    return mysql.connector.connect(
    host="localhost",
    user="root",
    password="@Sweety2007",
    database="healthcare_portal"

    )

def create_table():
    db = connect_db()
    cursor = db.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS vaccination (
            id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(100),
            age INT,
            gender VARCHAR(10),
            vaccine VARCHAR(50),
            dose_date DATE
        )
    """)
    db.close()

def insert_record(name, age, gender, vaccine, dose_date):
    db = connect_db()
    cursor = db.cursor()
    sql = "INSERT INTO vaccination (name, age, gender, vaccine, dose_date) VALUES (%s, %s, %s, %s, %s)"
    vals = (name, age, gender, vaccine, dose_date)
    cursor.execute(sql, vals)
    db.commit()
    db.close()

def fetch_all_records():
    db = connect_db()
    cursor = db.cursor()
    cursor.execute("SELECT * FROM vaccination")
    rows = cursor.fetchall()
    db.close()
    return rows

# ---------- Module 3: Data Organization and Manipulation ----------
def search_by_name(records, search_name):
    return list(filter(lambda x: x[1].lower() == search_name.lower(), records))

def selection_sort(records):
    sorted_records = records[:]
    for i in range(len(sorted_records)):
        min_idx = i
        for j in range(i+1, len(sorted_records)):
            if sorted_records[j][2] < sorted_records[min_idx][2]:  # Sort by age
                min_idx = j
        sorted_records[i], sorted_records[min_idx] = sorted_records[min_idx], sorted_records[i]
    return sorted_records

# ---------- Module 5: Data Processing using Numpy and Pandas ----------
def pandas_dataframe(records):
    df = pd.DataFrame(records, columns=['ID', 'Name', 'Age', 'Gender', 'Vaccine', 'Dose Date'])
    return df

def numpy_stats(records):
    ages = np.array([rec[2] for rec in records])
    return {
        'mean_age': np.mean(ages),
        'std_age': np.std(ages),
        'max_age': np.max(ages),
        'min_age': np.min(ages)
    }

# ---------- CLI Menu ----------
def main_menu():
    create_table()
    while True:
        print("\nHealthcare & Vaccination Portal")
        print("1. Add new vaccination record")
        print("2. View all records")
        print("3. Search by name")
        print("4. Sort records by age (Selection Sort)")
        print("5. View age statistics (Numpy)")
        print("6. Exit")
        choice = input("Enter choice: ")
        
        if choice == '1':
            name = input("Enter name: ")
            age = int(input("Enter age: "))
            gender = input("Enter gender: ")
            vaccine = input("Enter vaccine name: ")
            dose_date = input("Enter dose date (YYYY-MM-DD): ")
            insert_record(name, age, gender, vaccine, dose_date)
        elif choice == '2':
            records = fetch_all_records()
            df = pandas_dataframe(records)
            print(df)
        elif choice == '3':
            search_name = input("Enter name to search: ")
            records = fetch_all_records()
            result = search_by_name(records, search_name)
            print(pandas_dataframe(result))
        elif choice == '4':
            records = fetch_all_records()
            sorted_records = selection_sort(records)
            print(pandas_dataframe(sorted_records))
        elif choice == '5':
            records = fetch_all_records()
            stats = numpy_stats(records)
            print("Age statistics:")
            for key, value in stats.items():
                print(f"{key}: {value}")
        elif choice == '6':
            break

if __name__ == "__main__":
    main_menu()
