import json
import csv

json_file_path = "./data/News_Category_Dataset_v3.json"
csv_filename = "./data/csv_rawdata.csv"

json_file = open(json_file_path)
for line in json_file:
    #print(line)
    data = json.loads(line)
fieldnames = ["link", "headline", "category", "short_description", "authors", "date"]

with open(json_file_path, "r", encoding="utf-8") as json_file:
    with open(csv_filename, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for line in json_file:
            data = json.loads(line)
            writer.writerow(data)

print("CSV file has been created successfully.")
