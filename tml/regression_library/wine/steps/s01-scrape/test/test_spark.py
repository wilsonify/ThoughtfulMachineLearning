from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from collections import defaultdict
import csv


def transform_input_to_output(reader):
    all_tags = set([])
    all_times = set([])
    data = defaultdict(dict)
    for row in reader:
        time = row['time']
        tag = row['tag']
        value = row['value']
        all_tags.add(tag)
        all_times.add(time)
        if tag not in data[time]:
            data[time][tag] = [value]
        else:
            data[time][tag].append(value)

    result = {'time': sorted(all_times)}
    for tag in all_tags:
        result[tag] = []
        for time in result['time']:
            if time in data and tag in data[time]:
                result[tag].append(float(data[time][tag][0]))  # Take the first value
                data[time][tag] = data[time][tag][1:]  # Remove the taken value
            else:
                result[tag].append(None)  # Fill missing values with NaN

    return result


# Initialize Spark session
spark = SparkSession.builder.appName("Transform Input to Output").getOrCreate()

# Read the input CSV file into a DataFrame
df = spark.read.csv("stacked.csv", header=True)

# Convert DataFrame to Python dictionary
records = df.collect()
dict_reader = [{k: v for k, v in row.items()} for row in records]

# Apply transformation function
result = transform_input_to_output(dict_reader)

# Convert the result dictionary back to a Spark DataFrame
result_df = spark.createDataFrame([result])

# Show the result DataFrame
result_df.show(truncate=False)

# Stop the Spark session
spark.stop()
