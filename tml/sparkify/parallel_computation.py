from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession

# Initialize Spark
conf = SparkConf().setAppName("ComputationWithPySpark")
sc = SparkContext(conf=conf)
spark = SparkSession(sc)

# Create an RDD from the range of numbers
ps = sc.parallelize(range(10000), numSlices=100)


# Define the computation function
def process_csv(p):
    df = spark.read.csv(f"{p}.csv", header=True, inferSchema=True)
    result = expensive_computation(df)
    result.write.csv(f"{p}_result.csv", header=True)


# Apply the computation function to each element in the RDD
ps.foreach(process_csv)

# Stop Spark
spark.stop()
