import sys
import typing
# Mock typing.io and typing.re for Python 3.14 compatibility with PySpark 3.3
typing.io = sys.modules["typing.io"] = typing
typing.re = sys.modules["typing.re"] = typing

import os
# Force PySpark to use Java 8 to avoid modularity/reflection compatibility issues on Windows
# Using C:\PROGRA~2 short path to avoid parentheses in 'Program Files (x86)' breaking Spark CMD batch scripts
os.environ["JAVA_HOME"] = r"C:\PROGRA~2\Java\jre1.8.0_481"

# Set up HADOOP_HOME for local Spark execution on Windows
hadoop_dir = os.path.abspath("hadoop")
os.environ["HADOOP_HOME"] = hadoop_dir
os.environ["PATH"] = r"C:\PROGRA~2\Java\jre1.8.0_481\bin;" + os.path.join(hadoop_dir, "bin") + ";" + os.environ["PATH"]

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
import mleap.pyspark
from mleap.pyspark.spark_support import SimpleSparkSerializer

def train_and_export():
    print("Initializing Spark Session...")
    spark = SparkSession.builder \
        .appName("FraudDetectionMLeap") \
        .config("spark.jars.packages", "ml.combust.mleap:mleap-spark_2.12:0.20.0,commons-io:commons-io:2.5") \
        .getOrCreate()
        
    print("Loading data...")
    df = spark.read.csv("transactions.csv", header=True, inferSchema=True)
    df = df.drop("transaction_id")
    
    print("Building ML Pipeline...")
    cat_cols = ["merchant_category", "device_type"]
    indexers = [StringIndexer(inputCol=c, outputCol=c+"_idx", handleInvalid="keep") for c in cat_cols]
    
    assembler_inputs = ["amount", "time_of_day"] + [c+"_idx" for c in cat_cols]
    assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")
    
    rf = RandomForestClassifier(featuresCol="features", labelCol="is_fraud", numTrees=20)
    
    pipeline = Pipeline(stages=indexers + [assembler, rf])
    
    print("Training Model...")
    model = pipeline.fit(df)
    
    predictions = model.transform(df)
    accuracy = predictions.filter(predictions.is_fraud == predictions.prediction).count() / float(df.count())
    print(f"Model Training Complete. Training Accuracy: {accuracy:.4f}")
    
    print("Exporting to MLeap bundle...")
    bundle_name = "model.zip"
    # Replace backslashes with forward slashes and ensure a leading slash to form a valid hierarchical jar:file URI for Java
    bundle_path = f"jar:file:/{os.path.abspath(bundle_name).replace(os.sep, '/')}"
    
    if os.path.exists(bundle_name):
        os.remove(bundle_name)
        
    model.serializeToBundle(bundle_path, predictions)
    print(f"Successfully exported MLeap bundle to {bundle_name}")

if __name__ == '__main__':
    train_and_export()
