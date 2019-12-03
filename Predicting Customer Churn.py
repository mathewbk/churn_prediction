# Databricks notebook source
# MAGIC %md
# MAGIC ### Integrating with Source Code Management system (Version Control)
# MAGIC    1. Notebooks
# MAGIC       - Databricks notebooks integrate with Azure DevOps, GitHub and BitBucket
# MAGIC       - Push notebooks to any Source Code Management system
# MAGIC    2. Code via local development
# MAGIC       - Push code to any Source Code Management system

# COMMAND ----------

# MAGIC %md
# MAGIC ### Reference Architecture for integration with your Source Code Management system
# MAGIC <img src = "https://bmathew.blob.core.windows.net/bmathew-images/61.png" height="700" width="1000">

# COMMAND ----------

# MAGIC %md
# MAGIC ### Use Case: Predicting Customer Churn
# MAGIC - Update existing churn prediction code
# MAGIC - Train a new Random Forest classifier to preduct customer churn and update existing code
# MAGIC - Code will consist of Databricks Notebook and a new Python script
# MAGIC - After training model we will check-in Notebook, Python Script and the MLflow project that gets created 
# MAGIC - Existing code is in Source Code Management system and this code will be updated
# MAGIC    - https://github.com/mathewbk/churn_prediction

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create a branch/project on your machine using Source Code Management system
# MAGIC - I will be using GitHub for my example
# MAGIC - Run <br>
# MAGIC cd /Users/bmathew/Desktop/MY_FILES/Databricks/source_code_management <br>
# MAGIC rm -rf churn_prediction <br>
# MAGIC git clone https://github.com/mathewbk/churn_prediction.git <br>
# MAGIC cd churn_prediction <br>
# MAGIC git pull <br>
# MAGIC git checkout -b rf_model <br>
# MAGIC clear <br> 
# MAGIC ls

# COMMAND ----------

# MAGIC %md
# MAGIC ### Let's view the source data: Customer mobile phone usage data
# MAGIC - Comma delimited text file

# COMMAND ----------

# MAGIC %fs
# MAGIC head --maxBytes=1490 /bmathew/data/churn_data/customer_churn.csv

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create a schema for the data

# COMMAND ----------

from pyspark.sql.types import *
churnSchema =StructType([
    StructField("area_code",IntegerType(), False),
  	StructField("phone_number",StringType(), False),
    StructField("state",StringType(), False),
    StructField("account_length",DoubleType(), False),
	StructField("international_plan",StringType(), False),
	StructField("voice_mail_plan",StringType(), False),
	StructField("number_vmail_messages",DoubleType(), False),
	StructField("total_day_minutes",DoubleType(), False),
	StructField("total_day_calls",DoubleType(), False),
	StructField("total_day_charge",DoubleType(), False),
	StructField("total_eve_minutes",DoubleType(), False),
	StructField("total_eve_calls",DoubleType(), False),
	StructField("total_eve_charge",DoubleType(), False),
	StructField("total_night_minutes",DoubleType(), False),
	StructField("total_night_calls",DoubleType(), False),
	StructField("total_night_charge",DoubleType(), False),
	StructField("total_intl_minutes",DoubleType(), False),
	StructField("total_intl_calls",DoubleType(), False),
	StructField("total_intl_charge",DoubleType(), False),
    StructField("number_customer_service_calls",DoubleType(), False), 
    StructField("churned",IntegerType(), False)
])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load data into Spark DataFrame
# MAGIC - The field 'churned' is our label column
# MAGIC    - 1 means that the customer churned
# MAGIC    - 0 means that the customer did not churn

# COMMAND ----------

df = (sqlContext.read.option("delimiter", ",").schema(churnSchema).option("header","true").csv("/bmathew/data/churn_data/customer_churn.csv"))
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Databricks Delta: Performance, Reliability, and Consistency for your Data
# MAGIC <img src = "https://bmathew.blob.core.windows.net/bmathew-images/50.png" height="700" width="1000">

# COMMAND ----------

# MAGIC %md
# MAGIC ### Churn count by customer spend
# MAGIC - Query the data stored in non-Delta format
# MAGIC - Notice the time taken to run the query

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT CASE WHEN total_charges >= 0 AND total_charges <= 9.99 THEN '0 - 9.99' WHEN total_charges >= 10 AND total_charges <= 19.99 THEN '10 - 19.99' WHEN total_charges >= 20 AND total_charges <= 29.99 THEN '20 - 29.99' WHEN total_charges >= 30 AND total_charges <= 39.99 THEN '30 - 39.99' WHEN total_charges >= 40 AND total_charges <= 49.99 THEN '40 - 49.99'  WHEN total_charges >= 50 AND total_charges <= 59.99 THEN '50 - 59.99' WHEN total_charges >= 60 AND total_charges <= 69.99 THEN '60 - 69.99' WHEN total_charges >= 70 AND total_charges <= 79.99 THEN '70 - 79.99' WHEN total_charges >= 80 AND total_charges <= 89.99 THEN '80 - 89.99' WHEN total_charges >= 90 AND total_charges <= 99.99 THEN '90 - 99.99' 
# MAGIC  WHEN total_charges >= 100 then '100+' ELSE '0' END AS `$ Spend Amount`, COUNT(*) AS `Churn Count`
# MAGIC FROM (SELECT area_code, phone_number, sum(total_day_charge + total_night_charge +  total_eve_charge) total_charges FROM bmathew.churn_data_non_delta WHERE churned = 1 GROUP BY area_code, phone_number) m GROUP BY `$ Spend Amount` ORDER BY `$ Spend Amount` ASC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Querying the Delta table is 16x faster
# MAGIC - Data Skipping Indexes
# MAGIC - Caching
# MAGIC - File Layout Optimization (File Compaction)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT CASE WHEN total_charges >= 0 AND total_charges <= 9.99 THEN '0 - 9.99' WHEN total_charges >= 10 AND total_charges <= 19.99 THEN '10 - 19.99' WHEN total_charges >= 20 AND total_charges <= 29.99 THEN '20 - 29.99' WHEN total_charges >= 30 AND total_charges <= 39.99 THEN '30 - 39.99' WHEN total_charges >= 40 AND total_charges <= 49.99 THEN '40 - 49.99'  WHEN total_charges >= 50 AND total_charges <= 59.99 THEN '50 - 59.99' WHEN total_charges >= 60 AND total_charges <= 69.99 THEN '60 - 69.99' WHEN total_charges >= 70 AND total_charges <= 79.99 THEN '70 - 79.99' WHEN total_charges >= 80 AND total_charges <= 89.99 THEN '80 - 89.99' WHEN total_charges >= 90 AND total_charges <= 99.99 THEN '90 - 99.99' 
# MAGIC  WHEN total_charges >= 100 then '100+' ELSE '0' END AS `$ Spend Amount`, COUNT(*) AS `Churn Count`
# MAGIC FROM (SELECT area_code, phone_number, sum(total_day_charge + total_night_charge +  total_eve_charge) total_charges FROM bmathew.churn_data_delta WHERE churned = 1 GROUP BY area_code, phone_number) m GROUP BY `$ Spend Amount` ORDER BY `$ Spend Amount` ASC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Easily Update, Merge, and Delete data
# MAGIC - Delta is ANSI SQL 2003 Compliant

# COMMAND ----------

# MAGIC %md
# MAGIC ### Merge new incoming data with existing data
# MAGIC - Delta knows which files need to be replaced and will create new files for them

# COMMAND ----------

# MAGIC %sql
# MAGIC MERGE INTO bmathew.churn_data_delta target
# MAGIC USING bmathew.churn_data_updates as source
# MAGIC ON source.area_code = target.area_code AND source.phone_number = target.phone_number
# MAGIC WHEN MATCHED THEN UPDATE
# MAGIC SET target.total_day_minutes = source.total_day_minutes, target.total_eve_minutes = source.total_eve_minutes, target.total_intl_calls = source.total_intl_calls, target.phone_number = source.phone_number

# COMMAND ----------

# MAGIC %md
# MAGIC ### What if someone accidentally deletes data from the table?

# COMMAND ----------

# MAGIC %sql
# MAGIC DELETE FROM bmathew.churn_data_delta;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM bmathew.churn_data_delta;

# COMMAND ----------

# MAGIC %md
# MAGIC ### Delta provides audit trail of changes made

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE HISTORY bmathew.churn_data_delta

# COMMAND ----------

# MAGIC %md
# MAGIC ### Delta allows to view and rollback to a previous version of the data

# COMMAND ----------

# MAGIC %fs
# MAGIC head dbfs:/user/hive/warehouse/bmathew.db/churn_data_delta/_delta_log/00000000000000000003.json	

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM bmathew.churn_data_delta VERSION AS OF 1;

# COMMAND ----------

# MAGIC %sql
# MAGIC create table bmathew.ais_test_1 using delta as select * from bmathew.churn_data_delta

# COMMAND ----------

# MAGIC %sql
# MAGIC describe formatted bmathew.ais_test_1 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Train our model

# COMMAND ----------

# MAGIC %md
# MAGIC ###Split data into training and test datasets

# COMMAND ----------

df = sqlContext.sql("SELECT * FROM bmathew.churn_data_delta VERSION AS OF 1")
(trainingData, testData) = df.randomSplit([0.7, 0.3], seed=42)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Train Model
# MAGIC - Create a funtion to train Random Forest Model:
# MAGIC - Convert features into feature vector
# MAGIC - Train using training dataset
# MAGIC - Track model training to MLflow tracking server: Hyperparameters, Metrics, and Model
# MAGIC - Evaluate model using test dataset

# COMMAND ----------

import mlflow
from mlflow import spark
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.feature import StringIndexer, VectorAssembler 
from pyspark.ml.evaluation import BinaryClassificationEvaluator

def train_model(maxDepth, numTrees, featureCols):
  featureCols=featureCols
  maxDepth=maxDepth
  numTrees=numTrees

  with mlflow.start_run(experiment_id=787776453898653):
    rf = RandomForestClassifier(labelCol="churned", featuresCol="features", maxDepth=maxDepth, numTrees=numTrees)
    stages = [VectorAssembler(inputCols=featureCols, outputCol="features"), StringIndexer(inputCol="churned", outputCol="label")]
    pipeline = Pipeline(stages=stages+[rf])
    rfModel = pipeline.fit(trainingData)
    predictions = rfModel.transform(testData)
    evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
    evaluator.evaluate(predictions)
    auc = evaluator.evaluate(predictions)
    mlflow.log_param("maxDepth", maxDepth)
    mlflow.log_param("numTrees", numTrees)
    mlflow.log_param("features", featureCols)
    mlflow.log_metric("auc", auc)
    model_path = "Random_Forest_Churn_Prediction"
    mlflow.spark.log_model(spark_model=rfModel, artifact_path=model_path) 
   
  print('The model has an AUC value of {0}'.format(auc))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Call training function  with a set of parameters
# MAGIC - Input parameters: maxDepth, numTrees, and fields to use for features
# MAGIC - Output parameter: AUC, Area Under the Curve
# MAGIC - Training run will be logged to MLflow tracking server
# MAGIC    - Log model hyperparameters
# MAGIC    - Log model output metrics
# MAGIC    - Save model itself

# COMMAND ----------

maxDepth=10
numTrees=10
featureCols = ["account_length", "total_day_minutes", "total_day_calls",  "total_eve_calls", "total_eve_minutes","total_night_calls", "total_intl_calls", "total_intl_minutes", "number_customer_service_calls", "total_day_charge", "total_eve_charge","total_night_charge","total_intl_charge"]
train_model(maxDepth=maxDepth, numTrees=numTrees, featureCols=featureCols)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Example: Save code and artifacts to our Source Code Management system
# MAGIC - Easily integrate with any SCM system using Databricks Command Line Interface (CLI)
# MAGIC - Check-in local development and merge with master

# COMMAND ----------

# MAGIC %md
# MAGIC <img src = "https://bmathew.blob.core.windows.net/bmathew-images/70.png" height="600" width="800">

# COMMAND ----------

# MAGIC %md
# MAGIC ### Add Notebook, Code, and Model to local branch/project that we created
# MAGIC - Run: <br>
# MAGIC databricks --profile AZURE_FIELD_ENG workspace export --overwrite /Users/binu.mathew@databricks.com/AIS/"Predicting Customer Churn" . <br>
# MAGIC databricks --profile  AZURE_FIELD_ENG fs cp -r dbfs:/databricks/mlflow/787776453898653/91255b216468498788c3487b488c00a7/artifacts/Random_Forest_Churn_Prediction Random_Forest_Churn_Prediction/ <br>
# MAGIC cp /Users/bmathew/Desktop/MY_FILES/Databricks/source_code_management/prep/data_prep.py . <br>
# MAGIC clear <br>
# MAGIC ls

# COMMAND ----------

# MAGIC %md
# MAGIC ### Check-in changes and merge with Master
# MAGIC - Run: <br>
# MAGIC git add . <br>
# MAGIC git commit -am "updated code" <br>
# MAGIC git remote  set-url origin git@github.com:mathewbk/churn_prediction.git <br>
# MAGIC git push -u origin rf_model <br>
# MAGIC git checkout master <br>
# MAGIC git pull <br>
# MAGIC git merge rf_model <br>
# MAGIC git push -u origin master <br>
# MAGIC clear <br>
# MAGIC ls