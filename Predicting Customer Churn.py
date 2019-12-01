# Databricks notebook source
# MAGIC %md
# MAGIC ### Use Case: Predicting Customer Churn

# COMMAND ----------

# MAGIC %md
# MAGIC ### Demo Overview:
# MAGIC * Data Ingestion
# MAGIC * Data Preparation
# MAGIC * Data Exploration
# MAGIC * Feature Engineering
# MAGIC * MLflow
# MAGIC    - Model Training
# MAGIC    - Model Metrics
# MAGIC    - Model Deployment

# COMMAND ----------

# MAGIC %md
# MAGIC ### Raw customer mobile phone usage data
# MAGIC - Comma delimited text file

# COMMAND ----------

# MAGIC %fs
# MAGIC head --maxBytes=1490 /mnt/bmathew/customer_analysis_data/customer_churn.csv

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

df = (sqlContext.read.option("delimiter", ",").schema(churnSchema).option("header","true").csv("/mnt/bmathew/customer_analysis_data/customer_churn.csv"))
display(df)

# COMMAND ----------

df = (sqlContext.read.option("delimiter", ",").option("inferSchema","true").option("header","true").csv("/mnt/bmathew/customer_analysis_data/customer_churn.csv"))
df.createOrReplaceTempView("churn_analysis")
df.repartition(3000).write.mode("overwrite").parquet("/mnt/bmathew/customer_analysis_data/non_delta_parquet")

# COMMAND ----------

df = (sqlContext.read.option("delimiter", ",").schema(churnSchema).option("header","true").csv("/mnt/bmathew/customer_analysis_data/customer_churn.csv"))
df.createOrReplaceTempView("churn_analysis")
df.write.format("delta").mode("append").save("/mnt/bmathew/customer_analysis_data/delta")
sqlContext.sql("OPTIMIZE bmathew.churn_data_delta")

# COMMAND ----------

# MAGIC %sql
# MAGIC drop table if exists bmathew.churn_data_non_delta;
# MAGIC 
# MAGIC create table bmathew.churn_data_non_delta (
# MAGIC area_code int
# MAGIC ,phone_number string
# MAGIC ,state string
# MAGIC ,account_length double
# MAGIC ,international_plan string
# MAGIC ,voice_mail_plan string
# MAGIC ,number_vmail_messages double
# MAGIC ,total_day_minutes double
# MAGIC ,total_day_calls double
# MAGIC ,total_day_charge double
# MAGIC ,total_eve_minutes double
# MAGIC ,total_eve_calls double
# MAGIC ,total_eve_charge double
# MAGIC ,total_night_minutes double
# MAGIC ,total_night_calls double
# MAGIC ,total_night_charge double
# MAGIC ,total_intl_minutes double
# MAGIC ,total_intl_calls double
# MAGIC ,total_intl_charge double
# MAGIC ,number_customer_service_calls double 
# MAGIC ,churned integer)
# MAGIC STORED AS PARQUET 
# MAGIC LOCATION '/mnt/bmathew/customer_analysis_data/non_delta_parquet'

# COMMAND ----------

# MAGIC %sql
# MAGIC drop table if exists bmathew.churn_data_delta;
# MAGIC 
# MAGIC create table bmathew.churn_data_delta (
# MAGIC area_code int
# MAGIC ,phone_number string
# MAGIC ,state string
# MAGIC ,account_length double
# MAGIC ,international_plan string
# MAGIC ,voice_mail_plan string
# MAGIC ,number_vmail_messages double
# MAGIC ,total_day_minutes double
# MAGIC ,total_day_calls double
# MAGIC ,total_day_charge double
# MAGIC ,total_eve_minutes double
# MAGIC ,total_eve_calls double
# MAGIC ,total_eve_charge double
# MAGIC ,total_night_minutes double
# MAGIC ,total_night_calls double
# MAGIC ,total_night_charge double
# MAGIC ,total_intl_minutes double
# MAGIC ,total_intl_calls double
# MAGIC ,total_intl_charge double
# MAGIC ,number_customer_service_calls double 
# MAGIC ,churned integer)
# MAGIC USING DELTA 
# MAGIC LOCATION '/mnt/bmathew/customer_analysis_data/delta'

# COMMAND ----------

# MAGIC %md
# MAGIC ### Churn count by customer spend

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT CASE WHEN total_charges >= 0 AND total_charges <= 9.99 THEN '0 - 9.99' WHEN total_charges >= 10 AND total_charges <= 19.99 THEN '10 - 19.99' WHEN total_charges >= 20 AND total_charges <= 29.99 THEN '20 - 29.99' WHEN total_charges >= 30 AND total_charges <= 39.99 THEN '30 - 39.99' WHEN total_charges >= 40 AND total_charges <= 49.99 THEN '40 - 49.99'  WHEN total_charges >= 50 AND total_charges <= 59.99 THEN '50 - 59.99' WHEN total_charges >= 60 AND total_charges <= 69.99 THEN '60 - 69.99' WHEN total_charges >= 70 AND total_charges <= 79.99 THEN '70 - 79.99' WHEN total_charges >= 80 AND total_charges <= 89.99 THEN '80 - 89.99' WHEN total_charges >= 90 AND total_charges <= 99.99 THEN '90 - 99.99' 
# MAGIC  WHEN total_charges >= 100 then '100+' ELSE '0' END AS `$ Spend Amount`, COUNT(*) AS `Churn Count`
# MAGIC FROM (SELECT area_code, phone_number, sum(total_day_charge + total_night_charge +  total_eve_charge) total_charges FROM bmathew.churn_data_non_delta WHERE churned = 1 GROUP BY area_code, phone_number) m GROUP BY `$ Spend Amount` ORDER BY `$ Spend Amount` ASC

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT CASE WHEN total_charges >= 0 AND total_charges <= 9.99 THEN '0 - 9.99' WHEN total_charges >= 10 AND total_charges <= 19.99 THEN '10 - 19.99' WHEN total_charges >= 20 AND total_charges <= 29.99 THEN '20 - 29.99' WHEN total_charges >= 30 AND total_charges <= 39.99 THEN '30 - 39.99' WHEN total_charges >= 40 AND total_charges <= 49.99 THEN '40 - 49.99'  WHEN total_charges >= 50 AND total_charges <= 59.99 THEN '50 - 59.99' WHEN total_charges >= 60 AND total_charges <= 69.99 THEN '60 - 69.99' WHEN total_charges >= 70 AND total_charges <= 79.99 THEN '70 - 79.99' WHEN total_charges >= 80 AND total_charges <= 89.99 THEN '80 - 89.99' WHEN total_charges >= 90 AND total_charges <= 99.99 THEN '90 - 99.99' 
# MAGIC  WHEN total_charges >= 100 then '100+' ELSE '0' END AS `$ Spend Amount`, COUNT(*) AS `Churn Count`
# MAGIC FROM (SELECT area_code, phone_number, sum(total_day_charge + total_night_charge +  total_eve_charge) total_charges FROM bmathew.churn_data_delta WHERE churned = 1 GROUP BY area_code, phone_number) m GROUP BY `$ Spend Amount` ORDER BY `$ Spend Amount` ASC

# COMMAND ----------

# MAGIC %sql
# MAGIC drop table if exists bmathew.churn_data_updates;
# MAGIC create table bmathew.churn_data_updates using delta as select * from bmathew.churn_data_delta;

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from  bmathew.churn_data_updates 

# COMMAND ----------

# MAGIC %sql
# MAGIC update bmathew.churn_data_updates set total_day_minutes = 300, total_eve_minutes = 400, total_intl_calls = 5 where phone_number = "382-4657";
# MAGIC update bmathew.churn_data_updates set total_day_minutes = 300, total_eve_minutes = 400, total_intl_calls = 5 where phone_number = "370-3450";
# MAGIC update bmathew.churn_data_updates set total_day_minutes = 300, total_eve_minutes = 400, total_intl_calls = 5 where phone_number = "350-6639";
# MAGIC update bmathew.churn_data_updates set total_day_minutes = 300, total_eve_minutes = 400, total_intl_calls = 5 where phone_number = "378-1303";

# COMMAND ----------

# MAGIC %md
# MAGIC ### Merge new incoming data with existing data

# COMMAND ----------

# MAGIC %sql
# MAGIC MERGE INTO bmathew.churn_data_delta target
# MAGIC USING bmathew.churn_data_updates as source
# MAGIC ON source.area_code = target.area_code AND source.phone_number = target.phone_number
# MAGIC WHEN MATCHED THEN UPDATE
# MAGIC SET target.total_day_minutes = source.total_day_minutes, target.total_eve_minutes = source.total_eve_minutes, target.total_intl_calls = source.total_intl_calls, target.phone_number = source.phone_number

# COMMAND ----------

# MAGIC %sql
# MAGIC DELETE FROM bmathew.churn_data_delta;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM bmathew.churn_data_delta;

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE HISTORY bmathew.churn_data_delta

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM bmathew.churn_data_delta VERSION AS OF 2;

# COMMAND ----------

# MAGIC %md
# MAGIC ###Split data into training and test datasets

# COMMAND ----------

df = sqlContext.sql("SELECT * FROM bmathew.churn_data_delta VERSION AS OF 2")
(trainingData, testData) = df.randomSplit([0.7, 0.3], seed=42)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Train Model
# MAGIC   - Create a funtion to train Random Forest Model:
# MAGIC     - Convert features into feature vector
# MAGIC     - Train using training dataset
# MAGIC     - Track model training to MLflow tracking server: Hyperparameters, Metrics, and Model
# MAGIC     - Evaluate model using test dataset

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.sql.functions import *

train = trainingData.select("*")
test = trainingData.select("*")
trainCols = train.columns
testCols = test.columns
stages = [] 
featureCols = ["account_length", "total_day_minutes", "total_day_calls",  "total_eve_calls", "total_eve_minutes","total_night_calls", "total_intl_calls", "total_intl_minutes", "number_customer_service_calls", "total_day_charge", "total_eve_charge","total_night_charge","total_intl_charge"]
stages = [VectorAssembler(inputCols=featureCols, outputCol="features"), StringIndexer(inputCol="churned", outputCol="label")]
pipeline = Pipeline(stages=stages)
trainpipelineModel = pipeline.fit(train)
testpipelineModel = pipeline.fit(test)
trainData = trainpipelineModel.transform(train)
testData = testpipelineModel.transform(test)
selectedcols = trainCols + ["features"] 
data = trainData.select(selectedcols)
outputData = data.select("*")
display(outputData)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Train Model
# MAGIC   - Create a funtion to train Random Forest Model:
# MAGIC     - Train using training dataset
# MAGIC     - Track model training to MLflow tracking server: Hyperparameters, Metrics, and Model
# MAGIC     - Evaluate model using test dataset

# COMMAND ----------

def train_model(maxDepth, numTrees, featureCols):
  featureCols=featureCols
  #featureCols = ["account_length", "total_day_calls",  "total_eve_calls", "total_night_calls", "total_intl_calls",  "number_customer_service_calls"]
  maxDepth=maxDepth
  numTrees=numTrees

  with mlflow.start_run(experiment_id=3424774756056187):
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

maxDepth=1
numTrees=2
featureCols = ["account_length", "total_day_calls",  "total_eve_calls", "total_night_calls", "total_intl_calls",  "number_customer_service_calls"]
train_model(maxDepth=maxDepth, numTrees=numTrees, featureCols=featureCols)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Call training function  with a different set of parameters
# MAGIC - Input parameters: maxDepth, numTrees, and fields to use for features
# MAGIC - Output parameter: AUC, Area Under the Curve
# MAGIC - Training run will be logged to MLflow tracking server
# MAGIC    - Log model hyperparameters
# MAGIC    - Log model output metrics
# MAGIC    - Save model itself

# COMMAND ----------

maxDepth=1
numTrees=1
featureCols = ["account_length", "total_day_minutes", "total_day_calls",  "total_eve_calls", "total_eve_minutes","total_night_calls", "total_intl_calls", "total_intl_minutes", "number_customer_service_calls", "total_day_charge", "total_eve_charge","total_night_charge","total_intl_charge"]
train_model(maxDepth=maxDepth, numTrees=numTrees, featureCols=featureCols)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Batch Inference

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM  bmathew.churn_inference_data 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load one of the models that we saved to the MLflow Tracking server and score against the data

# COMMAND ----------

from mlflow import spark
df = sqlContext.sql("SELECT * FROM bmathew.churn_inference_data")
rfModel = mlflow.spark.load_model("dbfs:/databricks/mlflow/3424774756056187/26edf4f0eb134b669897e81b9319f55c/artifacts/Random_Forest_Churn_Prediction")
display(rfModel.transform(df))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Streaming Inference

# COMMAND ----------

from pyspark.sql.functions import *
from pyspark.sql.types import *
df = sqlContext.readStream.format("delta").option("maxFilesPerTrigger", "10").load("/user/hive/warehouse/bmathew.db/churn_inference_data_delta").select(col("*"))
df.registerTempTable("streaming_transactions")
display(df.limit(100))

# COMMAND ----------

display(rfModel.transform(df))