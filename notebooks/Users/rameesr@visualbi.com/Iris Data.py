# Databricks notebook source
#importing all required packages.
from pyspark.sql import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# COMMAND ----------

#Load Iris data into a dataframe.
iris=spark.read.format("csv").option("header", "true").load("FileStore/tables/iris.csv")

# COMMAND ----------

#Print the dataframe
display(iris)

# COMMAND ----------

#The first step we do after importing the data is to check what the data looks like. We will use describe function of databricks to summarize the data.
iris.describe().show()

# COMMAND ----------

type(iris)

# COMMAND ----------

#convering spark rdd to pandas dataframe
pdata=iris.toPandas()
pdata['sepal_length']=pd.to_numeric(pdata['sepal_length'])
pdata['sepal_width']=pd.to_numeric(pdata['sepal_width'])
pdata['petal_length']=pd.to_numeric(pdata['petal_length'])
pdata['petal_width']=pd.to_numeric(pdata['petal_width'])
print(pdata.describe())
pdata.dtypes


# COMMAND ----------

#Visualizations
fig=plt.figure()
fig.suptitle("Something")
fig, ax=plt.subplots(1,2)
pdata.plot(x="sepal_length",y="sepal_width",kind="scatter",ax=ax[0],label="sepal",color='g')
pdata.plot(x="petal_length",y="petal_width",kind="scatter",ax=ax[1],label="petal",color='g')
ax[0].set(title='Sepal Comparision')
ax[1].set(title='Petal Comparission')
display(fig)

# Basic comparssions Sepal length Vs Width | Petal Length vs Width