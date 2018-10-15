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
#Clearly there are two separate clusters here. But we dont know what is what.

# COMMAND ----------

#Now plotting each of the species separately to understand the distribution.

plt.figure()
fig,ax=plt.subplots(1,2,figsize=(15,10))
pdata[pdata['species']=='setosa'].plot(x="sepal_length",y="sepal_width",kind="scatter",ax=ax[0],label="setosa",color='g')
pdata[pdata['species']=='virginica'].plot(x="sepal_length",y="sepal_width",kind="scatter",ax=ax[0],label="virginica",color='b')
pdata[pdata['species']=='versicolor'].plot(x="sepal_length",y="sepal_width",kind="scatter",ax=ax[0],label="versicolor",color='r')
pdata[pdata['species']=='setosa'].plot(x="petal_length",y="petal_width",kind="scatter",ax=ax[1],label="setosa",color='g')
pdata[pdata['species']=='virginica'].plot(x="petal_length",y="petal_width",kind="scatter",ax=ax[1],label="virginica",color='b')
pdata[pdata['species']=='versicolor'].plot(x="petal_length",y="petal_width",kind="scatter",ax=ax[1],label="versicolor",color='r')
#print(pdata)
display(fig)

# Sepal
#Setosa has separate cluster in sepal length vs width and there is a linear relationship
#Its tough to differentiate virginica and versicolor

#Petal
#Again Sepal forms a separate cluster.
# virginica lies on the higher side when compared to versicolor.



# COMMAND ----------

#Lets check the distribution overlaps

plt.figure()
fig,ax = plt.subplots(2,2,figsize=(15,15))
pdata[pdata['species']=='setosa'].sepal_length.plot(kind="hist",ax=ax[0][0],label="setosa",color='g',fontsize=10,alpha=0.2,histtype='step',stacked=True,fill=True)
pdata[pdata['species']=='virginica'].sepal_length.plot(kind="hist",ax=ax[0][0],label="virginica",color='b',fontsize=10,alpha=1,histtype='step',stacked=True,fill=False)
pdata[pdata['species']=='versicolor'].sepal_length.plot(kind="hist",ax=ax[0][0],label="versicolor",color='r',fontsize=10,alpha=0.1,histtype='step',stacked=True,fill=True)
ax[0][0].legend(prop={'size': 10})
ax[0][0].set(title='Sepal Length frequency')
#ax[0][0].hist(pdata[pdata['species']=='setosa'].sepal_length,bins='auto',alpha=0.5)
############
pdata[pdata['species']=='setosa'].sepal_width.plot(kind="hist",ax=ax[0][1],label="setosa",color='g',fontsize=10,alpha=0.2,histtype='step',stacked=True,fill=True)
pdata[pdata['species']=='virginica'].sepal_width.plot(kind="hist",ax=ax[0][1],label="virginica",color='b',fontsize=10,alpha=1,histtype='step',stacked=True,fill=False)
pdata[pdata['species']=='versicolor'].sepal_width.plot(kind="hist",ax=ax[0][1],label="versicolor",color='r',fontsize=10,alpha=0.1,histtype='step',stacked=True,fill=True)
ax[0][1].legend(prop={'size': 10})
ax[0][1].set(title='Sepal Width frequency')

#############

pdata[pdata['species']=='setosa'].petal_length.plot(kind="hist",ax=ax[1][0],label="setosa",color='g',fontsize=10,alpha=0.2,histtype='step',stacked=True,fill=True)
pdata[pdata['species']=='virginica'].petal_length.plot(kind="hist",ax=ax[1][0],label="virginica",color='b',fontsize=10,alpha=1,histtype='step',stacked=True,fill=False)
pdata[pdata['species']=='versicolor'].petal_length.plot(kind="hist",ax=ax[1][0],label="versicolor",color='r',fontsize=10,alpha=0.1,histtype='step',stacked=True,fill=True)
ax[1][0].legend(prop={'size': 10})
ax[1][0].set(title='petal Length frequency')
####################
pdata[pdata['species']=='setosa'].petal_width.plot(kind="hist",ax=ax[1][1],label="setosa",color='g',fontsize=10,alpha=0.2,histtype='step',stacked=True,fill=True)
pdata[pdata['species']=='virginica'].petal_width.plot(kind="hist",ax=ax[1][1],label="virginica",color='b',fontsize=10,alpha=1,histtype='step',stacked=True,fill=False)
pdata[pdata['species']=='versicolor'].petal_width.plot(kind="hist",ax=ax[1][1],label="versicolor",color='r',fontsize=10,alpha=0.1,histtype='step',stacked=True,fill=True)
ax[1][1].legend(prop={'size': 10})
ax[1][1].set(title='petal Width frequency')

display(fig)

# Sepal length and width are having  overlapping values across species.
# Petal length and width are distinct for setosa and slight overlap between other two
