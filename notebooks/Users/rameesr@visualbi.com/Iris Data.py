# Databricks notebook source
iris=spark.read.format("csv").option("header", "true").load("FileStore/tables/iris.csv")