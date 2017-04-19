#coding:utf-8
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession

import pymongo_spark
# Important: activate pymongo_spark.
pymongo_spark.activate()


def main():
    conf = SparkConf().setAppName("pyspark test")
    sc = SparkContext(conf=conf)

    # pymongo_spark version
    # Create an RDD backed by the MongoDB collection.
    # This RDD *does not* contain key/value pairs, just documents.
    # If you want key/value pairs, use the mongoPairRDD method instead.
    rdd = sc.mongoRDD('mongodb://localhost:27017/stock.linrreg')
    print rdd.first()

    # Save this RDD back to MongoDB as a different collection.
    rdd.saveToMongoDB('mongodb://localhost:27017/stock.linrregother')

    # You can also read and write BSON:
    #bson_rdd = sc.BSONFileRDD('/path/to/file.bson')
    #bson_rdd.saveToBSON('/path/to/bson/output')

'''
#spark-mongo version not working
spark = SparkSession.builder.getOrCreate()
spark.sql("CREATE TEMPORARY VIEW linrreg_table USING com.stratio.datasource.mongodb OPTIONS (host 'localhost:27017', database 'stock', collection 'linrreg')")
spark.sql("SELECT * FROM linrreg_table").collect()
'''

'''
df = SparkSession.read.format('com.stratio.datasource.mongodb').options(host='localhost:27017', database='stock', collection='linrreg').load()
df.collect()

df.write.format("com.stratio.datasource.mongodb").mode('overwrite').options(host='localhost:27017', database='stock', collection='linrregother').save()
dfView = sparkSession.read.format('com.stratio.datasource.mongodb').options(host='localhost:27017', database='stock', collection='linrregother').load()
dfView.show()
'''

if __name__ == '__main__':
    main()