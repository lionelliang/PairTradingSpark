#coding:utf-8
from pyspark import SparkContext, SparkConf

import pymongo_spark
# Important: activate pymongo_spark.
pymongo_spark.activate()


def main():
    conf = SparkConf().setAppName("pyspark test")
    sc = SparkContext(conf=conf)

    # Create an RDD backed by the MongoDB collection.
    # This RDD *does not* contain key/value pairs, just documents.
    # If you want key/value pairs, use the mongoPairRDD method instead.
    rdd = sc.mongoRDD('mongodb://localhost:27017/stock.linrreg')
    print rdd.first()

    # Save this RDD back to MongoDB as a different collection.
    rdd.saveToMongoDB('mongodb://localhost:27017/stock.linrreg')

    # You can also read and write BSON:
    #bson_rdd = sc.BSONFileRDD('/path/to/file.bson')
    #bson_rdd.saveToBSON('/path/to/bson/output')

if __name__ == '__main__':
    main()