#coding:utf-8
import pandas as pd

from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession, SQLContext

from pymongo import MongoClient
import pymongo_spark
# Important: activate pymongo_spark.
pymongo_spark.activate()


def _connect_mongo(host, port, username, password, db):
    """ A util for making a connection to mongo """

    if username and password:
        mongo_uri = 'mongodb://%s:%s@%s:%s/%s' % (username, password, host, port, db)
        conn = MongoClient(mongo_uri)
    else:
        conn = MongoClient(host, port)

    return conn[db]

def read_mongo(db, collection, query={}, host='localhost', port=27017, username=None, password=None, no_id=True):
    """ Read from Mongo and Store into DataFrame """

    # Connect to MongoDB
    db = _connect_mongo(host=host, port=port, username=username, password=password, db=db)
    # Make a query to the specific DB and Collection
    cursor = db[collection].find(query)
    # Expand the cursor and construct the DataFrame
    df =  pd.DataFrame(list(cursor))
    # Delete the _id
    if no_id:
        del df['_id']
    return df

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
    #rdd.saveToMongoDB('mongodb://localhost:27017/stock.linrregother')

    rdd603989 = sc.mongoRDD('mongodb://localhost:27017/quotation.kline_603989')
    print rdd.first()

    sqlContext = SQLContext(sc)
    #print hasattr(rdd603989, "toDF")
    rddpure = rdd603989.map(lambda f: (f["date"], f["close"]))
    dfCollection = rddpure.toDF(['date', 'close']).toPandas()
    print dfCollection.head()

    # read pandas dataframe with pymongo
    pandasdf603989 = read_mongo("quotation", "kline_603989")
    print pandasdf603989.head()
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