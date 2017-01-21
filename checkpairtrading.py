## Spark Application - execute with spark-submit

## Imports
import csv
#import matplotlib.pyplot as plt

from StringIO import StringIO
from datetime import datetime
from collections import namedtuple
from operator import add, itemgetter
from pyspark import SparkConf, SparkContext

## Module Constants
APP_NAME = "ADF Spark Application"
fields = ('date', 'sym', 'open', 'high', 'low', 'clsoe', 'volume', 'amount')
Quotation = namedtuple('Quotation', fields)
APP_NAME = "Flight Delay Analysis"
DATE_FMT = "%Y/%m/%d"
TIME_FMT = "%H%M"

## Closure Functions
def parse(row):
	"""
	Parses a row and returns a named tuple.
	"""

	row[0] = datetime.strptime(row[0], DATE_FMT).date()
	row[2] = float(row[2])
	row[3] = float(row[3])
	row[4] = float(row[4])
	row[5] = float(row[5])
	return Quotation(*row[:8])

def split(line):
	"""
	Operator function for splitting a line with csv module
	"""
	reader = csv.reader(StringIO(line))
	return reader.next()

## Main functionality

def main(sc):
	# Read the CSV Data into an RDD
	kindleline1 = sc.textFile("600815.csv").map(split).map(parse)
	kindleline2 = sc.textFile("601002.csv").map(split).map(parse)
	print "%d, %d" %(kindleline1.count(), kindleline2.count())

if __name__ == "__main__":
	# Configure Spark
	conf = SparkConf().setAppName(APP_NAME)
	conf = conf.setMaster("local[*]")
	sc = SparkContext(conf=conf)

	# Execute Main functionality
	main(sc)
