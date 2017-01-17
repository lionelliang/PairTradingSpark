## Spark Application - execute with spark-submit

## Imports
from pyspark import SparkConf, SparkContext

## Module Constants
APP_NAME = "ADF Spark Application"

## Closure Functions

def isprime(n):
	"""
	check if integer n is a prime
	"""
	# make sure n is a positive integer
	n = abs(int(n))
	# 0 and 1 are not primes
	if n < 2:
		return False
	# 2 is the only even prime number
	if n == 2:
		return True
	# all other even numbers are not primes
	if not n & 1:
		return False
	# range starts with 3 and only needs to go up the square root of n
	# for all odd numbers
	for x in range(3, int(n**0.5)+1, 2):
		if n % x == 0:
			return False
	return True

## Main functionality

def main(sc):
	# Create an RDD of numbers from 0 to 1,000,000
	nums = sc.parallelize(xrange(1000000))

	# Compute the number of primes in the RDD
	cnt = nums.filter(isprime).count()
	print "%d primes\n"%cnt

if __name__ == "__main__":
	# Configure Spark
	conf = SparkConf().setAppName(APP_NAME)
	conf = conf.setMaster("local[*]")
	sc = SparkContext(conf=conf)

	# Execute Main functionality
	main(sc)
