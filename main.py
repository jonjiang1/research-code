"""
THIS IS THE MAIN FILE
- All operation and commands will be issued from here
- Methods are defined in ops.py (please refer)

- Authors: Edward Chiao and Jonathan Jiang (working with Lelia Glass, Ph.D)
"""

# module comtaining parsing methods
import ops
# module for file io
import os


def main():

	# the file that we are looking at
	tempFile = 'toy_reddit.txt'
	# parses the file and analyzes it
	pf = ops.parseFile(tempFile)
	# dependency parser used
	dc = ops.getDependencyCount(pf)
	# counts the number of verbs and calculates words per million within the file
	ops.countRoots('toy_reddit', pf)
	# gets sentences with null objects
	ops.getSentWithNullObject(pf)
	# gets sentences with null objects and the verb utilize
	# ops.getVerbWithNullObject(pf, u'utilize')
	# creates dict of all sentences with null objects and sorts it by verb
	ops.sortSentencesByVerb(pf)
	# print(pf)
	# print(dc)

	print ops.compare_csv("askreddit_output.csv", "business_output.csv")

main()