"""
THIS IS THE MAIN FILE
- All operation and commands will be issued from here
- Methods are defined in ops.py (please refer)

- Authors: Edward Chiao and Jon Jiang (working with Lelia Glass, Ph.D)
"""
import ops


def main():

	# the file that we are looking at
	tempFile = 'weightlifting.txt'

	# parses the file and analyzes it
	pf = ops.parseFile(tempFile)

	# dependency parser used
	dc = ops.getDependencyCount(pf)

	# counts the number of verbs and calculates words per million within the file
	ops.countRoots('weightlifting.csv', pf)

	# gets sentences with null objects
	print ops.getSentWithNullObject(pf)

	# print(pf)
	# print(dc)