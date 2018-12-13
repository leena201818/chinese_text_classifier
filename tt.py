import os

class dirIterator_tuple:
	def __iter__(self):
		for tName in os.listdir(os.getcwd()):
			fName = os.path.join(os.getcwd(),tName)
			if os.path.isdir(fName):
				for file in os.listdir(fName):
					# print os.path.join(fName,file),tName
					yield (os.path.join(fName,file),tName)

if __name__ == "__main__":
	for f in dirIterator_tuple():
		print(f[0].decode("gbk"))