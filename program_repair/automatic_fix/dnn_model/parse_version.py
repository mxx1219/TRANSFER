import os
import pickle

version_info = {}
for i in range(1, 11):
	with open("./model_save_d4j/{}/test_versions.txt".format(i), "r") as file:
		content = file.readlines()
	content = [line.strip() for line in content]
	for version in content:
		version_info[version] = str(i)

with open("version_info.pkl", "wb") as file:
	pickle.dump(version_info, file)

for version in version_info:
	print("{}:{}".format(version, version_info[version]))
