import os

env_dist = os.environ
if "D4J_HOME" not in env_dist:
	print("D4J_HOME has not been set.")
	exit(-1)
d4j_home = env_dist["D4J_HOME"]
if not d4j_home.endswith("/"):
	d4j_home += "/"

sus_pos_path = "../SuspiciousCodePositions/"
version_file = "../versions.txt"

with open(version_file, "r") as file:
	versions = file.readlines()
versions = [version.strip() for version in versions]

with open("run_repair.sh", "w") as file:
	root_dir = os.path.abspath('..')
	project_dir = os.path.join(root_dir, "projects/")
	log_dir = os.path.join(root_dir, "dnn_model/dnn_tbar_log/")
	file.write("#!/bin/bash\n\n")
	for version in versions:
		file.write("cd {} && timeout 3h java -Xmx1g -jar TBar_TRANSFER_PR.jar {} {} {} > {}{}.log \n".format(root_dir, project_dir, version, d4j_home, log_dir, version))
