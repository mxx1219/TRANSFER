import os

path = "../../../fault_localization/ranking_task/run_model/sus_pos_rerank/"
output_dir = "../SuspiciousCodePositions/"
for version in os.listdir(path):
	with open(os.path.join(path, version, "ranking.txt"), "r") as file:
		content = file.readlines()
	sus_list = [line.strip().split(" ")[0] for line in content]
	if not os.path.exists(os.path.join(output_dir, version)):
		os.makedirs(os.path.join(output_dir, version))
	with open(os.path.join(output_dir, version, "ranking.txt"), "w") as file:
		file.write("\n".join(sus_list))
		
	
	
