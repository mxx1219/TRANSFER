import os
import pickle
import sys


def parse_model_output_file():
	total_result = []
	version_list = []
	for group_index in range(1, 11):
		with open("./result/{}.txt".format(group_index), "r") as file:
			content = file.readlines()
		current_result = []
		first_flag = True
		for line in content:
			line = line.strip()
			if line.startswith("==="):
				if len(version_list) > 0:
					if len(current_result) == 0:
						if not first_flag:
							version_list.pop()
					else:
						total_result.append(current_result[:])
						current_result = []
				version_list.append(line.replace("===", ""))
			else:
				current_result.append(float(line))
			first_flag = False
		if len(current_result) > 0:
			total_result.append(current_result[:])
		else:
			version_list.pop()
	# print(len(total_result))
	# print(len(version_list))
	return total_result, version_list


if __name__ == "__main__":

	# data pre-processing for cross-validation setup
	status = os.system("python3 pipeline_group.py >/dev/null 2>&1")
	assert status == 0

	# 10-fold cross-validation training and testing
	for group_index in range(1,11):
		status = os.system("python3 train.py {} >/dev/null 2>&1".format(group_index))
		assert status == 0

	total_result, version_list = parse_model_output_file()

	top1_total = 0
	top3_total = 0
	top5_total = 0
	all_position_total = []
	first_position_total = []

	print("\nStatistics for each project.")
	for project in ["Chart", "Closure", "Math", "Mockito", "Lang", "Time"]:
		print("=" * 20)
		print(project)

		top1 = 0
		top3 = 0
		top5 = 0
		all_position = []
		first_position = []

		for i, version in enumerate(version_list):
			if not version.startswith(project):
				continue
			bugs = total_result[i]
			rank = []
			for bug in bugs:
				rank.append(bug)
			min_rank = min(rank)
			avg_rank = sum(rank) / len(rank)

			if min_rank <= 1:
				#print(version)
				top1 += 1
			if min_rank <= 3:
				top3 += 1
			if min_rank <= 5:
				top5 += 1
			first_position.append(min_rank)
			all_position.append(avg_rank)

		print("Top1\t{}".format(top1))
		print("Top3\t{}".format(top3))
		print("Top5\t{}".format(top5))
		print("MFR\t{}".format(round(sum(first_position) / len(first_position), 2)))
		print("MAR\t{}".format(round(sum(all_position) / len(all_position), 2)))

		top1_total += top1
		top3_total += top3
		top5_total += top5
		all_position_total += all_position
		first_position_total += first_position

	print("\nStatistics for all projects.")
	print("=" * 20)
	print("Top1\t{}".format(top1_total))
	print("Top3\t{}".format(top3_total))
	print("Top5\t{}".format(top5_total))
	print("MFR\t{}".format(round(sum(first_position_total) / len(first_position_total), 2)))
	print("MAR\t{}".format(round(sum(all_position_total) / len(all_position_total), 2)))

