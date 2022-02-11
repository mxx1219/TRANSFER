
# TRANSFER

I. Requirements
--------------------
 - [Java 1.7](https://www.oracle.com/technetwork/java/javase/downloads/java-archive-downloads-javase7-521261.html)
 - [Python 3.6](https://www.python.org/downloads/)
 - [Defects4J 1.2.0](https://github.com/rjust/defects4j/releases/tag/v1.2.0)
 - [SVN >= 1.8](https://subversion.apache.org/packages.html)
 - [Git >= 1.9](https://git-scm.com/)
 - [Perl >= 5.0.10](https://www.perl.org/get.html)
 - [PyTorch-1.5.1](https://pytorch.org/)


II. Overview of TRANSFER
--------------------

![The overview of TRANSFER to perform fault localization and automated program repair.\label{step}](./overview.png)



III. Download Dataset
---------------------------
1. Click the following url link and download the necessary dataset used in this research.

    [data_for_transfer.zip](https://mega.nz/file/S5w0BILC#Q2BHCuRD2aW_61vshVbcxj-ObYh2cyGhqOAmAXNn-T0)

2. Unzip it and put the extracted datasets in the corresponding path: 
    * Put `dataset_fl.pkl` into `./fault_localization/`
    * Put `dataset_pr.pkl` into `./program_repair/`
    * Put `src_code.pkl` into `./fault_localization/binary_classification/d4j_data/` 


IV. Prepare Defects4J Bugs
---------------------------
 1. Download and install [Defects4J 1.2.0](https://github.com/rjust/defects4j/releases/tag/v1.2.0), after which the required settings described in its readme file must be completed.
 
 2. Check out and compile each bug. There are 395 bugs in total, but only 71 bugs are necessary to be checked out for our repair experiments, which are listed in `./program_repair/automatic_fix/versions.txt`. All checked out bugs should be put into the directory `./program_repair/automatic_fix/projects/` and follow the same naming standard with `versions.txt` (i.g., Chart_1).
  
 3. Export environment variable `D4J_HOME` as the root dir of Defects4j.
 

V. Get Experimental Results Quickly
 --------------------------
1. For quickly obtaining the experimental results of our fault localization experiment, you can enter the path `./fault_localization/ranking_task/run_model/`, and run `python run_group.py` command. This command includes the training and testing phases. If gpus are used, the test reuslts can usually be obtained within 10 miniutes. If cpus are used, please replace `True` at line 46 with `False` in `train.py` file first.
2. For quickly obtaining the experimental results of our automated program repair experiment, you can enter the path `./program_repair/automatic_fix/shell_71_versions/`, and run `python generate_shell.py` to generate the shell file containing 71 serial commands for 71 bugs in Defects4J. Then, you can run `chomd +x run_repair.sh`to give the executable authority to the newly generated shell file. Finally, you can run `./run_repair.sh` to obtain the repair result. All repair logs are shown in `./program_repair/dnn_model/dnn_tbar_log/`, and all generated plausible patches are shown in `./program_repair/automatic_fix/OUTPUT/`. 


VI. Reproduce All Experiments from Scratch 
--------------------------
1. For fault localization:
    * Since three features groups are needed to run MLP-based ranking model for fault localization, the spectrum-based and mutation-based features are already extracted by using [GZoltar 1.7.2](https://github.com/GZoltar/gzoltar/releases/tag/v1.7.2) and [PIT 1.1.5](https://pitest.org/downloads/) tools respectively. While the semantic features are generated from 11 BiLSTM-based binary classifiers, you should train them on dataset_fl first to obtain the transferrd knowledge which are then used to generate 11-dimension semantic features for each suspicious statement in Defects4J bug. The steps to obtain the 11-dim semantic features are as follows:
        * Enter `./fault_localization/binary_classification/script/` and run `python data_preprocess.py <fix_template>`, then training data from dataset_fl are generated to train the corresponding  binary classifier of each fix template (bug type). The `<fix_template>` in the command is expected to replaced with one of the following 11 fix templates:
            * InsertMissedStmt
            * InsertNullPointerChecker
            * MoveStmt
            * MutateConditionalExpr
            * MutateDataType
            * MutateLiteralExpr
            * MutateMethodInvExpr
            * MutateOperators
            * MutateReturnStmt
            * MutateVariable
            * RemoveBuggyStmt
        * Enter `./fault_localization/binary_classification/code/` and run `python train.py <fix_tempalte>` to train the corresponding binary classifier. The optimal parameters are save in relative dir `./model_save/` of current path.
        * Enter `./fault_localization/binary_classification/script/` and run `python data_preprocess_for_d4j.py <fix_remplate>` to preprocess each suspicious statement in Defects4J dataset. `<fix_template>` should also be replaced with 11 fix templates.
        * Enter `./fault_localization/binary_classification/code/` and run `python predict_for_d4j.py` to obtain the final 11-dim semantic features which lotates in `../d4j_data/semantic.pkl`.
    * Having obtained the semantic features, we then generate samples for our fault localization task. You can enter `./fault_localization/ranking_task/` and run `python gen_data.py` to generate samples from Defects4J with 3 different feature groups.
    * Run the same command as in Section V(1).
2. For automated program repair:
    * We should first train a BiLSTM-based multi-classifier to learn the transferred knowledge for the subsequent fix template selection task in program repair. Thus, two steps are needed, which are as follows:
        * Enter `./program_repair/pattern_selection/` and run `python pipeline.py` to generate samples from dataset_pr to train the multi-classifier.
        * Enter the same dir, and run `python train_github.py` to train the multi-classifier. The optimal parameters are saved in the relative dir `./model_save/`.
        * Enter the same dir, and run `python train_d4j.py` to fine-tune the parameters of the model. The optimal parameters are saved in the relative dir `./model_save_d4j/`.
    * The trained multi-classifier can be embeded into the state-of-the-art template-based program repair technique [TBar](https://github.com/TruX-DTF/TBar) to optimize the fix template selection task.
        * Copy `./program_repair/pattern_selection/model_save_d4j/` and `./program_repair/pattern_selection/data/train/embedding/w2v_32` to `./program_repair/automatic_fix/dnn_model/`, and then run `python parse_version.py`.
        * Use the new generated suspicious files in Section VI(1) to program repair task. Specifically, to enter `./program_repair/automatic_fix/script/` and run `python get_sus_file.py`.
        * Run the same command as in Section V(2).

VII. Experiments on Defects4J-v2.0.0
--------------------------
we have conducted a preliminary experiment on a recent benchmark Defects4J-v2.0.0. The results show similar trend as that on Defects4J-v1.2.0 and confirm the effectiveness of the proposed approach. Since some bugs in Defects4J-v2.0.0 cannot be checked out (e.g. https://github.com/rjust/defects4j/issues/353) and reproduced (also mentioned in Grace(Lou et al. FSE’21)), we finally use 226 bugs that can be reproduced on our machine for the experiment. The details of bug versions can be seen in d4j_v2_versions.txt in the root directory.


### Fault Localization Experimental Results:

| Techniques | Top-1 | Top-3 | Top-5 | MFR | MAR |
|:------------|:----------:|:---------------:|----------|---------|:------:|
| Ochiai|24 | 43 | 55 | 104.74 | 132.25 |
| Tarantula | 23 | 41 | 53 | 107.9 | 135.62 |
| Dstar | 24 | 43 | 55 | 105.08 | 133.39 |
| Metallaxis | 8 | 18 | 28 | 342.99 | 415.71 |
| DeepFL |38 | 68 | 91| 83.42 | 115.39 |
| TRANSFER-FL |46 |80 | 102 |74.86 |106.72 |


### Program Repair Experimental Results:

TBar can fix 4 bugs: Cli_5, Compress_24, Csv_9, JxPath_6
TRANSFER-PR can fix 6 bugs: Cli_5, Compress_19, Compress_24, Compress_27, Csv_9, JxPath_6


VIII. Structure of the Directories
 -------------------------------
 ```powershell
  |--- README.md                :  user guidance
  |--- overview.png             :  overview of TRANSFER
  |--- d4j_v2_versions.txt      :  Defects4J-v2.0.0 versions
  |--- fault_localization       :  implementation of TRANSFER-FL
  |------ binary_classification :  implementation of BiLSTM-based binary classifier
  |------ ranking_task          :  implementation of MLP-based ranking model
  |------ script                :  script to see the structure of dataset_fl
  |--- program_repair           :  implementation of TRANSFER-PR
  |------ pattern_selection     :  implemeatation of Bilstm-based multi-classifier
  |------ automatic_fix         :  a new template-based apr technique enhanced by TRANSFER-PR
  |------ script                :  script to see the structure of dataset_pr  

```
