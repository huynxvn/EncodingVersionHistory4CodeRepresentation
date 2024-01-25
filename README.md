# Encoding Version History Context for Better Code Representation

This repository contains source code of research paper "Encoding Version History Context for Better Code Representation", which is submitted to MSR 2024


## Structure
The structure of our source code's repository is as follows:
- mining: contains our source code to extract version history context from Github;
    - version_history.py: script for data mining
- data: contain final data set, named "SeSaMe_VersionHistory_Callgraph.vFinal.json";
- astnn_*: contain script for ASTNN with different experiment settings
- codebert_*: contain script for CodeBERT with different experiment settings
- others: contains source code for: 
    - preprocess_clone.py: contains source code for data preprocessing for Code Clone Detection
    - preprocess_class.py: contains source code for data preprocessing for Code Classification    
- env.yml: contains the configuration for our enviroment. 

 

## Experiments
To replicate the result:
- for Code Clone Detection, please run the following commands
```
bash experiment_clone.sh
```
- for Code Classification, please run the following commands
```
bash experiment_class.sh
```
The experiment result will be stored in the file "result.txt"

## 📜 Citation
If you use our tool, please cite our paper as follows:

```
@inproceedings{nguyen2024encodingversionhistory,
  title={Encoding Version History Context for Better Code Representation},
  author={Nguyen, Huy and Treude, Christoph and Thongtanunam, Patanamon},
  booktitle={Proceedings of the 21st International Conference on Mining Software Repositories, 2024)},
  venue={Lisbon, Portugal}
  pages={x--y},
  year={2024}
}
```
