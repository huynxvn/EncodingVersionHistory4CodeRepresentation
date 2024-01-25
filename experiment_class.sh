#!/bin/bash

python -u astnn_versionall/class_pure_code.py 
python -u astnn_versionall/class_concat.py 
python -u astnn_versionall/class_max_pool.py 

# python -u astnn_callgraph/class_pure_code.py 
python -u astnn_callgraph/class_concat.py 
python -u astnn_callgraph/class_max_pool.py 

# python -u astnn_versionall_callgraph/class_pure_code.py 
python -u astnn_versionall_callgraph/class_concat.py 
python -u astnn_versionall_callgraph/class_max_pool.py 

# python -u astnn_versionall_numofdays/class_pure_code.py 
python -u astnn_versionall_numofdays/class_concat.py 
python -u astnn_versionall_numofdays/class_max_pool.py

# python -u astnn_versionall_cg_numofdays/class_pure_code.py 
python -u astnn_versionall_callgraph_numofdays/class_concat.py 
python -u astnn_versionall_callgraph_numofdays/class_max_pool.py 

python -u codebert_versionall/class_pure_code.py 
python -u codebert_versionall/class_concat.py 
python -u codebert_versionall/class_max_pool.py

# python -u codebert_callgraph/class_pure_code.py 
python -u codebert_callgraph/class_concat.py 
python -u codebert_callgraph/class_max_pool.py 

# python -u codebert_versionall_callgraph/class_pure_code.py 
python -u codebert_versionall_callgraph/class_concat.py 
python -u codebert_versionall_callgraph/class_max_pool.py 

# python -u codebert_versionall_numofdays/class_pure_code.py 
python -u codebert_versionall_numofdays/class_concat.py 
python -u codebert_versionall_numofdays/class_max_pool.py 

# python -u codebert_versionall_cg_numofdays/class_pure_code.py 
python -u ccodebert_versionall_callgraph_numofdays/class_concat.py 
python -u ccodebert_versionall_callgraph_numofdays/class_max_pool.py