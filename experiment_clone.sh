#!/bin/bash
python -u astnn_versionall/clone_pure_code.py 
python -u astnn_versionall/clone_concat.py 
python -u astnn_versionall/clone_max_pool.py 
python -u astnn_versionall/clone_diff_concat.py 

# python -u astnn_callgraph/clone_pure_code.py 
python -u astnn_callgraph/clone_concat.py 
python -u astnn_callgraph/clone_max_pool.py 
python -u astnn_callgraph/clone_diff_concat.py 

# python -u astnn_versionall_callgraph/clone_pure_code.py 
python -u astnn_versionall_callgraph/clone_concat.py 
python -u astnn_versionall_callgraph/clone_max_pool.py 
python -u astnn_versionall_callgraph/clone_diff_concat.py

# python -u astnn_versionall_numofdays/clone_pure_code.py 
python -u astnn_versionall_numofdays/clone_concat.py 
python -u astnn_versionall_numofdays/clone_max_pool.py 
python -u astnn_versionall_numofdays/clone_diff_concat.py 

# python -u astnn_versionall_callgraph_numofdays/clone_pure_code.py 
python -u astnn_versionall_callgraph_numofdays/clone_concat.py 
python -u astnn_versionall_callgraph_numofdays/clone_max_pool.py 
python -u astnn_versionall_callgraph_numofdays/clone_diff_concat.py

# python -u codebert_versionall/clone_pure_code.py 
python -u codebert_versionall/clone_concat.py 
python -u codebert_versionall/clone_max_pool.py 
python -u codebert_versionall/clone_diff_concat.py

# python -u codebert_callgraph/clone_pure_code.py 
python -u codebert_callgraph/clone_concat.py 
python -u codebert_callgraph/clone_max_pool.py 
python -u codebert_callgraph/clone_diff_concat.py

# python -u codebert_versionsall_callgraph/clone_pure_code.py 
python -u codebert_versionsall_callgraph/clone_concat.py 
python -u codebert_versionsall_callgraph/clone_max_pool.py 
python -u codebert_versionsall_callgraph/clone_diff_concat.py

# python -u codebert_versionall_numofdays/clone_pure_code.py 
python -u codebert_versionall_numofdays/clone_concat.py 
python -u codebert_versionall_numofdays/clone_max_pool.py 
python -u codebert_versionall_numofdays/clone_diff_concat.py

# python -u codebert_versionall_callgraph_numofdays/clone_pure_code.py 
python -u codebert_versionall_callgraph_numofdays/clone_concat.py 
python -u codebert_versionall_callgraph_numofdays/clone_max_pool.py 
python -u codebert_versionall_callgraph_numofdays/clone_diff_concat.py