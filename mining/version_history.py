import pandas as pd
import numpy as np
import re
import os
# import sys
import json

from pydriller import Git, Repository
import git
from git import Repo
# import hglib
import  lizard
from datetime import datetime

#=============================

def read_json_file(file_name):
    df_code = pd.read_json(file_name)
    return df_code

def extract_and_replace_pattern(input_str):
    pattern = r'<\?[^>]+>'

    def replace_question_marks(match):
        parameters = match.group().strip('<>').split(',')
        result = []
        for param in parameters:
            if '?' in param:
                result.append('?')
            else:
                result.append(param.strip())
        return '<' + ', '.join(result) + '>'

    result = re.sub(pattern, replace_question_marks, input_str)

    # special case when after '[]' is a var name
    s_output = result
    if bool(re.match(r'.*>\[\][\w ]*', result)) == True:
        s_output = re.sub(r'>\[\][\w ]*', '>[]', result)

    s_output = re.sub(r'>\w*', '>', s_output)
    return s_output

def fn_get_class_name(file_path):
    # Use regular expression to find the file name without extension
    match = re.search(r'[^/\\]*(?=\.[^.]+$|$)', file_path)
    
    if match:
        file_name = match.group(0)
        # Remove the file extension from the file name
        return file_name.split('.')[0]
    else:
        return None

def fn_fix_method_str(inputString):
    out = inputString    

    # Define the pattern to match
    pattern = r'@ \w+'
    # Use regex to replace the pattern with an empty string
    out = re.sub(pattern, '', out)
    
    # out = out.replace("@ NonNull", "").replace("@ Nonnull", "").replace("@ nonNull", "").replace("@ nonnull", "").replace("<?super K1,?super V1>", "<?, ?>")
    # out = out.replace("@ NonNull", "").replace("@ Nonnull", "").replace("@ nonNull", "").replace("@ nonnull", "")
    # out = out.replace("@ Nullable", "").replace("@ nullable", "").replace("@ NULLABLE", "")
    # out = out.replace("@ NonNegative", "").replace("@ nonnegative", "").replace("@ Nonnegative", "").replace("@ nonNegative", "").replace("@ NONNEGATIVE", "")

    out = out.replace("<?super K1,?super V1>", "<?, ?>")
    out = out.replace(" . ", ".").replace(" .", ".").replace(" (", "(").replace(" )", ")").replace("( ", "(").replace(") ", ")")
    out = out.replace(" [", "[").replace("[ ", "[").replace("...", "[]")
    out = out.replace(" <", "<").replace("< ", "<").replace(" >", ">").replace("> ", ">")
    out = out.replace("int []", "int[]")
    # while '  ' in out:
    #     out = out.replace('  ', ' ')
    out = extract_and_replace_pattern(out)
    return out

def fn_special_cases(inputString):
    dict_special_cases = {
        'LockFreeBitArray.bitSize()': 'BloomFilterStrategies.LockFreeBitArray.bitSize()',
        'ImmutableMultimap.Builder.putAll(Iterable<?, ?>>)': 'ImmutableMultimap.Builder.putAll(Iterable<?>)',
        'StringUtils.valueOf(char)': 'StringUtils.TraditionalBinaryPrefix.valueOf(char)',
        'TypeConstants.isOKbyJLS()': 'TypeConstants.BoundCheckStatus.isOKbyJLS()',
        'Entry.getResourcePoolEntryName()': 'Archive.Entry.getResourcePoolEntryName()',
        'IModuleAwareNameEnvironment.get(char[])': 'IModuleAwareNameEnvironment.LookupStrategy.get(char[])',
        'Infer.accepts(UndetVar, InferenceContext)': 'Infer.InferenceStep.accepts(UndetVar, InferenceContext)'
    }
    if inputString in list(dict_special_cases.keys()):
        out = dict_special_cases[inputString]
    else:
        out = inputString
    return out

def fn_detect_repository_type(repository_path):
    # return 'github'
    git_config = os.path.join(repository_path, '.git', 'config')
    hg_config = os.path.join(repository_path, '.hg', 'hgrc')

    try:
        if os.path.exists(git_config):
            return 'github'
        elif os.path.exists(hg_config):
            return 'mercurial'
        else:
            return 'unknown'
    except:
        return 'error_repor_type'
    
def fn_get_source_code_by_commit(repo_path, commit_sha, file_path):
    commit = git.Repo(repo_path).commit(commit_sha)

    # Access the file content using the file path at the specified commit
    file_blob = commit.tree[file_path] 

    output_commit_sha =  commit.hexsha
    output_commit_author = commit.author.name
    output_commit_email = commit.author.email    
    output_commit_date = commit.authored_datetime
    output_commit_timezone = commit.author_tz_offset
    output_commit_message = commit.message
    try:
        output_commit_sourcecode = file_blob.data_stream.read().decode()
    except:
        output_commit_sourcecode = file_blob.data_stream.read().decode('windows-1252')
        # print("Error: ", repo_path, commit_sha, file_path)
    output_commit_filepath = commit.tree[file_path].path
    output_commit_filename = commit.tree[file_path].name

    # Access the file content using the file path at the specified commit
    file_blob = commit.tree[file_path]        
    # file_content = file_blob.data_stream.read().decode()

    return {
        "commit_sha": output_commit_sha,
        "commit_author": output_commit_author,
        "commit_author_email": output_commit_email,
        "commit_author_date": output_commit_date,
        "commit_author_timezone": output_commit_timezone,
        "commit_message": output_commit_message,
        "commit_filepath": output_commit_filepath,
        "commit_filename": output_commit_filename,
        "commit_sourcecode": output_commit_sourcecode
    }



def fn_formalise_method_name6(input_str, input_filename):
    # check whether file having a class structure or not. if not using the file name as the class name
    if '::' not in input_str:
        input_str = fn_get_class_name(input_filename) + '::' + input_str

    pattern = r'([\w.]+)(::)?(.+)\s*\((.*?)\)'

    # Find the class name, method name, and parameters using the regular expression
    match = re.match(pattern, input_str)

    # if input_str == 'getWritableType()':
    #     return 'ColumnType.getWritableType()'

    if match:
        # if sCase == ".":
        #     has_separator = match.group(2) == '.'
        # else:
        has_separator = match.group(2) == '::'
        class_name = match.group(1) if has_separator else ''
        method_name = match.group(3)
        parameters = match.group(4)
        # print(parameters)
        # print()

        if parameters:
            # Extract parameter data types and remove extra spaces
            # parameter_data_types = [param.strip().split()[0] for param in parameters.split(',')]
            parameter_data_types = []
            for param in parameters.split(','):
                if bool(re.match('\w+\[\]\w+', param)): # fix the case: byte[]binaryData to byte[] binaryData
                    param = param.replace('[]', '[] ')
                c = param.strip().split()
                if c[0] == 'final':
                    sub_param_type = c[1].strip()
                else:
                    sub_param_type = c[0].strip()
                if '[]' in c[-1]:
                    sub_param_type = ''.join([sub_param_type, '[]'])
                    sub_param_type = sub_param_type.replace('[][]', '[]')
                    sub_param_type = sub_param_type.replace('[]>[]', '[]>')
                parameter_data_types.append(sub_param_type)
        else:
            parameter_data_types = []

        # Construct the formalized method string
        if len(parameter_data_types) > 0:
            formalized_method = f"{class_name.replace('::', '.')}.{method_name.replace('::', '.')}({', '.join(parameter_data_types)})"
        else:
            formalized_method = f"{class_name.replace('::', '.')}.{method_name.replace('::', '.')}()"
        
        # special cases when a method in SeSaMe dataset having a filename as a prefix
        formalized_method = fn_special_cases(formalized_method)

        return formalized_method
    else:
        return None

def fn_export_method_from_source_code(file_path, source_code, function_name):
    # Analyze the Python file using lizard
    analyzed_file = lizard.analyze_file.analyze_source_code(file_path, source_code)

    # Find the method with the specified name
    for function in analyzed_file.function_list:        
        if fn_formalise_method_name6(fn_fix_method_str(function.long_name), file_path) == function_name:
            # Get the start and end lines of the method
            start_line, end_line = function.start_line, function.end_line

            # Read the content of the source code version
            # lines = a_source_code.splitlines(True) # error found on Sep 6, 2023 ~ weird behaviour of splitlines() function
            lines = source_code.split("\n")            

            # Extract the source code of the method
            # method_source_code = "".join(lines[start_line-1:end_line])
            method_source_code = "\n".join(lines[start_line-1:end_line])

            return method_source_code

    return None

def fn_days_between_dates(date1, date2):
    # Get the date part from datetime objects
    date1_date = date1.date()
    date2_date = date2.date()

    # Calculate the difference between the two date objects
    delta = date2_date - date1_date

    # Extract the number of days from the timedelta object
    num_days = delta.days

    return num_days

def fn_export_methods_source_overloading(source_code, methods):
    code_snipets = []
    lines = source_code.split("\n")
    for method in methods:
        start_line, end_line = method[2], method[3]
        method_source_code = "\n".join(lines[start_line-1:end_line])
        code_snipets.append(method_source_code)

    return "\n".join(code_snipets)

def fn_get_method_name_only(method_name):
    # Use regular expression to remove parameters
    method_name_without_params = re.sub(r'\(.*\)', '', method_name)

    return method_name_without_params

def fn_get_method_overloading(file_path, source_code, function_name):
    # Analyze the Python file using lizard    
    analyzed_file = lizard.analyze_file.analyze_source_code(file_path, source_code)

    methods = analyzed_file.function_list

    met_list = [(m.name.replace("::", "."), m.long_name, m.start_line, m.end_line) for m in methods]
    
    extracted_methods = []

    for met in met_list:
        if met[0] ==  fn_get_method_name_only(function_name):
            extracted_methods.append(met)

    s_output = fn_export_methods_source_overloading(source_code, extracted_methods)

    return s_output

# =======================================

def fn_gather_version_history(inp_repo_path, inp_file_name, inp_method_name, data_item):
    # initiate 
    method_versions = []

    repository_type = fn_detect_repository_type(inp_repo_path)

    if repository_type == 'mercurial':
        # do nothing
        print("mercurial, do nothing")
        #=====================================================================================
    elif repository_type == 'github':        
        commits = Repository(inp_repo_path, order='reverse')
        # commits = Repository(a_repository_path, order='topo-order')
        
        tmp = [commit for commit in commits.traverse_commits()]
        # print(len(tmp))
        commits = sorted(tmp, key=lambda x: x.author_date, reverse=True)
        # print(type(commits))

        
        count_versions = 0

        prev_method_project = ""
        prev_method_file = ""
        prev_method_methodname = ""

        prev_method_version_no = 0
        prev_method_sha = ""
        prev_method_author = ""
        prev_method_author_email = ""
        prev_method_message = ""
        prev_method_date = None
        prev_method_content = ""
        # prev_method_days_to_prev_version = 0
        # prev_method_days_to_latest_version = 0

        flg_method_disappeared = False

        # Iterate over all modifications in the repository
        first_exist = ""
        checkpoint_date = None
        i = 0
        for commit in commits:
            if flg_method_disappeared == True:
                break
            
            if i == 0:
                # print("at checkpoint")
                checkpoint_date = commit.author_date
                    
            commit_id = commit.hash
            # print("-->", commit.author_date, commit_id)            

            try:
                objcommit = fn_get_source_code_by_commit(inp_repo_path, commit_id, inp_file_name)                
                
                file_path = objcommit['commit_filepath'] # file path

                # Convert file_path to string
                file_path_str = "/".join([inp_repo_path, str(file_path)]) # actual file path
                # print("Actual file path", file_path_str)

                source_code = objcommit['commit_sourcecode']
                
                # using lizard API to get the objects containing all methods
                # a filename is required to be passed for programming language detection based on extention
                parsed_source = lizard.analyze_file.analyze_source_code(inp_file_name, source_code)

                if len(parsed_source.function_list) > 0:
                    # print("====>", [fn_fix_method_str(item.long_name) for item in parsed_source.function_list])
                    methods_list = [fn_formalise_method_name6(fn_fix_method_str(item.long_name), file_path) for item in parsed_source.function_list]                    
                    if inp_method_name not in methods_list:
                        flg_method_disappeared = True
                    else:                        
                        methods = parsed_source.function_list
                        for m in methods:
                            # standardise method name from lizard to be similar to the one in the dataset
                            # e.g. Class1::Method1( int Param1) --> Class1.Method1(int Param1)
                            # print("--->", m.long_name, "\n", fn_formalise_method_name4(fn_fix_method_str(m.long_name)), "\n", inp_method_name)
                            formalised_method_name = fn_formalise_method_name6(fn_fix_method_str(m.long_name), file_path)

                            if formalised_method_name == inp_method_name: # method found                                
                                new_method_content = fn_export_method_from_source_code(file_path, 
                                                                                    source_code, 
                                                                                    formalised_method_name)

                                if new_method_content != prev_method_content:                                     
                                    # print("----start of method-----------------------")                                    
                                    
                                    prev_method_content = new_method_content

                                    new_method_date = objcommit['commit_author_date']
                                    # no_of_days = 0

                                    # if new_method_date != prev_method_date:
                                    #     if prev_method_date is not None:
                                    #         no_of_days = fn_days_between_dates(new_method_date, prev_method_date)
                                    #         prev_method_days_to_prev_version = 0
                                        
                                    #     prev_method_date = new_method_date

                                    dict_modified_methods = {}

                                    if i == 0:
                                        prev_method_sha = commit_id
                                        prev_method_author = objcommit['commit_author']
                                        prev_method_author_email = objcommit['commit_author_email']
                                        prev_method_message = objcommit['commit_message']

                                        dict_modified_methods['commit_version_no'] = count_versions
                                        
                                        dict_modified_methods['commit_project'] = data_item['project']
                                        dict_modified_methods['commit_file'] = data_item['file']
                                        dict_modified_methods['commit_method'] = data_item['method']
                                        
                                        dict_modified_methods['commit_sha'] = prev_method_sha
                                        dict_modified_methods['commit_author'] = prev_method_author
                                        dict_modified_methods['commit_author_email'] = prev_method_author_email
                                        dict_modified_methods['commit_message'] = prev_method_message
                                        dict_modified_methods['commit_date'] = objcommit['commit_author_date'] 
                                        dict_modified_methods['commit_source_code'] = prev_method_content
                                        # method overloading
                                        dict_modified_methods['commit_source_method_overloading'] = fn_get_method_overloading(file_path, objcommit['commit_sourcecode'], inp_method_name)
                                        dict_modified_methods['commit_days_to_prev_version'] = 0
                                        dict_modified_methods['commit_days_to_latest_version'] = 0

                                        prev_method_date = objcommit['commit_author_date']

                                    else:
                                        prior_commit = commits[i-1]
                                        prior = fn_get_source_code_by_commit(inp_repo_path, prior_commit.hash, inp_file_name)

                                        dict_modified_methods['commit_version_no'] = count_versions
                                        
                                        dict_modified_methods['commit_project'] = data_item['project']
                                        dict_modified_methods['commit_file'] = data_item['file']
                                        dict_modified_methods['commit_method'] = data_item['method']

                                        dict_modified_methods['commit_sha'] = prior['commit_sha']
                                        dict_modified_methods['commit_author'] = prior['commit_author']
                                        dict_modified_methods['commit_author_email'] = prior['commit_author_email']
                                        dict_modified_methods['commit_message'] = prior['commit_message']
                                        dict_modified_methods['commit_date'] = prior['commit_author_date']
                                        dict_modified_methods['commit_source_code'] = fn_export_method_from_source_code(prior['commit_filepath'], prior['commit_sourcecode'], inp_method_name)
                                        # method overloading
                                        dict_modified_methods['commit_source_method_overloading'] = fn_get_method_overloading(prior['commit_filepath'], prior['commit_sourcecode'], inp_method_name)
                                        # dict_modified_methods['commit_days_to_prev_version'] = no_of_days
                                        dict_modified_methods['commit_days_to_prev_version'] = fn_days_between_dates(prior['commit_author_date'], prev_method_date)
                                        dict_modified_methods['commit_days_to_latest_version'] = fn_days_between_dates(prior['commit_author_date'], checkpoint_date)                                       
                                        
                                        prev_method_date = prior['commit_author_date']

                                    method_versions.append(dict_modified_methods)

                                    count_versions += 1
                                    
                                    prev_method_version_no = count_versions

                                    prev_method_project = data_item['project']
                                    prev_method_file =  data_item['file']
                                    prev_method_methodname = data_item['method']

                                    prev_method_sha = commit_id
                                    prev_method_author = objcommit['commit_author']
                                    prev_method_author_email = objcommit['commit_author_email']
                                    prev_method_message = objcommit['commit_message']

                                    # prev_method_date = objcommit['commit_author_date']
                                    # prev_method_content = new_method_content

                                    # print("version: ", count_versions, "-days to previous version: ", no_of_days)
                                    # print(dict_modified_methods)
                                    # print("----end of method-----------------------")                                
                                    # print("========================")                                    
                                # else:                                    
                                #     # print("-->\n", "method is unchanged")
                                #     # print(new_method_content)
                                #     new_method_date = objcommit['commit_author_date']
                                #     no_of_days = fn_days_between_dates(new_method_date, prev_method_date)
                                #     prev_method_days_to_prev_version = no_of_days
                                #     prev_method_days_to_latest_version = fn_days_between_dates(new_method_date, checkpoint_date)
                                #     # print("----end of a file-----------------------")                                    
                                break # method found --> break for
                else:
                    flg_method_disappeared = True
                    first_exist = commits[i-1]
                    print("Log::No methods found in the file.", commit_id, file_path)
                    break # file found but method not found  --> break for
            except KeyError:
                flg_method_disappeared = True
                first_exist = commits[i-1]
                print(f"Log::Scan completed::{commit_id}, {inp_repo_path}, {inp_file_name}")
                break

            i += 1

        # finish for loop, check whether file and method might dissapeared
        # if True, add the last version of the method to the version history, re-calculate the number of days of its existence
        if flg_method_disappeared == True:
            # if prev_method_sha not in [k['commit_sha'] for k in method_versions]:

            dict_modified_methods = {}
            prior_commit = commits[i-2] # back to previous commit where the method still exists (since i +=1)
            
            prior = fn_get_source_code_by_commit(inp_repo_path, prior_commit.hash, inp_file_name)

            dict_modified_methods['commit_version_no'] = prev_method_version_no

            dict_modified_methods['commit_project'] = data_item['project']
            dict_modified_methods['commit_file'] = data_item['file']
            dict_modified_methods['commit_method'] = data_item['method']

            dict_modified_methods['commit_sha'] = prior['commit_sha']
            dict_modified_methods['commit_author'] = prior['commit_author']
            dict_modified_methods['commit_author_email'] = prior['commit_author_email']
            dict_modified_methods['commit_message'] = prior['commit_message']
            dict_modified_methods['commit_date'] = prior['commit_author_date']
            dict_modified_methods['commit_source_code'] = fn_export_method_from_source_code(prior['commit_filepath'], prior['commit_sourcecode'], inp_method_name)
            # method overloading
            dict_modified_methods['commit_source_method_overloading'] = fn_get_method_overloading(prior['commit_filepath'], prior['commit_sourcecode'], inp_method_name)
            dict_modified_methods['commit_days_to_prev_version'] = fn_days_between_dates(prior['commit_author_date'], prev_method_date)
            dict_modified_methods['commit_days_to_latest_version'] = fn_days_between_dates(prior['commit_author_date'], checkpoint_date)
            
            method_versions.append(dict_modified_methods)
            
            # print("method disappeared!")
            # print(dict_modified_methods)
            count_versions += 1
        else:
            print("method exists during the whole lifetime of the project!")
            
            dict_modified_methods = {}

            prior_commit = commits[i-1] # back to previous commit where the method still exists (since i +=1)
            
            prior = fn_get_source_code_by_commit(inp_repo_path, prior_commit.hash, inp_file_name)

            dict_modified_methods['commit_version_no'] = prev_method_version_no

            dict_modified_methods['commit_project'] = data_item['project']
            dict_modified_methods['commit_file'] = data_item['file']
            dict_modified_methods['commit_method'] = data_item['method']

            dict_modified_methods['commit_sha'] = prior['commit_sha']
            dict_modified_methods['commit_author'] = prior['commit_author']
            dict_modified_methods['commit_author_email'] = prior['commit_author_email']
            dict_modified_methods['commit_message'] = prior['commit_message']
            dict_modified_methods['commit_date'] = prior['commit_author_date']
            dict_modified_methods['commit_source_code'] = fn_export_method_from_source_code(prior['commit_filepath'], prior['commit_sourcecode'], inp_method_name)
            # method overloading
            dict_modified_methods['commit_source_method_overloading'] = fn_get_method_overloading(prior['commit_filepath'], prior['commit_sourcecode'], inp_method_name)
            dict_modified_methods['commit_days_to_prev_version'] = fn_days_between_dates(prior['commit_author_date'], prev_method_date)
            dict_modified_methods['commit_days_to_latest_version'] = fn_days_between_dates(prior['commit_author_date'], checkpoint_date)
            
            method_versions.append(dict_modified_methods)
            
            # print("method disappeared!")
            # print(dict_modified_methods)
            count_versions += 1

        print("total number of version:", count_versions)
        return method_versions

    else:
        print('Unknown repo type')
        return None
    
# =================================

if __name__ == "__main__":
    # datetime object containing current date and time
    now = datetime.now()
    
    # print("now =", now)

    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print("date and time =", dt_string)


    SOURCE_DATA_PATH = 'sesame/src/'
    FILE_DATA = 'unique-pairs.csv'
    REPOS_DIR = 'sesame/src/repos'
    OUTPUT_DIR = 'mining/output/'

    df = pd.read_csv(SOURCE_DATA_PATH + FILE_DATA)
    # print(df.shape)

    code_items = df.to_dict(orient='records')
    # print(len(code_items))
    
    # --------------------------------------------
    sesame_item = {}

    output_file = 'version_history_mining.json'

    for i, code_item in enumerate(code_items):        
        # if i >= 0:
        # if i >= 0 and i <= 600 :
        if i in [231,302,312,316,356,359,365,544,682,1080]:
            try:
                print(i, "|", code_item['project'], "|", code_item['file_and_method'])
                existing_items = []
                try:
                    existing_text = open(OUTPUT_DIR + output_file, 'r').read()
                    existing_items = json.loads(existing_text)
                    # print("Warning::Existing output file have some data")
                    print("total existing:", len(existing_items))
                except:
                    print("Warning::Existing output file have no data")
                    existing_items = []

                in_repo_path = "/".join([REPOS_DIR, code_item['project']])
                in_file_name = "/".join(code_item['file'].split('/')[2:])
                in_method_name = code_item['method']

                vh = fn_gather_version_history(in_repo_path, in_file_name, in_method_name, code_item)
                
                # print(vh)
                sesame_item = code_item
                if "version_history" not in sesame_item.keys():
                    sesame_item['version_history'] = vh

                # print(sesame_item)

                existing_items.extend([sesame_item])
                
                with open(OUTPUT_DIR + output_file, 'w') as f:
                    json.dump(existing_items, indent=4, default=str, fp=f)
                #     f.close()

                print("-----------------------------")
            except:
                print("Error::", i, "|", code_item['project'], "|", code_item['file_and_method'])
                with open(OUTPUT_DIR + "log.txt", 'a') as f:
                    f.write(str(i))
                    f.write("\n")
                    f.close()
                print("----------------------------")
        else:
            continue