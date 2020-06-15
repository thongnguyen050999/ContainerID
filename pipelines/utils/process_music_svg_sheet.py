import os, json, cairosvg,time
from PyPDF2 import PdfFileMerger
from natsort import natsorted
import pandas as pd

def merge_all_pdf_files(root, output_folder):
    '''
    From the pdf list, merge all into one pdf file,
    Then export to parent folder
    arg:
        1-dimentional vector with .pdf file name
        list = [song#1_directory/score_0.pdf, song#1_directory/score_1.pdf, ...]
    '''
    scores_list = find_all_file_available(root, ".pdf")
    os.system('mkdir -p ' + output_folder)

    total_dictionary = {}
    index = 0

    for pdf_paths in scores_list:
        merger = PdfFileMerger()
        for pdf in pdf_paths:
            merger.append(pdf)

        parent = os.path.dirname(pdf_paths[0])
        file_name = os.path.basename(parent)
        print('Merge files in directory: ' + parent)

        '''Find score info json'''
        info_file_path = os.path.join(parent, 'info.json')
        dictionary = None

        if not os.path.isfile(info_file_path):
            continue

        with open(info_file_path, 'r') as fp:
            dictionary = json.load(fp)

        total_dictionary[str(index)] = dictionary
        destination_path = os.path.join(output_folder, str(index) + ".pdf")

        index = index + 1
        merger.write(destination_path)
        print('Output into: ' + destination_path)

        '''Export to nesscessary file'''
        info_output_path = os.path.join(output_folder, 'info')
        try:
            with open(info_output_path + '.json', 'w') as fp:
                json.dump(total_dictionary, fp)
            print('Saved dictionary file')
        except:
            print('Cant save dict to json')

        save_dict_to_excel(total_dictionary, info_output_path + '.xls')
        save_download_dict_to_excel({}, os.path.join(output_folder, "downloaded.xls"), os.path.join(root, "downloaded.json"))
        save_download_dict_to_excel({}, os.path.join(output_folder, "downloaded_fail.xls"), os.path.join(root, "download_fail_scores.json"))

        merger.close()

def save_dict_to_excel(dictionary, output_file, input_file=None):
    '''
        Use to convert info.json in output folder
    '''
    if input_file is not None:
        if os.path.isfile(input_file):
            with open(input_file, 'r') as fp:
                dictionary = json.load(fp)

    if dictionary == {}:
        print('ERROR: Blank dictionary')
        return

    excel_dictionary = {}
    atts = ['Name', 'URL']
    for index in range(len(dictionary)):
        old_dict = dictionary[str(index)]
        new_dict = {}
        for att in atts:
            new_dict[att] = old_dict[att]
        
        new_dict['Genre'] = ''
        excel_dictionary[str(index)] = new_dict

    df = pd.DataFrame(excel_dictionary).T
    df.to_excel(output_file)

def save_download_dict_to_excel(dictionary, output_file, input_file=None):
    '''
        Use to convert download_fail_scores.json or downloaded.json to excel file
    '''
    if input_file is not None:
        if os.path.isfile(input_file):
            with open(input_file, 'r') as fp:
                dictionary = json.load(fp)

    if dictionary == {}:
        print('ERROR: Blank dictionary')
        return

    excel_dictionary = {}
    for index in range(len(dictionary)):
        new_dict = {}
        new_dict['Name'] = list(dictionary)[index]
        new_dict['URL'] = list(dictionary.values())[index]
        excel_dictionary[str(index)] = new_dict

    df = pd.DataFrame(excel_dictionary).T
    try:
        df.to_excel(output_file)
    except:
        print('ERROR: File is opening by another program')

def find_all_file_available(root_directory, extension):
    '''
    Find all available scores from root folder
    arg:
        root: Parent folder that store individual score folder
        root--
        --\song#1
        --\song#2
        --\...
    return:
        2-dimentional vector with each score folder and .svg file name
        list = [[song#1_directory/score_0.svg, song#1_directory/score_1.svg, ...],
                [song#2_directory/score_0.svg, song#2_directory/score_1.svg, ...],
                ...]
    '''
    scores_list = []
    #List all folder
    for folder in os.listdir(root_directory):
        if os.path.isdir(os.path.join(root_directory, folder)):
            files_path = []
            for file in os.listdir(os.path.join(root_directory, folder)):
                path = os.path.join(root_directory, folder, file)
                if path.endswith(extension):
                    files_path.append("{0}".format(path))

            if files_path !=[]:
                sorted_files_path = natsorted(files_path)
                scores_list.append(sorted_files_path)
                
    score_count = sum([len(score) for score in scores_list])
    print('Number of svg file: ' + str(score_count))

    return scores_list

def convert_all_avaialble_svg_to_pdf(root):
    '''
    From the svg list, convert all into pdf file
    arg:
        2-dimentional vector with each score folder and .svg file name
        list = [[song#1_directory/score_0.svg, song#1_directory/score_1.svg, ...],
                [song#2_directory/score_0.svg, song#2_directory/score_1.svg, ...],
                ...]
        
    return:
        2-dimentional vector with each score folder and .pdf file name
        list = [[song#1_directory/score_0.pdf, song#1_directory/score_1.pdf, ...],
                [song#2_directory/score_0.pdf, song#2_directory/score_1.pdf, ...],
                ...]
    '''
    score_pdf_paths = []
    scores_list = find_all_file_available(root, ".svg")
    score_count = sum([len(score) for score in scores_list])
    score_current_count = 0
    for score_folder in scores_list:
        paths = []

        parent = "{0}".format(os.path.dirname(score_folder[0]))
        json_flag_path = os.path.join(parent, "converted.json")

        if os.path.isfile(json_flag_path):
            continue

        for path in score_folder:
            filename_only = "{0}".format(os.path.basename(path).split('.')[0]) + ".pdf"
            save_path = os.path.join(parent, filename_only)
            print(save_path)
            try: 
                cairosvg.svg2pdf(
                    url=path, write_to=save_path)
                paths.append(save_path)
            except:
                print('Error converting')
                break

        score_current_count = score_current_count + len(paths)
        print('Convert svg to pdf. Progress: ' + str(round(score_current_count*100.0/score_count, 2))   +'%')

        '''Write a blank json file to mark whether all the file was converted'''
        with open(json_flag_path, 'w') as fp:
                json.dump({}, fp)


if __name__== "__main__":
    root = os.path.join("downloads", "scores")
    output_dir = os.path.join("downloads", "output")
    convert_all_avaialble_svg_to_pdf(root)
    merge_all_pdf_files(root, output_dir)