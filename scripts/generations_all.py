# -*- coding: utf-8 -*-
import subprocess
import os
import sys
import argparse

'''
generate results iteratively
'''
def parseArgs(argv):
    # Run parameters
    parser = argparse.ArgumentParser(description='Generate tokens from decoder models')
    parser.add_argument('--root_path', type=str, default = '/home/jliu/STELAWord/models/char_with',
                        help='model root path')
    parser.add_argument('--dict_path', type=str, default = '/home/jliu/STELAWord/data/preprocessed/EN',
                        help='dict root path')
    parser.add_argument('--data_path', type=str, default = '/scratch2/jliu/freq_bias_benchmark/data/generation/unprompted/sample_random/prompt',
                        help='reference transcript path')
    parser.add_argument('--out_path', type=str, default='/scratch2/jliu/freq_bias_benchmark/data/generation/unprompted/sample_random/generated/new',
                        help='output path')
    parser.add_argument('--gpu', type=bool, default = False,
                        help= 'whether to use gpu')
    parser.add_argument('--hour',  default = '800h',
                        help= 'whether to use gpu')

    return parser.parse_args(argv)

def run_command(command):
    subprocess.call(command, shell=True)

def main(argv):
    
    # Args parser
    args = parseArgs(argv)
    # loop the root folder path
    root_path = args.root_path
    dict_path = args.dict_path
    hours = args.hour
    data_path = args.data_path

    for chunk in os.listdir(root_path + '/' + hours):
            if os.path.exists(root_path + '/' + hours + '/' + chunk + '/checkpoints/checkpoint_best.pt'):
                if os.path.exists(dict_path + '/' + hours + '/' + chunk + '/bin_with/dict.txt'):
                    # loop over the folder
                    for file in os.listdir(data_path + '/' + hours):
                        generation_command = 'python generation.py --ModelPath /home/jliu/STELAWord/models/char_with/{hours_var}/{chunk_var} \
                              --DictPath /home/jliu/STELAWord/data/preprocessed/EN/{hours_var}/{chunk_var}/bin_with \
                              --DataPath {data_path_var}/{hours_name}/{file_name}\
                              --OutputPath {out_path_var}/{hours_var}/{chunk_var} \
                               --gpu {gpu_var}'.format(hours_var=hours,hours_name=hours[:-1],chunk_var=chunk, data_path_var = args.data_path
                              ,gpu_var=args.gpu,out_path_var=args.out_path,file_name=file)
                    run_command(generation_command)

if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)
    

