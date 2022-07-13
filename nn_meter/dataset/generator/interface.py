# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

def generate_dataset(model_faimly_list, num_of_samples=2000, output_folder=None, output_tflite_folder='', num_of_thread=12, input_nasbench201_descriptor=''):

    # nasbench201_descriptor = json.loads(open(args.input_nasbench201_descriptor, 'r').read())
    # nasbench201_acc_seq = sorted(nasbench201_descriptor, key = lambda x: nasbench201_descriptor[x]['acc'], reverse = True)
   
    # nasbench201_acc_seq = nasbench201_acc_seq[:args.num_of_samples]

    # #Pack args
    # build_tiny_net_args = []
    # for nasbench_keys in nasbench201_acc_seq:
    #     arch_str = nasbench201_descriptor[nasbench_keys]['config']['arch_str']
    #     pb_file_name = os.path.abspath(os.path.join(args.output_folder, 'nasbench201_%s.pb' % nasbench_keys))
    #     if args.output_tflite_folder !=  '':
    #         tflite_file_name = os.path.abspath(os.path.join(args.output_tflite_folder, 'nasbench201_%s.tflite' % nasbench_keys))
    #     else:
    #         tflite_file_name = ''
    #     build_tiny_net_args.append({'arch_str': arch_str, 'pb_file_name': pb_file_name, 'tflite_file_name': tflite_file_name})

    # if not os.path.isdir(args.output_folder):
    #     os.mkdir(args.output_folder)
   
    # if args.output_tflite_folder !=  '':
    #     if not os.path.isdir(args.output_tflite_folder):
    #         os.mkdir(args.output_tflite_folder)

    # with multiprocessing.Pool(processes = args.num_of_thread) as p:
    #     for _ in tqdm.tqdm(p.imap_unordered(build_tiny_net, build_tiny_net_args), total = len(build_tiny_net_args)):
    #             pass
    pass


