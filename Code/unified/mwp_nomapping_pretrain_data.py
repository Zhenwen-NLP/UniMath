import random
import preprocess
import sacrebleu
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import pickle
import torch
import numpy as np
import os
import pandas as pd
import json
from torch.utils.data.distributed import DistributedSampler
from tokenization import VLT5TokenizerFast
import nltk
from data_utils import process_image, create_patch, process_english_text, get_input_tablemwp, transfer_num_no_tokenize, load_raw_data, table_loader, transfer_num_keep_number
import re
import pandas as pd
from copy import deepcopy
import gc

project_dir = Path(__file__).resolve().parent.parent
workspace_dir = project_dir.parent

dataset_dir = workspace_dir.joinpath('datasets/').resolve()
geo_dir = dataset_dir.joinpath('UniGeo')
mwp_dir = dataset_dir.joinpath('MWP')
tab_dir = dataset_dir.joinpath('tab')



class GeoDataset(Dataset):
    def __init__(self, split='train', raw_dataset=None, verbose=True, args=None, mode='train'):
        super().__init__()

        self.raw_dataset = raw_dataset
        self.verbose = verbose
        self.args = args
        self.mode = mode

        # Loading datasets to data
        self.source = split.split(',')
        if self.verbose:
            print('Data source: ', self.source)

        if self.args.tokenizer is None:
            self.args.tokenizer = self.args.backbone

        self.tokenizer = VLT5TokenizerFast.from_pretrained(
            args.backbone,
            do_lower_case=self.args.do_lower_case)

        sub_dict_path = os.path.join(geo_dir, "sub_dataset_dict.pk")  # problems type
        with open(sub_dict_path, 'rb') as file:
            subset_dict = pickle.load(file)
        self.subset_dict = subset_dict

        # geo dataset
        target_text_list = []
        source_text_list = []
        image_list = []

        source_nums_list = []
        choice_nums_list = []
        label_list = []

        problem_form_list = []
        problem_type_list = []
        data = []
        if 'calculation' in str(self.source):
            ##########################MWP DATA###############################
            train_data, test_data = load_raw_data(os.path.join(mwp_dir, "train.csv"), os.path.join(mwp_dir, "test.csv"))
            if 'train' in split:
                mwp_pairs, _, _, _ = transfer_num_keep_number(train_data, test_data, False)
            else:
                _, mwp_pairs, _, _ = transfer_num_keep_number(train_data, test_data, False)
            
            for pair in mwp_pairs:
                r = random.random()
                if r > self.args.r_threshold:
                    datum = {
                        'image': None,
                        'source_text': deepcopy('Math word problem solving with number mapping: ' + ' '.join(pair[0])),
                        'target_text': deepcopy(' '.join(pair[1])),
                        'source_nums': deepcopy(pair[2]),
                        'choice_nums': None,
                        'label': deepcopy(' '.join(pair[1])),
                        'problem_form': 'mwp',
                        'problem_type': None
                    }
                    data.append(datum)
                else:
                    source_text, target_text = preprocess.corrupt_bart(
                            deepcopy(' '.join(pair[0])) + deepcopy(' '.join(pair[1])), mask_ratio=self.args.word_mask_rate, prefix='Denoise math word problem solving with number mapping: ')
                    datum = {
                        'image': None,
                        'source_text': source_text,
                        'target_text': target_text,
                        'source_nums': deepcopy(pair[2]),
                        'choice_nums': None,
                        'label': deepcopy(' '.join(pair[1])),
                        'problem_form': 'mwp',
                        'problem_type': None
                    }
                    data.append(datum)
            del mwp_pairs
            gc.collect()
            print(len(data),' MWPs are included.')
        
            ##########################TABLE DATA###############################
            
            problems, pids = table_loader(tab_dir, split)
            print(len(pids),' TabMWPs are included.')
            for pid in pids:
                r = random.random()
                if r > self.args.r_threshold:
                    datum = {
                        'image': None,
                        'source_text': 'Math word problem solving without mapping: ' + deepcopy(get_input_tablemwp(problems[pid])),
                        'target_text': deepcopy(problems[pid]['answer']),
                        'source_nums': deepcopy(problems[pid]['unit']),
                        'choice_nums': deepcopy(problems[pid]['choices']),
                        'label': deepcopy(problems[pid]['answer']),
                        'problem_form': 'table_mwp',
                        'problem_type': None
                    }
                    data.append(datum)
                else:
                    source_text, target_text = preprocess.corrupt_bart(
                           deepcopy(get_input_tablemwp(problems[pid])) + 'Chain of Thought:' + deepcopy(problems[pid]['solution']) + 'Answer:' + deepcopy(problems[pid]['answer']), 
                           mask_ratio=self.args.word_mask_rate, prefix='Denoise math word problem solving without mapping: ')
                    datum = {
                        'image': None,
                        'source_text': source_text,
                        'target_text': target_text,
                        'source_nums': deepcopy(pair[2]),
                        'choice_nums': None,
                        'label': deepcopy(' '.join(pair[1])),
                        'problem_form': 'table_mwp',
                        'problem_type': None
                    }
                    data.append(datum)
            del problems, pids
            gc.collect()

        for source in self.source:
            with open(geo_dir.joinpath(f'{source}.pk'), "rb") as f:
                dataset = pickle.load(f)
                for sample in dataset:
                    r = random.random()
                    if 'calculation' in source:
                        
                        if r > self.args.r_threshold:
                            problem_with_space = process_english_text(sample['English_problem'])
                            problem_with_space = 'Geometry calculation with number mapping: ' + problem_with_space
                            source_text_list.append(problem_with_space)

                            text_i = " ".join(sample["manual_program"])
                            target_text_list.append(text_i)
                        else:
                            problem_with_space = process_english_text(sample['English_problem'])
                            problem_with_space = problem_with_space
                            text_i = " ".join(sample["manual_program"])
                            source_text, target_text = preprocess.corrupt_bart(
                                problem_with_space + text_i, mask_ratio=self.args.word_mask_rate, prefix='Denoise geometry calculation with number mapping: ')
                            source_text_list.append(source_text)
                            target_text_list.append(target_text)

                        image = sample['image']
                        image = process_image(image)
                        img_rgb = np.zeros((3, image.shape[0], image.shape[1]))
                        for i in range(3):
                            img_rgb[i, :, :] = image
                        image_list.append(img_rgb)

                        source_nums_list.append(sample["numbers"])
                        choice_nums_list.append(sample["choice_nums"])
                        label_list.append(sample["label"])

                        problem_form_list.append('calculation')
                        type = self.subset_dict[sample['id']]
                        problem_type_list.append(type)

                    else:
                        assert 'proving' in source
                        if r > self.args.r_threshold:
                            problem_with_space = sample['input_text']
                            problem_with_space = 'Geometry proving with mapping: ' + problem_with_space
                            source_text_list.append(problem_with_space)

                            text_i = " ".join(sample['proving_sequence'])
                            target_text_list.append(text_i)
                        else:
                            problem_with_space = sample['input_text']
                            problem_with_space = problem_with_space
                            text_i = " ".join(sample['proving_sequence'])
                            source_text, target_text = preprocess.corrupt_bart(
                                problem_with_space + text_i, mask_ratio=self.args.word_mask_rate, prefix='Denoise geometry proving with mapping: ')
                            source_text_list.append(source_text)
                            target_text_list.append(target_text)


                        image = sample['img']
                        image = process_image(image)
                        image = image.transpose(2, 0, 1)
                        image_list.append(image)

                        source_nums_list.append(None)
                        choice_nums_list.append(None)
                        label_list.append(None)

                        problem_form_list.append('proving')
                        problem_type_list.append(sample['problem_type'])


        assert len(source_text_list) == len(target_text_list)

        
        for source_text, target_text, image, source_nums, choice_nums, label, problem_form, problem_type in \
                zip(source_text_list, target_text_list, image_list, source_nums_list, choice_nums_list, label_list, problem_form_list, problem_type_list):
            datum = {
                'image': image,
                'source_text': source_text.strip(),
                'target_text': target_text.strip(),
                'source_nums': source_nums,
                'choice_nums': choice_nums,
                'label': label,
                'problem_form': problem_form,
                'problem_type': problem_type
            }
            data.append(datum)

        if self.verbose:
            print(f"Loaded {len(data)} data from", split)

        self.n_gpus = torch.cuda.device_count()

        self.data = data

        if self.verbose:
            print("# all sentences:", len(self.data))
        



    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        out_dict = {}
        out_dict['args'] = self.args
        map_list = {'≠': 'not-equal', '≈': 'approximate', '△': 'triangle', '∠': 'angle', '∥': 'parallel',
                   '⊙': 'circle', '☉': 'circle', '⊥': 'perpendicular', '⟂': 'perpendicular', '≌': 'congruent', '≅': 'congruent',
                   '▱': 'parallelogram', '∽': 'similar', '⁀': 'arc', '⌒': 'arc', '\\frac':'frac', 'N_':'N', 'V_':'V', 'C_':'C', 'E_':'E', 'R_':'R'}
        #map_list_tar = {'+': 'cal_add','-': 'cal_minus', '*': 'cal_multiply', '/':'cal_divide'}
        map_list_tar = {'+': 'add','-': 'minus', '*': 'multiply', '/':'divide'}
        map_list_geo = {'g_equal': 'equal', 'g_double': 'double', 'g_half': 'half', 'g_add': 'add', 'g_minus': 'minus',
          'g_sin': 'sin', 'g_cos': 'cos', 'g_tan': 'tan', 'g_asin': 'asin', 'g_acos': 'acos',
          'gougu_add': 'g_add', 'gougu_minus': 'g_minus', 'g_bili': 'bili',
          'g_mul': 'multiply', 'g_divide': 'divide', 'cal_circle_area': 'area', 'cal_circle_perimeter': 'perimeter', 'cal_cone': 'cone'}
        #map_list_geo = {'g_equal': 'cal_equal', 'g_double': 'cal_double', 'g_half': 'cal_half', 'g_add': 'cal_add', 'g_minus': 'cal_minus',
        #  'g_sin': 'cal_sin', 'g_cos': 'cal_cos', 'g_tan': 'cal_tan', 'g_asin': 'cal_asin', 'g_acos': 'cal_acos',
        #  'gougu_add': 'g_add', 'gougu_minus': 'g_minus', 'g_bili': 'cal_bili',
        #  'g_mul': 'cal_multiply', 'g_divide': 'cal_divide', 'cal_circle_area': 'cal_area', 'cal_circle_perimeter': 'cal_perimeter', 'cal_cone': 'cal_cone'}
        datum = self.data[idx]

        ###### Image ######
        image = datum['image']
        out_dict['image'] = image
        boxes = create_patch(patch_num=7)
        boxes = torch.from_numpy(boxes)
        if 'mwp' in datum["problem_form"]:
            boxes = torch.zeros_like(boxes)
        boxes.clamp_(min=0.0, max=1.0)
        n_boxes = len(boxes)

        out_dict['n_boxes'] = n_boxes
        out_dict['boxes'] = boxes[:n_boxes]

        input_text = datum['source_text']
        #pattern = re.compile(r'N_\d+')
        #input_text = re.sub(pattern, 'num', input_text)
        target_text = datum['target_text']

        for s, t in map_list.items():
            input_text = input_text.replace(s, t)
            target_text = target_text.replace(s, t)
        
        if 'mwp' in datum["problem_form"]:
            for s, t in map_list_tar.items():
                pass
                #target_text = target_text.replace(s, t)
        else:
            for s, t in map_list_geo.items():
                target_text = target_text.replace(s, t)

        if 't5' in self.args.tokenizer:
            input_ids = self.tokenizer.encode(
                input_text,
                max_length=self.args.max_text_length, truncation=True)
        else:
            input_ids = self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.tokenize(input_text)[:self.args.max_text_length - 1] + ['[SEP]'])

        
        out_dict['input_text'] = input_text
        out_dict['input_ids'] = torch.LongTensor(input_ids)
        out_dict['input_length'] = len(input_ids)

        
        if 't5' in self.args.tokenizer:
            target_ids = self.tokenizer.encode(
                target_text, max_length=self.args.gen_max_length, truncation=True)
        else:
            target_ids = self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.tokenize(target_text)[:self.args.gen_max_length - 1] + ['[SEP]'])

        assert len(target_ids) <= self.args.gen_max_length, len(target_ids)
        out_dict['target_text'] = target_text
        out_dict['target_ids'] = torch.LongTensor(target_ids)
        out_dict['target_length'] = len(target_ids)

        out_dict['choice_nums'] = datum["choice_nums"]
        out_dict['source_nums'] = datum["source_nums"]
        out_dict['label'] = datum["label"]

        out_dict['problem_form'] = datum["problem_form"]
        out_dict['problem_type'] = datum["problem_type"]

        #if '<unk>' in self.tokenizer.convert_ids_to_tokens(torch.LongTensor(input_ids)):
        if 0:
            print('input',input_text)
            print('re_input',self.tokenizer.convert_ids_to_tokens(torch.LongTensor(input_ids)))
            
            #print('output',target_text)
            #print('re_output',self.tokenizer.convert_ids_to_tokens(torch.LongTensor(target_ids)))


        return out_dict


    def collate_fn(self, batch):
        batch_entry = {}
        B = len(batch)

        S_W_L = max(entry['input_length'] for entry in batch)
        input_ids = torch.ones(B, S_W_L, dtype=torch.long) * self.tokenizer.pad_token_id

        V_L = max(entry['n_boxes'] for entry in batch)
        boxes = torch.zeros(B, V_L, 4, dtype=torch.float)
        vis_attention_mask = torch.zeros(B, V_L, dtype=torch.float)

        if 'target_ids' in batch[0]:
            T_W_L = max(entry['target_length'] for entry in batch)
            target_ids = torch.ones(B, T_W_L, dtype=torch.long) * self.tokenizer.pad_token_id

        img_paths = []
        input_text = []
        target_text = []
        image_lists = []

        source_nums_list = []
        choice_nums_list = []
        label_list = []

        problem_form_list = []
        problem_type_list = []

        for i, entry in enumerate(batch):
            input_ids[i, :entry['input_length']] = entry['input_ids']

            n_boxes = entry['n_boxes']
            boxes[i, :n_boxes] = entry['boxes']
            if 'mwp' in entry["problem_form"]:
                vis_attention_mask[i, :n_boxes] = 0
            else:
                vis_attention_mask[i, :n_boxes] = 1
            if entry['image'] is not None:
                image_lists.append(entry['image'])
            else:
                image_lists.append(np.zeros((3, 224, 224)))

            if 'target_ids' in entry:
                target_ids[i, :entry['target_length']] = entry['target_ids']

            if 'input_text' in entry:
                input_text.append(entry['input_text'])

            if 'target_text' in entry:
                target_text.append(entry['target_text'])

            source_nums_list.append(entry["source_nums"])
            choice_nums_list.append(entry["choice_nums"])
            label_list.append(entry["label"])
            problem_form_list.append(entry['problem_form'])
            problem_type_list.append(entry['problem_type'])


        batch_entry['input_ids'] = input_ids
        if 'target_ids' in batch[0]:
            word_mask = target_ids != self.tokenizer.pad_token_id
            target_ids[~word_mask] = -100
            batch_entry['target_ids'] = target_ids

        batch_entry['boxes'] = boxes
        batch_entry['vis_attention_mask'] = vis_attention_mask
        batch_entry['image_list'] = torch.Tensor(np.array(image_lists))
        batch_entry['img_paths'] = img_paths

        batch_entry['input_text'] = input_text
        batch_entry['target_text'] = target_text

        batch_entry['source_nums'] = source_nums_list
        batch_entry['choice_nums'] = choice_nums_list
        batch_entry['label'] = label_list

        batch_entry['problem_form'] = problem_form_list
        batch_entry['problem_type'] = problem_type_list

        batch_entry['task'] = 'geo'

        return batch_entry


def get_loader(args, split='train', mode='train',
               batch_size=32, workers=4, distributed=False, gpu=0,
               ):

    verbose = True

    dataset = GeoDataset(
        split,
        verbose=verbose,
        args=args,
        mode=mode)

    if distributed and mode == 'train':
        train_sampler = DistributedSampler(dataset)

    else:
        train_sampler = None
    if mode == 'train':
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=(train_sampler is None),
            num_workers=workers, pin_memory=False, sampler=train_sampler,
            collate_fn=dataset.collate_fn)
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=False,
            sampler=None,
            collate_fn=dataset.collate_fn,
            drop_last=False)

    if verbose:
        loader.evaluator = GeoEvaluator()

    loader.task = 'geo'

    return loader


class GeoEvaluator:
    def __init__(self):
        pass

    def evaluate(self, predicts, answers):

        try:
            bleu = sacrebleu.corpus_bleu(predicts, answers,
                                     lowercase=True)
        except EOFError:
            print('# preds', len(predicts))
            print('# tgts', len(answers))
            exit()
        return {
            'BLEU': bleu.score
        }
