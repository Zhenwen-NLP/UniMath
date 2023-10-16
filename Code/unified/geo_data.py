import sacrebleu
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import pickle
import torch
import numpy as np
import os

from torch.utils.data.distributed import DistributedSampler
from tokenization import VLT5TokenizerFast

from data_utils import process_image, create_patch, process_english_text

project_dir = Path(__file__).resolve().parent.parent
workspace_dir = project_dir.parent

dataset_dir = workspace_dir.joinpath('datasets/').resolve()
geo_dir = dataset_dir.joinpath('UniGeo')


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

        for source in self.source:
            with open(geo_dir.joinpath(f'{source}.pk'), "rb") as f:
                dataset = pickle.load(f)
                for sample in dataset:
                    if 'calculation' in source:
                        problem_with_space = process_english_text(sample['English_problem'])
                        problem_with_space = 'Calculation: ' + problem_with_space
                        source_text_list.append(problem_with_space)

                        text_i = " ".join(sample["manual_program"])
                        target_text_list.append(text_i)

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
                        problem_with_space = sample['input_text']
                        problem_with_space = 'Proving: ' + problem_with_space

                        source_text_list.append(problem_with_space)

                        text_i = ' '.join(sample['proving_sequence'])
                        target_text_list.append(text_i)

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

        data = []
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
        map_list = {'∠': 'angle','△': 'triangle', '⟂': 'vertical', '≅':'congruent', '∥':'parallel', '~': 'similar', '\\frac':'frac'}
        
        datum = self.data[idx]

        ###### Image ######
        image = datum['image']
        out_dict['image'] = image
        boxes = create_patch(patch_num=7)
        boxes = torch.from_numpy(boxes)
        boxes.clamp_(min=0.0, max=1.0)
        n_boxes = len(boxes)

        out_dict['n_boxes'] = n_boxes
        out_dict['boxes'] = boxes[:n_boxes]


        input_text = datum['source_text']
        for s, t in map_list.items():
            input_text = input_text.replace(s, t)

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

        target_text = datum['target_text']
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
            
            print('output',target_text)
            print('re_output',self.tokenizer.convert_ids_to_tokens(torch.LongTensor(target_ids)))


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

            vis_attention_mask[i, :n_boxes] = 1
            image_lists.append(entry['image'])

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
            num_workers=workers, pin_memory=True, sampler=train_sampler,
            collate_fn=dataset.collate_fn)
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=True,
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
