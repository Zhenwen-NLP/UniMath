import re
import cv2 as cv
import numpy as np
import torch
import pandas as pd
import nltk
import json
import os
from copy import deepcopy

def table_loader(tab_dir, split):
    if 'train' in split:
        table_split = 'train'
    else:
        table_split = 'test'
    with open(os.path.join(tab_dir, f'problems_{table_split}.json')) as table_file:
        problems = json.load(table_file)
    print("number of problems for %s:" % (table_split), len(problems))

    pids = list(problems.keys())
    print("number of problems for %s:" % (table_split), len(pids))
    return problems, pids

def get_input_tablemwp(problem, option_inds = ["A", "B", "C", "D", "E", "F"]):
    # table
    table_title = problem['table_title']
    table = problem['table']
    if table_title:
        table = table_title + "\n" + table

    # question
    question = problem['question']
    unit = problem['unit']
    choices = problem['choices']

    if unit:
        question = question + f" (Unit: {unit})"

    if choices:
        for i, c in enumerate(choices):
            question += f" ({option_inds[i]}) {c}"

    # final input
    text = table + "\n" + question
    text = text.replace("\n", " \\n ").strip()

    return text


def load_mathqa_data(filename):  # load the json data to list(dict()) for MathQA
    print("Reading lines...")
    f = open(filename, encoding="utf-8")
    js = ""
    data = []
    for i, s in enumerate(f):
        js += s
        i += 1
        if i % 5 == 0:  # every 5 line is a json
            data_d = json.loads(js)
            data.append(data_d)
            js = ""

    return data

def load_tabmwp_dataset(data_root, split, option_inds = ["A", "B", "C", "D", "E", "F"]):
    """
    Load the dataset.
    """
    # load the data entries/annotations
    problems = json.load(open(os.path.join(data_root, f'problems_{split}.json')))
    print("number of problems for %s:" % (split), len(problems))

    pids = list(problems.keys())
    print("number of problems for %s:" % (split), len(pids))

    entries = []
    for pid in pids:
        prob = {}
        prob['pid'] = pid
        # prob['problem'] = problems[pid]
        prob['input_text'] = get_input_tablemwp(problems[pid], option_inds)
        prob['answer'] = problems[pid]['answer']
        entries.append(prob)

    return entries


## Create PyTorch dataset

def process_image(img, min_side=224):
    # fill the diagram with a white background and resize it
    size = img.shape
    h, w = size[0], size[1]

    scale = max(w, h) / float(min_side)
    new_w, new_h = int(w/scale), int(h/scale)
    resize_img = cv.resize(img, (new_w, new_h))

    top, bottom, left, right = 0, min_side-new_h, 0, min_side-new_w
    pad_img = cv.copyMakeBorder(resize_img, int(top), int(bottom), int(left), int(right),
                                cv.BORDER_CONSTANT, value=[255,255,255])
    pad_img = pad_img / 255

    return pad_img


def create_patch(patch_num=7):
    bboxes = []
    for i in range(patch_num):
        for j in range(patch_num):
            box = [1.0 * j / patch_num, 1.0 * i / patch_num, 1.0 * (j + 1) / patch_num, 1.0 * (i + 1) / patch_num]
            bboxes.append(box)
    bboxes = np.array(bboxes)
    return bboxes


def split_elements(sequence):
    new_sequence = []
    for token in sequence:
        if 'N_' in token or 'NS_' in token or 'frac' in token:
            new_sequence.append(token)
        elif token.istitle():
            new_sequence.append(token)
        elif re.search(r'[A-Z]', token):
            # split geometry elements with a space: ABC -> A B C
            new_sequence.extend(token)
        else:
            new_sequence.append(token)

    return new_sequence


def process_english_text(ori_text):
    text = re.split(r'([=≠≈+-/△∠∥⊙☉⊥⟂≌≅▱∽⁀⌒;,:.•?])', ori_text)
    text = ' '.join(text)

    text = text.split()
    text = split_elements(text)
    text = ' '.join(text)

    # The initial version of the calculation problem (GeoQA) is in Chinese.
    # The translated English version still contains some Chinese tokens,
    # which should be replaced by English words.
    replace_dict ={'≠': 'not-equal', '≈': 'approximate', '△': 'triangle', '∠': 'angle', '∥': 'parallel',
                   '⊙': 'circle', '☉': 'circle', '⊥': 'perpendicular', '⟂': 'perpendicular', '≌': 'congruent', '≅': 'congruent',
                   '▱': 'parallelogram', '∽': 'similar', '⁀': 'arc', '⌒': 'arc'
                   }
    for k, v in replace_dict.items():
        text = text.replace(k, v)

    return text


def process_Chinese_solving(ori_text):
    index = ori_text.find('故选')
    text = ori_text[:index]

    # delete special tokens
    delete_list = ['^{°}', '{', '}', '°', 'cm', 'm', '米', ',', ':', '．', '、', '′', '~', '″', '【', '】', '$']
    for d in delete_list:
        text = text.replace(d, ' ')
    # delete Chinese tokens
    zh_pattern = re.compile(u'[\u4e00-\u9fa5]+')
    text1 = re.sub(zh_pattern, ' ', text)

    # split
    pattern = re.compile(r'([=≠≈+-]|π|×|\\frac|\\sqrt|\\cdot|\√|[∵∴△∠∥⊙☉⊥⟂≌≅▱∽⁀⌒]|[.,，：；;,:.•?]|\d+\.?\d*|)')
    text2 = re.split(pattern, text1)
    # split elements: ABC -> A B C
    text2 = split_elements(text2)

    # store numbers
    text3 = []
    nums = []
    # replace only nums
    for t in text2:
        if re.search(r'\d', t):  # NS: number in solving
            if float(t) in nums:
                text3.append('NS_'+str(nums.index(float(t))))
            else:
                text3.append('NS_'+str(len(nums)))
                nums.append(float(t))
        else:
            text3.append(t)

    # replace
    text4 = ' '.join(text3)
    replace_dict = {'≠': 'not-equal', '≈': 'approximate', '△': 'triangle', '∠': 'angle', '∥': 'parallel',
                    '⊙': 'circle', '☉': 'circle', '⊥': 'perpendicular', '⟂': 'perpendicular', '≌': 'congruent', '≅': 'congruent',
                    '▱': 'parallelogram', '∽': 'similar', '⁀': 'arc', '⌒': 'arc',
                    '/ /': 'parallel', '∵': 'because', '∴': 'therefore', '²': 'square', '√': 'root'
                    }
    for k, v in replace_dict.items():
        text4 = text4.replace(k, v)
    text4 = text4.split()

    return text4, nums

def from_infix_to_prefix(expression):
    st = list()
    res = list()
    priority = {"+": 0, "-": 0, "*": 1, "/": 1, "^": 2}
    expression = deepcopy(expression)
    expression.reverse()
    for e in expression:
        if e in [")", "]"]:
            st.append(e)
        elif e == "(":
            c = st.pop()
            while c != ")":
                res.append(c)
                c = st.pop()
        elif e == "[":
            c = st.pop()
            while c != "]":
                res.append(c)
                c = st.pop()
        elif e in priority:
            while len(st) > 0 and st[-1] not in [")", "]"] and priority[e] < priority[st[-1]]:
                res.append(st.pop())
            st.append(e)
        else:
            res.append(e)
    while len(st) > 0:
        res.append(st.pop())
    res.reverse()
    return res

def load_raw_data(train_path, test_path, is_train = True):  # load the data to list(dict())
    train_ls = None
    if is_train:
        train_df = pd.read_csv(train_path, converters={'group_nums': eval})
        train_df['id'] = train_df.index
        train_ls = train_df.to_dict('records')

    dev_df = pd.read_csv(test_path, converters={'group_nums': eval})
    dev_df['id'] = dev_df.index
    dev_ls = dev_df.to_dict('records')

    return train_ls, dev_ls


def transfer_num_no_tokenize(train_ls, dev_ls, chall = False):  # transfer num into "NUM"
    print("Transfer numbers...")
    dev_pairs = []
    generate_nums = []
    generate_nums_dict = {}
    copy_nums = 0

    if train_ls != None:
        train_pairs = []
        for d in train_ls:
            # nums = []
            if 'Numbers' not in d.keys():
                continue
            nums = d['Numbers'].split()
            seg = nltk.word_tokenize(d["Question"].strip())
            equation = d["Equation"].split()
            
            input_seq = []
            

            numz = ['0','1','2','3','4','5','6','7','8','9']
            opz = ['+', '-', '*', '/']
            idxs = []
            num_idx = 0
            for s in range(len(seg)):
                if len(seg[s]) >= 7 and seg[s][:6] == "number" and seg[s][6] in numz:
                    input_seq.append("N_"+str(num_idx))
                    idxs.append(s)
                else:
                    input_seq.append(seg[s])
            if copy_nums < len(nums):
                copy_nums = len(nums)
            continue_flag = False
            out_seq = []
            for e1 in equation:
                if len(e1) >= 7 and e1[:6] == "number":
                    out_seq.append('N_'+e1[6:])
                elif e1 not in opz:
                    continue_flag = True
                    generate_nums.append(e1)
                    if e1 not in generate_nums_dict:
                        generate_nums_dict[e1] = 1
                    else:
                        generate_nums_dict[e1] += 1
                    out_seq.append(e1)
                else:
                    out_seq.append(e1)
            '''
            for iidx in range(len(input_seq)):
                if input_seq[iidx] == 'z':
                    input_seq[iidx] = 'x'
                if input_seq[iidx] == 'NUM':
                    input_seq[iidx] = 'z'
            '''
            if continue_flag:
                continue
            train_pairs.append((input_seq, out_seq, nums, idxs, d['group_nums'], d['id']))
    else:
        train_pairs = None

    for d in dev_ls:
        if 'Numbers' not in d.keys():
            continue
        # nums = []
        nums = d['Numbers'].split()
        input_seq = []
        try:
            seg = nltk.word_tokenize(d["Question"].strip())
        except:
            print('tokenize error')
        equation = d["Equation"].split()

        numz = ['0','1','2','3','4','5','6','7','8','9']
        opz = ['+', '-', '*', '/']
        idxs = []
        num_idx = 0
        for s in range(len(seg)):
            if len(seg[s]) >= 7 and seg[s][:6] == "number" and seg[s][6] in numz:
                input_seq.append("N_"+str(num_idx))
                idxs.append(s)
            else:
                input_seq.append(seg[s])
        if copy_nums < len(nums):
            copy_nums = len(nums)

        out_seq = []
        for e1 in equation:
            if len(e1) >= 7 and e1[:6] == "number":
                out_seq.append('N_'+e1[6:])
            elif e1 not in opz:
                generate_nums.append(e1)
                if e1 not in generate_nums_dict:
                    generate_nums_dict[e1] = 1
                else:
                    generate_nums_dict[e1] += 1
                out_seq.append(e1)
            else:
                out_seq.append(e1)
        '''
        for iidx in range(len(input_seq)):
            if input_seq[iidx] == 'z':
                input_seq[iidx] = 'x'
            if input_seq[iidx] == 'NUM':
                input_seq[iidx] = 'z'
        '''
        if chall:
            dev_pairs.append((input_seq, out_seq, nums, idxs, d['group_nums'], d['Type'], d['Variation Type'], d['Annotator'], d['Alternate'], d['id']))
    
        else:
            dev_pairs.append((input_seq, out_seq, nums, idxs, d['group_nums'], d['id']))
        

    temp_g = []
    for g in generate_nums_dict:
        if generate_nums_dict[g] >= 100:
            temp_g.append(g)
    return train_pairs, dev_pairs, temp_g, copy_nums


def transfer_num_keep_number(train_ls, dev_ls, chall = False):  # transfer num into "NUM"
    print("Transfer numbers...")
    dev_pairs = []
    generate_nums = []
    generate_nums_dict = {}
    copy_nums = 0

    if train_ls != None:
        train_pairs = []
        for d in train_ls:
            # nums = []
            if 'Numbers' not in d.keys():
                continue
            nums = d['Numbers'].split()
            seg = nltk.word_tokenize(d["Question"].strip())
            equation = d["Equation"].split()
            
            input_seq = []
            

            numz = ['0','1','2','3','4','5','6','7','8','9']
            opz = ['+', '-', '*', '/']
            idxs = []
            num_idx = 0
            for s in range(len(seg)):
                if len(seg[s]) >= 7 and seg[s][:6] == "number" and seg[s][6] in numz:
                    input_seq.append("N_"+str(num_idx))
                    idxs.append(s)
                else:
                    input_seq.append(seg[s])
            if copy_nums < len(nums):
                copy_nums = len(nums)
            continue_flag = False
            out_seq = []
            for e1 in equation:
                if len(e1) >= 7 and e1[:6] == "number":
                    out_seq.append('N_'+e1[6:])
                elif e1 not in opz:
                    continue_flag = True
                    generate_nums.append(e1)
                    if e1 not in generate_nums_dict:
                        generate_nums_dict[e1] = 1
                    else:
                        generate_nums_dict[e1] += 1
                    out_seq.append(e1)
                else:
                    out_seq.append(e1)
            '''
            for iidx in range(len(input_seq)):
                if input_seq[iidx] == 'z':
                    input_seq[iidx] = 'x'
                if input_seq[iidx] == 'NUM':
                    input_seq[iidx] = 'z'
            '''
            if continue_flag:
                continue
            for idx, iidx in enumerate(idxs):
                input_seq[iidx] = nums[idx]
            for idx, token in enumerate(out_seq):
                if token[0] == 'N':
                    out_seq[idx] = nums[int(token[2:])]
            train_pairs.append((input_seq, out_seq, nums, idxs, d['group_nums'], d['id']))
    else:
        train_pairs = None

    for d in dev_ls:
        if 'Numbers' not in d.keys():
            continue
        # nums = []
        nums = d['Numbers'].split()
        input_seq = []
        try:
            seg = nltk.word_tokenize(d["Question"].strip())
        except:
            print('tokenize error')
        equation = d["Equation"].split()

        numz = ['0','1','2','3','4','5','6','7','8','9']
        opz = ['+', '-', '*', '/']
        idxs = []
        num_idx = 0
        for s in range(len(seg)):
            if len(seg[s]) >= 7 and seg[s][:6] == "number" and seg[s][6] in numz:
                input_seq.append("N_"+str(num_idx))
                idxs.append(s)
            else:
                input_seq.append(seg[s])
        if copy_nums < len(nums):
            copy_nums = len(nums)

        out_seq = []
        for e1 in equation:
            if len(e1) >= 7 and e1[:6] == "number":
                out_seq.append('N_'+e1[6:])
            elif e1 not in opz:
                generate_nums.append(e1)
                if e1 not in generate_nums_dict:
                    generate_nums_dict[e1] = 1
                else:
                    generate_nums_dict[e1] += 1
                out_seq.append(e1)
            else:
                out_seq.append(e1)
        '''
        for iidx in range(len(input_seq)):
            if input_seq[iidx] == 'z':
                input_seq[iidx] = 'x'
            if input_seq[iidx] == 'NUM':
                input_seq[iidx] = 'z'
        '''
        for idx, iidx in enumerate(idxs):
            input_seq[iidx] = nums[idx]
        for idx, token in enumerate(out_seq):
            if token[0] == 'N':
                out_seq[idx] = nums[int(token[2:])]
        if chall:
            dev_pairs.append((input_seq, out_seq, nums, idxs, d['group_nums'], d['Type'], d['Variation Type'], d['Annotator'], d['Alternate'], d['id']))
    
        else:
            dev_pairs.append((input_seq, out_seq, nums, idxs, d['group_nums'], d['id']))
        

    temp_g = []
    for g in generate_nums_dict:
        if generate_nums_dict[g] >= 100:
            temp_g.append(g)
    return train_pairs, dev_pairs, temp_g, copy_nums


def transfer_num_mathqa_keepnum(data):  # transfer num into "NUM"
    print("Transfer numbers...")
    pattern = re.compile("\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?")
    pairs = []
    generate_nums = []
    generate_nums_dict = {}
    copy_nums = 0
    #count = 0
    for idx, d in enumerate(data):
        #count += 1
        #if count == 100:
        #    break
        nums = []
        input_seq = []
        seg = d["original_text"].strip().split(" ")
        equations = d["equation"][2:]
        ans = d['ans']

        for s in seg:
            pos = re.search(pattern, s)
            if pos and pos.start() == 0:
                nums.append(s[pos.start(): pos.end()])
                input_seq.append("NU")
                if pos.end() < len(s):
                    input_seq.append(s[pos.end():])
            else:
                input_seq.append(s)
        if copy_nums < len(nums):
            copy_nums = len(nums)

        nums_fraction = []
        #if len(input_seq) > 150:
        #    continue

        for num in nums:
            if re.search("\d*\(\d+/\d+\)\d*", num):
                nums_fraction.append(num)
        nums_fraction = sorted(nums_fraction, key=lambda x: len(x), reverse=True)

        def seg_and_tag(st):  # seg the equation and tag the num
            res = []
            for n in nums_fraction:
                if n in st:
                    p_start = st.find(n)
                    p_end = p_start + len(n)
                    if p_start > 0:
                        res += seg_and_tag(st[:p_start])
                    if nums.count(n) == 1:
                        res.append("N"+str(nums.index(n)))
                    else:
                        res.append(n)
                    if p_end < len(st):
                        res += seg_and_tag(st[p_end:])
                    return res
            pos_st = re.search("\d+\.\d+%?|\d+%?", st)
            if pos_st:
                p_start = pos_st.start()
                p_end = pos_st.end()
                if p_start > 0:
                    res += seg_and_tag(st[:p_start])
                st_num = st[p_start:p_end]
                if nums.count(st_num) == 1:
                    res.append("N"+str(nums.index(st_num)))
                else:
                    res.append(st_num)
                if p_end < len(st):
                    res += seg_and_tag(st[p_end:])
                return res
            for ss in st:
                res.append(ss)
            return res

        out_seq = seg_and_tag(equations)
        #if len(out_seq) > 30:
        #    continue
        continue_flag = 0
        op_list = ["+", "-", '*', "/", "^", '(', ')']
        num_list = ['100', '1', '2', '10', '1000', '3600', '4', '60', '5', '3', '12', '3.6', '0.2778']
        for out_token in out_seq:
            if out_token not in (op_list + num_list):
                if 'N' not in out_token:
                    continue_flag = True

        if continue_flag:
            continue
        for s in out_seq:  # tag the num which is generated
            if s[0].isdigit() and s not in generate_nums and s not in nums:
                generate_nums.append(s)
                generate_nums_dict[s] = 0
            if s in generate_nums and s not in nums:
                generate_nums_dict[s] = generate_nums_dict[s] + 1
        num_pos = []
        for i, j in enumerate(input_seq):
            if j == "NU":
                num_pos.append(i)
        assert len(nums) == len(num_pos)
        out_seq = from_infix_to_prefix(out_seq)
        for idx, iidx in enumerate(num_pos):
                input_seq[iidx] = nums[idx]
        for idx, token in enumerate(out_seq):
            if token[0] == 'N':
                out_seq[idx] = nums[int(token[1:])]
    

        pairs.append((input_seq, out_seq, nums, num_pos, idx))

    temp_g = []
    for g in generate_nums:
        if generate_nums_dict[g] >= 5:
            temp_g.append(g)
    return pairs, temp_g, copy_nums