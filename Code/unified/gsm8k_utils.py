from loguru import logger
from copy import deepcopy
import numpy as np
import torch
import decimal
from decimal import Decimal
from torch import Tensor
from torch.utils.data import Dataset
from copy import deepcopy
import re
import random
from collections import namedtuple
from typing import Dict, List, Any, AnyStr, Optional, Tuple, Union
import json
import math
import os
from tqdm import tqdm

Op = None

Tok = str

Expr = namedtuple('Expr', ['arg0', 'expr_toks', 'expr_str'])

MultiExpr = namedtuple('MultiExpr', ['args', 'expr_toks', 'expr_str'])

class MathDataset(Dataset):
    
    def __init__(self, data: List[Dict]) -> None:
        self.data = data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index) -> Dict:
        return self.data[index]   


class MathDataInstance:
    
    def __init__(
        self,
        question: str,
        nums: List[str],
        const_nums: List[str],
        expr_list: List[Expr],
        target: Optional[List[Expr]] = None,
        id: Optional[int] = None,
        end: bool = True
    ) -> None:
        self.question = question
        self.nums = nums
        self.const_nums = const_nums
        self.expr_list = expr_list
        self.target = target
        self.id = id
        self.end = end
    
    def parse_input(self, sep_token: str, use_expr: bool = True) -> str:
        input_text = [self.question]
        if use_expr:
            for expr in self.expr_list:
                input_text.append(sep_token)
                input_text.append("[num{}] = {}"\
                    .format(expr.arg0, expr.expr_str))
        input_text = " ".join(input_text)
        return input_text

    def parse_output(self, bos_token: str, eos_token: str) -> str:
        output_text = []
        for expr in self.target:
            output_text.extend(expr.expr_toks)
            output_text.append(bos_token)
        if self.end:
            output_text[-1] = eos_token
        output_text = " ".join(output_text)
        return output_text


class TemplateDataInstance:
    
    def __init__(
        self,
        question: str,
        nums: List[str],
        const_nums: List[str],
        expr_list: List[MultiExpr],
        target: Optional[List[MultiExpr]] = None,
        id: Optional[int] = None,
        end: bool = True
    ) -> None:
        self.question = question
        self.nums = nums
        self.const_nums = const_nums
        self.expr_list = expr_list
        self.target = target
        self.id = id
        self.end = end
    
    def parse_input(self, sep_token: str, use_expr: bool = True) -> str:
        input_text = [self.question]
        if use_expr:
            for expr in self.expr_list:
                input_text.append(sep_token)
                input_text.append("{} = {}"\
                    .format(" ".join([f"[num{i}]" for i in expr.args]), " ".join(expr.expr_toks)))
        input_text = " ".join(input_text)
        return input_text

    def parse_output(self, bos_token: str, eos_token: str) -> str:
        output_text = []
        for expr in self.target:
            output_text.extend(expr.expr_toks)
            output_text.append(bos_token)
        if self.end:
            output_text[-1] = eos_token
        output_text = " ".join(output_text)
        return output_text


def convert_const_nums(seg_expr: List[Tok], const_nums: List[str]) -> int:
    new_seg_expr: List[str] = []
    for tok in seg_expr:
        if tok in "+*/^()" or re.match("\[num\d+\]", tok):
            new_seg_expr.append(tok)
        elif tok == '-':
            if len(new_seg_expr) > 0 and \
                (re.match("\[num\d+\]", new_seg_expr[-1]) \
                    or re.match("\[c\d+\]", new_seg_expr[-1]) \
                    or new_seg_expr[-1] == ')'):
                new_seg_expr.append(tok)
            else:
                idx = const_nums.index('-1')
                new_seg_expr.append(f"[c{idx}]")
                new_seg_expr.append("*")
        else:
            if tok not in const_nums:
                print("tok:", tok)
                print("const_nums:", const_nums)
                raise ValueError
            idx = const_nums.index(tok)
            new_seg_expr.append(f"[c{idx}]")
    return new_seg_expr


def seq2seq_parse_num_index(num_token: str, nums: List[str]) -> int:
    m = re.match("\[num(\d+)\]", num_token)
    if m:
        return int(m.group(1))
    else:
        if num_token not in nums:
            raise ValueError
        return nums.index(num_token)


def parse_num_index(num_token: str) -> int:
    m = re.match("\[num(\d+)\]", num_token)
    if m:
        return int(m.group(1))
    else:
        raise ValueError


def parse_const_num_index(num_token: str) -> int:
    m = re.match("\[c(\d+)\]", num_token)
    if m:
        return int(m.group(1))
    else:
        raise ValueError


def parse_value(x: str) -> float:
    x = x.replace("%","*0.01")
    try:
        value = Decimal(eval(x))
        return value
    except:
        print(x)
        exit(-1)


def eval_expr(tokens: List[Tok]):
    op_stack = []
    v_stack = []

    def pop_stack():
        o = op_stack.pop()
        v1 = v_stack.pop()
        v0 = v_stack.pop()
        if o not in '+-*/^':
            raise SyntaxError
        if o == '^':
            v_stack.append(pow(v0, v1))
        elif o == '+':
            v_stack.append(v0 + v1)
        elif o == '-':
            v_stack.append(v0 - v1)
        elif o == '*':
            v_stack.append(v0 * v1)
        elif o == '/':
            v_stack.append(v0 / v1)

    for t in tokens:
        if t.replace(" ", "") == "":
            continue
        if t == '(':
            op_stack.append('(')
        elif t == ')':
            while len(op_stack) > 0 and op_stack[-1] != '(':
                pop_stack()
            op_stack.pop()
        elif t == '+' or t == '-':
            while len(op_stack) > 0 and op_stack[-1] in '+-*/^':
                pop_stack()
            op_stack.append(t)
        elif t == '*' or t == '/':
            while len(op_stack) > 0 and op_stack[-1] in '*/^':
                pop_stack()
            op_stack.append(t)
        elif t == '^':
            while len(op_stack) > 0 and op_stack[-1] in '^':
                pop_stack()
            op_stack.append(t)
        else:
            v_stack.append(Decimal(eval(t)))
    while len(op_stack) > 0:
        pop_stack()
    if len(v_stack) != 1:
        raise SyntaxError
    return v_stack[-1]


def convert_expr(expr: str, nums: List[str]):
    tokens = []
    while len(expr) > 0:
        m: re.Match = re.match("\[num\d+\]", expr)
        token_length = 0
        if m is None:
            token_length = 1
            tokens.append(expr[0])
        else:
            token_length = m.end()
            idx = seq2seq_parse_num_index(expr[:token_length], nums)
            num = nums[idx] if idx < len(nums) else '1'
            tokens.append("(" + num + ")")
        expr = expr[token_length:]
    expr = "".join(tokens)
    return expr


def compute_expr(expr: str, nums: List[str]):
    expr = convert_expr(expr, nums)
    expr = expr.replace("%", "*0.01")
    tokens = re.split(r"([\*\/\^\(\)\+\-])", expr)
    # print("".join(tokens))
    try:
        value = eval_expr(tokens)
    except:
        print("".join(tokens))
        value = None
    return value


def build_Expr_list_v1(seg_expr: List[Tok], nums_size: int) -> List[Expr]:
    # 根据括号对表达式进行切分
    Expr_list: List[Expr]   = []
    match_pos: Dict[int, int] = {}
    expr_dict: Dict[str, int] = {f'[num{i}]': i for i in range(nums_size)}

    def compute_match_pos():
        stk = []
        for i, tok in enumerate(seg_expr):
            if tok == '(':
                stk.append(i)
            elif tok == ')':
                j = stk.pop()
                match_pos[j] = i

    def rec_build_expr_list(l: int, r: int):
        expr_toks = []
        i = l
        while i < r:
            tok = seg_expr[i]
            if tok.replace(" ", "") != "":
                if tok == '(':
                    arg = rec_build_expr_list(i + 1, match_pos[i])
                    expr_toks.append('[num{}]'.format(arg))
                    i = match_pos[i]
                else:
                    expr_toks.append(tok)
            i += 1
        expr_str = " ".join(expr_toks)
        if expr_str not in expr_dict:
            arg0 = len(expr_dict)
            expr_dict[expr_str] = arg0
            Expr_list.append(
                Expr(arg0=arg0, expr_toks=expr_toks, expr_str=expr_str)
            )
        return expr_dict[expr_str]
    
    compute_match_pos()
    rec_build_expr_list(0, len(seg_expr))

    return Expr_list


def build_Expr_list_v2(seg_expr: List[Tok], nums_size: int) -> List[Expr]:
    # 不对表达式进行切分
    expr_toks = seg_expr
    Expr_list: List[Expr] = [Expr(arg0=nums_size, expr_toks=expr_toks, expr_str=" ".join(expr_toks))]
    return Expr_list


def build_Expr_list_v3(seg_expr: List[Tok], nums_size: int) -> List[Expr]:
    # 根据运算符对表达式进行切分
    if len(seg_expr) == 1:
        return [Expr(arg0=nums_size, expr_toks=seg_expr, expr_str=" ".join(seg_expr))]

    Expr_list: List[Expr] = []
    expr_dict: Dict[str, int] = {f'[num{i}]': f'[num{i}]' for i in range(nums_size)}

    op_stack = []
    v_stack = []

    def pop_stack():
        op = op_stack.pop()
        arg2 = v_stack.pop()
        arg1 = v_stack.pop()
        expr_toks=[arg1, op, arg2]
        expr_str=f'{arg1} {op} {arg2}'
        if expr_str not in expr_dict:
            arg0 = len(expr_dict)
            expr_dict[expr_str] = f'[num{arg0}]'
            Expr_list.append(Expr(
                arg0=arg0, 
                expr_toks=expr_toks,
                expr_str=expr_str
            ))
        v_stack.append(expr_dict[expr_str])
    for t in seg_expr:
        if t.replace(" ", "") == "":
            continue
        if t == '(':
            op_stack.append('(')
        elif t == ')':
            while len(op_stack) > 0 and op_stack[-1] != '(':
                pop_stack()
            op_stack.pop()
        elif t == '+' or t == '-':
            while len(op_stack) > 0 and op_stack[-1] in '+-*/^':
                pop_stack()
            op_stack.append(t)
        elif t == '*' or t == '/':
            while len(op_stack) > 0 and op_stack[-1] in '*/^':
                pop_stack()
            op_stack.append(t)
        elif t == '^':
            while len(op_stack) > 0 and op_stack[-1] in '^':
                pop_stack()
            op_stack.append(t)
        else:
            v_stack.append(t)
    while len(op_stack) > 0:
        pop_stack()
    if len(v_stack) != 1:
        raise SyntaxError

    return Expr_list


def compute_Expr_list(Expr_list: List[Expr], nums: List[str], const_nums: List[str], max_nums_size: int):
    if Expr_list is None:
        return None
    
    nums = [parse_value(x) for x in nums]
    nums_table = nums + [0.0] * (max_nums_size - len(nums))

    const_nums = [parse_value(x) for x in const_nums]

    def do_OpSeq(expr_toks: List[Tok]):
        # print("expr_tokens:", expr_toks)

        op_stack = []
        v_stack = []

        def pop_stack():
            o = op_stack.pop()
            v1 = v_stack.pop()
            v0 = v_stack.pop()
            # print("do_op: [{} {} {}]".format(v0, o, v1))
            if o not in '+-*/^':
                raise SyntaxError
            if o == '^':
                v_stack.append(pow(v0, v1))
            elif o == '+':
                v_stack.append(v0 + v1)
            elif o == '-':
                v_stack.append(v0 - v1)
            elif o == '*':
                v_stack.append(v0 * v1)
            elif o == '/':
                v_stack.append(v0 / v1)

        for t in expr_toks:
            if t.replace(" ", "") == "":
                continue
            if t == '(':
                op_stack.append('(')
            elif t == ')':
                while len(op_stack) > 0 and op_stack[-1] != '(':
                    pop_stack()
                if len(op_stack) == 0:
                    logger.warning("decimal.Error: {}".format(Expr_list))
                    return None
                op_stack.pop()
            elif t == '+' or t == '-':
                while len(op_stack) > 0 and op_stack[-1] in '+-*/^':
                    pop_stack()
                op_stack.append(t)
            elif t == '*' or t == '/':
                while len(op_stack) > 0 and op_stack[-1] in '*/^':
                    pop_stack()
                op_stack.append(t)
            elif t == '^':
                while len(op_stack) > 0 and op_stack[-1] in '^':
                    pop_stack()
                op_stack.append(t)
            else:
                if re.match('\[num\d+\]', t):
                    i = parse_num_index(t)
                    v_stack.append(Decimal(nums_table[i]))
                else:
                    i = parse_const_num_index(t)
                    v_stack.append(Decimal(const_nums[i]))
            
        while len(op_stack) > 0:
            pop_stack()
        if len(v_stack) != 1:
            raise SyntaxError
        return v_stack[-1]

    try:
        for opSeq in Expr_list:
            nums_table[opSeq.arg0] = do_OpSeq(opSeq.expr_toks)
    except:
        logger.warning("decimal.Error: {}".format(Expr_list))
        return None
    return nums_table[Expr_list[-1].arg0] if len(Expr_list) > 0 else None


def compute_MultiExpr_list(MultiExpr_list: List[MultiExpr], nums: List[str], const_nums: List[str], max_nums_size: int):
    if MultiExpr_list is None:
        return None
    
    nums = [float(x) for x in nums]
    nums_table = nums + [0.0] * (max_nums_size - len(nums))

    const_nums = [float(x) for x in const_nums]

    def parse_quants(tokens: List[Tok]):
        return [parse_num_index(token) for token in tokens]

    def parse_fn(tokens: List[Tok]):
        # print("op:", "".join(tokens))
        # print("tokens:", tokens)
        quantsIndex = parse_quants(tokens[1:])
        if tokens[0] == "[solve_linear_equation]":
            A = np.array(
                [
                    [nums_table[quantsIndex[0]], nums_table[quantsIndex[1]]],
                    [nums_table[quantsIndex[2]], nums_table[quantsIndex[3]]],
                ]
            )
            b = np.array(
                [nums_table[quantsIndex[4]], nums_table[quantsIndex[5]]]
            )
            return (np.linalg.inv(A) @ b).tolist()
        elif tokens[0] == "[quadratic_function_integral]":
            def poly3(a, b, c, d, x):
                return a * (x ** 3) + b * (x ** 2) + c * x + d
            l, r = nums_table[quantsIndex[0]], nums_table[quantsIndex[1]]
            a0, a1, a2 = nums_table[quantsIndex[2]], nums_table[quantsIndex[3]], nums_table[quantsIndex[4]]
            return [poly3(a0 / 3, a1 / 2, a2, 0, r) - poly3(a0 / 3, a1 / 2, a2, 0, l)]
        elif tokens[0] == "[quadratic_function_extremum]":
            def poly2(a, b, c, x):
                return a * (x ** 2) + b * x + c
            a0, a1, a2 = nums_table[quantsIndex[0]], nums_table[quantsIndex[1]], nums_table[quantsIndex[2]]
            return [poly2(a0, a1, a2, -a1 / (2 * a0))]
        elif tokens[0] == "[add]":
            a0, a1 = nums_table[quantsIndex[0]], nums_table[quantsIndex[1]]
            return [a0 + a1]
        elif tokens[0] == "[sub]":
            a0, a1 = nums_table[quantsIndex[0]], nums_table[quantsIndex[1]]
            return [a0 + a1]
        elif tokens[0] == "[mul]":
            a0, a1 = nums_table[quantsIndex[0]], nums_table[quantsIndex[1]]
            return [a0 * a1]
        elif tokens[0] == "[div]":
            a0, a1 = nums_table[quantsIndex[0]], nums_table[quantsIndex[1]]
            return [a0 / a1]
        else:
            a0, a1 = nums_table[quantsIndex[0]], nums_table[quantsIndex[1]]
            return [a0 ** a1]

    def do_OpSeq(expr_toks: List[Tok]):
        return parse_fn(expr_toks)

    try:
        for expr in MultiExpr_list:
            results = do_OpSeq(expr.expr_toks)
            for i, index in enumerate(expr.args):
                nums_table[index] = results[i]
    except:
        logger.warning("decimal.Error: {}".format(MultiExpr_list))
        return None
    return nums_table[MultiExpr_list[-1].args[-1]] if len(MultiExpr_list) > 0 else None

def compute_PreOrder(tokens: List[Tok], nums: List[float], const_nums: List[float], max_nums_size: int):    
    def compute_tokens(expr_toks: List[Tok]):

        op_stack = []
        v_stack = []

        def pop_stack():
            o = op_stack.pop()
            v1 = v_stack.pop()
            v0 = v_stack.pop()
            # print("do_op: [{} {} {}]".format(v0, o, v1))
            if o not in '+-*/^':
                raise SyntaxError
            if o == '^':
                v_stack.append(pow(v0, v1))
            elif o == '+':
                v_stack.append(v0 + v1)
            elif o == '-':
                v_stack.append(v0 - v1)
            elif o == '*':
                v_stack.append(v0 * v1)
            elif o == '/':
                v_stack.append(v0 / v1)

        for t in expr_toks[::-1]:
            if t.replace(" ", "") == "":
                continue
            if t in '+-*/^':
                op_stack.append(t)
                pop_stack()
            else:
                if re.match('\[num\d+\]', t):
                    i = parse_num_index(t)
                    v_stack.append(Decimal(nums[i]))
                else:
                    i = parse_const_num_index(t)
                    v_stack.append(Decimal(const_nums[i]))

        if not (len(v_stack) == 1 and len(op_stack) == 0):
            raise SyntaxError
        return v_stack[-1]

    try:
        value = compute_tokens(tokens)
    except:
        logger.warning("decimal.Error: {}".format(tokens))
        return None
    return value


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

def loadGSM8k_all(file_path: str = 'D:\\GSM8k\\data', head: Optional[int] = None):
    #file_path = './data'
    raw_test_dataset = []
    with open(os.path.join(file_path, "test.jsonl"), "r") as f:
        for line in f.readlines():
            raw_test_dataset.append(json.loads(line))
    raw_train_dataset = []
    with open(os.path.join(file_path, "train.jsonl"), "r") as f:
        for line in f.readlines():
            raw_train_dataset.append(json.loads(line))
    const_nums = []
    const_nums_dict = {}
    
    def gsm8k_filter(x: str) -> str:
        x = x.lstrip('0')
        if len(x) == 0:
            x = '0'
        return x
    def compress_Expr_list(Expr_list: List[Expr], nums_size: int):
        
        p0 = re.compile('\[num(\d+)\]')
        p1 = re.compile('\[c\d+\]')
        all_nums = [[f'[num{i}]'] for i in range(nums_size)]
        for expr in Expr_list:
            expr_toks = []
            for t in expr["expr_toks"]:
                if t in '+-*/()' or p1.match(t):
                    expr_toks.append(t)                    
                else:
                    try:
                        i = int(p0.match(t).group(1))
                    except:
                        print(t)
                    expr_toks.append('(')
                    expr_toks.extend(all_nums[i])
                    expr_toks.append(')')
            all_nums.append(expr_toks)
        return all_nums[-1]
    def parse_data(question_text: str, answer_text: str, const_count: dict, max_nums: int):
        #const_nums_truth = ['2', '60', '100', '10', '3', '7', '4', '0.01', '5', '1', '12', '0.2', '0.5', '6', '0.25', '30', '15', '24', '8', '16', '40', '20', '0.1', '0.8']
        const_nums_truth = ['2', '60', '100', '10', '3', '7', '4', '0.01', '5', '1', '12', '0.2', '0.5', '6', '0.25', '30', '0.1', '24', '8', '20']
        answer_value = str(eval(gsm8k_filter(answer_text.split("####")[-1].replace(",", ""))))
        for x, y in zip(
            ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten'],
            ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        ):
            question_text = question_text.replace(x, y)
        # parse interger
        p0 = re.compile('\d+/\d+')
        p1 = re.compile('\d+\.\d+')
        p2 = re.compile('\d+')
        nums_frac  = re.findall(p0, question_text)
        for num in sorted(nums_frac, key=lambda x: len(x), reverse=True):
            question_text = question_text.replace(num, " [num][frac] ")
        nums_float = re.findall(p1, question_text)
        for num in sorted(nums_float, key=lambda x: len(x), reverse=True):
            question_text = question_text.replace(num, " [num][float] ")
        nums_int   = re.findall(p2, question_text)
        for num in sorted(nums_int, key=lambda x: len(x), reverse=True):
            question_text = question_text.replace(num, " [num][int] ")
        nums = []
        question_text = question_text.replace('  ',' ')
        i_frac  = 0
        i_float = 0
        i_int   = 0
        q_texts = question_text.split('[num]')
        new_q_text = [q_texts[0]]
        for i in range(len(q_texts) - 1):
            #new_q_text.append('[num{}]'.format(len(nums)))
            new_q_text.append('[num]')
            new_q_text.append(q_texts[i + 1])
            if q_texts[i + 1].startswith('[frac]'):
                nums.append(str(eval(gsm8k_filter(nums_frac[i_frac]))))
                i_frac += 1
            elif q_texts[i + 1].startswith('[float]'):
                nums.append(str(eval(gsm8k_filter(nums_float[i_float]))))
                i_float += 1
            elif q_texts[i + 1].startswith('[int]'):
                nums.append(str(eval(gsm8k_filter(nums_int[i_int]))))
                i_int += 1
        question = "".join(new_q_text)
        p3 = re.compile('<<[^<>]*>>')
        p4 = re.compile('<<([^=<>]*)=([^=<>]*)>>')
        raw_Expr_list = re.findall(p3, answer_text)
        
        all_nums = [x for x in nums]
        if len(all_nums) > max_nums:
            max_nums = len(all_nums)
        Expr_list = []
        break_flag = False
        for opseq_text in raw_Expr_list:
            m = p4.match(opseq_text)
            if m is None:
                raise ValueError
            v0, v1 = m.group(1, 2)
            raw_expr_toks = re.split(r"([\*\/\+\-\(\)])", v0)
            expr_toks = []
            for x in raw_expr_toks:
                if x in "+-*/()":
                    expr_toks.append(x)
                else:
                    x = str(eval(x))
                    if x in all_nums:
                        expr_toks.append('[num{}]'.format(all_nums.index(x)))
                    else:
                        if x in const_count:
                            const_count[x] += 1
                        else:
                            const_count[x] = 1
                        if x not in const_nums:
                            const_nums.append(x)
                        else:
                            if x not in const_nums:
                                break_flag = True
                                break
                        expr_toks.append('[c{}]'.format(const_nums.index(x)))
                        const_nums_dict['[c{}]'.format(const_nums.index(x))] = x
            all_nums.append(str(eval(gsm8k_filter(v1))))
            Expr_list.append({
                "arg0": len(all_nums) - 1,
                "expr_toks": expr_toks,
                "expr_str": " ".join(expr_toks)
            })
        compress_expr_toks = compress_Expr_list(Expr_list, len(nums))
        for idx in range(len(compress_expr_toks)):
            if '[c' in compress_expr_toks[idx]:
                compress_expr_toks[idx] = const_nums_dict[compress_expr_toks[idx]]
            if 'num' in compress_expr_toks[idx]:
                compress_expr_toks[idx] = 'N' + compress_expr_toks[idx][-2]
        question = question.split()
        num_pos = []
        for idx in range(len(question)):
            if '[num]' in question[idx]:
                question[idx] = 'NUM'
                num_pos.append(idx)
        assert len(nums) == len(num_pos)
        Expr_list_v2 = [Expr(
            arg0=len(nums),
            expr_toks=compress_expr_toks,
            expr_str="".join(compress_expr_toks)
        )]
        if answer_value != all_nums[-1]:
            return None
        else:
        
            return {
                "seg_text": question,
                "answer": answer_text.split('####')[0],
                "nums": nums,
                "Expr_list": Expr_list,
                "Expr_list_v2": Expr_list_v2,
                "break_flag": break_flag,
                "max_num": max_nums,
                "num_pos": num_pos
            }
    train_dataset = []
    test_dataset = []
    train_const_count = {}
    test_const_count = {}
    max_nums = 0
    for dataset, raw_dataset, split in zip([train_dataset, test_dataset], [raw_train_dataset, raw_test_dataset], ['train', 'test']):
        for raw_obj in raw_dataset:
            if split == 'train':
                obj = parse_data(raw_obj["question"], raw_obj["answer"], train_const_count, max_nums)
                

            if split == 'test':
                obj = parse_data(raw_obj["question"], raw_obj["answer"], test_const_count, max_nums)
                
            if obj is not None:
                if not obj["break_flag"] and obj["max_num"] <= 9:
                    obj["const_nums"] = const_nums
                    dataset.append(obj)
                    if obj["max_num"] > max_nums:
                        max_nums = obj["max_num"]
        
    if head is not None and head != -1:
        train_dataset = train_dataset[:head]
        test_dataset = test_dataset[:head]
    #train_list = [i for i in train_const_count if train_const_count[i]>50]
    #test_list = [i for i in test_const_count if test_const_count[i]>10]
    #final_list = [i for i in train_list if i in test_list]
    #print(final_list)
    #print(train_const_count)
    return train_dataset, test_dataset, const_nums, max_nums

def load_gsm(data_path = './data'):
    train_dataset, test_dataset, const_nums, max_nums = loadGSM8k_all(file_path = data_path)

    pairs_tested = []
    #pairs_trained = valid_fold
    pairs_trained = []

    for i in train_dataset:
        for idx, iidx in enumerate(i['num_pos']):
            i['seg_text'][iidx] = i['nums'][idx]
        out_seq = from_infix_to_prefix(deepcopy(i['Expr_list_v2'][0][1]))
        if '' in out_seq:
            continue
        for idx, token in enumerate(out_seq):
            if token[0] == 'N':
                out_seq[idx] = i['nums'][int(token[1:])]
        pairs_trained.append([i['seg_text'], out_seq, i['nums'], i['num_pos'], i['answer']])
    for i in test_dataset:
        for idx, iidx in enumerate(i['num_pos']):
            i['seg_text'][iidx] = i['nums'][idx]
        out_seq = from_infix_to_prefix(deepcopy(i['Expr_list_v2'][0][1]))
        if '' in out_seq:
            continue
        for idx, token in enumerate(out_seq):
            if token[0] == 'N':
                out_seq[idx] = i['nums'][int(token[1:])]
        pairs_tested.append([i['seg_text'], out_seq, i['nums'], i['num_pos'], i['answer']])
    return pairs_trained, pairs_tested