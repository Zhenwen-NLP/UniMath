import os
import re
import json
import argparse
import numpy as np


def replace_punctuation(str):
    return str.replace("\"", "").replace("'", "")


# Temporary fix for bug where {}^<\` characters roundtrip into \u2047 (??) character
def fix_buggy_characters(str):
    return re.sub("[{}^\\\\`\u2047<]", " ", str)


def score_string_similarity(str1, str2):
    if str1 == str2:
        return 3.0  # Better than perfect token match
    str1 = fix_buggy_characters(replace_punctuation(str1))
    str2 = fix_buggy_characters(replace_punctuation(str2))
    if str1 == str2:
        return 2.0
    if " " in str1 or " " in str2:
        str1_split = str1.split(" ")
        str2_split = str2.split(" ")
        overlap = list(set(str1_split) & set(str2_split))
        return len(overlap) / max(len(str1_split), len(str2_split))
    else:
        if str1 == str2:
            return 1.0
        else:
            return 0.0


def extract_prediction(output, options):

    ## choose the most similar option
    if options:
        scores = [score_string_similarity(x, output) for x in options]
        max_idx = int(np.argmax(scores))  # json does not recognize NumPy data types
        prediction = options[max_idx]
        return prediction

    ## free_text QA problems, numeric answer
    else:
        patterns = [
            r' ([\d\$\.\,\/\:]+ [AP]\.M\.)',  # 7:25 P.M.
            r'([\-\d\$\.\,\/\:]{0,}[\d]+)',  # 14.5
        ]

        for p in patterns:
            pattern = re.compile(p)
            res = pattern.findall(output)
            if len(res) > 0:
                prediction = res[-1].strip()
                return prediction

    return output

def extract_cot_prediction(output, options):
    patterns = [
            r' ([\d\$\.\,\/\:]+ [AP]\.M\.)',  # 7:25 P.M.
            r'([\-\d\$\.\,\/\:]{0,}[\d]+)',  # 14.5
        ]
    if 'Answer:' in output:
        output = output[output.rindex('Answer:'):].replace('Answer:','').strip()

    ## choose the most similar option
    if options:
        scores = [score_string_similarity(x, output) for x in options]
        max_idx = int(np.argmax(scores))  # json does not recognize NumPy data types
        prediction = options[max_idx]
        return prediction

    ## free_text QA problems, numeric answer
    else:
        for p in patterns:
            pattern = re.compile(p)
            res = pattern.findall(output)
            if len(res) > 0:
                prediction = res[-1].strip()
                return prediction

    return output

def normalize_answer(text, unit):
    # ["1,000", "123", "3/4", "56.456", "$56.4", "-3", "-10.02", "-3/2"]

    text = re.sub("^[\$]", "", text)
    text = re.sub("[\,\.\,\/]$", "", text)

    result = re.match("^[-+]?[\d,./]+$", text)

    if result is not None:
        # is number?
        text = text.replace(",", "")
        result = re.match("[-+]?\d+$", text)

        if result is not None:
            number = int(text)
        elif "/" in text:
            try:
                nums = text.split("/")
                number = round(float(nums[0]) / float(nums[1]), 3)
            except:
                return text
        else:
            try:
                number = round(float(text), 3)
            except:
                return text
        number = str(number)
        number = re.sub(r"\.[0]+$", "", number)
        return number
    else:
        # is text
        if unit:
            text = text.replace(unit, "").strip()
        return text