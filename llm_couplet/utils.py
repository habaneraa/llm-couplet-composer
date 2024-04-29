import unicodedata


def count_chinese_characters(line: str) -> int:
    num = 0
    for ch in line:
        if is_punctuation(ch):
            continue
        if "\u4e00" <= ch <= "\u9fff":
            num += 1
    return num


def is_punctuation(char):
    category = unicodedata.category(char)
    return category.startswith("P")
