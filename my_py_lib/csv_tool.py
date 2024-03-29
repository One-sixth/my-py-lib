import csv
from typing import Union


def load_csv(csv_file, encoding='utf8'):
    '''
    从 csv 中载入为 list 双层列表
    :param csv_file:
    :return:
    '''
    if isinstance(csv_file, str):
        csv_file = open(csv_file, 'r', encoding=encoding)

    reader = csv.reader(csv_file)
    rows = []
    for row in reader:
        rows.append(row)

    return rows


def save_csv(rows, csv_file, encoding='utf8'):
    if isinstance(csv_file, str):
        csv_file = open(csv_file, 'w', encoding=encoding, newline='')

    csv_writer = csv.writer(csv_file)
    for row in rows:
        csv_writer.writerow(row)


if __name__ == '__main__':
    f = r"a.csv"
    rows = load_csv(f)
    save_csv(rows, 'o.csv')
    print(rows)
