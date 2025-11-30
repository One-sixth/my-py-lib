import openpyxl


def get_sheet_names(xlsx_file):
    return openpyxl.load_workbook(xlsx_file, True).sheetnames


def load_xlsx(xlsx_file, table_name=None):
    wb = openpyxl.load_workbook(xlsx_file, True)
    if table_name is None:
        ws = wb.active
    else:
        ws = wb[table_name]

    rows = []
    for row in ws:
        rows.append([v.value for v in row])

    return rows


def save_xlsx(rows, xlsx_file):
    wb = openpyxl.Workbook()
    ws = wb.active

    for row in rows:
        ws.append(row)

    wb.save(xlsx_file)


if __name__ == '__main__':
    print(get_sheet_names(r'tmp_1.xlsx'))
    rows = load_xlsx(r'tmp_1.xlsx')
    save_xlsx(rows, 'tmp_2.xlsx')
