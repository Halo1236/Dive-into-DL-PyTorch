# -*- coding: utf8 -*-

list = []
with open('id.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        line = line.strip('\n')  # 去掉列表中每一个元素的换行符
        if 'es' in line or '-' in line:
            continue
        list.append(line)
print('_'.join(list))
