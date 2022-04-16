import textstat

from utils import get_reading_level

lines = []

with open('single_cconj.txt') as f:
    for line in f.readlines():
        lines.append(line.strip('\n'))

from collections import defaultdict

reading_ease_line = defaultdict(list)

for line in lines:

    if len(line) <= 1:
        continue

    reading_ease_line[get_reading_level(line)].append(line)

a = 1