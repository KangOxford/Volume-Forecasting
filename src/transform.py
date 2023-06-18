with open("/src/SPY-15m-adj-20091201.txt") as f:
    head = f.readline()
    lines = f.readlines()
new_lines = []
for line in lines:
    splitted = line.split("\t")
    splitted[-1] = "1.01\n"
    new_line = '\t'.join(splitted)
    new_lines.append(new_line)

with open("/src/Test_SPY-15m-adj-20091201.txt", 'w+') as f:
    f.write(head)
    f.writelines(new_lines)
