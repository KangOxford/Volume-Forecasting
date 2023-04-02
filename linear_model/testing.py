with open("/Users/kang/dotfiles/docs/source/architecture.rst") as f:
    lines = f.readlines()

# for i, line in enumerate(lines):
#     lines[i] = line.replace("`", "``")
# for i, line in enumerate(lines):
#     lines[i] = line.replace("*", "-")
# for i, line in enumerate(lines):
#     lines[i] = line.rstrip("\n") + "\n\n"
# for i, line in enumerate(lines):
#     if i > 0 and lines[i-1].startswith("="):
#         lines[i-1] = lines[i-1].rstrip("\n")
#         break
for i, line in enumerate(lines):
    if " -" in line:
        lines[i] = line.replace(" -", "  -")

with open("/Users/kang/dotfiles/docs/source/architecture.rst", "w") as f:
    f.writelines(lines)



import os
directory = "/Users/kang/dotfiles/gym_exchange"

for root, dirs, files in os.walk(directory):
    for filename in files:
        if filename.endswith("__init__.py"):
            file_path = os.path.join(root, filename)
            with open(file_path, "r") as f:
                content = f.read()
            with open(file_path, "w") as f:
                f.write('"""Implementations of imitation and reward learning algorithms."""\n\n' + content)
