with open("datas\\dantzig42.txt","r+",encoding="UTF-8") as f:
    lines = f.readlines()
    newlines = []
    for line in lines:
        words = line.split(" ")
        newline = ",".join(words)
        newlines.append(newline)
    f.writelines(newlines)