import chardet

with open("faq.csv", "rb") as f:
    result = chardet.detect(f.read())
    print(result["encoding"])