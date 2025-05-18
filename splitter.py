import json
from pathlib import Path
from slugify import slugify



drugs=json.loads(Path("drugs.json").read_text())

# for drug in drugs['drugbank']['drug']:
#     filename = "drugs_split\\" + drug['name'] + ".json"
#     print(filename)
#     filename = slugify(filename)
#     file = open(filename, "w+")
#     json.dump(drug , file)


file = open("drugs_split\\Acetaminophen.json" , "r")
data = json.load(file)
#print(data['categories']['category']['category'])
for interaction in data["drug-interactions"]["drug-interaction"]:
    print(interaction["name"])
