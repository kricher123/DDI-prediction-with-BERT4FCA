import json
from pathlib import Path
from slugify import slugify





for drug in drugs['drugbank']['drug']:
    filename = "drugs_split\\" + drug['name'] + ".json"
    print(filename)
    filename = slugify(filename)
    file = open(filename, "w+")
    json.dump(drug , file)


