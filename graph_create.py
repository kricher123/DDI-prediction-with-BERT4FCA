import os
import json
from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx
import re


# drugs=json.loads(Path("drugs.json").read_text())
#TODO , split interactions per drug name for faster lookup
interactions = []
drugs = []
interactions_count = 0
directory = "drugs_subset2"

G = nx.Graph()
file_text = open("graph.txt" , "w+" , encoding="utf8")
file_interactions = open("graphs_subset2\graph_interactions.txt" , "w+" , encoding="utf8")
file_classifications = open("graphs_subset2\graph_classifications.txt" , "w+" , encoding="utf8")
file_mechanism = open("graphs_subset2\graph_mechanism.txt" , "w+" , encoding="utf8")
file_categories = open("graphs_subset2\graph_categories.txt" , "w+" , encoding="utf8")
file_enzymes = open("graphs_subset2\graph_enzymes.txt" , "w+" , encoding="utf8")
file_reactions = open("graphs_subset2\graph_reactions.txt" , "w+" , encoding="utf8")

for filename in os.listdir(directory):
    str = filename.strip(".json")
    drugs.append(str.lower())
    
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    if os.path.isfile(f):
        file = open(f,"r")
        drug = json.load(file)
        name = drug['name']
        obj_str = json.dumps(drug)
        insensitive_replace = re.compile(re.escape(drug['name']) , re.IGNORECASE)
        obj_str = insensitive_replace.sub("_" , obj_str)
        drug = json.loads(obj_str)
        drug['name'] = name
        #print(drug['name'])
        if drug['drug-interactions'] is not None:
            for interaction in drug['drug-interactions']['drug-interaction']:
                try:
                    G.add_edge(drug['name'] , interaction['name'] , label = "interacts")
                    file_text.write(drug['name'] + "|||" + interaction['name'] + "|||interacts|||\n")

                    obj_str = interaction['description']

                    insensitive_replace = re.compile(re.escape(drug['name']) , re.IGNORECASE)
                    obj_str = insensitive_replace.sub("_" , obj_str)

                    insensitive_replace = re.compile(re.escape(interaction['name']) , re.IGNORECASE)
                    obj_str = insensitive_replace.sub("_" , obj_str)

                    if interaction['name'].lower()  in drugs:
                        file_interactions.write(drug['name'] + "|||" + interaction['name'] + "|||"+obj_str+"|||\n")
                        interaction_row = []
                        #print(interaction['name'])
                        interaction_row.append(drug['name'])
                        interaction_row.append(interaction['name'])
                        interaction_row.append(interaction['description'])
                        interactions.append(interaction_row)
                        interactions_count+=1
                except TypeError:
                    if drug['drug-interactions']['drug-interaction']['name'].lower() in drugs:
                        G.add_edge(drug['name'] , drug['drug-interactions']['drug-interaction']['name'] , label = "interacts")
                        file_text.write(drug['name'] + "|||" + drug['drug-interactions']['drug-interaction']['name'] + "|||interacts|||\n")
                        file_interactions.write(drug['name'] + "|||" + drug['drug-interactions']['drug-interaction']['name'] + "|||interacts|||\n")
                        interaction_row.append(drug['name'])
                        interaction_row.append(drug['drug-interactions']['drug-interaction']['name'])
                        interaction_row.append(drug['drug-interactions']['drug-interaction']['description'])
                        interactions_count+=1
        if "classification" in drug:
            classification  = drug['classification']
            if "direct-parent" in classification:
                if classification['direct-parent'] is not None:
                    G.add_edge(drug['name'] , classification['direct-parent'] , label = "classification direct parent")
                    file_text.write(drug['name'] + "|||" + classification['direct-parent'] + "|||classification direct parent|||\n")
                    file_classifications.write(drug['name'] + "|||" + classification['direct-parent'] + "|||classification direct parent|||\n")
            if "kingdom" in classification:
                if classification['kingdom'] is not None:
                    G.add_edge(drug['name'] , classification['kingdom'] , label = "classification kingdom")
                    file_text.write(drug['name'] + "|||" + classification['kingdom'] + "|||classification kingdom|||\n")
                    file_classifications.write(drug['name'] + "|||" + classification['kingdom'] + "|||classification kingdom|||\n")
            if "superclass" in classification:
                if classification['superclass'] is not None:
                    G.add_edge(drug['name'] , classification['superclass'] , label = "classification superclass")
                    file_text.write(drug['name'] + "|||" + classification['superclass'] + "|||classification superclass|||\n")
                    file_classifications.write(drug['name'] + "|||" + classification['superclass'] + "|||classification superclass|||\n")
            if "class" in classification:
                if classification['class'] is not None:
                    G.add_edge(drug['name'] , classification['class'] , label = "classification class")
                    file_text.write(drug['name'] + "|||" + classification['class'] + "|||classification class|||\n")
                    file_classifications.write(drug['name'] + "|||" + classification['class'] + "|||classification class|||\n")
            if "subclass" in classification:
                if classification['subclass'] is not None:
                    G.add_edge(drug['name'] , classification['subclass'] , label = "classification subclass")
                    file_text.write(drug['name'] + "|||" + classification['subclass'] + "|||classification subclass|||\n")
                    file_classifications.write(drug['name'] + "|||" + classification['subclass'] + "|||classification subclass|||\n")
            if "alternative-parent" in classification:
                for alternative_parent in classification['alternative-parent']:
                    G.add_edge(drug['name'] , alternative_parent , label = "classification alternative parent")
                    file_text.write(drug['name'] + "|||" + alternative_parent + "|||classification alternative parent|||\n")
                    file_classifications.write(drug['name'] + "|||" + alternative_parent + "|||classification alternative parent|||\n")
            if "substituent" in classification:
                for substituent in classification['substituent']:
                    G.add_edge(drug['name'] , substituent , label = "classification substituent")
                    file_text.write(drug['name'] + "|||" + substituent + "|||classification substituent|||\n")
                    file_classifications.write(drug['name'] + "|||" + substituent + "|||classification substituent|||\n")
        if "mechanism-of-action" in drug:
            if drug['mechanism-of-action'] is not None:
                drug['mechanism-of-action'] = drug['mechanism-of-action'].replace('\n' , "")
                drug['mechanism-of-action'] = drug['mechanism-of-action'].replace(chr(13) , "")
                mechanism = drug['mechanism-of-action']
                insensitive_replace = re.compile(re.escape(drug['name']) , re.IGNORECASE)
                mechanism = insensitive_replace.sub('' , mechanism)
                # if drug['name'] == "19-norandrostenedione":
                #     print(ord(drug['mechanism-of-action'][476]))
                #     print(mechanism)
                #     test = open("temp.txt" , "w+")
                #     test.write(drug['mechanism-of-action'])
                G.add_edge(drug['name'] , mechanism , label = "mechanism of action")
                file_text.write(drug['name'] + "|||" + drug['mechanism-of-action'] + "|||mechanism of action|||\n")
                file_mechanism.write(drug['name'] + "|||" + drug['mechanism-of-action'] + "|||mechanism of action|||\n")
        if "categories" in drug:
            if drug['categories'] is not None:
                for category in drug['categories']['category']:
                    try:
                        G.add_edge(drug['name'] , category['category'] , label = "category")
                        file_text.write(drug['name'] + "|||" + category['category'] + "|||category|||\n")
                        file_categories.write(drug['name'] + "|||" + category['category'] + "|||category|||\n")
                    except:
                        G.add_edge(drug['name'] , drug['categories']['category']['category'] , label = "category")
                        file_text.write(drug['name'] + "|||" + drug['categories']['category']['category'] + "|||category|||\n")
                        file_categories.write(drug['name'] + "|||" + drug['categories']['category']['category'] + "|||category|||\n")
        if "enzymes" in drug:
            if drug['enzymes'] is not None:
                for enzyme in drug['enzymes']['enzyme']:
                    try:
                        G.add_edge(drug['name'] , enzyme['name'] , label = "enzyme-drug connection")
                        file_text.write(drug['name'] + "|||" + enzyme['name'] + "|||enzyme-drug connection|||\n")
                        file_enzymes.write(drug['name'] + "|||" + enzyme['name'] + "|||enzyme-drug connection|||\n")
                    except:
                        G.add_edge(drug['name'] , drug['enzymes']['enzyme']['name'] , label = "enzyme-drug connection") 
                        file_text.write(drug['name'] + "|||" + drug['enzymes']['enzyme']['name'] + "|||enzyme-drug connection|||\n")
                        file_enzymes.write(drug['name'] + "|||" + drug['enzymes']['enzyme']['name'] + "|||enzyme-drug connection|||\n")
        if "reactions" in drug:
            if drug['reactions'] is not None:
                for reaction in drug['reactions']['reaction']:
                    try:
                        name = ""
                        name1 = reaction['left-element']['name']
                        name2 = reaction['right-element']['name']
                        if name1>name2:
                            name = name1 + " " + name2 + " reaction"
                        else:
                            name = name2 + " " + name1 + " reaction"
                        G.add_edge(drug['name'] , name , label = "enzyme reactions")
                        file_text.write(drug['name'] + "|||" + name + "|||enzyme reactions|||\n")
                        file_reactions.write(drug['name'] + "|||" + name + "|||enzyme reactions|||\n")
                        if reaction['enzymes'] is not None:
                            for enzyme in reaction['enzymes']['enzyme']:
                                try:
                                    G.add_edge(enzyme['name'] , name , label = "enzyme reactions")
                                    file_text.write(enzyme['name'] + "|||" + name + "|||enzyme reactions|||\n")
                                    file_reactions.write(enzyme['name'] + "|||" + name + "|||enzyme reactions|||\n")
                                except:
                                    G.add_edge(reaction['enzymes']['enzyme']['name'] , name , label = "enzyme reactions")
                                    file_text.write(reaction['enzymes']['enzyme']['name'] + "|||" + name + "|||enzyme reactions|||\n")
                                    file_reactions.write(reaction['enzymes']['enzyme']['name'] + "|||" + name + "|||enzyme reactions|||\n")
                    except:
                        name = ""
                        name1 = drug['reactions']['reaction']['left-element']['name']
                        name2 = drug['reactions']['reaction']['right-element']['name']
                        if name1>name2:
                            name = name1 + " " + name2 + " reaction"
                        else:
                            name = name2 + " " + name1 + " reaction"
                        G.add_edge(drug['name'] , name , label = "enzyme reactions")
                        file_text.write(drug['name'] + "|||" + name + "|||enzyme reactions|||\n")
                        file_reactions.write(drug['name'] + "|||" + name + "|||enzyme reactions|||\n")
                        if drug['reactions']['reaction']['enzymes'] is not None:
                            for enzyme in drug['reactions']['reaction']['enzymes']['enzyme']:
                                try:
                                    G.add_edge(enzyme['name'] , name , label = "enzyme reactions")
                                    file_text.write(enzyme['name'] + "|||" + name + "|||enzyme reactions|||\n")
                                    file_reactions.write(enzyme['name'] + "|||" + name + "|||enzyme reactions|||\n")
                                except:
                                    G.add_edge(drug['reactions']['reaction']['enzymes']['enzyme']['name'] , name , label = "enzyme reactions")
                                    file_text.write(drug['reactions']['reaction']['enzymes']['enzyme']['name'] + "|||" + name + "|||enzyme reactions|||\n")
                                    file_reactions.write(drug['reactions']['reaction']['enzymes']['enzyme']['name'] + "|||" + name + "|||enzyme reactions|||\n")


print(G)
file = open("graph_visual.csv" , "wb")
nx.write_edgelist(G , file)

#os.environ["OPENAI_API_KEY"] = "sk-proj-BMrDaVZaTeEAUEhOlR27T3BlbkFJrsOdZq9Ymiurx839DQop"
