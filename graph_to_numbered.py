import networkx as nx

objects = []
attributes = []
indices = []
#CHANGE graphs_subset to graphs for the whole db
#interactions = []
graph = nx.Graph()
graph_interactions = nx.Graph()
file = open("graphs_subset2\graph_interactions.txt" , "r" , encoding="utf8")
lines = file.readlines()
file_translate = open("graphs_subset2\interactions_translate.txt" , "w+" , encoding="utf8")
file_translate_objects = open("graphs_subset2\objects_translate.txt" , "w+" , encoding="utf8")
#file_translate_attributes = open("graphs_subset2\attributes_translate.txt" , "w+" , encoding = "utf8")
print("##########################")
max_obj = 0
max_att = 0

for line in lines:
    res = line.split("|||")
    # if res[2] not in interactions:
    #     interactions.append(res[2])
    #graph_interactions.add_edge(res[0] , res[1] , label = res[2])
    replace0 = res[2].replace("_" , "!!!"+res[1])
    replace1 = res[2].replace("_" , "!!!"+res[0])
    if res[0] not in objects:
        objects.append(res[0])
    if res[1] not in objects:
        objects.append(res[1])
    if replace0 not in attributes:
        attributes.append(replace0)
        #file_translate.write(str(attributes.index(replace0)) + " " + replace0+"\n")
        file_translate.write(replace0+"\n")
    if replace1 not in attributes:
        attributes.append(replace1)
        #file_translate.write(str(attributes.index(replace1)) + " " + replace1+"\n")
        file_translate.write(replace1+"\n")
    indices.append([objects.index(res[0]) , attributes.index(replace0)])
    indices.append([objects.index(res[1]) , attributes.index(replace1)])
    graph_interactions.add_edge(res[0] , replace0)
    graph_interactions.add_edge(res[1] , replace1)
    graph.add_edge(res[0] , replace0)
    graph.add_edge(res[1] , replace1)


# file = open("graphs\interactions.txt" , "w+")
# print(interactions)
# for interaction in interactions:
#     file.write(interaction)
#     file.write("\n")


graph_classifications = nx.Graph()


print("##########################")
file = open("graphs_subset2\graph_classifications.txt" , "r" , encoding="utf8")
lines = file.readlines()

for line in lines:
    res = line.split("|||")
    if res[0] not in objects:
        objects.append(res[0])
    if res[1] not in objects:
        objects.append(res[1])
    if res[2] not in attributes:
        attributes.append(res[2])
    indices.append([objects.index(res[0]) , attributes.index(res[2])])
    indices.append([objects.index(res[1]) , attributes.index(res[2])])
    graph_classifications.add_edge(res[0] , res[2])
    graph_classifications.add_edge(res[1] , res[2])
    graph.add_edge(res[0] , res[2])
    graph.add_edge(res[1] , res[2])
    #graph_classifications.add_edge(res[0] , res[1] , label = res[2])

graph_mechanism = nx.Graph()
print("##########################")
file = open("graphs_subset2\graph_mechanism.txt" , "r" , encoding="utf8")
lines = file.readlines()

for line in lines:
    res = line.split("|||")
    if res[0] not in objects:
        objects.append(res[0])
    if res[1] not in objects:
        objects.append(res[1])
    if res[2] not in attributes:
        attributes.append(res[2])
    indices.append([objects.index(res[0]) , attributes.index(res[2])])
    indices.append([objects.index(res[1]) , attributes.index(res[2])])
    graph_mechanism.add_edge(res[0] , res[2])
    graph_mechanism.add_edge(res[1] , res[2])
    graph.add_edge(res[0] , res[2])
    graph.add_edge(res[1] , res[2])
    #graph_mechanism.add_edge(res[0] , res[1] , label = res[2])

graph_categories = nx.Graph()
print("##########################")
file = open("graphs_subset2\graph_categories.txt" , "r" , encoding="utf8")
lines = file.readlines()

for line in lines:
    res = line.split("|||")
    if res[0] not in objects:
        objects.append(res[0])
    if res[1] not in objects:
        objects.append(res[1])
    if res[2] not in attributes:
        attributes.append(res[2])
    indices.append([objects.index(res[0]) , attributes.index(res[2])])
    indices.append([objects.index(res[1]) , attributes.index(res[2])])
    graph_categories.add_edge(res[0] , res[2])
    graph_categories.add_edge(res[1] , res[2])
    graph.add_edge(res[0] , res[2])
    graph.add_edge(res[1] , res[2])
    #graph_categories.add_edge(res[0] , res[1] , label = res[2])

graph_enzymes = nx.Graph()
print("##########################")
file = open("graphs_subset2\graph_enzymes.txt" , "r" , encoding="utf8")
lines = file.readlines()

for line in lines:
    res = line.split("|||")
    if res[0] not in objects:
        objects.append(res[0])
    if res[1] not in objects:
        objects.append(res[1])
    if res[2] not in attributes:
        attributes.append(res[2])
    indices.append([objects.index(res[0]) , attributes.index(res[2])])
    indices.append([objects.index(res[1]) , attributes.index(res[2])])
    graph_enzymes.add_edge(res[0] , res[2])
    graph_enzymes.add_edge(res[1] , res[2])
    graph.add_edge(res[0] , res[2])
    graph.add_edge(res[1] , res[2])
    #graph_enzymes.add_edge(res[0] , res[1] , label = res[2])

graph_reactions = nx.Graph()
print("##########################")
file = open("graphs_subset2\graph_reactions.txt" , "r" , encoding="utf8")
lines = file.readlines()

for line in lines:
    res = line.split("|||")
    if res[0] not in objects:
        objects.append(res[0])
    if res[1] not in objects:
        objects.append(res[1])
    if res[2] not in attributes:
        attributes.append(res[2])
    indices.append([objects.index(res[0]) , attributes.index(res[2])])
    indices.append([objects.index(res[1]) , attributes.index(res[2])])
    graph_reactions.add_edge(res[0] , res[2])
    graph_reactions.add_edge(res[1] , res[2])
    graph.add_edge(res[0] , res[2])
    graph.add_edge(res[1] , res[2])
    #graph_reactions.add_edge(res[0] , res[1] , label = res[2])
# flag = 0
# for i in range(0 , len(indices)-1):
#     if flag == 1:
#         i = i - 1
#     for j in range(i+1 , len(indices)):
#         try :
#             flag = 0
#             if indices[i][0] == indices[j][0] and indices[i][1] == indices[j][1]:
#                 indices.pop(j)
#                 flag = 1
#                 break
        
#         except IndexError:
#             print(str(i) + " " + str(j))
#             exit()

import itertools
indices.sort()
indices = list(k for k,_ in itertools.groupby(indices))

file_numbers= open("graphs_subset2\graph_numbered.txt" , "w+" , encoding="utf8")
for index in indices:
    if index[0]>max_obj:
        max_obj = index[0]
    if index[1]>max_att:
        max_att = index[1]
    file_numbers.write(str(index[0]) + " " + str(index[1])+"\n")
# file_numbers= open("graphs_subset\graph_numbered.txt" , "w+" , encoding="utf8")
# for index in indices:
#     file_numbers.write(str(index[0]) + " " + str(index[1])+"\n")


print(max_obj)
print(max_att)

for object in objects:
    #file_translate_objects.write(str(objects.index(object)) + " " + object+"\n")
    file_translate_objects.write(object+"\n")



# final = nx.Graph()
# labels = {}

# p = nx.shortest_path(graph_interactions , drugs[0])
# temp = drugs[0]
# if drugs[1] in p:
#     for drug_interaction in p[drugs[1]]:
#         if drug_interaction not in drugs:
#             final.add_edge(temp , drug_interaction , label = graph_interactions.get_edge_data(temp,drug_interaction))
#             labels.update( {(temp,drug_interaction): "interact" }  )
#             temp = drug_interaction
#     #final.add_edge(temp , drugs[1] , label = graph_interactions.get_edge_data(temp , drugs[1] , graph_interactions.get_edge_data(temp,drugs[1])))
#     final.add_edge(temp , graph_interactions.get_edge_data(temp , graph_interactions.get_edge_data(temp,drugs[1])))
#     final.add_edge(temp , graph_interactions.get_edge_data(drugs[1] , graph_interactions.get_edge_data(temp,drugs[1])))
#     labels.update( {(temp,drug_interaction): "interact" }  )

# p = nx.shortest_path(graph_enzymes , drugs[0])
# temp = drugs[0]
# if drugs[1] in p:
#     for drug_interaction in p[drugs[1]]:
#         if drug_interaction not in drugs:
#             final.add_edge(temp , drug_interaction , label = graph_enzymes.get_edge_data(temp,drug_interaction))
#             labels.update( {(temp,drug_interaction): graph_enzymes.get_edge_data(temp,drug_interaction)['label'] }  )
#             temp = drug_interaction
#     final.add_edge(temp , drugs[1] , label = graph_enzymes.get_edge_data(temp , drugs[1] , graph_enzymes.get_edge_data(temp,drugs[1])))
#     labels.update( {(temp,drug_interaction): graph_enzymes.get_edge_data(temp,drug_interaction)['label'] }  )


# p = nx.shortest_path(graph_classifications , drugs[0])
# temp = drugs[0]
# if drugs[1] in p:
#     for drug_interaction in p[drugs[1]]:
#         if drug_interaction not in drugs:
#             final.add_edge(temp , drug_interaction , label = graph_classifications.get_edge_data(temp,drug_interaction))
#             labels.update( {(temp,drug_interaction): graph_classifications.get_edge_data(temp,drug_interaction)['label'] }  )
#             temp = drug_interaction
#     final.add_edge(temp , drugs[1] , label = graph_classifications.get_edge_data(temp , drugs[1] , graph_classifications.get_edge_data(temp,drugs[1])))
#     labels.update( {(temp,drug_interaction): graph_classifications.get_edge_data(temp,drug_interaction)['label'] }  )

# p = nx.shortest_path(graph_reactions , drugs[0])
# temp = drugs[0]
# if drugs[1] in p:
#     for drug_interaction in p[drugs[1]]:
#         if drug_interaction not in drugs:
#             final.add_edge(temp , drug_interaction , label = graph_reactions.get_edge_data(temp,drug_interaction))
#             labels.update( {(temp,drug_interaction): graph_reactions.get_edge_data(temp,drug_interaction)['label'] }  )
#             temp = drug_interaction
#     final.add_edge(temp , drugs[1] , label = graph_reactions.get_edge_data(temp , drugs[1] , graph_reactions.get_edge_data(temp,drugs[1])))
#     labels.update( {(temp,drug_interaction): graph_reactions.get_edge_data(temp,drug_interaction)['label'] }  )

# p = nx.shortest_path(graph_categories , drugs[0])
# temp = drugs[0]
# if drugs[1] in p:
#     for drug_interaction in p[drugs[1]]:
#         if drug_interaction not in drugs:
#             print(drug_interaction)
#             final.add_edge(temp , drug_interaction , label = graph_categories.get_edge_data(temp,drug_interaction))
#             labels.update( {(temp,drug_interaction): graph_categories.get_edge_data(temp,drug_interaction)['label'] }  )
#             temp = drug_interaction
#     final.add_edge(temp , drugs[1] , label = graph_categories.get_edge_data(temp , drugs[1] , graph_categories.get_edge_data(temp,drugs[1])))
#     labels.update( {(temp,drug_interaction): graph_categories.get_edge_data(temp,drug_interaction)['label'] }  )

# p = nx.shortest_path(graph_mechanism , drugs[0])
# temp = drugs[0]
# if drugs[1] in p:
#     for drug_interaction in p[drugs[1]]:
#         if drug_interaction not in drugs:
#             print(drug_interaction)
#             final.add_edge(temp , drug_interaction , label = graph_mechanism.get_edge_data(temp,drug_interaction))
#             labels.update( {(temp,drug_interaction): graph_mechanism.get_edge_data(temp,drug_interaction)['label'] }  )
#             temp = drug_interaction
#     final.add_edge(temp , drugs[1] , label = graph_mechanism.get_edge_data(temp , drugs[1] , graph_mechanism.get_edge_data(temp,drugs[1])))
#     labels.update( {(temp,drug_interaction): graph_mechanism.get_edge_data(temp,drug_interaction)['label'] }  )

# print(final)
# pos = nx.shell_layout(final)
# nx.draw_shell(final , with_labels=True)
# nx.draw_networkx_edge_labels(final , pos , edge_labels = labels , font_size = 10)
# print(labels)
# plt.savefig("images\graph_categories.png")

# print(graph_interactions.get_edge_data(drugs[0] , drugs[1]))
# print(graph_enzymes.get_edge_data(drugs[0],drugs[1]))
# print(graph_classifications.get_edge_data(drugs[0],drugs[1]))
# print(graph_reactions.get_edge_data(drugs[0],drugs[1]))
# print(graph_categories.get_edge_data(drugs[0],drugs[1]))
# print(graph_mechanism.get_edge_data(drugs[0],drugs[1]))

