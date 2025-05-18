
import os
import json
from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx
import re
import numpy as np

#OBJECTS = 3670
#ATTRIBUTES = 3245
OBJECTS = 1533
ATTRIBUTES = 419

matrix = np.zeros([OBJECTS, ATTRIBUTES] , dtype=int)
single_obj = np.zeros([OBJECTS] , dtype=int)
obj_att_map = []
final_obj = []
final_att = []
final = []
file_read = open("graphs_subset2/graph_numbered.txt" , "r")
file_write = open("graphs_subset2/graph_numbered_context.txt" , "w+" , encoding="utf8")

def merge(l1 , l2):
    #print(final_att)
    #print(l1)
    for att in final_att[l2]:
        if att not in final_att[l1]:
            final_att[l1].append(att)
    final_obj.pop(l2)
    final_att.pop(l2)

def check_list_eq(l1 , l2):
    if len(l1) != len(l2):
        return False
    for i in range(0 , len(l1)):
        if l1[i]!=l2[i]:
            return False
    return True

def calculate_attribute(list_att , list_obj , prev_count):
    list_object = []
    cur_count = 0
    #for our list of attributes find all objects with that attribute and add them to the list_object , cur_count is how many we added
    if not list_obj:
        for i in range(0 , OBJECTS):
            check = 1
            for j in range(0 , len(list_att)):
                if matrix[i , list_att[j]] == 0 :
                    check = 0
            if check == 1: 
                list_object.append(i)
                cur_count = cur_count+1
    else:
        for obj in list_obj:
            check = 1
            for j in range(0 , len(list_att)):
                if matrix[obj , list_att[j]] == 0 :
                    check = 0
            if  check == 1: 
                list_object.append(obj)
                cur_count = cur_count+1
    #if its empty return 0
    #print("OLD ATT")
    #print(list_att)
    if not list_object:
        #print("EMPTY LIST")
        return 0
    for obj1 in list_object:
        #print(obj_att_map[obj1])
        for att in obj_att_map[obj1]:
            # print("HERE")
            # print(att)
            check_att = 0
            for obj2 in list_object:
                if att not in obj_att_map[obj2]:
                    check_att = 1
            if check_att == 0:
                if att not in list_att:
                    list_att.append(att)

    if len(list_object) == 1:
        single_obj[list_object[0]] = 1
        return 0
    
    # if len(list_object) == 2:
    #     check0 = 0
    #     check1 = 0
    #     for att in obj_att_map[list_object[0]]:
    #         if att not in obj_att_map[list_object[1]]:
    #             check1 = 1
    #     for att in obj_att_map[list_object[1]]:
    #         if att not in obj_att_map[list_object[0]]:
    #             check0 = 1
    #     if check1 == check0 and check1 == 0:
    #         for att in obj_att_map[list_object[0]]:
    #             list_att.append(att)
    #             final.append(list_att , list_object)

    #print("lala")
    check = 0 
    new_att = []

    min_val = 10000 
    min = -1
    for obj in list_object:
        if min_val > len(obj_att_map[obj]):
            min = obj

    #print(list_object)
    #print("HERE")
    #print(list_att)
    for attribute in obj_att_map[min]:
        if attribute not in list_att:
            if attribute not in new_att :
                check_loop = 0
                for i in list_object:
                    if matrix[i,attribute] == 0:
                        check_loop = 1
                if check_loop == 0:
                    new_att.append(attribute)
    #print(new_att)
    #recure for every extra attribute 
    # print("NEW ATT")
    # print(new_att)
    #+print(list_object)
    for i in new_att:
        #print(list_att)
        next_list = list_att.copy()
        #print(next_list)
        if i not in next_list:
            next_list.append(i)
            tmp = calculate_attribute(next_list , list_object , cur_count)
            if tmp == 1:
                check = 1
    
    #print(list_att)

    if check == 0:
        final_obj.append(list_object)
        final_att.append(list_att)

    if cur_count == prev_count:
        return 1
    return 0

for i in range(0 , OBJECTS):
    obj_att_map.append([])

i=0
lines = file_read.readlines()
for line in lines:
    x , y = line.split()
    obj_att_map[int(x)].append(int(y))
    matrix[int(x)][int(y)]=1
    i=i+1


for i in range(0 , ATTRIBUTES):
    print("ITERATE " + str(i))
    arg = [i]
    calculate_attribute([i] , [] , -1)



count = 0
for objects in final_obj:
    i = count+1
    while i <len(final_obj)-2:
        #print(i)
        if i>=len(final_obj)-1:
            break
        #print(i)
        #print(len(final_obj))
        #print(final_obj)
        if(check_list_eq(objects , final_obj[i])):
            #print("MERGE")
            #print(count)
            #print(i)
            merge(count , i)
            i = count+1
        else:
            i = i + 1
    count = count+1

#print(final_obj[11])

#print(final)
#print("exiting")
i2 = 0
for i2 in range(0 , len(final_obj)-1) :
    for element in final_att[i2]:
        file_write.write(str(element) + " ")
    file_write.write("   ")
    for element in final_obj[i2]:
        file_write.write(str(element) + " ")
    file_write.write("\n")

for i in range(0 , OBJECTS):
    if single_obj[i] == 1:
        for att in obj_att_map[i]:
            file_write.write(str(att) + " ")
        file_write.write("   ")
        file_write.write(str(i))
        file_write.write("\n")


#i = 0
# for list in final:
#     #print(list)
#     for element in list:
#         file_write.write(str(element) + " ")
#     file_write.write("   ")
#     if i%2 == 1:
#         file_write.write("\n")
#     i = i + 1