import numpy as np

# model = torch.load("tensor_result.pt" , weights_only=False)

# np.savetxt('tensor_text.txt', torch.Tensor.cpu(model).numpy())
# print(model)

file_result = open("tensor_text.txt" , "r")
file_output = open("output_objects.txt" , "r")
file_translate_attribute = open("graphs_subset2\interactions_translate.txt" , "r" , encoding="utf-8")
file_translate_object = open("graphs_subset2\objects_translate.txt" , "r" , encoding="utf-8")
NUM_OBJECTS = 1999

file_final = open("final.txt" , "w+" , encoding="utf-8")
file_compare_high = open("output_compare_high.txt" , "w+" , encoding="utf-8")
file_compare_low = open("output_compare_low.txt" , "w+" , encoding="utf-8")

chance = []
objects = []
attributes = []

attribute_translate = []
object_translate = []

max_attribute = 0

for line in file_translate_attribute:
    attribute_translate.append(line)
    max_attribute = max_attribute + 1 

for line in file_translate_object:
    object_translate.append(line)

for line in file_result:
    chance.append(float(line.strip()))

for line in file_output:
    tmp = int(line.strip())
    if tmp>NUM_OBJECTS:
        attributes.append(tmp-NUM_OBJECTS-1)
    else:
        objects.append(tmp)

i=0
check_debug = 0
for chan in chance:
    if(attributes[i]<max_attribute):
        string_object = object_translate[objects[i]]
        string_attribute = attribute_translate[attributes[i]]
        check = 0
        final_line = ""
        final_cmp_line = ""
        cmp_obj1 = ""
        cmp_obj2 = string_object
        count_ex = 0
        cmp_check = 0 #0 we arent reading a drug
        for letter in string_attribute:
            if letter!="!":

                if count_ex == 3 and cmp_check == 1 :
                    cmp_obj1 = cmp_obj1 + letter

                if check != 1:
                    final_line = final_line + letter
                    if cmp_check == 0: 
                        final_cmp_line = final_cmp_line + letter
                    if cmp_check == 1:
                        if letter ==" " or letter == "." or letter == ",":
                            if cmp_check == 1: 
                                final_cmp_line = final_cmp_line + letter
                                cmp_check = 0
                else:
                    if letter == " " or letter == ".":
                        if cmp_check == 1: 
                            final_cmp_line = final_cmp_line + letter
                            cmp_check = 0
                        final_line = final_line + letter
                        check = 2
            else:
                count_ex = count_ex+1
                if count_ex == 3:
                    if cmp_check == 0:
                        final_cmp_line = final_cmp_line + "_"
                        cmp_check = cmp_check + 1
                    check = check + 1
                    final_line = final_line + string_object
                if count_ex == 6:
                    if cmp_check == 0:
                        final_cmp_line = final_cmp_line + "_"
                        cmp_check = cmp_check + 1
        final_line.replace("!!!" , "")
        check_debug = check_debug + 1

         
        #file_final.write(string_object.replace("\n", "") + " " + string_attribute.replace("\n", "") + " " + str(chan).replace("\n", "") + "\n")
        file_final.write(final_line.replace("\n", "") + " " + str(chan).replace("\n", "") + "\n")

        if(chan>0.75):
            file_compare_high.write(cmp_obj1.strip() + "|||" +cmp_obj2.strip() + "|||" + final_cmp_line.strip() + "|||\n")
        if(chan<0.25):
            file_compare_low.write(cmp_obj1.strip() + "|||" +cmp_obj2.strip() + "|||" + final_cmp_line.strip() + "|||\n")


    i = i + 1 