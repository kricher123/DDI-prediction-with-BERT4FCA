These python files are taking the drugbank dataset for drugs as a json which im not allowed to upload. Splitter splits it into single json files per drug. Graph_create takes the directory with the set of drugs and creates the graph with all drugs as objects and attributes
as edges, it also creates seperate graphs per attribute. Graph_to_numbered transforms the graph from string to numbers each of which corresponds to an object or an attribute. we also hold a way to translate them back. context_creator transforms the graph into a formal 
context with concept lattices, it finds all maximal bi-cliques. attribute and object pretrain are the pretraining phases of the BERT4FCA method tailored to work with our data and finetune is the final part of the BERT4FCA method which works with our 2 pretraining files
and predicts new edges. output_reader and comparator are there to translate the final output into new interactions we predicted and compare those with the ones we already knew about

The original algorithm can be found at :https://github.com/kricher123/DDI-prediction-with-BERT4FCA/blob/main/graph_create.py and the paper on it at:https://arxiv.org/abs/2402.08236
