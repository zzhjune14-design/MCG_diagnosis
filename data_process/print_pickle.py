import pickle

pickle_file = r"E:\Pythonpro\MCG_quexue_xinshuai\data_pickle\1701.pickle"

with open(pickle_file, "rb") as f:
    data = pickle.load(f)

print(data)
