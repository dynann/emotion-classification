emotions = sorted(list(set(dataset["train"]["emotion"])))
label2id = {e: i for i, e in enumerate(emotions)}
id2label = {i: e for e, i in label2id.items()}
num_labels = len(label2id)
