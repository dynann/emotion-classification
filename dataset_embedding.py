model.classifier = torch.nn.Sequential(
    torch.nn.Linear(768 + 8, 256),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.3),
    torch.nn.Linear(256, num_labels)
)
