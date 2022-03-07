def single_input_single_output(dataset, device):
    data, target = dataset[0].to(device), dataset[1].to(device)
    return data, target


def single_input_multiple_output(dataset, device):
    data, targets = dataset[0], dataset[1:]
    data, targets = data.to(device), [target.to(device) for target in targets]
    return data, targets


def multiple_input_multiple_output(dataset, device, n):
    data, targets = dataset[:n], dataset[n:]
    data, targets = [datum.to(device) for datum in data], [target.to(device) for target in targets]
    return data, targets
