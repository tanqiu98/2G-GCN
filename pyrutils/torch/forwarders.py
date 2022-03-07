def basic_forward(model, data, **kwargs):
    return model(data)


def multiple_input_forward(model, data, **kwargs):
    return model(*data)
