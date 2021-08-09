def isinstance_(value, type_):
    return isinstance(value, type_)

def if_(condition, true_result, false_result):
    return true_result() if condition else false_result()

def zip_(*iterables):
    return zip(*iterables)

def tuple_(arg):
    return tuple(arg)