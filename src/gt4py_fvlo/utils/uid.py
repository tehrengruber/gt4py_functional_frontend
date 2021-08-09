_uid_counters = {}

def uid(key):
    global _uid_counters
    if key not in _uid_counters:
        _uid_counters[key] = 0
    _uid_counters[key]+=1
    return _uid_counters[key]
