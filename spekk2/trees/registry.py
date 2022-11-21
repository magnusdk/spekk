registry = {}


def register(type, get_keys, get_value, create, repr_f=None):
    registry[type] = get_keys, get_value, create, repr_f


def treedef(tree):
    from spekk2.spec import Spec

    # Special case for Spec since we know they are just wrappers of trees
    if isinstance(tree, Spec):
        keys, get_value, create_tree, repr = treedef(tree.tree)
        create = lambda k, v: Spec(create_tree(k, v))
        return keys, get_value, create, repr
    else:
        get_keys, get_value, create, repr = registry.get(
            type(tree), (None, None, None, None)
        )
        return get_keys(tree) if get_keys else None, get_value, create, repr


def get_values(tree):
    keys, get_value, create, repr = treedef(tree)
    return [get_value(tree, k) for k in keys]


def get_keys(tree):
    keys, get_value, create, repr = treedef(tree)
    return keys


def repr(tree):
    keys, get_value, create, repr = treedef(tree)
    return repr(tree) if repr else tree.__repr__()


register(
    dict,
    lambda d: d.keys(),
    lambda d, k: d[k],
    lambda keys, values: {k: v for k, v in zip(keys, values)},
    lambda d: "{" + ", ".join(f"{k}: {v}" for k, v in d.items()) + "}",
)
register(
    list,
    lambda l: range(len(l)),
    lambda l, i: l[i],
    lambda keys, values: list(values),
    lambda l: "[" + ", ".join(f"{v}" for v in l) + "]",
)
register(
    tuple,
    lambda t: range(len(t)),
    lambda t, i: t[i],
    lambda keys, values: tuple(values),
    lambda t: "(" + ", ".join(f"{v}" for v in t) + (")" if len(t) > 1 else ",)"),
)
