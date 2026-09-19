from scipy.io import loadmat

data = loadmat("data/annotations.mat")

labels = data["labels"]
names = data["names"]

print("=== STRUCTURE ===")

print("labels dtype:")
print(labels.dtype)

print()
print("names dtype:")
print(names.dtype)

print()
print("=== FIELD SHAPES ===")

for field in labels.dtype.names:
    value = labels[field][0, 0]

    print(
        field,
        "labels:", type(value),
        getattr(value, "shape", None),
        getattr(value, "dtype", None)
    )

print()
print("=== NAMES FIELD SHAPES ===")

for field in names.dtype.names:
    value = names[field][0, 0]

    print(
        field,
        "names:", type(value),
        getattr(value, "shape", None),
        getattr(value, "dtype", None)
    )