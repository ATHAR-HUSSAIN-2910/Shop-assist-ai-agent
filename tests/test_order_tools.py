# from src.tools.order_tools import track_order


# print("Testing existing order...")

# result = track_order("1024")

# print(result)


# print("\nTesting non-existing order...")

# result = track_order("9999")

# print(result)


from src.tools.order_tools import cancel_order


print("Testing cancellable order...")
result = cancel_order("1025")
print(result)


print("\nTesting non-cancellable order...")
result = cancel_order("1024")
print(result)


print("\nTesting non-existing order...")
result = cancel_order("9999")
print(result)