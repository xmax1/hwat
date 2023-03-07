from prettytable import PrettyTable

def dict_depth(data: dict):
	depth = 0
	for key, value in data.items():
		if isinstance(value, dict):
			depth = max(depth, dict_depth(value))
	return depth + 1
	
def get_keys_at_depth(d: dict, depth: int):
	""" Get all keys in a nested dictionary at a certain depth level. """
	if depth == 0:
		return d.keys()
	else:
		keys = []
		for key, value in d.items():
			if isinstance(value, dict):
				keys.extend(get_keys_at_depth(value, depth - 1))
		return keys

def get_kpath_v(d: dict, parent: str= None):
	items = []
	for key, value in d.items():
		label = key if parent is None else f'{parent}.{key}'
		if isinstance(value, dict):
			items.extend(get_kpath_v(value, parent= label))
		else:
			items.append((label, value))
	return items

def print_table(data: dict | list, header: list=None):
	if not isinstance(data, list):
		depth = dict_depth(data)
		if header is None:
			header = [f'depth-{i}' for i in range(depth)] + ['value']
		data = get_kpath_v(data)
	if header is None:
		header = ['setting', 'value']
	table = PrettyTable(header)
	table.add_rows(data)
	print(table)