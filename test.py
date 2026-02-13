import json


def save_data_to_json(original_data, output):
    new_data = {}
    for (prefix, index), value in original_data.items():
        new_key = f"{prefix},Filter{index}"
        new_data[new_key] = value

    # Save to JSON file
    with open(output, 'w') as f:
        json.dump(new_data, f, indent=4)  # indent for readability

def sort_json_values(json_file,output):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # data = json.loads(json_file)

    sorted_items = sorted(data.items(), key=lambda item: item[1], reverse=True)
    sorted_data = dict(sorted_items)
    with open(output, 'w') as f:
        json.dump(sorted_data, f, indent=4)