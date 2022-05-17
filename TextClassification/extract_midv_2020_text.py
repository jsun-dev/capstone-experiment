import os
import json

MIDV_2020_PATH = os.path.join('MIDV-2020-Annotations', 'annotations')

JSON_FILES = os.listdir(MIDV_2020_PATH)

# Extract all non-empty key-value pairs
for jf in JSON_FILES:
    # Make the subdirectory
    subdirectory = os.path.splitext(jf)[0]
    os.makedirs(subdirectory)

    # Read the current JSON file
    f = open(os.path.join(MIDV_2020_PATH, jf), encoding='utf-8')
    data = json.load(f)
    imgMetadata = data['_via_img_metadata']

    # Search for any key-value pairs
    for i in imgMetadata:
        filename = imgMetadata[i]['filename']
        filename = os.path.splitext(filename)[0] + '.json'
        regions = imgMetadata[i]['regions']

        # Extract non-empty key-value pairs
        kvp = []
        for region in regions:
            attributes = region['region_attributes']
            field_name = attributes['field_name']
            value = attributes['value']
            if value:
                kvp.append((field_name, value))

        # Write the key-value pairs to its own JSON file
        with open(os.path.join(subdirectory, filename), 'w') as fp:
            json.dump(dict(kvp), fp, indent=4)

    # Close the JSON file
    f.close()