import json
import os


def main():
    dir_path = os.path.dirname(os.path.abspath(__file__))
    postcode = []
    for domain in ['attraction', 'hotel', 'hospital', 'police', 'restaurant']:
        db = json.load(open(os.path.join(dir_path, "{}_db.json".format(domain))))
        for entry in db:
            if entry['postcode'] not in postcode:
                postcode.append(entry['postcode'])
    json.dump(postcode, open(os.path.join(dir_path, "postcode.json"), 'w'), indent=2)


if __name__ == '__main__':
    main()
