import json
import zipfile
from convlab2.dst.sumbt.crosswoz_en.sumbt_config import *

null = 'none'

def trans_value(value):
    trans = {
        '': 'none',
    }
    value = value.strip()
    value = trans.get(value, value)
    value = value.replace('’', "'")
    value = value.replace('‘', "'")
    return value

def convert_to_glue_format(data_dir, sumbt_dir):

    if not os.path.isdir(os.path.join(sumbt_dir, args.tmp_data_dir)):
        os.mkdir(os.path.join(sumbt_dir, args.tmp_data_dir))

    ### Read ontology file
    with open(os.path.join(data_dir, "ontology.json"), "r") as fp_ont:
        data_ont = json.load(fp_ont)
    ontology = {}
    facilities = []
    for domain_slot in data_ont:
        domain, slot = domain_slot.split('-', 1)
        if domain not in ontology:
            ontology[domain] = {}
        if slot.startswith('Hotel Facilities'):
            facilities.append(slot.split(' - ')[1])
        ontology[domain][slot] = set(map(str.lower, data_ont[domain_slot]))

    ### Read woz logs and write to tsv files
    tsv_filename = os.path.join(sumbt_dir, args.tmp_data_dir, "train.tsv")
    print('tsv file: ', os.path.join(sumbt_dir, args.tmp_data_dir, "train.tsv"))
    if os.path.exists(os.path.join(sumbt_dir, args.tmp_data_dir, "train.tsv")):
        print('data has been processed!')
        return 0
    else:
        print('processing data')

    with open(os.path.join(sumbt_dir, args.tmp_data_dir, "train.tsv"), "w") as fp_train, \
        open(os.path.join(sumbt_dir, args.tmp_data_dir, "dev.tsv"), "w") as fp_dev,      \
        open(os.path.join(sumbt_dir, args.tmp_data_dir, "test.tsv"), "w") as fp_test:

        fp_train.write('# Dialogue ID\tTurn Index\tUser Utterance\tSystem Response\t')
        fp_dev.write('# Dialogue ID\tTurn Index\tUser Utterance\tSystem Response\t')
        fp_test.write('# Dialogue ID\tTurn Index\tUser Utterance\tSystem Response\t')

        for domain in sorted(ontology.keys()):
            for slot in sorted(ontology[domain].keys()):
                fp_train.write(f'{str(domain)}-{str(slot)}\t')
                fp_dev.write(f'{str(domain)}-{str(slot)}\t')
                fp_test.write(f'{str(domain)}-{str(slot)}\t')

        fp_train.write('\n')
        fp_dev.write('\n')
        fp_test.write('\n')

        # fp_data = open(os.path.join(SELF_DATA_DIR, "data.json"), "r")
        # data = json.load(fp_data)

        file_split = ['train', 'val', 'test']
        fp = [fp_train, fp_dev, fp_test]

        for split_type, split_fp in zip(file_split, fp):

            zipfile_name = "{}.json.zip".format(split_type)
            zip_fp = zipfile.ZipFile(os.path.join(data_dir, zipfile_name))
            data = json.loads(str(zip_fp.read(zip_fp.namelist()[0]), 'utf-8'))

            for file_id in data:
                user_utterance = ''
                system_response = ''
                turn_idx = 0
                messages = data[file_id]['messages']
                for idx, turn in enumerate(messages):
                    if idx % 2 == 0:        # user turn
                        user_utterance = turn['content']
                    else:                   # system turn
                        user_utterance = user_utterance.replace('\t', ' ')
                        user_utterance = user_utterance.replace('\n', ' ')
                        user_utterance = user_utterance.replace('  ', ' ')

                        system_response = system_response.replace('\t', ' ')
                        system_response = system_response.replace('\n', ' ')
                        system_response = system_response.replace('  ', ' ')

                        split_fp.write(str(file_id))                   # 0: dialogue ID
                        split_fp.write('\t' + str(turn_idx))           # 1: turn index
                        split_fp.write('\t' + str(user_utterance))     # 2: user utterance
                        split_fp.write('\t' + str(system_response))    # 3: system response

                        # hardcode the value of facilities as 'yes' and 'no'
                        belief = {f'Hotel-Hotel Facilities - {str(facility)}': null for facility in facilities}
                        sys_state_init = turn['sys_state_init']
                        for domain, slots in sys_state_init.items():
                            for slot, value in slots.items():
                                # skip selected results
                                if isinstance(value, list):
                                    continue
                                if domain not in ontology:
                                    print("domain (%s) is not defined" % domain)
                                    continue

                                if slot == 'Hotel Facilities':
                                    for facility in value.split(','):
                                        belief[f'{str(domain)}-Hotel Facilities - {str(facility)}'] = 'yes'
                                else:
                                    if slot not in ontology[domain]:
                                        print("slot (%s) in domain (%s) is not defined" % (slot, domain))   # bus-arriveBy not defined
                                        continue

                                    value = trans_value(value).lower()

                                    if value not in ontology[domain][slot] and value != null:
                                        print("%s: value (%s) in domain (%s) slot (%s) is not defined in ontology" %
                                            (file_id, value, domain, slot))
                                        value = null

                                    belief[f'{str(domain)}-{str(slot)}'] = value

                        for domain in sorted(ontology.keys()):
                            for slot in sorted(ontology[domain].keys()):
                                key = str(domain) + '-' + str(slot)
                                if key in belief:
                                    val = belief[key]
                                    split_fp.write('\t' + val)
                                else:
                                    split_fp.write(f'\t{null}')

                        split_fp.write('\n')
                        split_fp.flush()

                        system_response = turn['content']
                        turn_idx += 1
    print('data has been processed!')
