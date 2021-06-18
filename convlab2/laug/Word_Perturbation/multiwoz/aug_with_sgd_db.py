import os
from convlab2.laug.Word_Perturbation.multiwoz.multiwoz_eda import MultiwozEDA
from convlab2.laug.Word_Perturbation.multiwoz.db.slot_value_replace import MultiSourceDBLoader, MultiSourceDBLoaderArgs
from convlab2.laug.Word_Perturbation.multiwoz.util import load_json, dump_json
from convlab2 import DATA_ROOT,get_root_path
import json
import zipfile
def read_zipped_json(filepath, filename):
    print("zip file path = ", filepath)
    archive = zipfile.ZipFile(filepath, 'r')
    return json.load(archive.open(filename))


class multiwoz_eda_config:
    def __init__(self,):
        self.multiwoz=read_zipped_json(os.path.join(DATA_ROOT, 'multiwoz','train.json.zip'),'train.json')
        multiwoz_db_dir = os.path.join(DATA_ROOT, 'multiwoz', 'db')
        multiwoz_multiwoz_domain_slot_map = {
        ('attraction', 'area'): ('attraction', 'Area'),
        ('attraction', 'type'): ('attraction', 'Type'),
        ('attraction', 'name'): ('attraction', 'Name'),
        ('attraction', 'address'): ('attraction', 'Addr'),
        ('hospital', 'department'): ('hospital', 'Department'),
        ('hospital', 'address'): ('hospital', 'Addr'),
        ('hotel', 'type'): ('hotel', 'Type'),
        ('hotel', 'area'): ('hotel', 'Area'),
        ('hotel', 'name'): ('hotel', 'Name'),
        ('hotel', 'address'): ('hotel', 'Addr'),
        ('restaurant', 'food'): ('restaurant', 'Food'),
        ('restaurant', 'area'): ('restaurant', 'Area'),
        ('restaurant', 'name'): ('restaurant', 'Name'),
        ('restaurant', 'address'): ('restaurant', 'Addr'),
        ('train', 'destination'): ('train', 'Dest'),
        ('train', 'departure'): ('train', 'Depart')
    }

        multiwoz_sgd_domain_slot_map = {
        ('train', 'dest'): ('train', 'to'),
        ('train', 'depart'): ('train', 'from'),
        ('hotel', 'name'): ('hotels', 'hotel_name'),
        ('hotel', 'addr'): ('hotels', 'address'),
        ('attraction', 'name'): ('travel', 'attraction_name'),
        ('restaurant', 'name'): ('restaurants', 'restaurant_name'),
        ('restaurant', 'addr'): ('restaurants', 'street_address')
    }
        loader_args = [MultiSourceDBLoaderArgs(multiwoz_db_dir, multiwoz_multiwoz_domain_slot_map)]
        sgd_db_dir=os.path.join(get_root_path(),"convlab2/laug/Word_Perturbation/db/sgd-db/")
        loader_args.append(MultiSourceDBLoaderArgs(
            sgd_db_dir,
            multiwoz_sgd_domain_slot_map
        ))
        self.db_loader = MultiSourceDBLoader(loader_args)

def main(multiwoz_filepath, output_filepath,
         sgd_db_dir=None,
         alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, num_aug=2,
         p_slot_value_replacement=0.25):
    multiwoz = load_json(multiwoz_filepath)

    multiwoz_db_dir = os.path.join(DATA_ROOT, 'multiwoz', 'db')
    multiwoz_multiwoz_domain_slot_map = {
        ('attraction', 'area'): ('attraction', 'Area'),
        ('attraction', 'type'): ('attraction', 'Type'),
        ('attraction', 'name'): ('attraction', 'Name'),
        ('attraction', 'address'): ('attraction', 'Addr'),
        ('hospital', 'department'): ('hospital', 'Department'),
        ('hospital', 'address'): ('hospital', 'Addr'),
        ('hotel', 'type'): ('hotel', 'Type'),
        ('hotel', 'area'): ('hotel', 'Area'),
        ('hotel', 'name'): ('hotel', 'Name'),
        ('hotel', 'address'): ('hotel', 'Addr'),
        ('restaurant', 'food'): ('restaurant', 'Food'),
        ('restaurant', 'area'): ('restaurant', 'Area'),
        ('restaurant', 'name'): ('restaurant', 'Name'),
        ('restaurant', 'address'): ('restaurant', 'Addr'),
        ('train', 'destination'): ('train', 'Dest'),
        ('train', 'departure'): ('train', 'Depart')
    }

    multiwoz_sgd_domain_slot_map = {
        ('train', 'dest'): ('train', 'to'),
        ('train', 'depart'): ('train', 'from'),
        ('hotel', 'name'): ('hotels', 'hotel_name'),
        ('hotel', 'addr'): ('hotels', 'address'),
        ('attraction', 'name'): ('travel', 'attraction_name'),
        ('restaurant', 'name'): ('restaurants', 'restaurant_name'),
        ('restaurant', 'addr'): ('restaurants', 'street_address')
    }
    loader_args = [MultiSourceDBLoaderArgs(multiwoz_db_dir, multiwoz_multiwoz_domain_slot_map)]
    assert sgd_db_dir is not None
    loader_args.append(MultiSourceDBLoaderArgs(
        sgd_db_dir,
        multiwoz_sgd_domain_slot_map
    ))
    db_loader = MultiSourceDBLoader(loader_args)

    eda = MultiwozEDA(multiwoz, db_loader,
                      slot_value_replacement_probability=p_slot_value_replacement,
                      alpha_sr=alpha_sr, alpha_ri=alpha_ri, alpha_rs=alpha_rs, p_rd=p_rd, num_aug=num_aug)
    result = eda.augment_multiwoz_dataset('usr')

    dump_json(result, output_filepath, indent=4)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--multiwoz_filepath", '--multiwoz', default='multiwoz.json')
    parser.add_argument('--output_filepath', '--output', '-o', default='augmented_multiwoz.json')
    parser.add_argument('--alpha_sr', type=float, default=0.1, help='probability of replacement')
    parser.add_argument('--alpha_ri', type=float, default=0.1, help='probability of insertion')
    parser.add_argument('--alpha_rs', type=float, default=0.1, help='probability of swap')
    parser.add_argument('--p_rd', type=float, default=0.1, help="probability of deletion")
    parser.add_argument('--num_aug', type=int, default=2,
                        help="generate `num_aug` candidates with EDA and randomly choose one dialog as augmented dialog.")
    parser.add_argument('--p_slot_value_replacement', '-p_svr', type=float, default=0.25,
                        help='probability to replace a slot value.')
    parser.add_argument('--sgd_db_dir', '--sgd', help='dir of sgd db.')
    opts = parser.parse_args()
    main(**vars(opts))
