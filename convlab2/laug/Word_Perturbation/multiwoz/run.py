import os, json
from convlab2.laug.Word_Perturbation.multiwoz.multiwoz_eda import MultiwozEDA
from convlab2.laug.Word_Perturbation.multiwoz.db.slot_value_replace import MultiSourceDBLoader, MultiSourceDBLoaderArgs
from convlab2.laug.Word_Perturbation.multiwoz.util import load_json
from convlab2 import DATA_ROOT


def main(multiwoz_filepath, output_filepath, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, num_aug=2,
         p_slot_value_replacement=0.25):
    multiwoz = load_json(multiwoz_filepath)

    db_dir = os.path.join(DATA_ROOT, 'multiwoz', 'db')
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
    loader_args = MultiSourceDBLoaderArgs(db_dir, multiwoz_multiwoz_domain_slot_map)
    db_loader = MultiSourceDBLoader(loader_args)

    eda = MultiwozEDA(multiwoz, db_loader,
                      slot_value_replacement_probability=p_slot_value_replacement,
                      alpha_sr=alpha_sr, alpha_ri=alpha_ri, alpha_rs=alpha_rs, p_rd=p_rd, num_aug=num_aug)
    result = eda.augment_multiwoz_dataset('usr')

    os.makedirs(os.path.dirname(os.path.abspath(output_filepath)), exist_ok=True)
    with open(output_filepath, 'w', encoding='utf-8') as out:
        json.dump(result, out, indent=4)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--multiwoz_filepath", '--multiwoz', default='multiwoz.json')
    parser.add_argument('--output_filepath', '--output', '-o', default='augmented_multiwoz.json')
    parser.add_argument('--alpha_sr', type=float, default=0.1, help='probability of replacement')
    parser.add_argument('--alpha_ri', type=float, default=0.1, help='probability of insertion')
    parser.add_argument('--alpha_rs', type=float, default=0.1, help='probability of swap')
    parser.add_argument('--p_rd', type=float, default=0.1, help="probability of deletion")
    parser.add_argument('--num_aug', type=int, default=2, help="generate `num_aug` candidates with EDA and randomly choose one dialog as augmented dialog.")
    parser.add_argument('--p_slot_value_replacement', '-p_svr', type=float, default=0.25, help='probability to replace a slot value.')
    opts = parser.parse_args()
    main(**vars(opts))
