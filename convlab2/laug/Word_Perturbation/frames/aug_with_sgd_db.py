from convlab2.laug.Word_Perturbation.multiwoz.multiwoz_eda import MultiwozEDA
from convlab2.laug.Word_Perturbation.multiwoz.db.slot_value_replace import MultiSourceDBLoader, MultiSourceDBLoaderArgs
from convlab2.laug.Word_Perturbation.multiwoz.util import load_json, dump_json
from convlab2 import DATA_ROOT,get_root_path

def read_zipped_json(filepath, filename):
    print("zip file path = ", filepath)
    archive = zipfile.ZipFile(filepath, 'r')
    return json.load(archive.open(filename))


class frames_eda_config:
    def __init__(self,):
        self.frames=read_zipped_json(os.path.join(DATA_ROOT, 'frames/Ori','train.json.zip'),'train.json')
        
        frames_frames_domain_slot_map = {
        # ('frame', 'category'): ('hotel', 'category'),
        ('frame', 'dst_city'): ('hotel', 'location'),
        # ('frame', 'gst_rating'): ('hotel', 'gst_rating'),
        ('frame', 'name'): ('hotel', 'name'),

        ('frame', 'or_city'): ('trip', 'or_city'),
        # ('frame', 'seat'): ('trip', 'seat'),
    }

        frames_sgd_domain_slot_map = {
        ('frame', 'dst_city'): ('hotels', 'dst_city'),
        ('frame', 'name'): ('hotels', 'hotel_name'),

        ('frame', 'or_city'): ('travel', 'location'),
    }
        frames_db_dir=os.path.join(get_root_path(),"convlab2/laug/Word_Perturbation/db/frames-db/")
        sgd_db_dir=os.path.join(get_root_path(),"convlab2/laug/Word_Perturbation/db/sgd-db/")
        
        loader_args = [
        MultiSourceDBLoaderArgs(frames_db_dir, frames_frames_domain_slot_map),
        MultiSourceDBLoaderArgs(sgd_db_dir, frames_sgd_domain_slot_map)
    ]
        self.db_loader = MultiSourceDBLoader(loader_args)




def main(frames_filepath, output_filepath,
         frames_db_dir,
         sgd_db_dir,
         alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, num_aug=2,
         p_slot_value_replacement=0.25):
    frames = load_json(frames_filepath)

    frames_frames_domain_slot_map = {
        # ('frame', 'category'): ('hotel', 'category'),
        ('frame', 'dst_city'): ('hotel', 'location'),
        # ('frame', 'gst_rating'): ('hotel', 'gst_rating'),
        ('frame', 'name'): ('hotel', 'name'),

        ('frame', 'or_city'): ('trip', 'or_city'),
        # ('frame', 'seat'): ('trip', 'seat'),
    }

    frames_sgd_domain_slot_map = {
        ('frame', 'dst_city'): ('hotels', 'dst_city'),
        ('frame', 'name'): ('hotels', 'hotel_name'),

        ('frame', 'or_city'): ('travel', 'location'),
    }
    loader_args = [
        MultiSourceDBLoaderArgs(frames_db_dir, frames_frames_domain_slot_map),
        MultiSourceDBLoaderArgs(sgd_db_dir, frames_sgd_domain_slot_map)
    ]
    db_loader = MultiSourceDBLoader(loader_args)

    eda = MultiwozEDA(frames, db_loader,
                      inform_intents=('inform', 'switch_frame', 'confirm'),
                      slot_value_replacement_probability=p_slot_value_replacement,
                      alpha_sr=alpha_sr, alpha_ri=alpha_ri, alpha_rs=alpha_rs, p_rd=p_rd, num_aug=num_aug)
    result = eda.augment_multiwoz_dataset('usr')

    dump_json(result, output_filepath, indent=4)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--frames_filepath", default='multiwoz.json')
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
    parser.add_argument('--frames_db_dir', help='dir of frames db')
    opts = parser.parse_args()
    main(**vars(opts))
