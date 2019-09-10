import logging
import pickle
import os
import glob

from concurrent.futures import ThreadPoolExecutor

from tqdm.auto import tqdm

from eolearn.core import EOPatch

from eoflow.data_loader import EOMultiTempDataGenerator
from eoflow.utils import process_config
from eoflow.utils import create_dirs
from eoflow.utils import get_args

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)
    except RuntimeError:
        print("missing or invalid arguments")
        exit(0)

    # create the experiments dirs
    create_dirs([config.log_dir, config.checkpoint_dir])

    data_creator = EOMultiTempDataGenerator(config)

    def save_patchlets(filename):
        eopatch = EOPatch.load(filename, lazy_loading=True)
        patch_name = filename.split('/')[-1]

        if not os.path.isdir(f'{config.save_patchlet_dir}{patch_name}/'):
            os.system(f'mkdir {config.save_patchlet_dir}{patch_name}/')

        if len(glob.glob(f'{config.save_patchlet_dir}{patch_name}/*')) == config.n_patchlets:
            return None

        for index in range(config.n_patchlets):
            seed = data_creator.get_seed_for_patch(filename, index)
            patchlet = data_creator.read_eopatch(filename, seed, eopatch)
            if patchlet is not None:
                pickle.dump(patchlet,
                            open(f'{config.save_patchlet_dir}{patch_name}/patchlet_{patch_name}_{index}.pkl', 'wb'))

    with open(os.path.join(config.log_dir, config.train_cval_log), 'rb') as f:
        train_set, _ = pickle.load(f)

    with ThreadPoolExecutor(max_workers=config.workers) as executor:
        _ = list(tqdm(executor.map(save_patchlets, train_set), total=len(train_set)))
        

if __name__ == '__main__':
    main()
