import sys
import time
import os
import json
import torch as th
import logging
from convlab2.policy.larl.multiwoz.experiments_woz.dialog_utils import task_generate
import convlab2.policy.larl.multiwoz.latent_dialog.domain as domain
from convlab2.policy.larl.multiwoz.latent_dialog.main import train, validate
from convlab2.policy.larl.multiwoz.latent_dialog.models_task import SysPerfectBD2Cat
from convlab2.policy.larl.multiwoz.latent_dialog.evaluators import MultiWozEvaluator
from convlab2.policy.larl.multiwoz.latent_dialog.data_loaders import BeliefDbDataLoaders
import convlab2.policy.larl.multiwoz.latent_dialog.corpora as corpora
from convlab2.policy.larl.multiwoz.latent_dialog.utils import Pack, prepare_dirs_loggers, set_seed




domain_name = 'object_division'
domain_info = domain.get_domain(domain_name)
config = Pack(
    seed=10,
    train_path='../data/train_dials.json',
    valid_path='../data/val_dials.json',
    test_path='../data/test_dials.json',
    max_vocab_size=1000,
    last_n_model=5,
    max_utt_len=50,
    max_dec_len=50,
    backward_size=2,
    batch_size=32,
    use_gpu=False,
    op='adam',
    init_lr=0.001,
    l2_norm=1e-05,
    momentum=0.0,
    grad_clip=5.0,
    dropout=0.5,
    max_epoch=100,
    embed_size=100,
    num_layers=1,
    utt_rnn_cell='gru',
    utt_cell_size=300,
    bi_utt_cell=True,
    enc_use_attn=True,
    dec_use_attn=True,
    dec_rnn_cell='lstm',
    dec_cell_size=300,
    dec_attn_mode='cat',
    y_size=10,
    k_size=20,
    beta=0.001,
    simple_posterior=True,
    contextual_posterior=True,
    use_mi=False,
    use_pr=True,
    use_diversity=False,
    #
    beam_size=20,
    fix_batch=True,
    fix_train_batch=False,
    avg_type='word',
    print_step=300,
    ckpt_step=1416,
    improve_threshold=0.996,
    patient_increase=2.0,
    save_model=True,
    early_stop=False,
    gen_type='greedy',
    preview_batch_num=None,
    k=domain_info.input_length(),
    init_range=0.1,
    pretrain_folder='2020-01-05-11-23-06-sl_cat',
    forward_only=False
)
set_seed(config.seed)
start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
stats_path = 'sys_config_log_model'
if config.forward_only:
    saved_path = os.path.join(stats_path, config.pretrain_folder)
    config = Pack(json.load(open(os.path.join(saved_path, 'config.json'))))
    config['forward_only'] = True
else:
    saved_path = os.path.join(
        stats_path, start_time+'-'+os.path.basename(__file__).split('.')[0])
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)
config.saved_path = saved_path

prepare_dirs_loggers(config)
logger = logging.getLogger()
logger.info('[START]\n{}\n{}'.format(start_time, '=' * 30))
config.saved_path = saved_path

# save configuration
with open(os.path.join(saved_path, 'config.json'), 'w') as f:
    json.dump(config, f, indent=4)  # sort_keys=True

corpus = corpora.NormMultiWozCorpus(config)
train_dial, val_dial, test_dial = corpus.get_corpus()

train_data = BeliefDbDataLoaders('Train', train_dial, config)
val_data = BeliefDbDataLoaders('Val', val_dial, config)
test_data = BeliefDbDataLoaders('Test', test_dial, config)

evaluator = MultiWozEvaluator('SysWoz')

model = SysPerfectBD2Cat(corpus, config)

if config.use_gpu:
    model.cuda()

best_epoch = None
if not config.forward_only:
    try:
        best_epoch = train(model, train_data, val_data,
                           test_data, config, evaluator, gen=task_generate)
    except KeyboardInterrupt:
        print('Training stopped by keyboard.')
if best_epoch is None:
    model_ids = sorted([int(p.replace('-model', ''))
                        for p in os.listdir(saved_path) if 'model' in p and 'rl' not in p])
    best_epoch = model_ids[-3]

print("$$$ Load {}-model".format(best_epoch))
config.batch_size = 32
model.load_state_dict(th.load(os.path.join(
    saved_path, '{}-model'.format(best_epoch))))


logger.info("Forward Only Evaluation")

validate(model, val_data, config)
validate(model, test_data, config)

with open(os.path.join(saved_path, '{}_{}_valid_file.txt'.format(start_time, best_epoch)), 'w') as f:
    task_generate(model, val_data, config, evaluator, num_batch=None, dest_f=f)

with open(os.path.join(saved_path, '{}_{}_test_file.txt'.format(start_time, best_epoch)), 'w') as f:
    task_generate(model, test_data, config,
                  evaluator, num_batch=None, dest_f=f)

end_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
print('[END]', end_time, '=' * 30)
