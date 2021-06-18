import tqdm

from .task_oriented_eda import eda
from .types import MultiwozSampleType, SentenceType, MultiwozDatasetType
from .util import AugmentationRecorder, iter_dialogues, Helper, choice, is_span_info_consistent_with_text, p_str
from .tokenize_util import tokenize, convert_tokens_to_string, convert_sentence_to_tokens
from .db.slot_value_replace import replace_slot_values_in_turn, MultiSourceDBLoader, assert_correct_turn


class MultiwozEDA:
    def __init__(self, multiwoz: MultiwozDatasetType,
                 db_loader: MultiSourceDBLoader,
                 inform_intents=('inform',),
                 slot_value_replacement_probability=0.25,
                 alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, num_aug=2):
        # attributes for slot value replacement
        self.db_loader = db_loader
        self.inform_intents = inform_intents
        self.slot_value_replacement_probability = slot_value_replacement_probability

        # attributes for EDA.
        self.eda_config = dict(alpha_sr=alpha_sr, alpha_ri=alpha_ri, alpha_rs=alpha_rs, p_rd=p_rd, num_aug=num_aug)
        self.multiwoz = multiwoz
        self.helper = Helper(multiwoz)

    def _get_excluding_indexes(self, words, span_info, dialog_act):
        return self.helper._get_excluding_indexes(words, span_info, dialog_act)

    def _augment_sentence_only(self, sentence: SentenceType, span_info, dialog_act):
        """don't change DA (span indexes may change)"""
        words = convert_sentence_to_tokens(sentence)
        excluding_indexes = self._get_excluding_indexes(words, span_info, dialog_act)

        for new_words, index_map in eda(words, **self.eda_config, excluding_indexes=excluding_indexes):
            new_span_info = []
            for x in span_info:
                new_span_info.append([*x[:3], index_map[x[3]], index_map[x[4]]])
            yield convert_tokens_to_string(new_words), new_span_info, dialog_act

    def augment_sentence_only(self, sentence: SentenceType, span_info, dialog_act):
        return list(self._augment_sentence_only(sentence, span_info, dialog_act))

    def _augment_sample(self, sample: MultiwozSampleType, mode='usr') -> AugmentationRecorder:
        recorder = AugmentationRecorder(sample)

        for turn_index, turn in iter_dialogues(sample, mode=mode):
            if not is_span_info_consistent_with_text(turn['text'], turn['span_info']):
                continue
            try:
                assert_correct_turn(turn)
            except:
                continue
            from copy import deepcopy
            orig_turn = deepcopy(turn)
            new_turn = replace_slot_values_in_turn(
                turn,
                self.db_loader,
                p=self.slot_value_replacement_probability,
                inform_intents=self.inform_intents
            )
            augmented = new_turn != turn
            turn = new_turn

            try:
                text = turn['text']
                span_info = turn['span_info']
                dialog_act = turn['dialog_act']
                tokens = tokenize(text)
                augmented_sentence, augmented_span_info, augmented_dialog_act = choice(
                    self._augment_sentence_only(tokens, span_info, dialog_act)
                )
            except (ValueError, IndexError):
                pass
            else:
                assert is_span_info_consistent_with_text(augmented_sentence, augmented_span_info), p_str(
                    [orig_turn, turn])
                augmented = True
                turn = {
                    'text': augmented_sentence,
                    'span_info': augmented_span_info,
                    'dialog_act': augmented_dialog_act,
                    **{k: v for k, v in turn.items() if k not in ('text', 'span_info', 'dialog_act')}
                }

            if augmented:
                recorder.add_augmented_dialog(turn_index, turn)
        return recorder

    def augment_sample(self, sample: MultiwozSampleType, mode='usr') -> MultiwozSampleType:
        return self._augment_sample(sample, mode=mode).get_augmented_sample()

    __call__ = augment_sample

    def augment_multiwoz_dataset(self, mode='usr', progress_bar=True):
        assert mode in ('usr', 'user', 'sys', 'all')
        res = {}
        if progress_bar:
            items = tqdm.tqdm(self.multiwoz.items(), total=len(self.multiwoz))
        else:
            items = self.multiwoz.items()
        for sample_id, sample in items:
            res[sample_id] = self.augment_sample(sample, mode=mode)
        return res
