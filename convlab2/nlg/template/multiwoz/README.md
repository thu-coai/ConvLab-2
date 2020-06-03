# Template-based NLG on Multiwoz

Template NLG for Multiwoz dataset. The templates are extracted from data and modified manually.

We select the utterances that have only one dialog act to extract templates. For `auto` mode, the templates may have several slot, while for `manual` mode, the templates only have one slot. As a result, `auto` templates can fail when some slot combination don't appear in dataset, while for `manual` mode, we generate utterance slot by slot, which could not fail but may be verbose. Notice that `auto` templates could be inappropriate.

## How to run

There are three mode:

- `auto`: templates extracted from data without manual modification, may have no match (return 'None');
- `manual`: templates with manual modification, sometimes verbose;
- `auto_manual`: use auto templates first. When fails, use manual templates.

Example:

```python
from convlab2.nlg.template.multiwoz import TemplateNLG

# dialog act
dialog_acts = [['Inform', 'Train', 'Day', 'wednesday'], ['Inform', 'Train', 'Leave', '10:15']]

print(dialog_acts)

# system model for manual, auto, auto_manual
nlg_sys_manual = TemplateNLG(is_user=False, mode='manual')
nlg_sys_auto = TemplateNLG(is_user=False, mode='auto')
nlg_sys_auto_manual = TemplateNLG(is_user=False, mode='auto_manual')

# generate
print('manual      : ', nlg_sys_manual.generate(dialog_acts))
print('auto        : ', nlg_sys_auto.generate(dialog_acts))
print('auto_manual : ', nlg_sys_auto_manual.generate(dialog_acts))
```
Result:
```
[['Inform', 'Train', 'Day', 'wednesday'], ['Inform', 'Train', 'Leave', '10:15']]
manual      :  The train is for wednesday you are all set. How about 10:15 will that work for you ?
auto        :  You can depart on wednesday at 10:15 , how does that sound ?
auto_manual :  There is a train leaving at 10:15 on wednesday .
```
