#!/usr/bin/env python
# coding: utf-8

# # Prepare 

import json
import re
zh_pattern = re.compile(u'[\u4e00-\u9fa5]+')


vocab_dict_path = 'vocab_dict.json'
vocab_dict = json.load(open(vocab_dict_path))

# # CrossWOZ ontology translate

def domain_translate(domain):
    if not zh_pattern.findall(domain):
        return domain
    return vocab_dict['domain_set'][domain]


def slot_translate(slot):
    if not zh_pattern.findall(slot):
        return slot
    return vocab_dict['slot_set'][slot.strip().lower()]


def value_translate(domain, slot, value):
    
    added_name = ['北京工体a. hotel酒店',
                  'bhg kitchen(欧陆广场店)',
                  'berry beans',
                  'costa coffee(天通苑)',
                  'cafe mignon谜鸟咖啡馆',
                  "let's burger plus(三里屯太古里店)",
                  'palms l.a. kitchen and bar洛杉矶厨房酒吧(亮马桥店)',
                  "yuan'  mousse& flower",
                  '北京万达文华酒店lobby lounge',
                  '北京亮china grill',
                  '千只鹤手作り料理 居楽屋(昌平店)',
                  '博璨德国啤酒餐厅 brotzeit bier bar&restaurant(亮马桥官舍店)',
                  '咖啡陪你caffe bene(华联力宝店)',
                  '庆丰包子铺 八达岭长城店(滚天沟停车场店)',
                  '恰·牛扒房char bar&grill',
                  '珍宝海鲜jumbo seafood(北京skp店)',
                  '福楼法餐厅 flo']
    
    error_tup = [
        ('餐馆', '电话', '人均消费'),
        ('酒店', '电话', '推荐锦江之星(北京奥体中心店)'),
        ('酒店', '地址', '酒店设施是否包含健身房')
    ]
    
    if (domain, slot, value) in error_tup:
        print(f'    Skip mismatching tuple: {domain}-{slot}-{value}')
        return ''
    
    if domain in ['greet', 'thank', 'welcome', 'bye', 'reqmore']:
        return value
    
    if domain not in vocab_dict['value_set'] or slot not in vocab_dict['value_set'][domain]:
        print(f'{domain}-{slot} has no vocab dict')
        raise Exception

    elif not zh_pattern.findall(value):
        return value
    else:
        try:
            if slot == '名称':
                # 名称-added里的单独拎出来翻译
                if value.lower() in added_name:
                    trans_value = vocab_dict['value_set'][domain][slot][value.strip().lower()]
                elif len(value.split()) > 1:
                    # 空格分割
                    ls = value.split()
                    trans_ls = [vocab_dict['value_set'][domain][slot][x] for x in ls]
                    trans_value = ','.join(trans_ls)
                else:
                    trans_value = vocab_dict['value_set'][domain][slot][value.strip().lower()]
            elif slot in ['推荐菜', '酒店设施']:
                # 空格分割
                ls = value.split()
                trans_ls = [vocab_dict['value_set'][domain][slot][x.strip().lower()] for x in ls]
                trans_value = ','.join(trans_ls)
            elif slot == 'selectedResults' and domain in ['出租', '地铁']:
                # 出租："出租（xx - yy）"
                if domain == '出租':
                    ls = value[4:-1].split(' - ')
                    trans_value = 'Taxi (' + vocab_dict['value_set'][domain][slot][ls[0].strip().lower()] + ' - ' + vocab_dict['value_set'][domain][slot][ls[1].strip().lower()] + ')'
                # 地铁："(起点) /(终点) xx"
                elif domain == '地铁':
                    assert '起点' in value or '终点' in value, f'地铁："(起点) /(终点) xx" format error: {value}'
                    v = value[5:]
                    if '起点' in value:
                        trans_value = '(Starting point) ' + vocab_dict['value_set'][domain][slot][v.strip().lower()]
                    elif '终点' in value:
                        trans_value = '(Termination) ' + vocab_dict['value_set'][domain][slot][v.strip().lower()]
            else:
                trans_value = vocab_dict['value_set'][domain][slot][value.strip().lower()]
            return trans_value
        except Exception as e:
            print(f'{domain}-{slot}-{value} translation failed')
            print(e)


def ontology_translate(typ, *args):
    assert typ in ['domain', 'slot', 'value'], 'Function translate() requires 1st argument: domain|slot|value.'
    if typ == 'domain':
        assert len(args) == 1, 'Needs 1 argument: domain.'
        return domain_translate(args[0])
    elif typ == 'slot':
        assert len(args) == 1, 'Needs 1 argument: slot.'
        return slot_translate(args[0])
    elif typ == 'value':
        assert len(args) == 3, 'Needs 3 argument: domain, slot, value.'
        return value_translate(args[0], args[1], args[2])

if __name__ == '__main__':
    print(ontology_translate('domain', '出租'))
    print(ontology_translate('slot', '推荐菜'))
    print(ontology_translate('value', '景点', '名称', '阳台山自然风景区'))
    print(ontology_translate('value', 'greet', 'none', 'none'))
