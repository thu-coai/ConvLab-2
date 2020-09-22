from convlab2.dst import DST


class ExampleModel(DST):
    def init_session(self):
        self.history = []
        self.state = {
            "出租车": {
                "出发时间": "",
                "目的地": "",
                "出发地": "",
                "到达时间": "",
            },
            "餐厅": {
                "时间": "",
                "日期": "",
                "人数": "",
                "食物": "",
                "价格范围": "",
                "名称": "",
                "区域": "",
            },
            "公共汽车": {
                "人数": "",
                "出发时间": "",
                "目的地": "",
                "日期": "",
                "到达时间": "",
                "出发地": "",
            },
            "旅馆": {
                "停留天数": "",
                "日期": "",
                "人数": "",
                "名称": "",
                "区域": "",
                "停车处": "",
                "价格范围": "",
                "星级": "",
                "互联网": "",
                "类型": "",
            },
            "景点": {
                "类型": "",
                "名称": "",
                "区域": "",
            },
            "列车": {
                "票价": "",
                "人数": "",
                "出发时间": "",
                "目的地": "",
                "日期": "",
                "到达时间": "",
                "出发地": "",
            },
        }

    def update_turn(self, sys_utt, user_utt):
        if sys_utt is not None:
            self.history.append(sys_utt)
        self.history.append(user_utt)
        # model can do some modification to state here
        return self.state
