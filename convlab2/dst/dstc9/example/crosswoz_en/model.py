from convlab2.dst import DST


class ExampleModel(DST):
    def init_session(self):
        self.history = []
        self.state = {
            "Attraction": {
                "name": "",
                "fee": "",
                "duration": "",
                "rating": "",
                "nearby attract.": "",
                "nearby rest.": "",
                "nearby hotels": ""
            },
            "Restaurant": {
                "name": "",
                "dishes": "",
                "cost": "",
                "rating": "",
                "nearby attract.": "",
                "nearby rest.": "",
                "nearby hotels": ""
            },
            "Hotel": {
                "name": "",
                "type": "",
                "Hotel Facilities": "",
                "price": "",
                "rating": "",
                "nearby attract.": "",
                "nearby rest.": "",
                "nearby hotels": ""
            },
            "Metro": {
                "from": "",
                "to": ""
            },
            "Taxi": {
                "from": "",
                "to": ""
            }
        }

    def update_turn(self, sys_utt, user_utt):
        if sys_utt is not None:
            self.history.append(sys_utt)
        self.history.append(user_utt)
        # model can do some modification to state here
        return self.state
