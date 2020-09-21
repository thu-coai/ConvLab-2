from convlab2.dst import DST


class ExampleModel(DST):
    def update_turn(self, sys_utt, user_utt):
        return {
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
