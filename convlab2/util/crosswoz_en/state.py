def default_state():
    state = dict(
        user_action=[],
        system_action=[],
        belief_state={},
        request_state={},
        terminated=False,
        history=[]
    )
    state['belief_state'] = {
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
            "dishes": "Food court",
            "cost": "50-100 yuan",
            "rating": "",
            "nearby attract.": "",
            "nearby rest.": "",
            "nearby hotels": ""
        },
        "Hotel": {
            "name": "",
            "type": "",
            "Hotel Facilities - Wake Up Service": "",
            "Hotel Facilities - Non-smoking Room": "",
            "Hotel Facilities - Business Center": "",
            "Hotel Facilities - Chinese Restaurant": "",
            "Hotel Facilities - Pick-up Service": "",
            "Hotel Facilities - Hair Dryer": "",
            "Hotel Facilities - International Call": "",
            "Hotel Facilities - Meeting Room": "",
            "Hotel Facilities - Broadband Internet": "",
            "Hotel Facilities - Childcare Services": "",
            "Hotel Facilities - WiFi throughout the Hotel": "",
            "Hotel Facilities - Heating": "",
            "Hotel Facilities - SPA": "",
            "Hotel Facilities - Luggage Storage": "",
            "Hotel Facilities - Western Restaurant": "",
            "Hotel Facilities - Bar": "",
            "Hotel Facilities - Breakfast Service": "",
            "Hotel Facilities - Gym": "",
            "Hotel Facilities - Free Local Calls": "",
            "Hotel Facilities - Disabled Facilities": "",
            "Hotel Facilities - Foreign Guests Reception": "",
            "Hotel Facilities - WiFi in Some Rooms": "",
            "Hotel Facilities - Laundry Service": "",
            "Hotel Facilities - Car Rental": "",
            "Hotel Facilities - WiFi in Public Areas and Some Rooms": "",
            "Hotel Facilities - 24-hour Hot Water": "",
            "Hotel Facilities - Hot Spring": "",
            "Hotel Facilities - Sauna": "",
            "Hotel Facilities - Pay Parking": "",
            "Hotel Facilities - WiFi in All Rooms": "",
            "Hotel Facilities - Chess-Poker Room": "",
            "Hotel Facilities - Free Domestic Long Distance Call": "",
            "Hotel Facilities - Indoor Swimming Pool": "",
            "Hotel Facilities - Free Breakfast Service": "",
            "Hotel Facilities - WiFi in Public Areas": "",
            "Hotel Facilities - Outdoor Swimming Pool": "",
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
    return state
