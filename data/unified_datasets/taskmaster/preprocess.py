import json
import os
import copy
import zipfile
from tqdm import tqdm
import re
from convlab2.util.file_util import read_zipped_json, write_zipped_json
from pprint import pprint

descriptions = {
    "uber_lyft": {
        "uber_lyft": "order a car for a ride inside a city",
        "location.from": "pickup location",
        "location.to": "destination of the ride",
        "type.ride": "type of ride",
        "num.people": "number of people",
        "price.estimate": "estimated cost of the ride",
        "duration.estimate": "estimated duration of the ride",
        "time.pickup": "time of pickup",
        "time.dropoff": "time of dropoff",
    },
    "movie_ticket": {
        "movie_ticket": "book movie tickets for a film",
        "name.movie": "name of the movie",
        "name.theater": "name of the theater",
        "num.tickets": "number of tickets",
        "time.start": "start time of the movie",
        "location.theater": "location of the theater",
        "price.ticket": "price of the ticket",
        "type.screening": "type of the screening",
        "time.end": "end time of the movie",
        "time.duration": "duration of the movie",
    },
    "restaurant_reservation": {
        "restaurant_reservation": "searching for a restaurant and make reservation",
        "name.restaurant": "name of the restaurant",
        "name.reservation": "name of the person who make the reservation",
        "num.guests": "number of guests",
        "time.reservation": "time of the reservation",
        "type.seating": "type of the seating",
        "location.restaurant": "location of the restaurant",
    },
    "coffee_ordering": {
        "coffee_ordering": "order a coffee drink from either Starbucks or Peets for pick up",
        "location.store": "location of the coffee store",
        "name.drink": "name of the drink",
        "size.drink": "size of the drink",
        "num.drink": "number of drinks",
        "type.milk": "type of the milk",
        "preference": "user preference of the drink",
    },
    "pizza_ordering": {
        "pizza_ordering": "order a pizza",
        "name.store": "name of the pizza store",
        "name.pizza": "name of the pizza",
        "size.pizza": "size of the pizza",
        "type.topping": "type of the topping",
        "type.crust": "type of the crust",
        "preference": "user preference of the pizza",
        "location.store": "location of the pizza store",
    },
    "auto_repair": {
        "auto_repair": "set up an auto repair appointment with a repair shop",
        "name.store": "name of the repair store",
        "name.customer": "name of the customer",
        "date.appt": "date of the appointment",
        "time.appt": "time of the appointment",
        "reason.appt": "reason of the appointment",
        "name.vehicle": "name of the vehicle",
        "year.vehicle": "year of the vehicle",
        "location.store": "location of the repair store",
    },
    "flights": {
        "flights": "find a round trip or multi-city flights",
        "type": "type of the flight",
        "destination1": "the first destination city of the trip",
        "destination2": "the second destination city of the trip",
        "origin": "the origin city of the trip",
        "date.depart_origin": "date of departure from origin",
        "date.depart_intermediate": "date of departure from intermediate",
        "date.return": "date of return",
        "time_of_day": "time of the flight",
        "seating_class": "seat type (first class, business class, economy class, etc.",
        "seat_location": "location of the seat",
        "stops": "non-stop, layovers, etc.",
        "price_range": "price range of the flight",
        "num.pax": "number of people",
        "luggage": "luggage information",
        "total_fare": "total cost of the trip",
        "other_description": "other description of the flight",
        "from": "departure of the flight",
        "to": "destination of the flight",
        "airline": "airline of the flight",
        "flight_number": "the number of the flight",
        "date": "date of the flight",
        "from.time": "departure time of the flight",
        "to.time": "arrival time of the flight",
        "stops.location": "location of the stop",
        "fare": "cost of the flight",
    },
    "food_order": {
        "food_order": "order take-out for a particular cuisine choice",
        "name.item": "name of the item",
        "other_description.item": "other description of the item",
        "type.retrieval": "type of the retrieval method",
        "total_price": "total price",
        "time.pickup": "pick up time",
        "num.people": "number of people",
        "name.restaurant": "name of the restaurant",
        "type.food": "type of food",
        "type.meal": "type of meal",
        "location.restaurant": "location of the restaurant",
        "rating.restaurant": "rating of the restaurant",
        "price_range": "price range of the food",
    },
    "hotel": {
        "hotel": "find a hotel using typical preferences",
        "name.hotel": "name of the hotel",
        "location.hotel": "location of the hotel",
        "sub_location.hotel": "rough location of the hotel",
        "star_rating": "star rating of the hotel",
        "customer_rating": "customer rating of the hotel",
        "price_range": "price range of the hotel",
        "amenity": "amenity of the hotel",
        "num.beds": "number of beds to book",
        "type.bed": "type of the bed",
        "num.rooms": "number of rooms to book",
        "check-in_date": "check-in date",
        "check-out_date": "check-out date",
        "date_range": "date range of the reservation",
        "num.guests": "number of guests",
        "type.room": "type of the room",
        "price_per_night": "price per night",
        "total_fare": "total fare",
        "location": "location of the hotel",
    },
    "movie": {
        "movie": "find a movie to watch in theaters or using a streaming service at home",
        "name.movie": "name of the movie",
        "genre": "genre of the movie",
        "name.theater": "name of the theater",
        "location.theater": "location of the theater",
        "time.start": "start time of the movie",
        "time.end": "end time of the movie",
        "price.ticket": "price of the ticket",
        "price.streaming": "price of the streaming",
        "type.screening": "type of the screening",
        "audience_rating": "audience rating",
        "movie_rating": "film rating",
        "release_date": "release date of the movie",
        "runtime": "running time of the movie",
        "real_person": "name of actors, directors, etc.",
        "character": "name of character in the movie",
        "streaming_service": "streaming service that provide the movie",
        "num.tickets": "number of tickets",
        "seating": "type of seating",
    },
    "music": {
        "music": "find several tracks to play and then comment on each one",
        "name.track": "name of the track",
        "name.artist": "name of the artist",
        "name.album": "name of the album",
        "name.genre": "music genre",
        "type.music": "rough type of the music",
        "describes_track": "description of a track to find",
        "describes_artist": "description of a artist to find",
        "describes_album": "description of an album to find",
        "describes_genre": "description of a genre to find",
        "describes_type.music": "description of the music type",
    },
    "restaurant": {
        "restaurant": "ask for recommendations for a particular type of cuisine",
        "name.restaurant": "name of the restaurant",
        "location": "location of the restaurant",
        "sub-location": "rough location of the restaurant",
        "type.food": "the cuisine of the restaurant",
        "menu_item": "item in the menu",
        "type.meal": "type of meal",
        "rating": "rating of the restaurant",
        "price_range": "price range of the restaurant",
        "business_hours": "business hours of the restaurant",
        "name.reservation": "name of the person who make the reservation",
        "num.guests": "number of guests",
        "time.reservation": "time of the reservation",
        "date.reservation": "date of the reservation",
        "type.seating": "type of the seating",
    },
    "sport": {
        "sport": "discuss facts and stats about players, teams, games, etc. in EPL, MLB, MLS, NBA, NFL",
        "name.team": "name of the team",
        "record.team": "record of the team (number of wins and losses)",
        "record.games_ahead": "number of games ahead",
        "record.games_back": "number of games behind",
        "place.team": "ranking of the team",
        "result.match": "result of the match",
        "score.match": "score of the match",
        "date.match": "date of the match",
        "day.match": "day of the match",
        "time.match": "time of the match",
        "name.player": "name of the player",
        "position.player": "position of the player",
        "record.player": "record of the player",
        "name.non_player": "name of non-palyer such as the manager, coach",
        "venue": "venue of the match take place",
    }
}


def normalize_domain_name(domain):
    if domain == 'auto':
        return 'auto_repair'
    elif domain == 'pizza':
        return 'pizza_ordering'
    elif domain == 'coffee':
        return 'coffee_ordering'
    elif domain == 'uber':
        return 'uber_lyft'
    elif domain == 'restaurant':
        return 'restaurant_reservation'
    elif domain == 'movie':
        return 'movie_ticket'
    elif domain == 'flights':
        return 'flights'
    elif domain == 'food-ordering':
        return 'food_order'
    elif domain == 'hotels':
        return 'hotel'
    elif domain == 'movies':
        return 'movie'
    elif domain == 'music':
        return 'music'
    elif domain == 'restaurant-search':
        return 'restaurant'
    elif domain == 'sports':
        return 'sport'
    assert 0


def format_turns(ori_turns):
    new_turns = []
    previous_speaker = None
    utt_idx = 0
    for i, turn in enumerate(ori_turns):
        speaker = 'system' if turn['speaker'] == 'ASSISTANT' else 'user'
        turn['speaker'] = speaker
        if utt_idx == 0 and speaker == 'system':
            continue
        if turn['text'] == '(deleted)':
            continue
        if not previous_speaker:
            assert speaker != previous_speaker
        if speaker != previous_speaker:
            previous_speaker = speaker
            new_turns.append(copy.deepcopy(turn))
            utt_idx += 1
        else:
            # continuous speaking
            last_turn = new_turns[-1]
            # if ori_turns[i-1]['text'] == turn['text']:
            #     # skip repeat turn
            #     continue
            if turn['text'] in ori_turns[i-1]['text']:
                continue
            index_shift = len(last_turn['text']) + 1
            last_turn['text'] += ' '+turn['text']
            if 'segments' in turn:
                last_turn.setdefault('segments', [])
                for segment in turn['segments']:
                    segment['start_index'] += index_shift
                    segment['end_index'] += index_shift
                last_turn['segments'] += turn['segments']
    if new_turns and new_turns[-1]['speaker'] == 'system':
        new_turns = new_turns[:-1]
    return new_turns


def log_ontology(acts, ontology, ori_ontology):
    for item in acts:
        intent, domain, slot, value = item['intent'], item['domain'], item['slot'], item['value']
        if domain not in ontology['domains']:
            ontology['domains'][domain] = {'description': "", 'slots': {}}
        if slot not in ontology['domains'][domain]['slots']:
            ontology['domains'][domain]['slots'][slot] = {
                'description': '',
                'is_categorical': False,
                'possible_values': [],
                'count': 1
            }
        else:
            ontology['domains'][domain]['slots'][slot]['count'] += 1
        ontology['domains'][domain]['slots'][slot]['in original ontology'] = slot in ori_ontology[domain]
        if intent is not None and intent not in ontology['intents']:
            ontology['intents'][intent] = {
                "description": ''
            }


def preprocess():
    self_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dialogue = []
    ontology = {'domains': {},
                'intents': {},
                'binary_dialogue_act': [],
                'state': {}}
    original_zipped_path = os.path.join(self_dir, 'original_data.zip')
    new_dir = os.path.join(self_dir, 'original_data')
    if not os.path.exists(os.path.join(self_dir, 'data.zip')) or not os.path.exists(os.path.join(self_dir, 'ontology.json')):
        print('unzip to', new_dir)
        print('This may take several minutes')
        archive = zipfile.ZipFile(original_zipped_path, 'r')
        archive.extractall(self_dir)
        files = [
            ('TM-1-2019/woz-dialogs.json', 'TM-1-2019/ontology.json'),
            ('TM-1-2019/self-dialogs.json', 'TM-1-2019/ontology.json'),
            ('TM-2-2020/data/flights.json', 'TM-2-2020/ontology/flights.json'),
            ('TM-2-2020/data/food-ordering.json', 'TM-2-2020/ontology/food-ordering.json'),
            ('TM-2-2020/data/hotels.json', 'TM-2-2020/ontology/hotels.json'),
            ('TM-2-2020/data/movies.json', 'TM-2-2020/ontology/movies.json'),
            ('TM-2-2020/data/music.json', 'TM-2-2020/ontology/music.json'),
            ('TM-2-2020/data/restaurant-search.json', 'TM-2-2020/ontology/restaurant-search.json'),
            ('TM-2-2020/data/sports.json', 'TM-2-2020/ontology/sports.json')
        ]
        idx_count = 1
        total = 0

        for filename, ontology_filename in files:
            data = json.load(open(os.path.join(new_dir, filename)))
            ori_ontology = {}
            if 'TM-1' in filename:
                for domain, item in json.load(open(os.path.join(new_dir, ontology_filename))).items():
                    ori_ontology[item["id"]] = {}
                    for slot in item["required"] + item["optional"]:
                        ori_ontology[item["id"]][slot] = 0
            else:
                domain = normalize_domain_name(filename.split('/')[-1].split('.')[0])
                ori_ontology[domain] = {}
                for _, item in json.load(open(os.path.join(new_dir, ontology_filename))).items():
                    for group in item:
                        for anno in group["annotations"]:
                            ori_ontology[domain][anno] = 0
            for d in ori_ontology:
                if d not in ontology['domains']:
                    ontology['domains'][d] = {'description': descriptions[d][d], 'slots': {}}
                for s in ori_ontology[d]:
                    if s not in ontology['domains'][d]['slots']:
                        ontology['domains'][d]['slots'][s] = {
                            'description': descriptions[d][s],
                            'is_categorical': False,
                            'possible_values': [],
                            'count': 0,
                            'in original ontology': True
                        }
            # pprint(ori_ontology)
            for ori_sess in tqdm(data, desc='processing taskmaster-{}'.format(filename)):
                total += 1
                turns = format_turns(ori_sess['utterances'])
                if not turns:
                    continue
                if 'TM-2' in filename:
                    dial_domain = normalize_domain_name(filename.split('/')[-1].split('.')[0])
                else:
                    dial_domain = normalize_domain_name(ori_sess['instruction_id'].split('-', 1)[0])
                dialogue = {
                    "dataset": "taskmaster",
                    "data_split": "train",
                    "dialogue_id": 'taskmaster_' + str(idx_count),
                    "original_id": ori_sess['conversation_id'],
                    "instruction_id": ori_sess['instruction_id'],
                    "domains": [
                        dial_domain
                    ],
                    "turns": []
                }
                idx_count += 1
                assert turns[0]['speaker'] == 'user' and turns[-1]['speaker'] == 'user', print(turns)
                for utt_idx, uttr in enumerate(turns):
                    speaker = uttr['speaker']
                    turn = {
                        'speaker': speaker,
                        'utterance': uttr['text'],
                        'utt_idx': utt_idx,
                        'dialogue_act': {
                            'binary': [],
                            'categorical': [],
                            'non-categorical': [],
                        },
                    }
                    if speaker == 'user':
                        turn['state'] = {}
                        turn['state_update'] = {'categorical': [], 'non-categorical': []}

                    if 'segments' in uttr:
                        for segment in uttr['segments']:
                            for item in segment['annotations']:
                                # domain = item['name'].split('.', 1)[0]
                                domain = dial_domain

                                # if domain != item['name'].split('.', 1)[0]:
                                #     print(domain, item['name'].split('.', 1), dialogue["original_id"])
                                #     assert domain in item['name'].split('.', 1)[0]

                                # if item['name'].split('.', 1)[0] != domain:
                                #     print(domain, item['name'].split('.', 1), dialogue["original_id"])
                                slot = item['name'].split('.', 1)[-1]
                                if slot.endswith('.accept') or slot.endswith('.reject'):
                                    slot = slot[:-7]
                                if slot not in ori_ontology[domain]:
                                    # print(domain, item['name'].split('.', 1), dialogue["original_id"])
                                    continue
                                # if domain in ori_ontology:
                                #     ori_ontology[domain][slot] += 1
                                # else:
                                #     print(domain, item['name'].split('.', 1), dialogue["original_id"])
                                # assert domain in ori_ontology, print(domain, item['name'].split('.', 1), dialogue["original_id"])

                                if not segment['text']:
                                    print(slot)
                                    print(segment)
                                    print()
                                assert turn['utterance'][segment['start_index']:segment['end_index']] == segment['text']
                                turn['dialogue_act']['non-categorical'].append({
                                    'intent': 'inform',
                                    'domain': domain,
                                    'slot': slot,
                                    'value': segment['text'].lower(),
                                    'start': segment['start_index'],
                                    'end': segment['end_index']
                                })
                        log_ontology(turn['dialogue_act']['non-categorical'], ontology, ori_ontology)
                    dialogue['turns'].append(turn)
                processed_dialogue.append(dialogue)
            # pprint(ori_ontology)
        # save ontology json
        json.dump(ontology, open(os.path.join(self_dir, 'ontology.json'), 'w'), indent=2)
        json.dump(processed_dialogue, open('data.json', 'w'), indent=2)
        write_zipped_json(os.path.join(self_dir, 'data.zip'), 'data.json')
        os.remove('data.json')
    else:
        # read from file
        processed_dialogue = read_zipped_json(os.path.join(self_dir, 'data.zip'), 'data.json')
        ontology = json.load(open(os.path.join(self_dir, 'ontology.json')))
    return processed_dialogue, ontology

if __name__ == '__main__':
    preprocess()
