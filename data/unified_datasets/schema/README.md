# README

## Features

- Annotations: dialogue act, belief state, character-level span for non-categorical slots.
- Unseen domains and slots in the test set to quantify the performance in zero-shot or few shot settings.

Statistics: 

|       | \# dialogues | \# utterances | avg. turns | avg. tokens | \# domains |
| ----- | ------------ | ------------- | ---------- | ----------- | ---------- |
| train | 16142        | 313822        | 19.44      | 10.02       | 16         |
| val   | 2482         | 46244         | 18.63      | 9.94        | 16         |
| test  | 4201         | 80393         | 19.14      | 10.7        | 18         |

## Main changes

1. download the original data as `original_data.zip`

2. run `python preprocess` to unzip `original_data.zip` and get processed `data.zip` & `ontology.json`.

Main changes:

- extract intent from domains.
- ~~numerical slot => non-categorical, use string match to get the span.~~
- add binary_dialogue_act for those binary intents such as 'goodbye', 'request'.
- add **count** non-categorical, numerical slot for each domain, but not appear in belief state.
- sys state are updated by previous user frame['state']. 
- calculate the state update according to prev state and slot spans in current turn slot_vals and all previous dialogue acts. 99.6% non-categorical state update have spans while the rest of them are like "Could you help me search for songs from **two years back** too?" 
- values in possible values, dialogue act, state, and state_update are in **lowercase**. 

Notice:

- for categorical slot, value maybe **dontcare**, which is not presented in **possible_values**.

## Original data

The Schema-Guided Dialogue (SGD) dataset consists of over 20k annotated
multi-domain, task-oriented conversations between a human and a virtual
assistant. These conversations involve interactions with services and APIs
spanning 20 domains, ranging from banks and events to media, calendar, travel,
and weather. For most of these domains, the dataset contains multiple different
APIs, many of which have overlapping functionalities but different interfaces,
which reflects common real-world scenarios. The wide range of available
annotations can be used for intent prediction, slot filling, dialogue state
tracking, policy imitation learning, language generation, user simulation
learning, among other tasks in large-scale virtual assistants. Besides these,
the dataset has unseen domains and services in the evaluation set to quantify
the performance in zero-shot or few shot settings.

[[paper]](https://arxiv.org/abs/1909.05855) [[download link]](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue)

### Scheme Representation

A service or API is essentially a set of functions (called intents), each taking
a set of parameters (called slots). A schema is a normalized representation of
the interface exposed by a service/API. In addition, the schema also includes
natural language description of the included functions and their parameters to
outline the semantics of each element. The schemas have been manually generated
by the dataset creators. The schema for a service contains the following fields:

*   **service_name** - A unique name for the service.
*   **description** - A natural language description of the tasks supported by
    the service.
*   **slots** - A list of slots/attributes corresponding to the entities present
    in the service. Each slot contains the following fields:
    *   **name** - The name of the slot.
    *   **description** - A natural language description of the slot.
    *   **is_categorical** - A boolean value. If it is true, the slot has a
        fixed set of possible values.
    *   **possible_values** - List of possible values the slot can take. If the
        slot is a categorical slot, it is a complete list of all the possible
        values. If the slot is a non categorical slot, it is either an empty
        list or a small sample of all the values taken by the slot.
*   **intents** - The list of intents/tasks supported by the service. Each
    method contains the following fields:
    *   **name** - The name of the intent.
    *   **description** - A natural language description of the intent.
    *   **is_transactional** - A boolean value. If true, indicates that the
        underlying API call is transactional (e.g, a booking or a purchase), as
        opposed to a search call.
    *   **required_slots** - A list of slot names whose values must be provided
        before making a call to the service.
    *   **optional_slots** - A dictionary mapping slot names to the default
        value taken by the slot. These slots may be optionally specified by the
        user and the user may override the default value. An empty default value
        allows that slot to take any value by default, but the user may override
        it.
    *   **result_slots** - A list of slot names which are present in the results
        returned by a call to the service or API.

### Dialogue Representation

The dialogue is represented as a list of turns, where each turn contains either
a user or a system utterance. The annotations for a turn are grouped into
frames, where each frame corresponds to a single service. Each turn in the
single domain dataset contains exactly one frame. In multi-domain datasets, some
turns may have multiple frames.

Each dialogue is represented as a json object with the following fields:

*   **dialogue_id** - A unique identifier for a dialogue.
*   **services** - A list of services present in the dialogue.
*   **turns** - A list of annotated system or user utterances.

Each turn consists of the following fields:

*   **speaker** - The speaker for the turn. Possible values are "USER" or
    "SYSTEM".
*   **utterance** - A string containing the natural language utterance.
*   **frames** - A list of frames, each frame containing annotations for a
    single service.

Each frame consists of the fields listed below. The fields marked with * will
be excluded from all user turns in the test data released to the participants.

*   **service** - The name of the service corresponding to the frame. The slots
    and intents used in the following fields are taken from the schema of this
    service.
*   **slots** - A list of slot spans in the utterance, only provided for
    non-categorical slots. Each slot span contains the following fields:
    *   **slot** - The name of the slot.
    *   **start** - The index of the starting character in the utterance
        corresponding to the slot value.
    *   **exclusive_end** - The index of the character just after the last
        character corresponding to the slot value in the utterance. In python,
        `utterance[start:exclusive_end]` gives the slot value.
*   **actions** - A list of actions corresponding to the system. Each action has
    the following fields:
    *   **act** - The type of action. The list of all possible system acts is
        given below.
    *   **slot** (optional) - A slot argument for some of the actions.
    *   **values** (optional) - A list of values assigned to the slot. If the
        values list is non-empty, then the slot must be present.
    *   **canonical_values** (optional) - The values in their canonicalized form
        as used by the service. It is a list of strings of the same length as
        values.
*   **service_call** (system turns only, optional) - The request sent to the
    service. It consists of the following fields:
    *   **method** - The name of the intent or function of the service or API
        being executed.
    *   **parameters** - A dictionary mapping slot name (all required slots and
        possibly some optional slots) to a value in its canonicalized form.
*   **service_results** (system turns only, optional) - A list of entities
    containing the results obtained from the service. It is only available for
    turns in which a service call is made. Each entity is represented as a
    dictionary mapping a slot name to a string containing its canonical value.
*   **state** (user turns only) - The dialogue state corresponding to the
    service. It consists of the following fields:
    *   **active_intent** - The intent corresponding to the service of the frame
        which is currently being fulfilled by the system. It takes the value
        "NONE" if none of the intents are active.
    *   **requested_slots** - A list of slots requested by the user in the
        current turn.
    *   **slot_values** - A dictionary mapping slot name to a list of strings.
        For categorical slots, this list contains a single value assigned to the
        slot. For non-categorical slots, all the values in this list are spoken
        variations of each other and are equivalent (e.g, "6 pm", "six in the
        evening", "evening at 6" etc.).

List of possible system acts:

*   **INFORM** - Inform the value for a slot to the user. The slot and values
    fields in the corresponding action are always non-empty.
*   **REQUEST** - Request the value of a slot from the user. The corresponding
    action always contains a slot, but values are optional. When values are
    present, they are used as examples for the user e.g, "Would you like to eat
    indian or chinese food or something else?"
*   **CONFIRM** - Confirm the value of a slot before making a transactional
    service call.
*   **OFFER** - Offer a certain value for a slot to the user. The corresponding
    action always contains a slot and a list of values for that slot offered to
    the user.
*   **NOTIFY_SUCCESS** - Inform the user that their request was successful. Slot
    and values are always empty in the corresponding action.
*   **NOTIFY_FAILURE** - Inform the user that their request failed. Slot and
    values are always empty in the corresponding action.
*   **INFORM_COUNT** - Inform the number of items found that satisfy the user's
    request. The corresponding action always has "count" as the slot, and a
    single element in values for the number of results obtained by the system.
*   **OFFER_INTENT** - Offer a new intent to the user. Eg, "Would you like to
    reserve a table?". The corresponding action always has "intent" as the slot,
    and a single value containing the intent being offered. The offered intent
    belongs to the service corresponding to the frame.
*   **REQ_MORE** - Asking the user if they need anything else. Slot and values
    are always empty in the corresponding action.
*   **GOODBYE** - End the dialogue. Slot and values are always empty in the
    corresponding action.

List of possible user acts:

*   **INFORM_INTENT** - Express the desire to perform a certain task to the
    system. The action always has "intent" as the slot and a single value
    containing the intent being informed.
*   **NEGATE_INTENT** - Negate the intent which has been offered by the system.
*   **AFFIRM_INTENT** - Agree to the intent which has been offered by the
    system.
*   **INFORM** - Inform the value of a slot to the system. The slot and values
    fields in the corresponding action are always non-empty.
*   **REQUEST** - Request the value of a slot from the system. The corresponding
    action always contains a slot parameter. It may optionally contain a value,
    in which case, the user asks the system if the slot has the specified value.
*   **AFFIRM** - Agree to the system's proposition. Slot and values are always
    empty.
*   **NEGATE** - Deny the system's proposal. Slot and values are always empty.
*   **SELECT** - Select a result being offered by the system. The corresponding
    action may either contain no parameters, in which case all the values
    proposed by the system are being accepted, or it may contain a slot and
    value parameters, in which case the specified slot and value are being
    accepted.
*   **REQUEST_ALTS** - Ask for more results besides the ones offered by the
    system. Slot and values are always empty.
*   **THANK_YOU** - Thank the system. Slot and values are always empty.
*   **GOODBYE** - End the dialogue. Slot and values are always empty.

### Dataset Statistics

The dataset consists of two kinds of dialogues.

| Type of Dialogue |                 Train files                  |                  Dev files                   |                  Test Files                  |
| ---------------- | :------------------------------------------: | :------------------------------------------: | :------------------------------------------: |
| Single Domain    | `dialogues_001.json` to `dialogues_043.json` | `dialogues_001.json` to `dialogues_007.json` | `dialogues_001.json` to `dialogues_011.json` |
| Multi Domain     | `dialogues_044.json` to `dialogues_127.json` | `dialogues_008.json` to `dialogues_020.json` | `dialogues_012.json` to `dialogues_034.json` |

The single domain dialogues involve interactions with a single service, possibly
over multiple intents. The multi-domain dialogues have interactions involving
intents belonging to two or more different services. The multi-domain dialogues
also involve transfer of dialogue state values from one service to the other
wherever such a transfer is deemed natural. Eg, if a user finds a restaurant and
searches for a movie next, the dialogue state for movie service is already
initialized with the location from the dialogue state for restaurant service.

The overall statistics of the train and dev sets are given below. The term
*informable slots* refers to the slots over which the user can specify a
constraint. For example, slots like *phone_number* are not informable.

<table>
    <tr>
        <th rowspan="2"></th>
        <th colspan="3">Train</th><th colspan="3">Dev</th><th colspan="3">Test</th>
    </tr>
    <tr>
        <td>Single-domain</td>
        <td>Multi-domain</td>
        <td>Combined</td>
        <td>Single-domain</td>
        <td>Multi-domain</td>
        <td>Combined</td>
        <td>Single-domain</td>
        <td>Multi-domain</td>
        <td>Combined</td>
    </tr>
    <tr>
        <td>No. of dialogues</td>
        <td align="center">5,403</td>
        <td align="center">10,739</td>
        <td align="center">16,142</td>
        <td align="center">836</td>
        <td align="center">1,646</td>
        <td align="center">2,482</td>
        <td align="center">1,331</td>
        <td align="center">2,870</td>
        <td align="center">4,201</td>
    </tr>
    <tr>
        <td>No. of turns</td>
        <td align="center">82,588</td>
        <td align="center">247,376</td>
        <td align="center">329,964</td>
        <td align="center">11,928</td>
        <td align="center">36,798</td>
        <td align="center">48,726</td>
        <td align="center">16,850</td>
        <td align="center">67,744</td>
        <td align="center">84,594</td>
    </tr>
    <tr>
        <td>No. of tokens (lower-cased)</td>
        <td align="center">807,562</td>
        <td align="center">2,409,857</td>
        <td align="center">3,217,419</td>
        <td align="center">117,492</td>
        <td align="center">353,381</td>
        <td align="center">470,873</td>
        <td align="center">166,329</td>
        <td align="center">713,731</td>
        <td align="center">880,060</td>
    </tr>
     <tr>
        <td>Average turns per dialogue</td>
        <td align="center">15.286</td>
        <td align="center">23.035</td>
        <td align="center">20.441</td>
        <td align="center">14.268</td>
        <td align="center">22.356</td>
        <td align="center">19.632</td>
        <td align="center">12.660</td>
        <td align="center">23.604</td>
        <td align="center">20.137</td>
    </tr>
    <tr>
        <td>Average tokens per turn</td>
        <td align="center">9.778</td>
        <td align="center">9.742</td>
        <td align="center">9.751</td>
        <td align="center">9.850</td>
        <td align="center">9.603</td>
        <td align="center">9.664</td>
        <td align="center">9.871</td>
        <td align="center">10.536</td>
        <td align="center">10.403</td>
    </tr>
    <tr>
        <td>Total unique tokens (lower-cased)</td>
        <td align="center">16,350</td>
        <td align="center">25,459</td>
        <td align="center">30,349</td>
        <td align="center">6,803</td>
        <td align="center">10,533</td>
        <td align="center">12,719</td>
        <td align="center">7,213</td>
        <td align="center">14,888</td>
        <td align="center">16,382</td>
    </tr>
    <tr>
        <td>Total no. of slots</td>
        <td align="center">201</td>
        <td align="center">214</td>
        <td align="center">214</td>
        <td align="center">134</td>
        <td align="center">132</td>
        <td align="center">136</td>
        <td align="center">157</td>
        <td align="center">158</td>
        <td align="center">159</td>
    </tr>
    <tr>
        <td>Total no. of informable slots</td>
        <td align="center">138</td>
        <td align="center">144</td>
        <td align="center">144</td>
        <td align="center">89</td>
        <td align="center">87</td>
        <td align="center">89</td>
        <td align="center">109</td>
        <td align="center">110</td>
        <td align="center">111</td>
    </tr>
    <tr>
        <td>Total unique slot values (lower-cased)</td>
        <td align="center">7,070</td>
        <td align="center">11,635</td>
        <td align="center">14,139</td>
        <td align="center">2,418</td>
        <td align="center">4,182</td>
        <td align="center">5,101</td>
        <td align="center">2,492</td>
        <td align="center">5,847</td>
        <td align="center">6,533</td>
    </tr>
    <tr>
        <td>Total unique informable slot values (lower-cased)</td>
        <td align="center">3,742</td>
        <td align="center">6,348</td>
        <td align="center">7,661</td>
        <td align="center">1,137</td>
        <td align="center">2,118</td>
        <td align="center">2,524</td>
        <td align="center">1,387</td>
        <td align="center">3,323</td>
        <td align="center">3,727</td>
    </tr>
    <tr>
        <td>Total domains</td>
        <td align="center">14</td>
        <td align="center">16</td>
        <td align="center">16</td>
        <td align="center">16</td>
        <td align="center">15</td>
        <td align="center">16</td>
        <td align="center">17</td>
        <td align="center">18</td>
        <td align="center">18</td>
    </tr>
    <tr>
        <td>Total services</td>
        <td align="center">24</td>
        <td align="center">26</td>
        <td align="center">26</td>
        <td align="center">17</td>
        <td align="center">16</td>
        <td align="center">17</td>
        <td align="center">20</td>
        <td align="center">21</td>
        <td align="center">21</td>
    </tr>
    <tr>
        <td>Total intents</td>
        <td align="center">35</td>
        <td align="center">37</td>
        <td align="center">37</td>
        <td align="center">28</td>
        <td align="center">26</td>
        <td align="center">28</td>
        <td align="center">33</td>
        <td align="center">34</td>
        <td align="center">35</td>
    </tr>
</table>


The following table shows how the dialogues and services are distributed among
different domains for the train and dev sets. In this table, each multi-domain
dialogue contirbutes to the count of every service present in the dialogue.
Please note that a few domains like *Travel* and *Weather* are only present in
the dev set. This is to test the generalization of models on unseen domains. The
test set will similarly have some unseen domains which are neither present in
the training nor in the dev set. Also, the number in parenthesis represents the
number of unique services belonging to the corresponding domain.

* In the first column, it indicates the number of unique services for the
  domain in Train, Dev and Test datasets combined.
* In the fourth column, it indicates the number of such unique services in the
  Train dataset only.
* In the seventh column, it indicates the number of such unique services in
  the Dev dataset only.
* In the last column, it indicates the number of such unique services in the
  Test dataset only.

<table>
    <tr>
        <th rowspan="2"></th>
        <th colspan="3"># Dialogues <br> Train</th>
        <th colspan="3"># Dialogues <br> Dev</th>
        <th colspan="3"># Dialogues <br> Test</th>
    </tr>
    <tr>
        <td>Single-domain</td>
        <td>Multi-domain</td>
        <td>Combined</td>
        <td>Single-domain</td>
        <td>Multi-domain</td>
        <td>Combined</td>
        <td>Single-domain</td>
        <td>Multi-domain</td>
        <td>Combined</td>
    </tr>
    <tr>
        <td>Alarm (1)</td>
        <td align="center">NA</td>
        <td align="center">NA</td>
        <td align="center">NA</td>
        <td align="center">37</td>
        <td align="center">NA</td>
        <td align="center">37 (1)</td>
        <td align="center">47</td>
        <td align="center">240</td>
        <td align="center">287 (1)</td>
    </tr>
    <tr>
        <td>Banks (2)</td>
        <td align="center">207</td>
        <td align="center">520</td>
        <td align="center">727 (1)</td>
        <td align="center">42</td>
        <td align="center">252</td>
        <td align="center">294 (1)</td>
        <td align="center">NA</td>
        <td align="center">NA</td>
        <td align="center">NA</td>
    </tr>
    <tr>
        <td>Buses (3)</td>
        <td align="center">310</td>
        <td align="center">1,970</td>
        <td align="center">2,280 (2)</td>
        <td align="center">44</td>
        <td align="center">285</td>
        <td align="center">329 (1)</td>
        <td align="center">88</td>
        <td align="center">438</td>
        <td align="center">526 (1)</td>
    </tr>
    <tr>
        <td>Calendar (1)</td>
        <td align="center">169</td>
        <td align="center">1,433</td>
        <td align="center">1,602 (1)</td>
        <td align="center">NA</td>
        <td align="center">NA</td>
        <td align="center">NA</td>
        <td align="center">NA</td>
        <td align="center">NA</td>
        <td align="center">NA</td>
    </tr>
    <tr>
        <td>Events (3)</td>
        <td align="center">788</td>
        <td align="center">2,721</td>
        <td align="center">3,509 (1)</td>
        <td align="center">73</td>
        <td align="center">345</td>
        <td align="center">418 (1)</td>
        <td align="center">76</td>
        <td align="center">516</td>
        <td align="center">592 (1)</td>
    </tr>
    <tr>
        <td>Flights (4)</td>
        <td align="center">985</td>
        <td align="center">1,762</td>
        <td align="center">2,747 (2)</td>
        <td align="center">94</td>
        <td align="center">297</td>
        <td align="center">391 (1)</td>
        <td align="center">87</td>
        <td align="center">419</td>
        <td align="center">506 (1)</td>
    </tr>
        <tr>
        <td>Homes (2)</td>
        <td align="center">268</td>
        <td align="center">579</td>
        <td align="center">847 (1)</td>
        <td align="center">81</td>
        <td align="center">99</td>
        <td align="center">180 (1)</td>
        <td align="center">89</td>
        <td align="center">157</td>
        <td align="center">246 (1)</td>
    </tr>
        <tr>
        <td>Hotels (4)</td>
        <td align="center">457</td>
        <td align="center">2,896</td>
        <td align="center">3,353 (3)</td>
        <td align="center">56</td>
        <td align="center">521</td>
        <td align="center">577 (2)</td>
        <td align="center">177</td>
        <td align="center">885</td>
        <td align="center">1062 (2)</td>
    </tr>
        <tr>
        <td>Media (3)</td>
        <td align="center">281</td>
        <td align="center">832</td>
        <td align="center">1,113 (1)</td>
        <td align="center">46</td>
        <td align="center">133</td>
        <td align="center">179 (1)</td>
        <td align="center">80</td>
        <td align="center">284</td>
        <td align="center">364 (1)</td>
    </tr>
        <tr>
        <td>Messaging (1)</td>
        <td align="center">NA</td>
        <td align="center">NA</td>
        <td align="center">NA</td>
        <td align="center">NA</td>
        <td align="center">NA</td>
        <td align="center">NA</td>
        <td align="center">NA</td>
        <td align="center">298</td>
        <td align="center">298 (1)</td>
    </tr>
        <tr>
        <td>Movies (2)</td>
        <td align="center">292</td>
        <td align="center">1,325</td>
        <td align="center">1,617 (1)</td>
        <td align="center">47</td>
        <td align="center">94</td>
        <td align="center">141 (1)</td>
        <td align="center">132</td>
        <td align="center">449</td>
        <td align="center">581</td>
    </tr>
        <tr>
        <td>Music (3)</td>
        <td align="center">394</td>
        <td align="center">896</td>
        <td align="center">1,290 (2)</td>
        <td align="center">35</td>
        <td align="center">161</td>
        <td align="center">196 (1)</td>
        <td align="center">25</td>
        <td align="center">322</td>
        <td align="center">347 (2)</td>
    </tr>
        <tr>
        <td>Payment (1)</td>
        <td align="center">NA</td>
        <td align="center">NA</td>
        <td align="center">NA</td>
        <td align="center">NA</td>
        <td align="center">NA</td>
        <td align="center">NA</td>
        <td align="center">36</td>
        <td align="center">186</td>
        <td align="center">222 (1)</td>
    </tr>
        <tr>
        <td>RentalCars (3)</td>
        <td align="center">215</td>
        <td align="center">1,370</td>
        <td align="center">1,585 (2)</td>
        <td align="center">39</td>
        <td align="center">342</td>
        <td align="center">381 (1)</td>
        <td align="center">64</td>
        <td align="center">480</td>
        <td align="center">544 (1)</td>
    </tr>
        <tr>
        <td>Restaurants (2)</td>
        <td align="center">367</td>
        <td align="center">2052</td>
        <td align="center">2,419 (1)</td>
        <td align="center">73</td>
        <td align="center">263</td>
        <td align="center">336 (1)</td>
        <td align="center">73</td>
        <td align="center">390</td>
        <td align="center">463 (1)</td>
    </tr>
        <tr>
        <td>RideSharing (2)</td>
        <td align="center">119</td>
        <td align="center">1,584</td>
        <td align="center">1,703 (2)</td>
        <td align="center">45</td>
        <td align="center">225</td>
        <td align="center">270 (1)</td>
        <td align="center">34</td>
        <td align="center">216</td>
        <td align="center">250 (1)</td>
    </tr>
        <tr>
        <td>Services (4)</td>
        <td align="center">551</td>
        <td align="center">1,338</td>
        <td align="center">1,889 (3)</td>
        <td align="center">44</td>
        <td align="center">157</td>
        <td align="center">201 (1)</td>
        <td align="center">167</td>
        <td align="center">489</td>
        <td align="center">656 (2)</td>
    </tr>
        <tr>
        <td>Trains (1)</td>
        <td align="center">NA</td>
        <td align="center">NA</td>
        <td align="center">NA</td>
        <td align="center">NA</td>
        <td align="center">NA</td>
        <td align="center">NA</td>
        <td align="center">84</td>
        <td align="center">266</td>
        <td align="center">350 (1)</td>
    </tr>
        <tr>
        <td>Travel (1)</td>
        <td align="center">NA</td>
        <td align="center">1,871</td>
        <td align="center">1,871 (1)</td>
        <td align="center">45</td>
        <td align="center">238</td>
        <td align="center">283 (1)</td>
        <td align="center">24</td>
        <td align="center">630</td>
        <td align="center">654 (1)</td>
    </tr>
        <tr>
        <td>Weather (1)</td>
        <td align="center">NA</td>
        <td align="center">951</td>
        <td align="center">951 (1)</td>
        <td align="center">35</td>
        <td align="center">322</td>
        <td align="center">357 (1)</td>
        <td align="center">48</td>
        <td align="center">427</td>
        <td align="center">475 (1)</td>
    </tr>
</table>

