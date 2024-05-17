"""Fundamental Configuration for Data Preprocessing"""

"""
A map from channel name to channel id 
"""
CHANNEL_TO_ID = {
    'FP1-F7': 0,
    'F7-T3': 1,
    'T3-T5': 2,
    'T5-O1': 3,
    'FP2-F8': 4,
    'F8-T4': 5,
    'T4-T6': 6,
    'T6-O2': 7,
    'A1-T3': 8,
    'T3-C3': 9,
    'C3-CZ': 10,
    'CZ-C4': 11,
    'C4-T4': 12,
    'T4-A2': 13,
    'FP1-F3': 14,
    'F3-C3': 15,
    'C3-P3': 16,
    'P3-O1': 17,
    'FP2-F4': 18,
    'F4-C4': 19,
    'C4-P4': 20,
    'P4-O2': 21,
    'EKG': 22
}
'''
"TUAR": A map from label name to label id
'''
ARTIFACT_TO_ID = {
    'null': 0,
    'eyem': 1,
    'chew': 2,
    'shiv': 3,
    'musc': 4,
    'elpp': 5,
    'elec': 6,
    'eyem_chew': 7,
    'eyem_shiv': 8,
    'eyem_musc': 9,
    'eyem_elec': 10,
    'chew_musc': 11,
    'chew_elec': 12,
    'shiv_elec': 13,
    'musc_elec': 14
}

'''
Single label to multi label
'''
SINGLE_LABEL_TO_MULTI_LABEL = {
    0: [0,],
    1: [1,],
    2: [2,],
    3: [3,],
    4: [4,],
    5: [5,],
    6: [5,],
    7: [1, 2],
    8: [1, 3],
    9: [1, 4],
    10: [1, 5],
    11: [2, 4],
    12: [2, 5],
    13: [3, 5],
    14: [4, 5]
}