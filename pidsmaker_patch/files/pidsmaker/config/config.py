# Datasets

DATASET_DEFAULT_CONFIG = {
    "THEIA_E5": {
        "raw_dir": "",
        "database": "theia_e5",
        "database_all_file": "theia_e5",
        # "database_all_file": "theia_e5_all", # NOTE: the whole dataset is too huge
        "num_node_types": 3,
        "num_edge_types": 10,
        "start_date": "2019-05-08",
        "end_date": "2019-05-18",
        "train_dates": ["2019-05-08", "2019-05-09", "2019-05-10"],
        "val_dates": ["2019-05-11"],
        "test_dates": ["2019-05-14", "2019-05-15"],
        "unused_dates": ["2019-05-12", "2019-05-13", "2019-05-16", "2019-05-17"],
        "ground_truth_relative_path": [
            "E5-THEIA/node_THEIA_1_Firefox_Drakon_APT_BinFmt_Elevate_Inject.csv"
        ],
        "attack_to_time_window": [
            [
                "E5-THEIA/node_THEIA_1_Firefox_Drakon_APT_BinFmt_Elevate_Inject.csv",
                "2019-05-15 14:47:00",
                "2019-05-15 15:08:00",
            ],
        ],
    },
    "THEIA_E3": {
        "raw_dir": "",
        "database": "theia_e3",
        "database_all_file": "theia_e3",
        "num_node_types": 3,
        "num_edge_types": 10,
        "start_date": "2018-04-02",
        "end_date": "2018-04-14",
        "train_dates": [
            "2018-04-02",
            "2018-04-03",
            "2018-04-04",
            "2018-04-05",
            "2018-04-06",
            "2018-04-07",
            "2018-04-08",
        ],
        "val_dates": ["2018-04-09"],
        "test_dates": ["2018-04-10", "2018-04-12", "2018-04-13"],
        "unused_dates": ["2018-04-11"],
        "ground_truth_relative_path": [
            "E3-THEIA/node_Browser_Extension_Drakon_Dropper.csv",
            "E3-THEIA/node_Firefox_Backdoor_Drakon_In_Memory.csv",
            # "E3-THEIA/node_Phishing_E_mail_Executable_Attachment.csv", # attack failed so we don't use it
            # "E3-THEIA/node_Phishing_E_mail_Link.csv" # attack only at network level, not system
        ],
        "attack_to_time_window": [
            [
                "E3-THEIA/node_Browser_Extension_Drakon_Dropper.csv",
                "2018-04-12 12:40:00",
                "2018-04-12 13:30:00",
            ],
            [
                "E3-THEIA/node_Firefox_Backdoor_Drakon_In_Memory.csv",
                "2018-04-10 14:30:00",
                "2018-04-10 15:00:00",
            ],
        ],
    },
    "CADETS_E5": {
        "raw_dir": "",
        "database": "cadets_e5",
        "database_all_file": "cadets_e5",
        "num_node_types": 3,
        "num_edge_types": 10,
        "start_date": "2019-05-08",
        "end_date": "2019-05-18",
        "train_dates": ["2019-05-08", "2019-05-09", "2019-05-11"],
        "val_dates": ["2019-05-12"],
        "test_dates": ["2019-05-16", "2019-05-17"],
        "unused_dates": ["2019-05-15", "2019-05-10", "2019-05-13", "2019-05-14"],
        "ground_truth_relative_path": [
            "E5-CADETS/node_Nginx_Drakon_APT.csv",
            "E5-CADETS/node_Nginx_Drakon_APT_17.csv",
        ],
        "attack_to_time_window": [
            ["E5-CADETS/node_Nginx_Drakon_APT.csv", "2019-05-16 09:31:00", "2019-05-16 10:12:00"],
            [
                "E5-CADETS/node_Nginx_Drakon_APT_17.csv",
                "2019-05-17 10:15:00",
                "2019-05-17 15:33:00",
            ],
        ],
    },
    "CADETS_E3": {
        "raw_dir": "",
        "database": "cadets_e3",
        "database_all_file": "cadets_e3",
        "num_node_types": 3,
        "num_edge_types": 10,
        "start_date": "2018-04-02",
        "end_date": "2018-04-14",
        "train_dates": [
            "2018-04-02",
            "2018-04-03",
            "2018-04-04",
            "2018-04-05",
            "2018-04-07",
            "2018-04-08",
            "2018-04-09",
        ],
        "val_dates": ["2018-04-10"],
        "test_dates": ["2018-04-06", "2018-04-11", "2018-04-12", "2018-04-13"],
        "unused_dates": [],
        "ground_truth_relative_path": [
            # "E3-CADETS/node_E_mail_Server.csv",
            "E3-CADETS/node_Nginx_Backdoor_06.csv",
            # "E3-CADETS/node_Nginx_Backdoor_11.csv",
            "E3-CADETS/node_Nginx_Backdoor_12.csv",
            "E3-CADETS/node_Nginx_Backdoor_13.csv",
        ],
        "attack_to_time_window": [
            ["E3-CADETS/node_Nginx_Backdoor_06.csv", "2018-04-06 11:20:00", "2018-04-06 12:09:00"],
            # ["E3-CADETS/node_Nginx_Backdoor_11.csv" , '2018-04-11 15:07:00', '2018-04-11 15:16:00'],
            ["E3-CADETS/node_Nginx_Backdoor_12.csv", "2018-04-12 13:59:00", "2018-04-12 14:39:00"],
            ["E3-CADETS/node_Nginx_Backdoor_13.csv", "2018-04-13 09:03:00", "2018-04-13 09:16:00"],
        ],
    },
    # Shadow/target variants of CADETS_E3 for membership-inference work. The two roles train on
    # DISJOINT day sets (target 04-02/03/04, shadow 04-07/08/09) so a shadow model is a legitimate
    # stand-in for the target: same distribution, no shared training records. Each role keeps its own
    # val day and its own attack days, so both produce a self-contained detection table. Everything
    # else (database, node/edge type counts, date range) is identical to CADETS_E3 on purpose --
    # only the split differs, so preprocessing artifacts stay comparable.
    "CADETS_E3_TARGET": {
        "raw_dir": "",
        "database": "cadets_e3",
        "database_all_file": "cadets_e3",
        "num_node_types": 3,
        "num_edge_types": 10,
        "start_date": "2018-04-02",
        "end_date": "2018-04-14",
        "train_dates": [
            "2018-04-02",
            "2018-04-03",
            "2018-04-04",
        ],
        "val_dates": ["2018-04-05"],
        "test_dates": ["2018-04-06", "2018-04-12"],
        "unused_dates": [],
        "ground_truth_relative_path": [
            "E3-CADETS/node_Nginx_Backdoor_06.csv",
            "E3-CADETS/node_Nginx_Backdoor_12.csv",
        ],
        "attack_to_time_window": [
            ["E3-CADETS/node_Nginx_Backdoor_06.csv", "2018-04-06 11:20:00", "2018-04-06 12:09:00"],
            ["E3-CADETS/node_Nginx_Backdoor_12.csv", "2018-04-12 13:59:00", "2018-04-12 14:39:00"],
        ],
    },
    "CADETS_E3_SHADOW": {
        "raw_dir": "",
        "database": "cadets_e3",
        "database_all_file": "cadets_e3",
        "num_node_types": 3,
        "num_edge_types": 10,
        "start_date": "2018-04-02",
        "end_date": "2018-04-14",
        "train_dates": [
            "2018-04-07",
            "2018-04-08",
            "2018-04-09",
        ],
        "val_dates": ["2018-04-10"],
        "test_dates": ["2018-04-13"],
        "unused_dates": [],
        "ground_truth_relative_path": [
            "E3-CADETS/node_Nginx_Backdoor_13.csv",
        ],
        "attack_to_time_window": [
            ["E3-CADETS/node_Nginx_Backdoor_13.csv", "2018-04-13 09:03:00", "2018-04-13 09:16:00"],
        ],
    },
    # Cross-role membership split: the detector trains on ALL of the target's days (no
    # `mia_split_frac` subsampling), and the non-members are ALL of the shadow's days.
    #
    # Why this exists. In `random` mode `mia_split_frac` is what *creates* the non-member pool --
    # the withheld 1-frac of batches. That costs training data: a frac=0.5 target sees ~1.5 of
    # its 3 days, so its detection numbers are those of a data-starved model. Here instead the
    # negatives come from days the model never trained on, so the detector gets 100% of its
    # training days and is full strength.
    #
    # The cost is that membership now correlates with the calendar, and a structure-only floor
    # has to absorb day-to-day drift before any AUC counts as memorisation. That is the trade:
    # `random` buys a confound-free split with a weaker model; this buys a strong model with a
    # confound the floor must rule out. Use `mia_split_mode: xrole`, which takes the val split
    # as the non-member pool.
    #
    # The non-member days sit in `val_dates` (not `test_dates`) so that `test_dates` keeps only
    # the two attack days and the detection table stays directly comparable to CADETS_E3_TARGET.
    # Validation here is 3 benign days rather than 1, which is strictly better threshold
    # calibration; the threshold is a scalar fit on benign losses and takes no part in the MIA,
    # which reads the raw per-node dumps.
    "CADETS_E3_NOFRAC": {
        "raw_dir": "",
        "database": "cadets_e3",
        "database_all_file": "cadets_e3",
        "num_node_types": 3,
        "num_edge_types": 10,
        "start_date": "2018-04-02",
        "end_date": "2018-04-14",
        # members: identical to CADETS_E3_TARGET, trained on in full
        "train_dates": [
            "2018-04-02",
            "2018-04-03",
            "2018-04-04",
        ],
        # non-members: the shadow role's training days, benign and never trained on
        "val_dates": [
            "2018-04-07",
            "2018-04-08",
            "2018-04-09",
        ],
        "test_dates": ["2018-04-06", "2018-04-12"],
        "unused_dates": [],
        "ground_truth_relative_path": [
            "E3-CADETS/node_Nginx_Backdoor_06.csv",
            "E3-CADETS/node_Nginx_Backdoor_12.csv",
        ],
        "attack_to_time_window": [
            ["E3-CADETS/node_Nginx_Backdoor_06.csv", "2018-04-06 11:20:00", "2018-04-06 12:09:00"],
            ["E3-CADETS/node_Nginx_Backdoor_12.csv", "2018-04-12 13:59:00", "2018-04-12 14:39:00"],
        ],
    },
    # The mirror image of CADETS_E3_NOFRAC: train and val days swapped, everything else identical.
    #
    # Why a second model is needed. The published node-level MIAs this pilot is benchmarked against
    # -- He et al. 2021 and Conti et al. 2022 (AISec) -- are both SHADOW-model attacks: the adversary
    # trains their own model on data whose membership they know, fits an attack classifier on its
    # outputs, and transfers that classifier to the target. Scoring the target with its own
    # membership labels (cross-validation) measures separability, not an attack, and overstates what
    # an adversary can do. Without this dataset there is no second model to be the shadow, because
    # the cross-role design already spent the shadow role's days as the target's non-member pool.
    #
    # Roles are symmetric by construction, so the attack can be run in both directions and any large
    # asymmetry between them is itself a finding rather than something to average away.
    "CADETS_E3_NOFRAC_SHADOW": {
        "raw_dir": "",
        "database": "cadets_e3",
        "database_all_file": "cadets_e3",
        "num_node_types": 3,
        "num_edge_types": 10,
        "start_date": "2018-04-02",
        "end_date": "2018-04-14",
        # members of the SHADOW model: the days that are non-members for the target
        "train_dates": [
            "2018-04-07",
            "2018-04-08",
            "2018-04-09",
        ],
        # non-members of the SHADOW model: the target's training days
        "val_dates": [
            "2018-04-02",
            "2018-04-03",
            "2018-04-04",
        ],
        "test_dates": ["2018-04-06", "2018-04-12"],
        "unused_dates": [],
        "ground_truth_relative_path": [
            "E3-CADETS/node_Nginx_Backdoor_06.csv",
            "E3-CADETS/node_Nginx_Backdoor_12.csv",
        ],
        "attack_to_time_window": [
            ["E3-CADETS/node_Nginx_Backdoor_06.csv", "2018-04-06 11:20:00", "2018-04-06 12:09:00"],
            ["E3-CADETS/node_Nginx_Backdoor_12.csv", "2018-04-12 13:59:00", "2018-04-12 14:39:00"],
        ],
    },
    # CADETS_E3_NOFRAC_SHADOW re-pointed at the shadow role's own attack day, for RE-EVALUATION ONLY.
    #
    # Identical to CADETS_E3_NOFRAC_SHADOW in every field that enters an artifact hash -- same
    # `train_dates`, same construction/featurization/training args -- so the already-trained
    # checkpoints are reused verbatim and nothing is retrained. Only `test_dates` moves, and
    # `test_dates` enters no hash (pidsmaker/config/pipeline.py:208-231).
    #
    # WHY IT IS A SEPARATE ENTRY rather than an edit in place. The 04-06/12 evaluation of that same
    # model is a finished result. Because `test_dates` is not hashed, editing the original would
    # silently rebuild over its artifacts in the same directories, leaving no way to regenerate the
    # old numbers. A distinct name keeps both evaluations of one model addressable at once.
    #
    # What this answers: how the shadow model detects on the day the committed CADETS_E3_SHADOW
    # calls its test day. The MIA is untouched -- `mia_split_mode: xrole` reads members from
    # `train_dates` and non-members from `val_dates`, neither of which moves here.
    "CADETS_E3_NOFRAC_SHADOW_T13": {
        "raw_dir": "",
        "database": "cadets_e3",
        "database_all_file": "cadets_e3",
        "num_node_types": 3,
        "num_edge_types": 10,
        "start_date": "2018-04-02",
        "end_date": "2018-04-14",
        "train_dates": [
            "2018-04-07",
            "2018-04-08",
            "2018-04-09",
        ],
        "val_dates": [
            "2018-04-02",
            "2018-04-03",
            "2018-04-04",
        ],
        "test_dates": ["2018-04-13"],
        "unused_dates": [],
        "ground_truth_relative_path": [
            "E3-CADETS/node_Nginx_Backdoor_13.csv",
        ],
        "attack_to_time_window": [
            ["E3-CADETS/node_Nginx_Backdoor_13.csv", "2018-04-13 09:03:00", "2018-04-13 09:16:00"],
        ],
    },
    # The NEUTRAL reference model for difficulty calibration. Trains on three days that are neither
    # the target's members nor its non-members, and validates on all six days the MIA is scored over.
    #
    # Why it is needed. CADETS_E3_NOFRAC_SHADOW cannot serve as a calibration reference, because its
    # training days ARE the target's non-member pool: it is biased towards exactly the half the
    # calibration is supposed to treat neutrally, and any attack classifier fitted on it transfers to
    # the target with inverted sign. Measured on the epoch-49 pair: the target scores AUC 0.645
    # against its own membership labels, but the shadow -- which never trained on those members --
    # scores 0.617 against the same labels. Nearly all of the apparent leak is the two day sets being
    # unequally easy to reconstruct, not memorisation.
    #
    # This model is neutral to both halves, so its loss on a node estimates that node's intrinsic
    # difficulty. Subtracting it isolates what is specific to the target having trained on the node.
    # It also gives the shadow-model attacks (He et al. 2021, Conti et al. 2022) a reference whose
    # in/out split is not anti-correlated with the target's.
    #
    # Day choice: 04-06/12/13 carry APT ground truth and are excluded; 04-14 has no events at all.
    # That leaves 04-05, 04-10 and 04-11, which are ~3M events each and so comparable in size to the
    # days both roles train on.
    "CADETS_E3_NOFRAC_CAL": {
        "raw_dir": "",
        "database": "cadets_e3",
        "database_all_file": "cadets_e3",
        "num_node_types": 3,
        "num_edge_types": 10,
        "start_date": "2018-04-02",
        "end_date": "2018-04-14",
        # days used by neither role
        "train_dates": [
            "2018-04-05",
            "2018-04-10",
            "2018-04-11",
        ],
        # every day the MIA is scored over, so this model reports a loss for all of them
        "val_dates": [
            "2018-04-02",
            "2018-04-03",
            "2018-04-04",
            "2018-04-07",
            "2018-04-08",
            "2018-04-09",
        ],
        "test_dates": ["2018-04-06", "2018-04-12"],
        "unused_dates": [],
        "ground_truth_relative_path": [
            "E3-CADETS/node_Nginx_Backdoor_06.csv",
            "E3-CADETS/node_Nginx_Backdoor_12.csv",
        ],
        "attack_to_time_window": [
            ["E3-CADETS/node_Nginx_Backdoor_06.csv", "2018-04-06 11:20:00", "2018-04-06 12:09:00"],
            ["E3-CADETS/node_Nginx_Backdoor_12.csv", "2018-04-12 13:59:00", "2018-04-12 14:39:00"],
        ],
    },
    # ---------------------------------------------------------------------------------------------
    # V2: the SHADOW-MODEL design. Supersedes the CADETS_E3_NOFRAC_* trio above for MIA work.
    #
    # WHAT CHANGED AND WHY. In the NOFRAC trio the two roles are mirrors: the target's `val_dates`
    # ARE the shadow's `train_dates`. Under `mia_split_mode: xrole` that makes the target's
    # non-member pool identical to the shadow's member pool, which is elegant but means the shadow
    # is not an independent model -- it is the target's negative class, trained. Here each role owns
    # a private validation day instead (target 04-05, shadow 04-10), matching the day assignment of
    # the committed CADETS_E3_TARGET / CADETS_E3_SHADOW pair.
    #
    # That makes this the textbook shadow-model threat model (He et al. 2021, Conti et al. 2022):
    # the adversary trains their own model on their own days with known membership, fits an attack
    # classifier on its outputs, and transfers it to the target. Target and shadow share no days.
    #
    # KNOWN LIMITATION, stated up front: membership is still perfectly collinear with the calendar
    # (members 04-02/03/04 vs non-members 04-05). This design does not remove the day confound that
    # the NOFRAC campaign identified -- only a within-day randomised split would. The neutral arm
    # below is what keeps the confound measurable.
    #
    # COST OF THE PRIVATE VAL DAY. The non-member pool shrinks from three days to one, so the MIA
    # scores far fewer negatives than the NOFRAC design did.
    "CADETS_E3_V2_TARGET": {
        "raw_dir": "",
        "database": "cadets_e3",
        "database_all_file": "cadets_e3",
        "num_node_types": 3,
        "num_edge_types": 10,
        "start_date": "2018-04-02",
        "end_date": "2018-04-14",
        # members of the TARGET model
        "train_dates": [
            "2018-04-02",
            "2018-04-03",
            "2018-04-04",
        ],
        # non-members of the TARGET model, private to this role (the shadow never touches 04-05)
        "val_dates": ["2018-04-05"],
        "test_dates": ["2018-04-06", "2018-04-12"],
        "unused_dates": [],
        "ground_truth_relative_path": [
            "E3-CADETS/node_Nginx_Backdoor_06.csv",
            "E3-CADETS/node_Nginx_Backdoor_12.csv",
        ],
        "attack_to_time_window": [
            ["E3-CADETS/node_Nginx_Backdoor_06.csv", "2018-04-06 11:20:00", "2018-04-06 12:09:00"],
            ["E3-CADETS/node_Nginx_Backdoor_12.csv", "2018-04-12 13:59:00", "2018-04-12 14:39:00"],
        ],
    },
    # The adversary's own model. Day-for-day disjoint from CADETS_E3_V2_TARGET, including the test
    # day: 04-13 is the shadow role's attack day in the committed CADETS_E3_SHADOW too.
    #
    # The detection columns of the two roles are therefore NOT head-to-head -- 04-13 lists 24
    # ground-truth malicious nodes against 51 across 04-06+04-12 (49 of which are realized in the
    # graphs). Read them as two separate detection results, not as a comparison.
    "CADETS_E3_V2_SHADOW": {
        "raw_dir": "",
        "database": "cadets_e3",
        "database_all_file": "cadets_e3",
        "num_node_types": 3,
        "num_edge_types": 10,
        "start_date": "2018-04-02",
        "end_date": "2018-04-14",
        # members of the SHADOW model
        "train_dates": [
            "2018-04-07",
            "2018-04-08",
            "2018-04-09",
        ],
        # non-members of the SHADOW model, private to this role
        "val_dates": ["2018-04-10"],
        "test_dates": ["2018-04-13"],
        "unused_dates": [],
        "ground_truth_relative_path": [
            "E3-CADETS/node_Nginx_Backdoor_13.csv",
        ],
        "attack_to_time_window": [
            ["E3-CADETS/node_Nginx_Backdoor_13.csv", "2018-04-13 09:03:00", "2018-04-13 09:16:00"],
        ],
    },
    # The NEUTRAL reference: a model that trained on NEITHER role's members NOR either role's
    # non-members, so its loss on a node estimates that node's intrinsic difficulty. Subtracting it
    # is what separates memorisation from day-to-day drift, which is the whole methodological point
    # of the campaign.
    #
    # WHY IT TRAINS ON A SINGLE DAY, unlike CADETS_E3_NOFRAC_CAL's three. The v2 split spends 04-05
    # and 04-10 as the two roles' non-member pools, and CADETS_E3_NOFRAC_CAL trained on exactly
    # those two days plus 04-11. A neutral model that trains on the non-members is worse than no
    # neutral at all: it fits the negative class, which deflates its own member-vs-non-member AUC
    # and therefore INFLATES `target - neutral` in the direction of an apparent leak. So they are
    # excluded, and the day inventory leaves only 04-11 -- 04-06/11/12/13 carry APT ground truth
    # (04-11 is kept because its ground truth is 4 lines and it is not used as anyone's test day),
    # and 04-14 has no events at all.
    #
    # CAVEAT TO CARRY INTO ANY WRITE-UP: one training day is ~80k unique nodes against the target's
    # ~178k, so this reference is a weaker model than the roles it calibrates. dAUC is a ranking
    # difference, so a uniform loss offset does not bias it, but a weaker model is noisier.
    "CADETS_E3_V2_CAL": {
        "raw_dir": "",
        "database": "cadets_e3",
        "database_all_file": "cadets_e3",
        "num_node_types": 3,
        "num_edge_types": 10,
        "start_date": "2018-04-02",
        "end_date": "2018-04-14",
        # the only benign day neither role uses as members or non-members
        "train_dates": ["2018-04-11"],
        # every day the MIA is scored over, so this model reports a loss for all of them
        "val_dates": [
            "2018-04-02",
            "2018-04-03",
            "2018-04-04",
            "2018-04-05",
            "2018-04-07",
            "2018-04-08",
            "2018-04-09",
            "2018-04-10",
        ],
        "test_dates": ["2018-04-06", "2018-04-12"],
        "unused_dates": [],
        "ground_truth_relative_path": [
            "E3-CADETS/node_Nginx_Backdoor_06.csv",
            "E3-CADETS/node_Nginx_Backdoor_12.csv",
        ],
        "attack_to_time_window": [
            ["E3-CADETS/node_Nginx_Backdoor_06.csv", "2018-04-06 11:20:00", "2018-04-06 12:09:00"],
            ["E3-CADETS/node_Nginx_Backdoor_12.csv", "2018-04-12 13:59:00", "2018-04-12 14:39:00"],
        ],
    },
    # ---------------------------------------------------------------------------------------------
    # THEIA_E3 split into two roles with *disjoint training days*, so membership is well defined.
    #
    # The day inventory drives this split and is not obvious from the base THEIA_E3 entry above:
    # event_table has NO rows for 2018-04-02, 04-06, 04-07 and 04-08. The base config lists them as
    # train_dates, where they are harmless no-ops, but a role built on them would train on an empty
    # graph. Only 8 days carry data:
    #
    #   04-03 8.19M | 04-04 7.98M | 04-05 2.43M | 04-09 0.71M
    #   04-10 5.80M (attack) | 04-11 7.35M | 04-12 6.79M (attack) | 04-13 5.12M
    #
    # Two usable attack days, so each role gets one and the detection tables stay self-contained.
    # That leaves 6 benign days for 2 roles x (train + val), i.e. 2 train days each - one fewer than
    # CADETS, but on far larger days (target 16.2M / shadow 8.1M events, vs CADETS 5.8M / 11.2M).
    # Days are assigned so that every training day falls strictly BEFORE that role's attack day.
    # The one compromise: the shadow role's validation day (04-13) is after its attack day, because
    # no earlier benign day is left. Validation only selects a threshold on benign data and takes no
    # part in the membership definition, so this does not contaminate the MIA split.
    "THEIA_E3_TARGET": {
        "raw_dir": "",
        "database": "theia_e3",
        "database_all_file": "theia_e3",
        "num_node_types": 3,
        "num_edge_types": 10,
        "start_date": "2018-04-02",
        "end_date": "2018-04-14",
        "train_dates": [
            "2018-04-03",
            "2018-04-04",
        ],
        "val_dates": ["2018-04-05"],
        "test_dates": ["2018-04-10"],
        "unused_dates": [],
        "ground_truth_relative_path": [
            "E3-THEIA/node_Firefox_Backdoor_Drakon_In_Memory.csv",
        ],
        "attack_to_time_window": [
            [
                "E3-THEIA/node_Firefox_Backdoor_Drakon_In_Memory.csv",
                "2018-04-10 14:30:00",
                "2018-04-10 15:00:00",
            ],
        ],
    },
    "THEIA_E3_SHADOW": {
        "raw_dir": "",
        "database": "theia_e3",
        "database_all_file": "theia_e3",
        "num_node_types": 3,
        "num_edge_types": 10,
        "start_date": "2018-04-02",
        "end_date": "2018-04-14",
        "train_dates": [
            "2018-04-09",
            "2018-04-11",
        ],
        "val_dates": ["2018-04-13"],
        "test_dates": ["2018-04-12"],
        "unused_dates": [],
        "ground_truth_relative_path": [
            "E3-THEIA/node_Browser_Extension_Drakon_Dropper.csv",
        ],
        "attack_to_time_window": [
            [
                "E3-THEIA/node_Browser_Extension_Drakon_Dropper.csv",
                "2018-04-12 12:40:00",
                "2018-04-12 13:30:00",
            ],
        ],
    },
    "CLEARSCOPE_E5": {
        "raw_dir": "",
        "database": "clearscope_e5",
        "database_all_file": "clearscope_e5",
        "num_node_types": 3,
        "num_edge_types": 10,
        "start_date": "2019-05-08",
        "end_date": "2019-05-18",
        "train_dates": ["2019-05-08", "2019-05-09", "2019-05-10", "2019-05-11", "2019-05-12"],
        "val_dates": ["2019-05-13"],
        "test_dates": ["2019-05-14", "2019-05-15", "2019-05-17"],
        "unused_dates": ["2019-05-16"],
        "ground_truth_relative_path": [
            "E5-CLEARSCOPE/node_clearscope_e5_appstarter_0515.csv",
            # "E5-CLEARSCOPE/node_clearscope_e5_firefox_0517.csv",
            # "E5-CLEARSCOPE/node_clearscope_e5_lockwatch_0517.csv",
            "E5-CLEARSCOPE/node_clearscope_e5_tester_0517.csv",
        ],
        "attack_to_time_window": [
            [
                "E5-CLEARSCOPE/node_clearscope_e5_appstarter_0515.csv",
                "2019-05-15 15:38:00",
                "2019-05-15 16:19:00",
            ],
            # ["E5-CLEARSCOPE/node_clearscope_e5_firefox_0517.csv", '2019-05-17 11:49:00', '2019-05-17 15:32:00'],
            [
                "E5-CLEARSCOPE/node_clearscope_e5_lockwatch_0517.csv",
                "2019-05-17 15:48:00",
                "2019-05-17 16:01:00",
            ],
            [
                "E5-CLEARSCOPE/node_clearscope_e5_tester_0517.csv",
                "2019-05-17 16:20:00",
                "2019-05-17 16:28:00",
            ],
        ],
    },
    "CLEARSCOPE_E3": {
        "raw_dir": "",
        "database": "clearscope_e3",
        "database_all_file": "clearscope_e3",
        "num_node_types": 3,
        "num_edge_types": 10,
        "start_date": "2018-04-02",
        "end_date": "2018-04-14",
        "train_dates": [
            "2018-04-03",
            "2018-04-04",
            "2018-04-05",
            "2018-04-07",
            "2018-04-08",
            "2018-04-09",
            "2018-04-10",
        ],
        "val_dates": ["2018-04-02"],
        "test_dates": ["2018-04-11", "2018-04-12"],
        "unused_dates": ["2018-04-06", "2018-04-13"],
        "ground_truth_relative_path": [
            "E3-CLEARSCOPE/node_clearscope_e3_firefox_0411.csv",
            # "E3-CLEARSCOPE/node_clearscope_e3_firefox_0412.csv", # due to malicious file downloaded but failed to exec and feture missing, there is no malicious nodes found in database
        ],
        "attack_to_time_window": [
            [
                "E3-CLEARSCOPE/node_clearscope_e3_firefox_0411.csv",
                "2018-04-11 13:54:00",
                "2018-04-11 14:48:00",
            ],
            # ["E3-CLEARSCOPE/node_clearscope_e3_firefox_0412.csv", '2018-04-12 15:18:00', '2018-04-12 15:25:00'],
        ],
    },
    # CLEARSCOPE_E3 split into two roles with *disjoint training days*, so membership is well
    # defined. Two things make this split different from the CADETS and THEIA ones above.
    #
    # 1. Only ONE usable attack day exists (04-11; the 04-12 ground truth is commented out in the
    #    base entry above because no malicious node survives into the database). Both roles
    #    therefore share test_dates 04-11. That is sound: MIA membership is defined by train_dates
    #    (mia_train_in / mia_heldout are drawn from them) and 04-11 is held out of BOTH roles'
    #    training, so a shared attack day cannot contaminate the membership split. It also makes
    #    the two roles' detection rows directly comparable.
    #
    # 2. The days are volume-balanced on purpose. The detection threshold is max_val_loss - the
    #    MAXIMUM loss over the whole validation split (evaluation_utils.py:120) - which is an
    #    extreme-value statistic and therefore grows with validation-set size. A lopsided split
    #    gives the two roles thresholds that are not comparable, which shows up as spurious
    #    asymmetry in the TP/FP/TN/FN columns. Day inventory (event_table rows):
    #
    #      04-02 0.04M | 04-03 1.19M | 04-04 1.46M | 04-05 0.94M | 04-06 1.19M
    #      04-07 1.84M | 04-08 1.10M | 04-09 2.09M | 04-10 2.81M | 04-11 2.13M (attack)
    #      04-12 2.00M | 04-13 1.39M
    #
    #    The assignment below balances train volume to 1.015x, val volume to 1.076x and total
    #    role volume to 1.001x, and keeps each role's val day strictly INSIDE its own training
    #    span so no role is thresholded on data from outside the period it was trained on.
    #    04-02 is dropped (0.04M rows, two orders of magnitude thin); 04-12 and 04-13 are dropped
    #    because they fall after the shared attack day.
    "CLEARSCOPE_E3_TARGET": {
        "raw_dir": "",
        "database": "clearscope_e3",
        "database_all_file": "clearscope_e3",
        "num_node_types": 3,
        "num_edge_types": 10,
        "start_date": "2018-04-02",
        "end_date": "2018-04-14",
        # 1.46M + 0.94M + 2.81M = 5.21M
        "train_dates": [
            "2018-04-04",
            "2018-04-05",
            "2018-04-10",
        ],
        "val_dates": ["2018-04-08"],  # 1.10M, inside the 04-04..04-10 training span
        "test_dates": ["2018-04-11"],
        "unused_dates": ["2018-04-02", "2018-04-12", "2018-04-13"],
        "ground_truth_relative_path": [
            "E3-CLEARSCOPE/node_clearscope_e3_firefox_0411.csv",
        ],
        "attack_to_time_window": [
            [
                "E3-CLEARSCOPE/node_clearscope_e3_firefox_0411.csv",
                "2018-04-11 13:54:00",
                "2018-04-11 14:48:00",
            ],
        ],
    },
    "CLEARSCOPE_E3_SHADOW": {
        "raw_dir": "",
        "database": "clearscope_e3",
        "database_all_file": "clearscope_e3",
        "num_node_types": 3,
        "num_edge_types": 10,
        "start_date": "2018-04-02",
        "end_date": "2018-04-14",
        # 1.19M + 1.84M + 2.09M = 5.13M
        "train_dates": [
            "2018-04-03",
            "2018-04-07",
            "2018-04-09",
        ],
        "val_dates": ["2018-04-06"],  # 1.19M, inside the 04-03..04-09 training span
        "test_dates": ["2018-04-11"],
        "unused_dates": ["2018-04-02", "2018-04-12", "2018-04-13"],
        "ground_truth_relative_path": [
            "E3-CLEARSCOPE/node_clearscope_e3_firefox_0411.csv",
        ],
        "attack_to_time_window": [
            [
                "E3-CLEARSCOPE/node_clearscope_e3_firefox_0411.csv",
                "2018-04-11 13:54:00",
                "2018-04-11 14:48:00",
            ],
        ],
    },
    # ---------------------------------------------------------------------------------------------
    # V2: the SHADOW-MODEL design for CLEARSCOPE_E3. Supersedes the CLEARSCOPE_E3_TARGET/SHADOW pair
    # above for MIA work. Same threat model as CADETS_E3_V2_* and THEIA_E3_TARGET/SHADOW: each role
    # trains on its own days, owns a private non-member pool, and shares no day with the other role.
    #
    # SHAPE IS 2 TRAIN + 2 VAL, not 3 + 1. Under `mia_split_mode: xrole` `val_dates` IS the
    # non-member class, so the structure floor is set by the WITHIN-role distance between a role's
    # train pool and its val pool. Searching all 560 (3+1) and 1260 (2+2) assignments of the eight
    # usable benign days (see /data/clearscope_split_design/) gives:
    #
    #     shape                    best floor   best class ratio
    #     3 train + 1 val              0.4254   1.84:1 / 2.16:1
    #     2 train + 2 val              0.2834   1.42:1 / 1.03:1   <- this
    #     committed CLEARSCOPE_E3_*    0.8313   2.42:1 / 2.19:1
    #
    # The 2+2 shape cuts the floor by a third against the best 3+1 and ~2.9x against the committed
    # pair, and is the only shape that gets a role near 1:1 class balance. Measured pools:
    # target 45,522 members vs 32,154 non-members (node-type L1 0.0271, 2,589 dropped for appearing
    # in both); shadow 40,688 vs 39,657 (L1 0.0051, 3,218 dropped). It costs one training day, which
    # THEIA_E3 already pays for the same reason, and CLEARSCOPE detection is weak regardless --
    # only ~1.4% of either role's members reappear on the attack day.
    #
    # BOTH ROLES SHARE test_dates 04-11, and unlike CADETS/THEIA no private attack day is possible:
    # node_clearscope_e3_firefox_0412.csv is an EMPTY file (see the base CLEARSCOPE_E3 entry), so
    # 04-11 is the only day with attack labels. This does not touch the MIA -- `xrole` excludes test
    # days from the non-member pool on purpose (training_loop.py:72: they carry the APT ground truth,
    # so scoring them as non-members would measure the attack rather than membership). It is in fact
    # an advantage for the detection table: both roles are scored on the identical 57,569-node
    # evaluation set, which the CADETS v2 pair (04-06/12 vs 04-13) cannot offer. Test-day exposure is
    # balanced between the roles: 977 member nodes (1.7%) for target, 635 (1.1%) for shadow.
    #
    # KNOWN LIMITATION, same as every other v2 pair: membership remains collinear with the calendar.
    # Day-to-day Jaccard on CLEARSCOPE is 0.010-0.091 and 85-88% of each day's nodes occur on exactly
    # one day, so the structure floor must be measured and subtracted before any AUC counts as
    # memorisation. Only a within-day randomised node split removes the confound outright.
    "CLEARSCOPE_E3_V2_TARGET": {
        "raw_dir": "",
        "database": "clearscope_e3",
        "database_all_file": "clearscope_e3",
        "num_node_types": 3,
        "num_edge_types": 10,
        "start_date": "2018-04-02",
        "end_date": "2018-04-14",
        # members: 1.10M + 2.09M events, 45,522 nodes after dropping the train/val intersection
        "train_dates": [
            "2018-04-08",
            "2018-04-09",
        ],
        # non-members: 1.19M + 1.19M events, 32,154 nodes
        "val_dates": [
            "2018-04-03",
            "2018-04-06",
        ],
        "test_dates": ["2018-04-11"],
        "unused_dates": [
            "2018-04-02",
            "2018-04-04",
            "2018-04-05",
            "2018-04-07",
            "2018-04-10",
            "2018-04-12",
            "2018-04-13",
        ],
        "ground_truth_relative_path": [
            "E3-CLEARSCOPE/node_clearscope_e3_firefox_0411.csv",
        ],
        "attack_to_time_window": [
            [
                "E3-CLEARSCOPE/node_clearscope_e3_firefox_0411.csv",
                "2018-04-11 13:54:00",
                "2018-04-11 14:48:00",
            ],
        ],
    },
    "CLEARSCOPE_E3_V2_SHADOW": {
        "raw_dir": "",
        "database": "clearscope_e3",
        "database_all_file": "clearscope_e3",
        "num_node_types": 3,
        "num_edge_types": 10,
        "start_date": "2018-04-02",
        "end_date": "2018-04-14",
        # members: 0.94M + 2.81M events, 40,688 nodes after dropping the train/val intersection
        "train_dates": [
            "2018-04-05",
            "2018-04-10",
        ],
        # non-members: 1.46M + 1.84M events, 39,657 nodes
        "val_dates": [
            "2018-04-04",
            "2018-04-07",
        ],
        "test_dates": ["2018-04-11"],
        "unused_dates": [
            "2018-04-02",
            "2018-04-03",
            "2018-04-06",
            "2018-04-08",
            "2018-04-09",
            "2018-04-12",
            "2018-04-13",
        ],
        "ground_truth_relative_path": [
            "E3-CLEARSCOPE/node_clearscope_e3_firefox_0411.csv",
        ],
        "attack_to_time_window": [
            [
                "E3-CLEARSCOPE/node_clearscope_e3_firefox_0411.csv",
                "2018-04-11 13:54:00",
                "2018-04-11 14:48:00",
            ],
        ],
    },
    "optc_h201": {
        "raw_dir": "",
        "database": "optc_201",
        "database_all_file": "optc_201",
        "num_node_types": 3,
        "num_edge_types": 10,
        "start_date": "2019-09-15",
        "end_date": "2019-09-26",
        "train_dates": ["2019-09-19", "2019-09-20", "2019-09-21"],
        "val_dates": ["2019-09-22"],
        "test_dates": ["2019-09-23", "2019-09-24", "2019-09-25"],
        "unused_dates": ["2019-09-16", "2019-09-17", "2019-09-18"],
        "ground_truth_relative_path": [
            "h201/node_h201_0923.csv",
        ],
        "attack_to_time_window": [
            ["h201/node_h201_0923.csv", "2019-09-23 11:23:00", "2019-09-23 13:25:00"],
        ],
    },
    "optc_h501": {
        "raw_dir": "",
        "database": "optc_501",
        "database_all_file": "optc_501",
        "num_node_types": 3,
        "num_edge_types": 10,
        "start_date": "2019-09-15",
        "end_date": "2019-09-26",
        "train_dates": ["2019-09-19", "2019-09-20", "2019-09-21"],
        "val_dates": ["2019-09-22"],
        "test_dates": ["2019-09-23", "2019-09-24", "2019-09-25"],
        "unused_dates": ["2019-09-16", "2019-09-17", "2019-09-18"],
        "ground_truth_relative_path": [
            "h501/node_h501_0924.csv",
        ],
        "attack_to_time_window": [
            ["h501/node_h501_0924.csv", "2019-09-24 10:28:00", "2019-09-24 15:29:00"],
        ],
    },
    "optc_h051": {
        "raw_dir": "",
        "database": "optc_051",
        "database_all_file": "optc_051",
        "num_node_types": 3,
        "num_edge_types": 10,
        "start_date": "2019-09-15",
        "end_date": "2019-09-26",
        "train_dates": ["2019-09-19", "2019-09-20", "2019-09-21"],
        "val_dates": ["2019-09-22"],
        "test_dates": ["2019-09-23", "2019-09-24", "2019-09-25"],
        "unused_dates": ["2019-09-16", "2019-09-17", "2019-09-18"],
        "ground_truth_relative_path": [
            "h051/node_h051_0925.csv",
        ],
        "attack_to_time_window": [
            ["h051/node_h051_0925.csv", "2019-09-25 10:29:00", "2019-09-25 14:25:00"],
        ],
    },
    "TRACE_E5": {
        "raw_dir": "",
        "database": "trace_e5",
        "database_all_file": "trace_e5",
        "num_node_types": 3,
        "num_edge_types": 10,
        "start_date": "2019-05-08",
        "end_date": "2019-05-18",
        "train_dates": ["2019-05-08", "2019-05-09"],
        "val_dates": ["2019-05-11"],
        "test_dates": ["2019-05-14", "2019-05-15"],
        "unused_dates": ["2019-05-10", "2019-05-12", "2019-05-13", "2019-05-16", "2019-05-17"],
        "ground_truth_relative_path": [
            "E5-TRACE/node_Trace_Firefox_Drakon.csv",
        ],
        "attack_to_time_window": [
            [
                "E5-TRACE/node_Trace_Firefox_Drakon.csv",
                "2019-05-14 10:17:00",
                "2019-05-14 11:45:00",
            ],
        ],
    },
    "TRACE_E3": {
        "raw_dir": "",
        "database": "trace_e3",
        "database_all_file": "trace_e3",
        "num_node_types": 3,
        "num_edge_types": 10,
        "start_date": "2018-04-02",
        "end_date": "2018-04-14",
        "train_dates": [
            "2018-04-03",
            "2018-04-04",
            "2018-04-05",
            "2018-04-07",
            "2018-04-08",
            "2018-04-09",
            "2018-04-06",
            "2018-04-11",
            "2018-04-12",
        ],
        "val_dates": ["2018-04-02"],
        "test_dates": [
            "2018-04-10",
            "2018-04-13",
        ],
        "unused_dates": [],
        "ground_truth_relative_path": [
            "E3-TRACE/node_trace_e3_firefox_0410.csv",
            "E3-TRACE/node_trace_e3_phishing_executable_0413.csv",
            "E3-TRACE/node_trace_e3_pine_0413.csv",
        ],
        "attack_to_time_window": [
            [
                "E3-TRACE/node_trace_e3_firefox_0410.csv",
                "2018-04-10 09:45:00",
                "2018-04-10 11:10:00",
            ],
            [
                "E3-TRACE/node_trace_e3_phishing_executable_0413.csv",
                "2018-04-13 14:14:00",
                "2018-04-13 14:29:00",
            ],
            ["E3-TRACE/node_trace_e3_pine_0413.csv", "2018-04-13 12:42:00", "2018-04-13 12:54:00"],
        ],
    },
    "FIVEDIRECTIONS_E5": {
        "raw_dir": "",
        "database": "fivedirections_e5",
        "database_all_file": "fivedirections_e5",
        "num_node_types": 3,
        "num_edge_types": 10,
        "start_date": "2019-05-08",
        "end_date": "2019-05-18",
        "train_dates": ["2019-05-08", "2019-05-10", "2019-05-11", "2019-05-13", "2019-05-14"],
        "val_dates": ["2019-05-12"],
        "test_dates": [
            "2019-05-09",
            "2019-05-15",
            "2019-05-17",
        ],
        "unused_dates": ["2019-05-16"],
        "ground_truth_relative_path": [
            "E5-FIVEDIRECTIONS/node_fivedirections_e5_bits_0515.csv",
            "E5-FIVEDIRECTIONS/node_fivedirections_e5_copykatz_0509.csv",
            "E5-FIVEDIRECTIONS/node_fivedirections_e5_dns_0517.csv",
            "E5-FIVEDIRECTIONS/node_fivedirections_e5_drakon_0517.csv",
        ],
        "attack_to_time_window": [
            [
                "E5-FIVEDIRECTIONS/node_fivedirections_e5_bits_0515.csv",
                "2019-05-15 13:14:00",
                "2019-05-15 13:35:00",
            ],
            [
                "E5-FIVEDIRECTIONS/node_fivedirections_e5_copykatz_0509.csv",
                "2019-05-09 13:25:00",
                "2019-05-09 13:57:00",
            ],
            [
                "E5-FIVEDIRECTIONS/node_fivedirections_e5_dns_0517.csv",
                "2019-05-17 12:46:00",
                "2019-05-17 12:57:00",
            ],
            [
                "E5-FIVEDIRECTIONS/node_fivedirections_e5_drakon_0517.csv",
                "2019-05-17 16:10:00",
                "2019-05-17 16:16:00",
            ],
        ],
    },
    "FIVEDIRECTIONS_E3": {
        "raw_dir": "",
        "database": "fivedirections_e3",
        "database_all_file": "fivedirections_e3",
        "num_node_types": 3,
        "num_edge_types": 10,
        "start_date": "2018-04-02",
        "end_date": "2018-04-14",
        "train_dates": [
            "2018-04-03",
            "2018-04-05",
            "2018-04-06",
            "2018-04-07",
            "2018-04-08",
            "2018-04-10",
            "2018-04-13",
        ],
        "val_dates": ["2018-04-04"],
        "test_dates": ["2018-04-09", "2018-04-11"],
        "unused_dates": ["2018-04-12"],
        "ground_truth_relative_path": [
            "E3-FIVEDIRECTIONS/node_fivedirections_e3_firefox_0411.csv",
            # "E3-FIVEDIRECTIONS/node_fivedirections_e3_browser_0412.csv",
            "E3-FIVEDIRECTIONS/node_fivedirections_e3_excel_0409.csv",
        ],
        "attack_to_time_window": [
            [
                "E3-FIVEDIRECTIONS/node_fivedirections_e3_firefox_0411.csv",
                "2018-04-11 09:59:00",
                "2018-04-11 10:41:00",
            ],
            # ["E3-FIVEDIRECTIONS/node_fivedirections_e3_browser_0412.csv", '2018-04-12 11:12:00', '2018-04-12 11:15:00'],
            [
                "E3-FIVEDIRECTIONS/node_fivedirections_e3_excel_0409.csv",
                "2018-04-09 15:06:00",
                "2018-04-09 15:43:00",
            ],
        ],
    },
    # See https://arxiv.org/pdf/2401.01341
    "ATLASV2_EDR": {
        "raw_dir": "",
        "database": "atlasv2_edr",
        "database_all_file": "atlasv2_edr",
        "num_node_types": 3,
        "num_edge_types": 33,
        "start_date": "2022-07-15",
        "end_date": "2022-07-21",
        "train_dates": [
            "2022-07-16",
            # Arbitrarly picked 2022-07-17 for the validation/threshold calibration
            "2022-07-18",
        ],
        "val_dates": [
            "2022-07-17"
        ],
        "test_dates": [
            "2022-07-19",
            "2022-07-20"
        ],
        "unused_dates": [
            "2022-07-15"
        ],
        "ground_truth_relative_path": [
            "atlasv2_edr/atlasv2_edr_s1.csv",
            "atlasv2_edr/atlasv2_edr_s2.csv",
            "atlasv2_edr/atlasv2_edr_s3.csv",
            "atlasv2_edr/atlasv2_edr_s4.csv",
            "atlasv2_edr/atlasv2_edr_m1.csv",
            "atlasv2_edr/atlasv2_edr_m2.csv",
            "atlasv2_edr/atlasv2_edr_m3.csv",
            "atlasv2_edr/atlasv2_edr_m4.csv",
            "atlasv2_edr/atlasv2_edr_m5.csv",
            "atlasv2_edr/atlasv2_edr_m6.csv",
        ],
        "attack_to_time_window": [
            # NOTE: the reported attack windows are somewhat inaccurate (i.e., the first and last
            # true-positive malicious alerts occur OUTSIDE the reported attack windows), so we start
            # at 2022-07-19 13:00:00 (Rather than 2022-07-19 13:12:00) and end at
            # 2022-07-20 01:15:00 (Rather than 2022-07-20 01:00:00) to capture all of the true
            # positives.
            ["atlasv2_edr/atlasv2_edr_s1.csv", "2022-07-19 13:00:00", "2022-07-19 13:40:00"],
            ["atlasv2_edr/atlasv2_edr_s2.csv", "2022-07-19 13:45:00", "2022-07-19 14:20:00"],
            ["atlasv2_edr/atlasv2_edr_s3.csv", "2022-07-19 14:20:00", "2022-07-19 15:05:00"],
            ["atlasv2_edr/atlasv2_edr_s4.csv", "2022-07-20 00:31:00", "2022-07-20 01:15:00"],
            ["atlasv2_edr/atlasv2_edr_m1.csv", "2022-07-19 16:00:00", "2022-07-19 17:50:00"],
            ["atlasv2_edr/atlasv2_edr_m2.csv", "2022-07-19 19:32:00", "2022-07-19 20:02:00"],
            ["atlasv2_edr/atlasv2_edr_m3.csv", "2022-07-19 20:06:00", "2022-07-19 20:40:00"],
            ["atlasv2_edr/atlasv2_edr_m4.csv", "2022-07-19 22:31:00", "2022-07-19 23:04:00"],
            ["atlasv2_edr/atlasv2_edr_m5.csv", "2022-07-19 23:16:00", "2022-07-19 23:46:00"],
            ["atlasv2_edr/atlasv2_edr_m6.csv", "2022-07-19 23:54:00", "2022-07-20 00:27:00"],
        ],
    },
    # See https://www.ndss-symposium.org/wp-content/uploads/prism2026-12.pdf
    "CARBANAKV2_EDR": {
        "raw_dir": "",
        "database": "carbanakv2_edr",
        "database_all_file": "carbanakv2_edr",
        "num_node_types": 3,
        "num_edge_types": 33,
        "start_date": "2024-04-18",
        "end_date": "2024-05-13",
        "train_dates": [
            "2024-04-20",
            "2024-04-21",
            # Arbitrarly picked 2024-04-22 for the validation/threshold calibration
            "2024-04-23",
            "2024-04-24",
            "2024-04-25",
            # Arbitrarly picked 2024-04-26 for the validation/threshold calibration
            "2024-04-27",
            "2024-04-28",
            "2024-04-29",
        ],
        "val_dates": [
            "2024-04-22",
            "2024-04-26",
        ],
        "test_dates": [
            "2024-04-30",
            "2024-05-01",
            "2024-05-02",
            "2024-05-07",
            "2024-05-08",
            "2024-05-09",
            "2024-05-10",
        ],
        "unused_dates": [
            "2024-04-18",
            "2024-04-19",
            "2024-05-03",
            "2024-05-04",
            "2024-05-05",
            "2024-05-06",
            "2024-05-11",
            "2024-05-12",
            "2024-05-13",
        ],
        "ground_truth_relative_path": [
            "carbanakv2_edr/carbanakv2_edr.csv",
        ],
        "attack_to_time_window": [
            ["carbanakv2_edr/carbanakv2_edr.csv", "2024-04-30 17:30:00", "2024-05-10 20:30:00"]
        ],
    },
}

# Arguments

TASK_DEPENDENCIES = {
    "construction": [],
    "transformation": ["construction"],
    "featurization": ["transformation"],
    "feat_inference": ["featurization"],
    "batching": ["feat_inference"],
    "training": ["batching"],
    "evaluation": ["training"],
    "triage": ["evaluation"],
}


class AND(list):
    pass


class OR(list):
    pass


class Arg:
    def __init__(self, type, vals: list = None, desc: str = None):
        self.type = type
        self.vals = vals
        self.desc = desc


FEATURIZATIONS_CFG = {
    "word2vec": {
        "alpha": Arg(float),
        "window_size": Arg(int),
        "min_count": Arg(int),
        "use_skip_gram": Arg(bool),
        "num_workers": Arg(int),
        "epochs": Arg(int),
        "compute_loss": Arg(bool),
        "negative": Arg(int),
        "decline_rate": Arg(int),
    },
    "doc2vec": {
        "include_neighbors": Arg(bool),
        "epochs": Arg(int),
        "alpha": Arg(float),
    },
    "fasttext": {
        "min_count": Arg(int),
        "alpha": Arg(float),
        "window_size": Arg(int),
        "negative": Arg(int),
        "num_workers": Arg(int),
        "use_pretrained_fb_model": Arg(bool),
    },
    "alacarte": {
        "walk_length": Arg(int),
        "num_walks": Arg(int),
        "epochs": Arg(int),
        "context_window_size": Arg(int),
        "min_count": Arg(int),
        "use_skip_gram": Arg(bool),
        "num_workers": Arg(int),
        "compute_loss": Arg(bool),
        "add_paths": Arg(bool),
    },
    "temporal_rw": {
        "walk_length": Arg(int),
        "num_walks": Arg(int),
        "trw_workers": Arg(int),
        "time_weight": Arg(str),
        "half_life": Arg(int),
        "window_size": Arg(int),
        "min_count": Arg(int),
        "use_skip_gram": Arg(bool),
        "wv_workers": Arg(int),
        "epochs": Arg(int),
        "compute_loss": Arg(bool),
        "negative": Arg(int),
        "decline_rate": Arg(int),
    },
    "flash": {
        "min_count": Arg(int),
        "workers": Arg(int),
    },
    "hierarchical_hashing": {},
    "magic": {},
    "only_type": {},
    "only_ones": {},
}

ENCODERS_CFG = {
    "tgn": {
        "tgn_memory_dim": Arg(int),
        "tgn_time_dim": Arg(int),
        "use_node_feats_in_gnn": Arg(bool),
        "use_memory": Arg(bool),
        "use_time_order_encoding": Arg(bool),
        "project_src_dst": Arg(bool),
    },
    "graph_attention": {
        "activation": Arg(str),
        "num_heads": Arg(int),
        "concat": Arg(bool),
        "flow": Arg(str),
        "num_layers": Arg(int),
    },
    "sage": {
        "activation": Arg(str),
        "num_layers": Arg(int),
    },
    "gat": {
        "activation": Arg(str),
        "num_heads": Arg(int),
        "concat": Arg(bool),
        "flow": Arg(str),
        "num_layers": Arg(int),
    },
    "gin": {
        "activation": Arg(str),
        "num_layers": Arg(int),
    },
    "sum_aggregation": {},
    "rcaid_gat": {},
    "magic_gat": {
        "num_layers": Arg(int),
        "num_heads": Arg(int),
        "negative_slope": Arg(float),
        "alpha_l": Arg(float),
        "activation": Arg(str),
    },
    "glstm": {},
    "custom_mlp": {
        "architecture_str": Arg(str),
    },
    "none": {},
}

DECODERS_NODE_LEVEL = ["node_mlp", "none", "magic_gat", "nodlink"]
DECODERS_EDGE_LEVEL = ["edge_mlp"]
DECODERS_CFG = {
    "edge_mlp": {
        "architecture_str": Arg(
            str,
            desc="A string describing a simple neural network. Example: if the encoder's output has shape `node_out_dim=128` \
                                setting `architecture_str=linear(2) | relu | linear(0.5)` creates this MLP: nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 128), nn.Linear(128, y). \
                                Precisely, in linear(x), x is the multiplier of input neurons. The final layer `nn.Linear(128, y)` is added automatically such that `y` is the \
                                output size matching the downstream objective (e.g. edge type prediction involves predicting 10 edge types, so the output of the decoder should be 10).",
        ),
        "src_dst_projection_coef": Arg(
            int, desc="Multiplier of input neurons to project src and dst nodes."
        ),
    },
    "node_mlp": {
        "architecture_str": Arg(str),
    },
    "magic_gat": {
        "num_layers": Arg(int),
        "num_heads": Arg(int),
        "negative_slope": Arg(float),
        "alpha_l": Arg(float),
        "activation": Arg(str),
    },
    "nodlink": {},
    "inner_product": {},
    "none": {},
}

RECON_LOSSES = ["SCE", "MSE", "MSE_sum", "MAE", "none"]
PRED_LOSSES = ["cross_entropy", "BCE"]
OBJECTIVES_NODE_LEVEL = [
    "predict_node_type",
    "reconstruct_node_features",
    "reconstruct_node_embeddings",
    "reconstruct_masked_features",
]
OBJECTIVES_EDGE_LEVEL = [
    "predict_edge_type",
    "reconstruct_edge_embeddings",
    "predict_edge_contrastive",
]
OBJECTIVES = OBJECTIVES_NODE_LEVEL + OBJECTIVES_EDGE_LEVEL
OBJECTIVES_CFG = {
    # Prediction-based
    "predict_edge_type": {
        "decoder": Arg(
            str, vals=OR(list(DECODERS_CFG.keys())), desc="Decoder used before computing loss."
        ),
        **DECODERS_CFG,
        "balanced_loss": Arg(bool),
        "use_triplet_types": Arg(bool),
    },
    "predict_node_type": {
        "decoder": Arg(
            str, vals=OR(list(DECODERS_CFG.keys())), desc="Decoder used before computing loss."
        ),
        **DECODERS_CFG,
        "balanced_loss": Arg(bool),
    },
    "predict_masked_struct": {
        "loss": Arg(str, vals=OR(PRED_LOSSES)),
        "decoder": Arg(
            str, vals=OR(list(DECODERS_CFG.keys())), desc="Decoder used before computing loss."
        ),
        **DECODERS_CFG,
        "balanced_loss": Arg(bool),
    },
    "detect_edge_few_shot": {
        "decoder": Arg(
            str, vals=OR(list(DECODERS_CFG.keys())), desc="Decoder used before computing loss."
        ),
        **DECODERS_CFG,
    },
    "predict_edge_contrastive": {
        "decoder": Arg(
            str, vals=OR(list(DECODERS_CFG.keys())), desc="Decoder used before computing loss."
        ),
        **DECODERS_CFG,
        "inner_product": {
            "dropout": Arg(float),
        },
    },
    # Reconstruction-based
    "reconstruct_node_features": {
        "loss": Arg(str, vals=OR(RECON_LOSSES)),
        "decoder": Arg(
            str, vals=OR(list(DECODERS_CFG.keys())), desc="Decoder used before computing loss."
        ),
        **DECODERS_CFG,
    },
    "reconstruct_node_embeddings": {
        "loss": Arg(str, vals=OR(RECON_LOSSES)),
        "decoder": Arg(
            str, vals=OR(list(DECODERS_CFG.keys())), desc="Decoder used before computing loss."
        ),
        **DECODERS_CFG,
    },
    "reconstruct_edge_embeddings": {
        "loss": Arg(str, vals=OR(RECON_LOSSES)),
        "decoder": Arg(
            str, vals=OR(list(DECODERS_CFG.keys())), desc="Decoder used before computing loss."
        ),
        **DECODERS_CFG,
    },
    "reconstruct_masked_features": {
        "loss": Arg(str, vals=OR(RECON_LOSSES)),
        "mask_rate": Arg(float),
        "decoder": Arg(
            str, vals=OR(list(DECODERS_CFG.keys())), desc="Decoder used before computing loss."
        ),
        **DECODERS_CFG,
    },
}

SYNTHETIC_ATTACKS = {
    "synthetic_attack_naive": {
        "num_attacks": Arg(int),
        "num_malicious_process": Arg(int),
        "num_unauthorized_file_access": Arg(int),
        "process_selection_method": Arg(str),
    },
}

THRESHOLD_METHODS = ["max_val_loss", "mean_val_loss", "threatrace", "magic", "flash", "nodlink"]

# --- Tasks, subtasks, and argument configurations ---
TASK_ARGS = {
    "construction": {
        "used_method": Arg(
            str, vals=OR(["default", "magic"]), desc="The method to build time window graphs."
        ),
        "use_all_files": Arg(bool),
        "mimicry_edge_num": Arg(int),
        "time_window_size": Arg(
            float,
            desc="The size of each graph in minutes. The notation should always be float (e.g. 10.0). Supports sizes < 1.0.",
        ),
        "use_hashed_label": Arg(bool, desc="Whether to hash the textual features."),
        "fuse_edge": Arg(
            bool, desc="Whether to fuse duplicate sequential edges into a single edge."
        ),
        "node_label_features": {
            "subject": Arg(
                str,
                vals=AND(["type", "path", "cmd_line"]),
                desc="Which features use for process nodes. Features will be concatenated.",
            ),
            "file": Arg(
                str,
                vals=AND(["type", "path"]),
                desc="Which features use for file nodes. Features will be concatenated.",
            ),
            "netflow": Arg(
                str,
                vals=AND(["type", "remote_ip", "remote_port"]),
                desc="Which features use for netflow nodes. Features will be concatenated.",
            ),
        },
        "multi_dataset": Arg(
            str,
            vals=OR(list(DATASET_DEFAULT_CONFIG.keys()) + ["none"]),
            desc="A comma-separated list of datasets on which training is performed. Evaluation is done only the primary dataset run in CLI.",
        ),
    },
    "transformation": {
        "used_methods": Arg(
            str,
            vals=AND(
                ["undirected", "dag", "rcaid_pseudo_graph", "none"] + list(SYNTHETIC_ATTACKS.keys())
            ),
            desc="Applies transformations to graphs after their construction. Multiple transformations can be applied sequentially. Example: `used_methods=undirected,dag`",
        ),
        "rcaid_pseudo_graph": {
            "use_pruning": Arg(bool),
        },
        **SYNTHETIC_ATTACKS,
    },
    "featurization": {
        "emb_dim": Arg(
            int,
            desc="Size of the text embedding. Arg not used by some featurization methods that do not build embeddings.",
        ),
        "epochs": Arg(
            int, desc="Epochs to train the embedding method. Arg not used by some methods."
        ),
        "seed": Arg(int),
        "training_split": Arg(
            str,
            vals=OR(["train", "all"]),
            desc="The partition of data used to train the featurization method.",
        ),
        "multi_dataset_training": Arg(
            bool,
            desc="Whether the featurization method should be trained on all datasets in `multi_dataset`.",
        ),
        "used_method": Arg(
            str,
            vals=OR(list(FEATURIZATIONS_CFG.keys())),
            desc="Algorithms used to create node and edge features.",
        ),
        **FEATURIZATIONS_CFG,
    },
    "feat_inference": {
        "to_remove": Arg(bool),  # TODO: remove
    },
    "batching": {
        "save_on_disk": Arg(
            bool,
            desc="Whether to store the graphs on disk upon building the graphs. \
            Used to avoid re-computation of very complex batching operations that take time. Can take up to 300GB storage for CADETS_E5.",
        ),
        "node_features": Arg(
            str,
            vals=AND(["node_type", "node_emb", "only_ones", "edges_distribution"]),
            desc="Node features to use during GNN training. `node_type` is a one-hot encoded entity type vector, \
                                    `node_emb` refers to the embedding generated during the `featurization` task, `only_ones` is a vector of ones \
                                    with length `node_type`, `edges_distribution` counts emitted and received edges.",
        ),
        "edge_features": Arg(
            str,
            vals=AND(["edge_type", "edge_type_triplet", "msg", "time_encoding", "none"]),
            desc="Edge features to used during GNN training. `edge_type` refers to the system call type, `edge_type_triplet` \
                                considers a same edge type as a new type if source or destination node types are different, `msg` is the message vector \
                                used in the TGN, `time_encoding` encodes temporal order of events with their timestamps in the TGN, `none` uses no features.",
        ),
        "multi_dataset_training": Arg(
            bool, desc="Whether the GNN should be trained on all datasets in `multi_dataset`."
        ),
        "fix_buggy_graph_reindexer": Arg(
            bool,
            desc="A bug has been found in the first version of the framework, where reindexing graphs in shape (N, d) \
                                                slightly modify node features. Setting this to true fixes the bug.",
        ),
        "global_batching": {
            "used_method": Arg(
                str,
                vals=OR(["edges", "minutes", "unique_edge_types", "none"]),
                desc="Flattens the time window-based graphs into a single large \
                            temporal graph and recreate graphs based on the given method. `edges` creates contiguous graphs of size `global_batching_batch_size` edges, \
                            the same applies for `minutes`, `unique_edge_types` builds graphs where each pair of connected nodes share edges with distinct edge types, \
                            `none` uses the default time window-based batching defined in minutes with arg `time_window_size`.",
            ),
            "global_batching_batch_size": Arg(
                int,
                desc="Controls the value associated with `global_batching.used_method` (training+inference).",
            ),
            "global_batching_batch_size_inference": Arg(
                int,
                desc="Controls the value associated with `global_batching.used_method` (inference only).",
            ),
        },
        "intra_graph_batching": {
            "used_methods": Arg(
                str,
                vals=AND(["edges", "tgn_last_neighbor", "none"]),
                desc="Breaks each previously computed graph into even smaller graphs. \
                                `edges` creates contiguous graphs of size `intra_graph_batch_size` edges (if a graph has 2000 edges and `intra_graph_batch_size=1500` \
                                creates two graphs: one with 1500 edges, the other with 500 edges), `tgn_last_neighbor` computes for each graph its associated graph \
                                based on the TGN last neighbor loader, namely a new graph where each node is connected with its last `tgn_neighbor_size` incoming edges.\
                                `none` does not alter any graph.",
            ),
            "edges": {
                "intra_graph_batch_size": Arg(
                    int,
                    desc="Controls the value associated with `global_batching.used_method`.",
                ),
            },
            "tgn_last_neighbor": {
                "tgn_neighbor_size": Arg(
                    int, desc="Number of last neighbors to store for each node."
                ),
                "tgn_neighbor_n_hop": Arg(
                    int,
                    desc="If greater than one, will also gather the last neighbors of neighbors.",
                ),
                "fix_buggy_orthrus_TGN": Arg(
                    bool,
                    desc="A bug has been in the first version of the framework, where the features of last neighbors not appearing \
                                            in the input graph have zero node feature vectors. Setting this arg to true includes the features of all nodes in the TGN graph.",
                ),
                "fix_tgn_neighbor_loader": Arg(
                    bool,
                    desc="We found a minor bug in the original TGN code (https://github.com/pyg-team/pytorch_geometric/issues/10100). This \
                                                is an unofficial fix.",
                ),
                "directed": Arg(
                    bool,
                    desc="The original TGN's loader builds graphs in an undirected way. This makes the graphs purely directed.",
                ),
                "insert_neighbors_before": Arg(
                    bool,
                    desc="Whether to insert the edges of the current graph before loading last neighbors.",
                ),
            },
        },
        "inter_graph_batching": {
            "used_method": Arg(
                str,
                vals=OR(["graph_batching", "none"]),
                desc="Batches multiple graphs into a single large one for parallel training. \
                                Does not support TGN. `graph_batching` batches `inter_graph_batch_size` together, `none` doesn't batch graphs.",
            ),
            "inter_graph_batch_size": Arg(
                int,
                desc="Controls the value associated with `inter_graph_batching.used_method`.",
            ),
        },
    },
    "training": {
        "seed": Arg(int),
        "deterministic": Arg(
            bool, desc="Whether to force PyTorch to use deterministic algorithms."
        ),
        "num_epochs": Arg(int),
        "patience": Arg(int),
        "lr": Arg(float),
        "weight_decay": Arg(float),
        "node_hid_dim": Arg(int, desc="Number of neurons in the middle layers of the encoder."),
        "node_out_dim": Arg(int, desc="Number of neurons in the last layer of the encoder."),
        "grad_accumulation": Arg(int, desc="Number of epochs to gather gradients before backprop."),
        "inference_device": Arg(str, vals=OR(["cpu", "cuda"]), desc="Device used during testing."),
        "used_method": Arg(str, vals=OR(["default"]), desc="Which training pipeline use."),
        "save_final_model": Arg(bool, desc="Whether to checkpoint the final (best-epoch-restored) model to `training._trained_models_dir`. Only the final model is saved, not one per epoch."),
        "mia_dump": Arg(bool, desc="Whether to dump per-node inference losses for MIA."),
        "mia_rich_dump": Arg(bool, desc="Whether to also dump per-node encoder embeddings (and node types) next to the MIA loss CSVs, as .npz. Enables white-box representation-based attacks."),
        "mia_split_mode": Arg(str, vals=OR(["random", "calendar", "xrole"]), desc="Strategy for splitting training graphs: 'random' (probabilistic, mia_split_frac), 'calendar' (all in-training, non-members = val + test days), 'xrole' (all in-training, non-members = val days only; pair with a dataset whose val_dates are another role's training days, e.g. CADETS_E3_NOFRAC)."),
        "mia_split_frac": Arg(float, desc="Fraction of training graphs to use as in-training (for random mode)."),
        "mia_split_seed": Arg(int, desc="Random seed for deterministic graph splitting."),
        "encoder": {
            "dropout": Arg(float),
            "used_methods": Arg(
                str,
                vals=AND(list(ENCODERS_CFG.keys())),
                desc="First part of the neural network. Usually GNN encoders to capture complex patterns.",
            ),
            "x_is_tuple": Arg(
                bool, desc="Whether to consider nodes differently when being source or destination."
            ),
            **ENCODERS_CFG,
        },
        "decoder": {
            "used_methods": Arg(
                str,
                vals=AND(list(OBJECTIVES_CFG.keys())),
                desc="Second part of the neural network. Usually MLPs specific to the downstream task (e.g. reconstruction of prediction)",
            ),
            **OBJECTIVES_CFG,
            "use_few_shot": Arg(bool, desc="Old feature: need some work to update it."),
            "few_shot": {
                "include_attacks_in_ssl_training": Arg(bool),
                "freeze_encoder": Arg(bool),
                "num_epochs_few_shot": Arg(int),
                "patience_few_shot": Arg(int),
                "lr_few_shot": Arg(float),
                "weight_decay_few_shot": Arg(float),
                "decoder": {
                    "used_methods": Arg(str),
                    **OBJECTIVES_CFG,
                },
            },
        },
    },
    "evaluation": {
        "viz_malicious_nodes": Arg(
            bool,
            desc="Whether to generate images of malicious nodes' neighborhoods (not stable).",
        ),
        "ground_truth_version": Arg(str, vals=OR(["orthrus", "reapr"])),
        "best_model_selection": Arg(
            str,
            vals=OR(["best_adp", "best_discrimination"]),
            desc="Strategy to select the best model across epochs. `best_adp` selects the best model based on the highest ADP score, `best_discrimination` \
                                    selects the model that does the best separation between top-score TPs and top-score FPs.",
        ),
        "used_method": Arg(str),
        "node_evaluation": {
            "threshold_method": Arg(
                str,
                vals=OR(THRESHOLD_METHODS),
                desc="Method to calculate the threshold value used to detect anomalies.",
            ),
            "use_dst_node_loss": Arg(
                bool,
                desc="Whether to consider the loss of destination nodes when computing the node-level scores (maximum loss of a node).",
            ),
            "use_kmeans": Arg(
                bool, desc="Whether to cluster nodes after thresholding as done in Orthrus"
            ),
            "kmeans_top_K": Arg(int, desc="Number of top-score nodes selected before clustering."),
        },
        "tw_evaluation": {
            "threshold_method": Arg(
                str,
                vals=OR(THRESHOLD_METHODS),
                desc="Time-window detection. The code is broken and needs work to be updated.",
            ),
        },
        "node_tw_evaluation": {
            "threshold_method": Arg(
                str,
                vals=OR(THRESHOLD_METHODS),
                desc="Node-level detection where a same node in multiple time windows is \
                    considered as multiple unique nodes. More realistic evaluation for near real-time detection. The code is broken and needs work to be updated.",
            ),
            "use_dst_node_loss": Arg(bool),
            "use_kmeans": Arg(bool),
            "kmeans_top_K": Arg(int),
        },
        "queue_evaluation": {
            "used_method": Arg(
                str,
                vals=OR(["kairos_idf_queue", "provnet_lof_queue"]),
                desc="Queue-level detection as in Kairos. The code is broken and needs work to be updated.",
            ),
            "queue_threshold": Arg(int),
            "kairos_idf_queue": {
                "include_test_set_in_IDF": Arg(bool),
            },
            "provnet_lof_queue": {
                "queue_arg": Arg(str),
            },
        },
        "edge_evaluation": {
            "malicious_edge_selection": Arg(
                str,
                vals=OR(["src_node", "dst_node", "both_nodes"]),
                desc="The ground truth only contains node-level labels. \
                This arg controls the strategy to label edges. `src_nodes` and `dst_nodes` consider an edge as malicious if only its source or only its destination \
                node is malicious. `both` labels an edge as malicious if both end nodes are malicious.",
            ),
            "threshold_method": Arg(str, vals=OR(THRESHOLD_METHODS)),
        },
    },
    "triage": {
        "used_method": Arg(
            str,
            vals=OR(["depimpact"]),
            desc="Post-processing step to reconstruct attack paths or reduce false positives. `depimpact` is used in Orthrus.",
        ),
        "depimpact": {
            "used_method": Arg(
                str, vals=OR(["component", "shortest_path", "1-hop", "2-hop", "3-hop"])
            ),
            "score_method": Arg(str, vals=OR(["degree", "recon_loss", "degree_recon"])),
            "workers": Arg(int),
            "visualize": Arg(bool),
        },
    },
    "postprocessing": {},
}

EXPERIMENTS_CONFIG = {
    "training_loop": {
        "run_evaluation": Arg(
            str, vals=OR(["each_epoch", "best_epoch"])
        ),  # (when to run inference on test set)
    },
    "experiment": {
        "used_method": Arg(str, vals=OR(["uncertainty", "none"])),
        "uncertainty": {
            "hyperparameter": {
                "hyperparameters": Arg(str, vals=AND(["lr, num_epochs, text_h_dim, gnn_h_dim"])),
                "iterations": Arg(int),
                "delta": Arg(float),
            },
            "mc_dropout": {
                "iterations": Arg(int),
                "dropout": Arg(float),
            },
            "deep_ensemble": {
                "method": Arg(str),
                "iterations": Arg(int),
                "restart_from": Arg(str),
            },
            "bagged_ensemble": {
                "iterations": Arg(int),
                "min_num_days": Arg(int),
            },
        },
    },
}
UNCERTAINTY_EXP_YML_FOLDER = "experiments/uncertainty/"
