import dataset_walker as ds
from semantic_tag_parser import SemanticTagParser
import websocket, argparse, urllib, codecs, os, sys, re, json
from HTMLParser import HTMLParseError
from calc_amfm_bleu import calcScoresBleuAMFM
import logging
import logging.handlers

info_teams = {
    'team1': {
        'url': 'ws://127.0.0.1:8080/DSTC4',  # Address for each system to evaluate
        'tasks': ['SLU', 'SAP', 'SLG', 'EES'],  # Different pilot tasks that these team are evaluating
        'roles': ['GUIDE', 'TOURIST'],  # Different roles that these team are evaluating
    },
}

root_dir = './'
dir_output = root_dir + 'output_systems/'  # Directory to save the CSV and log files for each team
MAXBYTESLOG = 10048576  # Maximum size for the general log file before doing the backup
BACKUPCOUNT = 1000  # Maximum number of backups for logs before starting the shift
MAX_TIMEOUT = 1 * 60  # Maximum time to wait to establish the connection and send requests to the server, default: 1 min

# Configuration of logging systems
logger = logging.getLogger('general_logger')  # The general logger
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
ch = logging.StreamHandler(sys.stdout)  # To display the information also through the stdout
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
logger.addHandler(ch)

def main(argv):
    install_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    utils_dirname = os.path.join(install_path,'lib')

    sys.path.append(utils_dirname)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', dest='dataroot', help='The directory where to find the data [default: data]', default='data')
    parser.add_argument('--dataset', dest='dataset', action='store', metavar='DATASET', required=True, help='The dataset to analyze')
    args = parser.parse_args()

    if not os.path.exists(dir_output):
        print("...creating " + dir_output)
        os.makedirs(dir_output)

    dataset = args.dataset
    # Save first the information, then later we start calling the team
    info_for_tasks = fntInitStruct(dataset)
    sessions = ds.dataset_walker(dataset, dataroot=args.dataroot, labels=True)
    print('Collecting information for all pilot tasks')
    for session in sessions:
        session_id = session.log['session_id']
        info_for_tasks = fntInitSession(info_for_tasks, session_id)

        utter_span = []  # Saves the utterance indexes for different concatenated turns
        # Saves the temporal information along several continuous turns for a given user
        concatenated_info = {'transcripts':[], 'speech_acts':[], 'slots': [], 'uniq_sa': {}}
        utter_index = 0
        # now iterate through turns
        current_role = session.log['utterances'][0]['speaker'].upper()
        for id, (log_utter, label_utter) in enumerate(session):
            # We concatenate the utterance + semantic slots + speech_acts
            if log_utter['speaker'].upper() == current_role:
                utter_span.append(str(log_utter['utter_index']))
                concatenated_info = fntConcatInfo(concatenated_info, log_utter['transcript'], ' '.join(label_utter['semantic_tagged']), label_utter['speech_act'])
            else:  # Change the user
                info_for_tasks = fntAddUtterance(info_for_tasks, session_id, utter_index, '_'.join(utter_span),
                                                      current_role, concatenated_info['speech_acts'],
                                                      ' '.join(concatenated_info['slots']),
                                                      ' '.join(concatenated_info['transcripts']))
                concatenated_info = {'transcripts':[], 'speech_acts':[], 'slots': [], 'uniq_sa': {}}
                # Restart the process for the next user
                utter_index += 1
                current_role = log_utter['speaker'].upper()
                utter_span = []
                utter_span.append(str(log_utter['utter_index']))
                concatenated_info = fntConcatInfo(concatenated_info, log_utter['transcript'], ' '.join(label_utter['semantic_tagged']), label_utter['speech_act'])
        info_for_tasks = fntAddUtterance(info_for_tasks, session_id, utter_index, '_'.join(utter_span),
                                                      current_role, concatenated_info['speech_acts'],
                                                      ' '.join(concatenated_info['slots']),
                                                      ' '.join(concatenated_info['transcripts']))

    # Now we start the process of asking information for each team
    for team in info_teams:
        # Configuration of logger for each team
        genlog_fh = logging.handlers.RotatingFileHandler(dir_output + '/' + team + '.log', mode='a', maxBytes=MAXBYTESLOG, backupCount=BACKUPCOUNT, encoding="latin-1")  # save up to 1 GB before rotation
        genlog_fh.setLevel(logging.DEBUG)
        genlog_fh.setFormatter(formatter)
        logger.addHandler(genlog_fh)
        logger.info('Processing team ' + team)
        url = info_teams[team]['url']
        logger.info('Connecting to ' + url + ' for ' + team)
        # websocket.enableTrace(True)  # To check the content of each data send to the server
        ws = websocket.create_connection(url)
        ws.settimeout(MAX_TIMEOUT)
        stats = {}
        for pilottask in info_teams[team]['tasks']:
            logger.info('Doing task: ' + pilottask)
            for roletype in info_teams[team]['roles']:
                if pilottask == 'SLU':
                    stats['semantic_tagged'] = {}
                    stats['semantic_tagged']['detection'] = Stat_Precision_Recall()
                    stats['semantic_tagged']['class'] = Stat_Precision_Recall()
                    stats['semantic_tagged']['all'] = Stat_Precision_Recall()

                if pilottask == 'SLU' or pilottask == 'SAP':
                    stats['speech_act'] = {}
                    stats['speech_act']['act'] = Stat_Precision_Recall()
                    stats['speech_act']['all'] = Stat_Precision_Recall()

                if pilottask == 'SLG' or pilottask == 'EES':
                    stats['utt_transcriptions'] = {}
                    stats['utt_transcriptions']['all'] = Stat_BLEU_AM_FM()

                logger.info('Doing role: ' + roletype)

                for session_id in info_for_tasks['sessions']:
                    logger.info('Processing session: ' + str(session_id))
                    for n_utt in sorted(info_for_tasks['sessions'][session_id]['utterances']):
                        utterance = info_for_tasks['sessions'][session_id]['utterances'][n_utt]
                        logger.info('utterance: ' + str(n_utt))
                        if pilottask == 'SLU':
                            # INPUT: The user's utterance,
                            # OUTPUT: The current user's slots and speech acts
                            if utterance['role_type'] == roletype:
                                ref_sa_list = utterance['speech_act']
                                ref_tagged = utterance['semantic_tagged']
                            else:
                                ref_sa_list = []
                                ref_tagged = ''

                            jsonMsg = fntCreateJSONMessage(info_for_tasks['dataset'], session_id, n_utt, roletype, utterance['role_type'], pilottask, utterance['transcript'], None, None, None)
                            pred_sa_list, pred_tagged = fntSendMessage(ws, pilottask, jsonMsg)
                            eval_acts(ref_sa_list, pred_sa_list, stats['speech_act'])
                            eval_semantics(ref_tagged, pred_tagged, stats['semantic_tagged'])

                        elif pilottask == 'SAP':
                            # Here we need to concatenate the turns for a given role user + the slots for the following user
                            # INPUT: The user's utterance + speech acts and semantic tags for the current user + semantic tags for the next user
                            # OUTPUT: The next user's speech acts
                            if utterance['role_type'] == roletype:
                                if n_utt + 1 in info_for_tasks['sessions'][session_id]['utterances']:  # Check there is a next turn
                                    ref_sa_list = info_for_tasks['sessions'][session_id]['utterances'][n_utt + 1]['speech_act']
                                    jsonMsg = fntCreateJSONMessage(info_for_tasks['dataset'], session_id, n_utt, roletype, utterance['role_type'], pilottask, utterance['transcript'], utterance['speech_act'], utterance['semantic_tagged'], info_for_tasks['sessions'][session_id]['utterances'][n_utt + 1]['semantic_tagged'])
                                    pred_sa_list = fntSendMessage(ws, pilottask, jsonMsg)
                                    eval_acts(ref_sa_list, pred_sa_list, stats['speech_act'])

                        elif pilottask == 'SLG':
                            # Here we need to concatenate the turns for a given role user + the slots for the following user
                            # INPUT: Speech acts and semantic tags for the current user
                            # OUTPUT: The user's utterance
                            if utterance['role_type'] == roletype:
                                ref = info_for_tasks['sessions'][session_id]['utterances'][n_utt]['transcript']
                                jsonMsg = fntCreateJSONMessage(info_for_tasks['dataset'], session_id, n_utt, roletype, utterance['role_type'], pilottask, None, utterance['speech_act'], utterance['semantic_tagged'], None)
                                pred = fntSendMessage(ws, pilottask, jsonMsg)
                                eval_utt(ref, pred, stats['utt_transcriptions'])

                        elif pilottask == 'EES':
                            # Here we need to concatenate the turns for a given role user + the slots for the following user
                            # INPUT: The current user's utterance
                            # OUTPUT: The next user's utterance
                            if utterance['role_type'] == roletype:
                                if n_utt + 1 in info_for_tasks['sessions'][session_id]['utterances']:  # Check there is a next turn
                                    ref = info_for_tasks['sessions'][session_id]['utterances'][n_utt+1]['transcript']
                                    jsonMsg = fntCreateJSONMessage(info_for_tasks['dataset'], session_id, n_utt, roletype, utterance['role_type'], pilottask, utterance['transcript'], None, None, None)
                                    pred = fntSendMessage(ws, pilottask, jsonMsg)
                                    eval_utt(ref, pred, stats['utt_transcriptions'])

                # Save the final results in a CSV file
                with codecs.open(dir_output + '/' + team + '_' + pilottask + '_' + roletype + '.csv', 'w', 'utf-8') as f:
                    f.write("task, subtask, schedule, stat, N, result\n")
                    for subtask in stats:
                        for schedule in stats[subtask]:
                            for measure, N, result in stats[subtask][schedule].results():
                                f.write("%s, %s, %s, %s, %i, %s\n" %(pilottask, subtask, schedule, measure, N, result))

        logger.info('Closing the connection with team ' + team)
        logger.removeHandler(genlog_fh)
        ws.close()

# function to concatenate continuous turns by a same user
def fntConcatInfo(dictionary, transcript, sent_slots, speech_act):
    dictionary['transcripts'].append(transcript)
    if sent_slots is not None:
        slots = re.findall(r'(<.+?>)(.+?)(</.+?>)', sent_slots)
        if len(slots) > 0:
            for slot in slots:
                dictionary['slots'].append(' '.join(slot))

    if speech_act is not None:
        for sp in speech_act:
            dmp = json.dumps(sp)
            if dmp not in dictionary['uniq_sa']:
                dictionary['uniq_sa'] = dmp
                dictionary['speech_acts'].append(sp)

    return dictionary

# Function to initialize the structure containing all turns and sessions
def fntInitStruct(dataset):
    return {'dataset': dataset}


# Function to initialize a session
def fntInitSession(dictionary, session_id):
    if 'sessions' not in dictionary:
        dictionary['sessions'] = {}

    if session_id not in dictionary['sessions']:
        dictionary['sessions'][session_id] = {}
        dictionary['sessions'][session_id]['utterances'] = {}

    return dictionary


# Function to add a new utterance in a given session
def fntAddUtterance(dictionary, session_id, utter_index, utter_span, role_type, speech_acts, semantic_tagged, transcript):
    dictionary['sessions'][session_id]['utterances'][utter_index] = {
            'utter_span': utter_span,
            'role_type': role_type,
            'speech_act': speech_acts,
            'semantic_tagged': semantic_tagged,
            'transcript': transcript
    }
    return dictionary


# Function to create the JSON message to send to the server
def fntCreateJSONMessage(dataset, session_id, utter_index, role_task, role_type, task_type, utterance, speech_act, current_semantic_tagged, next_semantic_tagged):
    jsonObj = {}
    jsonObj['dataset'] = dataset
    jsonObj['session_id'] = session_id
    jsonObj['utter_index'] = utter_index
    jsonObj['role_task'] = role_task
    jsonObj['role_type'] = role_type
    jsonObj['task_type'] = task_type
    jsonObj['transcript'] = utterance
    jsonObj['speech_act'] = speech_act
    jsonObj['current_semantic_tagged'] = current_semantic_tagged
    jsonObj['next_semantic_tagged'] = next_semantic_tagged
    return json.dumps(jsonObj, sort_keys=True, ensure_ascii=True)


# Function to send the message to a connected server using websockets
def fntSendMessage(ws, pilottask, json_msg):
    logger.info("Send {}".format(json_msg))
    ws.send(urllib.quote(json_msg.encode('utf-8')))
    result_msg = ws.recv()
    logger.info("Received {}".format(result_msg))
    result = json.loads(result_msg)

    # Return only what is relevant for the given task
    if pilottask == 'SLU':
        return result['speech_act'], result['semantic_tagged']
    elif pilottask == 'SAP':
        return result['speech_act']
    elif pilottask == 'SLG':
        return result['transcript']
    elif pilottask == 'EES':
        return result['transcript']

# Functions to evaluate the sentences one by one
def eval_utt(ref, pred, stat_text):
    stat_text['all'].add(ref, pred)

def eval_acts(ref_act_objs, pred_act_objs, stat_acts):
    ref_act_tag_list = []
    ref_act_attr_list = []
    for act_obj in ref_act_objs:
        act_tag = act_obj['act']
        ref_act_tag_list.append(act_tag)
        for attr in act_obj['attributes']:
            ref_act_attr_list.append((act_tag, attr))

    ref_act_tag_list = sorted(set(ref_act_tag_list))
    ref_act_attr_list = sorted(set(ref_act_attr_list))

    pred_act_tag_list = []
    pred_act_attr_list = []
    for act_obj in pred_act_objs:
        act_tag = act_obj['act']
        pred_act_tag_list.append(act_tag)
        for attr in act_obj['attributes']:
            pred_act_attr_list.append((act_tag, attr))

    pred_act_tag_list = sorted(set(pred_act_tag_list))
    pred_act_attr_list = sorted(set(pred_act_attr_list))

    if 'act' in stat_acts:
        stat_acts['act'].add(pred_act_tag_list, ref_act_tag_list, list_mode=True)
    if 'all' in stat_acts:
        stat_acts['all'].add(pred_act_attr_list, ref_act_attr_list, list_mode=True)


def eval_semantics(ref_tagged, pred_tagged, stat_semantics):
    parser_ref = SemanticTagParser()
    parser_pred = SemanticTagParser()
    try:
        parser_ref.feed(ref_tagged)
        ref_chr_seq = parser_ref.get_chr_seq()
        ref_space_seq = parser_ref.get_chr_space_seq()

        parser_pred.feed(pred_tagged)
        pred_chr_seq = parser_pred.get_chr_seq()
        pred_space_seq = parser_pred.get_chr_space_seq()

        if ref_chr_seq != pred_chr_seq:
            raise

        merged_space_seq = [x or y for x,y in zip(ref_space_seq, pred_space_seq)]

        parser_ref.tokenize(merged_space_seq)
        parser_pred.tokenize(merged_space_seq)

        ref_word_tag_seq = parser_ref.get_word_tag_seq()
        pred_word_tag_seq = parser_pred.get_word_tag_seq()

        for (ref_bio, ref_tag, ref_attrs), (pred_bio, pred_tag, pred_attrs) in zip(ref_word_tag_seq, pred_word_tag_seq):
            pred_obj = None
            ref_obj = None

            if pred_bio is not None:
                pred_obj = {'bio': pred_bio}
            if ref_bio is not None:
                ref_obj = {'bio': ref_bio}

            if 'detection' in stat_semantics:
                stat_semantics['detection'].add(pred_obj, ref_obj)

            if pred_obj is not None and pred_tag is not None:
                pred_obj['tag'] = pred_tag
            if ref_obj is not None and ref_tag is not None:
                ref_obj['tag'] = ref_tag

            if 'class' in stat_semantics:
                stat_semantics['class'].add(pred_obj, ref_obj)

            if pred_obj is not None and pred_attrs is not None:
                for (s,v) in pred_attrs:
                    if v != 'NONE':
                        pred_obj[s] = v

            if ref_obj is not None and ref_attrs is not None:
                for (s,v) in ref_attrs:
                    if v != 'NONE':
                        ref_obj[s] = v

            if 'all' in stat_semantics:
                stat_semantics['all'].add(pred_obj, ref_obj)

        parser_ref.close()
        parser_pred.close()
    except HTMLParseError, err:
        print "HTMLParseError: %s" % err

# Class to keep the statistics for the evaluation
class Stat(object):
    def __init__(self,):
        pass

    def add(self, pred, ref):
        pass

    def results(self,):
        return []


class Stat_Precision_Recall(Stat):
    def __init__(self,):
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0

    def add(self, pred, ref, list_mode = False):
        if list_mode:
            for pred_obj in pred:
                if pred_obj in ref:
                    self.tp += 1
                else:
                    self.fp += 1
            for ref_obj in ref:
                if ref_obj not in pred:
                    self.fn += 1
        else:
            if pred is not None:
                self.tp += int(pred == ref)
                self.fp += int(pred != ref)
            if ref is not None:
                self.fn += int(pred != ref)

    def results(self,):
        precision = None
        recall = None
        fscore = None

        if (self.tp+self.fp) > 0.0:
            precision = self.tp/(self.tp+self.fp)
        if (self.tp+self.fn) > 0.0:
            recall = self.tp/(self.tp+self.fn)
        if precision is not None and recall is not None and (precision+recall) > 0.0:
            fscore = 2*precision*recall/(precision+recall)

        return [("precision", self.tp+self.fp, precision),("recall", self.tp+self.fn, recall),("f1", self.tp+self.fp+self.fn, fscore)]

# Class to evaluate the results in terms of BLEU and AM-FM scores
class Stat_BLEU_AM_FM(Stat):
    def __init__(self,):
        self.bleu = 0.0
        self.am_fm = 0.0
        self.alpha = 0.6
        self.num_sent = 0
        self.cs = calcScoresBleuAMFM()

    def add(self, pred, ref):
        self.num_sent += 1
        ref, pred = self.cs.doProcessFromStrings(ref, pred, self.num_sent)
        b = self.cs.calculateBLEUMetric(ref, pred)[0][-1]
        self.bleu += b
        am = self.cs.calculateAMMetric(ref, pred)
        fm = self.cs.calculateFMMetric(ref, pred)
        self.am_fm += (self.alpha)*am + (1.0 - self.alpha)*fm
        logger.info('num:%d ref: %s | pred: %s | bleu: %f | am: %f | fm: %f' %(self.num_sent, ref, pred, b, am, fm))

    def results(self,):
        logger.info("am_fm_avg", self.num_sent, self.am_fm[0]/self.num_sent)
        logger.info("bleu_avg", self.num_sent, self.bleu/self.num_sent)
        return [("am_fm_avg", self.num_sent, self.am_fm[0]/self.num_sent), ("bleu_avg", self.num_sent, self.bleu/self.num_sent)]


if __name__ == '__main__':
    main(sys.argv)