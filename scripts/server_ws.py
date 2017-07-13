import tornado.web
import tornado.websocket
import tornado.httpserver
import tornado.ioloop
import argparse
import urllib
import logging
import logging.handlers
import sys
import datetime
import dataset_walker as ds
import json,re


info_for_tasks = {}  # Variable to save information about the turns in the testing file

# Configuration for the logger
MAXBYTESLOG = 10048576  # Maximum size for the general log file before doing the backup
BACKUPCOUNT = 1000  # Maximum number of backups for logs before starting the shift
logger = logging.getLogger('general_logger')  # The general logger
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

# To print the logging information on the stdout
ch = logging.StreamHandler(sys.stdout)  # To display the information also through the stdout
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
logger.addHandler(ch)

# To save the logging information on a file
genlog_fh = logging.handlers.RotatingFileHandler('logging.log', mode='a', maxBytes=MAXBYTESLOG, backupCount=BACKUPCOUNT, encoding="latin-1")  # save up to 1 GB before rotation
genlog_fh.setLevel(logging.DEBUG)
genlog_fh.setFormatter(formatter)
logger.addHandler(genlog_fh)


class WebSocketHandler(tornado.websocket.WebSocketHandler):
    def open(self):
        logger.info('There system is now connected')
        pass

    # The function to process all the incoming messages
    def on_message(self, message):
        inputMsg = urllib.unquote(message.encode('ascii')).decode('utf-8')
        objInput = json.loads(inputMsg)
        logger.info('Received ' + inputMsg)

        # Prepare an answer for the given input. In this demo we send the correct answer since we are using the same dstc4_dev file
        # In the real setup the system must point to the dstc4_test_team file
        outputMsg = {}
        outputMsg['dataset'] = dataset
        outputMsg['session_id'] = objInput['session_id']
        outputMsg['utter_index'] = objInput['utter_index']
        outputMsg['role_type'] = objInput['role_type']  # The user in the current turn
        outputMsg['task_type'] = objInput['task_type']
        if objInput['task_type'] == 'SLU':
            # INPUT: The user's utterance,
            # OUTPUT: The current user's slots and speech acts

            t1 = datetime.datetime.now()
            input_utt = objInput['transcript']  # This is the only information needed in this task

            # *******************  TODO: START MODIFYING FROM HERE ****************************
            if objInput['role_task'] != objInput['role_type']:  # The role task is which user we really want to evaluate
                outputMsg['speech_act'] = []  # In this case we do not do anything, but a more advanced tracker can use this information too
                outputMsg['semantic_tagged'] = ''
            else:
                outputMsg['speech_act'] = info_for_tasks['sessions'][objInput['session_id']]['utterances'][objInput['utter_index']]['speech_act']
                outputMsg['semantic_tagged'] = info_for_tasks['sessions'][objInput['session_id']]['utterances'][objInput['utter_index']]['semantic_tagged']

            # *******************  TODO: END MODIFICATIONS HERE ****************************
            t2 = datetime.datetime.now()
            outputMsg['wall_time'] = str(t2-t1)
            logger.info("Execution time: %s" % (t2-t1))

        elif objInput['task_type'] == 'SAP':
            # INPUT: The user's utterance + speech acts and semantic tags for the current user + semantic tags for the next user
            # OUTPUT: The next user's speech acts

            t1 = datetime.datetime.now()
            input_utt = objInput['transcript']
            input_speech_acts = objInput['speech_act']
            input_semantic_current = objInput['current_semantic_tagged']
            input_semantic_next = objInput['next_semantic_tagged']

            # Now prepare the answer with the utterance for the following user
            # *******************  TODO: START MODIFYING FROM HERE ****************************

            outputMsg['speech_act'] = info_for_tasks['sessions'][objInput['session_id']]['utterances'][objInput['utter_index']+1]['speech_act']

            # *******************  TODO: END MODIFICATIONS HERE ****************************

            t2 = datetime.datetime.now()
            outputMsg['wall_time'] = str(t2-t1)
            logger.info("Execution time: %s" % (t2-t1))

        elif objInput['task_type'] == 'SLG':
            # INPUT: Speech acts and semantic tags for the current user
            # OUTPUT: The user's utterance

            t1 = datetime.datetime.now()
            input_speech_acts = objInput['speech_act']
            input_semantic_current = objInput['current_semantic_tagged']

            # *******************  TODO: START MODIFYING FROM HERE ****************************

            # Now prepare the answer with the utterance for the following user
            outputMsg['transcript'] = info_for_tasks['sessions'][objInput['session_id']]['utterances'][objInput['utter_index']]['transcript']

            # *******************  TODO: END MODIFICATIONS HERE ****************************

            t2 = datetime.datetime.now()
            outputMsg['wall_time'] = str(t2-t1)
            logger.info("Execution time: %s" % (t2-t1))

        elif objInput['task_type'] == 'EES':
            # INPUT: The current user's utterance
            # OUTPUT: The next user's utterance

            t1 = datetime.datetime.now()
            input_utterance = objInput['transcript']

            # *******************  TODO: START MODIFYING FROM HERE ****************************

            # Now prepare the answer with the utterance for the following user
            outputMsg['transcript'] = info_for_tasks['sessions'][objInput['session_id']]['utterances'][objInput['utter_index']+1]['transcript']

            # *******************  TODO: END MODIFICATIONS HERE ****************************

            t2 = datetime.datetime.now()
            outputMsg['wall_time'] = str(t2-t1)
            logger.info("Execution time: %s" % (t2-t1))

        # Send back the answer to the client
        out_message = json.dumps(outputMsg, sort_keys=True, ensure_ascii=True, indent=3)
        logger.info(u'Your message is ' + out_message)
        self.write_message(urllib.unquote(out_message.encode('ascii')).decode('utf-8'))
  
    def on_close(self):
        logger.info('The connection was closed')
        pass
  
class Application(tornado.web.Application):
    def __init__(self):
        handlers = [
            (r'/DSTC4', WebSocketHandler)  # The exposed URL for communicating with the client
        ]
        tornado.web.Application.__init__(self, handlers)



# ***************  THE FOLLOWING FUNCTIONS ARE NOT NEEDED DURING THE TEST EVALUATION SINCE THERE IS NOT A GENERATED FILE
# ***************  CONTAINING THE TURNS IN JSON FORMAT. INSTEAD, EACH TIME THE PROGRAM MUST PROCESS THE INPUT
# ***************  FROM THE CLIENT AND SENDS BACK THE RESULT


# Function to read the test file generated by the team into memory. Turns from the same user are concatenated together
def fntLoadInfoSubmissionDemo():
    global info_for_tasks
    # Save first the information, then later we start calling the team
    info_for_tasks = fntInitStruct(dataset)
    sessions = ds.dataset_walker(dataset, dataroot=dataroot, labels=True)
    print('Collecting information for all pilot tasks')
    for session in sessions:
        session_id = session.log['session_id']
        info_for_tasks = fntInitSession(info_for_tasks, session_id)

        utter_span = []  # Saves the indexes of different concatenated turns
        concatenated_info = {'transcripts':[], 'speech_acts':[], 'slots': [], 'uniq_sa': {}}  # Saves the temporal information along several continuous turns for a given user
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

# Function to concatenate information from different turns for the same user
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

# Function to initialize the structure containing all the information for sessions and turns and users
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

# Function to add a new utterance
def fntAddUtterance(dictionary, session_id, utter_index, utter_span, role_type, speech_acts, semantic_tagged, transcript):
    dictionary['sessions'][session_id]['utterances'][utter_index] = {
            'utter_index': utter_index,
            'utter_span': utter_span,
            'role_type': role_type,
            'speech_act': speech_acts,
            'semantic_tagged': semantic_tagged,
            'transcript': transcript
    }
    return dictionary


# **********************************************************************************************************************

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-port', '--port', help='Port for incoming connections', default=8080)
    parser.add_argument('--dataset', dest='dataset', action='store', metavar='DATASET', required=True, help='The dataset to analyze')
    parser.add_argument('--dataroot',dest='dataroot',action='store', metavar='PATH', required=True, help='Will look for corpus in <destroot>/<dataset>/...')
    args = parser.parse_args()
    dataset = args.dataset
    dataroot = args.dataroot
    fntLoadInfoSubmissionDemo()  # NO NEEDED IN THE REAL TEST EVALUATION

    print('Listening to incomming data')
    ws_app = Application()
    server = tornado.httpserver.HTTPServer(ws_app)
    server.listen(args.port)
    tornado.ioloop.IOLoop.instance().start()