import argparse, sys, os, json, types, ontology_reader
from semantic_tag_parser import SemanticTagParser
from HTMLParser import HTMLParseError
from calc_amfm_bleu import calcScoresBleuAMFM

def main(argv):
	install_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
	utils_dirname = os.path.join(install_path,'lib')

	sys.path.append(utils_dirname)
	from dataset_walker import dataset_walker

	parser = argparse.ArgumentParser(description='Evaluate output from an SLU system.')
	parser.add_argument('--dataset', dest='dataset', action='store', metavar='DATASET', required=True, help='The dataset to analyze')
	parser.add_argument('--dataroot',dest='dataroot',action='store', metavar='PATH', required=True, help='Will look for corpus in <destroot>/<dataset>/...')
	parser.add_argument('--pilotfile',dest='pilotfile',action='store',metavar='JSON_FILE',required=True, help='File containing JSON output')
	parser.add_argument('--ontology',dest='ontology',action='store',metavar='JSON_FILE',required=True, help='JSON Ontology file')
	parser.add_argument('--pilottask',dest='pilottask',action='store',choices=['SLU', 'SAP', 'SLG', 'EES'],required=True, help='Target task')
	parser.add_argument('--roletype',dest='roletype',action='store',choices=['GUIDE', 'TOURIST'],required=True, help='Target role')
	parser.add_argument('--scorefile',dest='scorefile',action='store',metavar='JSON_FILE',required=True,help='File to write with CSV scoring data')	

	args = parser.parse_args()

	sessions = dataset_walker(args.dataset,dataroot=args.dataroot,labels=True)

	system_output = json.load(open(args.pilotfile))

	tagsets = ontology_reader.OntologyReader(args.ontology).get_tagsets()

	stats = {}
	if args.pilottask == 'SLU':
		stats['semantic_tagged'] = {}
		stats['semantic_tagged']['detection'] = Stat_Precision_Recall()
		stats['semantic_tagged']['class'] = Stat_Precision_Recall()
		stats['semantic_tagged']['all'] = Stat_Precision_Recall()

	if args.pilottask == 'SLU' or args.pilottask == 'SAP':
		stats['speech_act'] = {}
		stats['speech_act']['act'] = Stat_Precision_Recall()
		stats['speech_act']['all'] = Stat_Precision_Recall()

	if args.pilottask == 'SLG' or args.pilottask == 'EES':
		stats['utt_transcriptions'] = {}
		stats['utt_transcriptions']['all'] = Stat_BLEU_AM_FM()

	for session, track_session in zip(sessions, system_output["sessions"]):
		session_id = session.log['session_id']

		log_utter_list = []
		label_utter_list = []

		for log_utter, label_utter in session:
			if (args.roletype == 'GUIDE' and log_utter['speaker'] == 'Guide') or (args.roletype == 'TOURIST' and log_utter['speaker'] == 'Tourist'):
				log_utter_list.append(log_utter)
				label_utter_list.append(label_utter)

		# now iterate through turns
		for log_utter, label_utter, track_utter in zip(log_utter_list, label_utter_list, track_session["utterances"]):
			for subtask in stats:
				if subtask == 'speech_act':
					ref_sa_list = label_utter['speech_act']
					pred_sa_list = track_utter['speech_act']
					eval_acts(ref_sa_list, pred_sa_list, stats[subtask])
				elif subtask == 'semantic_tagged':
					ref_tagged = ' '.join(label_utter['semantic_tagged'])
					pred_tagged = track_utter['semantic_tagged']
					eval_semantics(ref_tagged, pred_tagged, stats[subtask])
				elif subtask == 'utt_transcriptions':
					ref = log_utter['transcript']
					pred = track_utter['generated_sentence']
					eval_utt(ref, pred, stats[subtask])

	csvfile = open(args.scorefile,'w')
	print >> csvfile,("task, subtask, schedule, stat, N, result")

	for subtask in stats:
		for schedule in stats[subtask]:
			for measure, N, result in stats[subtask][schedule].results():
				print >>csvfile,("%s, %s, %s, %s, %i, %s"%(args.pilottask, subtask, schedule, measure, N, result))
	csvfile.close()			


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
		stat_acts['act'].add(pred_act_tag_list, ref_act_tag_list, list_mode = True)
	if 'all' in stat_acts:
		stat_acts['all'].add(pred_act_attr_list, ref_act_attr_list, list_mode = True)



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
		print('num:%d ref: %s | pred: %s | bleu: %f | am: %f | fm: %f' %(self.num_sent, ref, pred, b, am, fm))
	def results(self,):
		return ("am_fm_avg", self.num_sent, self.am_fm/self.num_sent), ("bleu_avg", self.num_sent, self.bleu/self.num_sent)

if (__name__ == '__main__'):
	main(sys.argv)