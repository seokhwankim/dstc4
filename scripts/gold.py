import argparse, dataset_walker, json, time, copy
from collections import defaultdict

target_topics = ['ATTRACTION', 'ACCOMMODATION', 'FOOD', 'SHOPPING', 'TRANSPORTATION']

class Tracker(object):
    def __init__(self):
        self.reset()

    def addUtter(self, utter, ref):
        hyp = {'utter_index': utter['utter_index']}        
        if utter['segment_info']['topic'] in target_topics:
            hyp['frame_label'] = ref['frame_label']
        return hyp

    def reset(self):
        pass

def main() :
    parser = argparse.ArgumentParser(description='Simple hand-crafted dialog state tracker baseline.')
    parser.add_argument('--dataset', dest='dataset', action='store', metavar='DATASET', required=True,
                        help='The dataset to analyze')
    parser.add_argument('--dataroot',dest='dataroot',action='store',required=True,metavar='PATH',
                        help='Will look for corpus in <destroot>/<dataset>/...')
    parser.add_argument('--trackfile',dest='trackfile',action='store',required=True,metavar='JSON_FILE',
                        help='File to write with tracker output')

    args = parser.parse_args()
    dataset = dataset_walker.dataset_walker(args.dataset, dataroot=args.dataroot, labels=True)
    track_file = open(args.trackfile, "wb")
    track = {"sessions":[]}
    track["dataset"]  = args.dataset
    start_time = time.time()

    tracker = Tracker()

    for call in dataset :
        this_session = {"session_id":call.log["session_id"], "utterances":[]}
        tracker.reset()
        for in_obj, ref_obj in call :
            # this_session['utterances'].append(ref_obj['frame_label'])
            tracker_result = tracker.addUtter(in_obj, ref_obj)
            if tracker_result is not None:
                this_session["utterances"].append(tracker_result)
        
        track["sessions"].append(this_session)
    end_time = time.time()
    elapsed_time = end_time - start_time
    track["wall_time"] = elapsed_time
   
    json.dump(track, track_file,indent=4)
    
if __name__ == '__main__':
    main()
