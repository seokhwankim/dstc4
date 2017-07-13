import os, json, re

patch_list = []


class dataset_walker(object):
    def __init__(self,dataset,labels=False,dataroot=None):
        if "[" in dataset :
            self.datasets = json.loads(dataset)
        elif type(dataset) == type([]) :
            self.datasets= dataset
        else:
            self.datasets = [dataset]
            self.dataset = dataset
        self.install_root = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
        self.dataset_session_lists = [os.path.join(self.install_root,'config',dataset + '.flist') for dataset in self.datasets]
           
        self.labels = labels
        if (dataroot == None):
            install_parent = os.path.dirname(self.install_root)
            self.dataroot = os.path.join(install_parent,'data')
        else:
            self.dataroot = os.path.join(os.path.abspath(dataroot))

        # load dataset (list of calls)
        self.session_list = []
        for dataset_session_list in self.dataset_session_lists :
            f = open(dataset_session_list)
            for line in f:
                line = line.strip()
                if (line in self.session_list):
                    raise RuntimeError,'Call appears twice: %s' % (line)
                self.session_list.append(line)
            f.close()   
        
    def __iter__(self):
        for session_id in self.session_list:
            session_id_list = session_id.split('/')
            session_dirname = os.path.join(self.dataroot,*session_id_list)
            applog_filename = os.path.join(session_dirname,'log.json')
            if (self.labels):
                labels_filename = os.path.join(session_dirname,'label.json')
                if (not os.path.exists(labels_filename)):
                    raise RuntimeError,'Cant score : cant open labels file %s' % (labels_filename)
            else:
                labels_filename = None
            call = Call(applog_filename,labels_filename)
            call.dirname = session_dirname
            yield call
    def __len__(self, ):
        return len(self.session_list)
    

class Call(object):
    def __init__(self,applog_filename,labels_filename):
        self.applog_filename = applog_filename
        self.labels_filename = labels_filename
        f = open(applog_filename)
        self.log = json.load(f)
        f.close()
        if (labels_filename != None):
            f = open(labels_filename)
            self.labels = json.load(f)
            f.close()
        else:
            self.labels = None

    def __iter__(self):
        if (self.labels_filename != None):
            for (log,labels) in zip(self.log['utterances'],self.labels['utterances']):
                if 'speech_act' in labels:
                    for i in range(len(labels['speech_act'])):
                        act = labels['speech_act'][i]['act'].strip().upper()
                        if act == '':
                            act = 'NONE'
                        labels['speech_act'][i]['act'] = act
                        for j in range(len(labels['speech_act'][i]['attributes'])):
                            attr = labels['speech_act'][i]['attributes'][j].strip()
                            if attr is None or attr == '':
                                attr = 'NONE'
                            labels['speech_act'][i]['attributes'][j] = attr

                yield (log,labels)
        else: 
            for log in self.log['utterances']:
                yield (log,None)
                
    def __len__(self, ):
        return len(self.log['utterances'])
    
        
if __name__ == '__main__':
    dataset = dataset_walker("dstc4_train", dataroot="data", labels=True)
    for call in dataset :
        for utter, label in call:
            print utter
