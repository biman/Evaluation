#!/usr/bin/env python
# coding=utf-8
import argparse # optparse is deprecated
from itertools import islice # slicing for iterators
from sklearn import svm
from nltk.stem.porter import PorterStemmer
import nltk,re
def word_matches(h, ref):
    sum1 =0.0 #sum(1 for w in h if w in ref)
    sum2=0.0
    sum3=0.0
    sum4=0.0
    for i in range(len(h)-3):
        for j in range(len(ref)-3):
            if(h[i] == ref[j] and  h[i+1] == ref[j+1] and  h[i+2] == ref[j+2] and  h[i+3] == ref[j+3]):
                sum4+=1
    for i in range(len(h)-2):
        for j in range(len(ref)-2):
            if(h[i] == ref[j] and  h[i+1] == ref[j+1] and  h[i+2] == ref[j+2]):
                sum3+=1
    for i in range(len(h)-1):
        for j in range(len(ref)-1):
            if(h[i] == ref[j] and  h[i+1] == ref[j+1]):
                sum2+=1
    for i in range(len(h)):
        for j in range(len(ref)):
            if(h[i] == ref[j]):
                sum1+=1
    return (sum1,sum2,sum3,sum4)
def tag_matches(h, ref):
    sum1 =0.0 #sum(1 for w in h if w in ref)
    sum2=0.0
    sum3=0.0
    sum4=0.0
    h_content = 0.0
    ref_content=0.0
    for i in range(len(h)-3):
        for j in range(len(ref)-3):
            if(h[i][1] == ref[j][1] and  h[i+1][1] == ref[j+1][1] and  h[i+2][1] == ref[j+2][1] and  h[i+3][1] == ref[j+3][1]):
                sum4+=1
    for i in range(len(h)-2):
        for j in range(len(ref)-2):
            if(h[i][1] == ref[j][1] and  h[i+1][1] == ref[j+1][1] and  h[i+2][1] == ref[j+2][1]):
                sum3+=1
    for i in range(len(h)-1):
        for j in range(len(ref)-1):
            if(h[i][1] == ref[j][1] and  h[i+1][1] == ref[j+1][1]):
                sum2+=1
    for i in range(len(h)):
        if('NN' in h[i][1] or 'VB' in h[i][1] or 'JJ' in h[i][1] or 'RB' in h[i][1] ):
            h_content+=1
        for j in range(len(ref)):
            if(h[i][1] == ref[j][1]):
                sum1+=1
    for i in range(len(ref)):
        if('NN' in ref[i][1] or 'VB' in ref[i][1] or 'JJ' in ref[i][1] or 'RB' in ref[i][1]):
            ref_content+=1
    return (sum1,sum2,sum3,sum4,h_content,ref_content)

def main():
    ps = PorterStemmer()
    lbl=[]
    fv = []
    op = []
    tagger = nltk.data.load(nltk.tag._POS_TAGGER)
    output_file = open("data/output",'w')
    label_file = open("data/dev.answers")
    labels = label_file.readlines()
    parser = argparse.ArgumentParser(description='Evaluate translation hypotheses.')
    parser.add_argument('-i', '--input', default='data/hyp1-hyp2-ref',
            help='input file (default data/hyp1-hyp2-ref)')  #hyp1-hyp2-ref
    parser.add_argument('-n', '--num_sentences', default=None, type=int,
            help='Number of hypothesis pairs to evaluate')
    # note that if x == [1, 2, 3], then x[:None] == x[:] == x (copy); no need for sys.maxint
    opts = parser.parse_args()
 
    # we create a generator and avoid loading all sentences into a list
    def sentences():
        with open(opts.input) as f:
            for pair in f:
                yield [sentence.strip().split() for sentence in pair.split(' ||| ')]
 
    # note: the -n option does not work in the original code
    ##Train and test
    i=0
    len_lbl = len(labels)
    first_flag=1
    clf = svm.SVC()
    count=1
    for h1, h2, ref in islice(sentences(), opts.num_sentences):
        if  i > len_lbl:
            break
        count+=1
        if count==5000:
            print "."
            count =1
        rset = set(ref)
        lh1 =float(len(h1))
        lh2=float(len(h2))
        lref = float(len(ref))
        (h1_1,h1_2,h1_3,h1_4) = (word_matches(h1, ref))#rset))
        (h2_1,h2_2,h2_3,h2_4) = (word_matches(h2, ref))#rset))
        '''for i in range(len(h1)):
            h1[i]= ''.join(character for character in h1[i] if ord(character)<128)
        for i in range(len(h2)):
            h2[i] = ''.join(character for character in h2[i] if ord(character)<128)
        for i in range(len(ref)):
            ref[i]= ''.join(character for character in ref[i] if ord(character)<128)'''
        for j in range(len(h1)):
            #if "x" in h1[i]:
            h1[j]= re.sub(r'[^\x20-\x7e]+', r'',h1[j])
        for j in range(len(h2)):
            #if "x" in h2[i]:
            h2[j] = re.sub(r'[^\x20-\x7e]+', r'',h2[j])
        for j in range(len(ref)):
            #if "x" in ref[i]:
            ref[j] = re.sub(r'[^\x20-\x7e]+', r'',ref[j])
        tags_all = tagger.tag_sents([h1,h2,ref])
        #tags_h2 = tagger.tag(h2)
        #tags_ref = tagger.tag(ref)
        if(count%1000==0):
            print "TAGGED a "+ str(count)
        (t1_1,t1_2,t1_3,t1_4,h1_c,r_f) = (tag_matches(tags_all[0], tags_all[2]))#rset))
        (t2_1,t2_2,t2_3,t2_4,h2_c,r_f) = (tag_matches(tags_all[1], tags_all[2]))#rset))
        if(i<len_lbl):
            lbl.append(int(labels[i].strip(' ').strip('\n').strip(' ')))
            i+=1
            fv.append([h1_1/lh1,h1_2/lh1,h1_3/lh1,h1_4/lh1,h1_1/lref,h1_2/lref,h1_3/lref,h1_4/lref,\
            (2*h1_1)/(lref*lh1),(2*h1_2)/(lref*lh1),(2*h1_3)/(lref*lh1),(2*h1_4)/(lref*lh1), (lh1-lref)/lref,(h1_c-r_f)/lref,\
            h2_1/lh2,h2_2/lh2,h2_3/lh2,h2_4/lh2,h2_1/lref,h2_2/lref,h2_3/lref,h2_4/lref,\
            (2*h2_1)/(lref*lh2),(2*h2_2)/(lref*lh2),(2*h2_3)/(lref*lh2),(2*h2_4)/(lref*lh2),(lh2-lref)/lref,(h2_c-r_f)/lref,\
            t1_1/lh1,t1_2/lh1,t1_3/lh1,t1_4/lh1,t1_1/lref,t1_2/lref,t1_3/lref,t1_4/lref,\
            (2*t1_1)/(lref*lh1),(2*t1_2)/(lref*lh1),(2*t1_3)/(lref*lh1),(2*t1_4)/(lref*lh1),\
            t2_1/lh2,t2_2/lh2,t2_3/lh2,t2_4/lh2,t2_1/lref,t2_2/lref,t2_3/lref,t2_4/lref,\
            (2*t2_1)/(lref*lh2),(2*t2_2)/(lref*lh2),(2*t2_3)/(lref*lh2),(2*t2_4)/(lref*lh2)])
        elif(i==len_lbl):
            clf.fit(fv, lbl)
            i+=1
            break
    i=0
    for h1, h2, ref in islice(sentences(), opts.num_sentences):
        count+=1
        if count==5000:
            print "."
            count =1
        if(i<len(fv)):
            y = ( clf.decision_function(fv[i]))
            #print y
            if first_flag==1:
                first_flag=0
            else:
                output_file.write("\n")
            if(y[0][0]<0 and y[0][2]>0):
              output_file.write("0")
            if(y[0][0]>0 and y[0][1]>0):
             output_file.write("-1")
            if(y[0][1]<0 and y[0][2]<0):
             output_file.write("1")
            i+=1;
        else:
            lh1 =float(len(h1))
            lh2=float(len(h2))
            lref = float(len(ref))
            (h1_1,h1_2,h1_3,h1_4) = (word_matches(h1, ref))#rset))
            (h2_1,h2_2,h2_3,h2_4) = (word_matches(h2, ref))#rset))
            '''for i in range(len(h1)):
                h1[i]= ''.join(character for character in h1[i] if ord(character)<128)
            for i in range(len(h2)):
                h2[i] = ''.join(character for character in h2[i] if ord(character)<128)
            for i in range(len(ref)):
                ref[i]= ''.join(character for character in ref[i] if ord(character)<128)'''
            for j in range(len(h1)):
            #if "x" in h1[i]:
                h1[j]= re.sub(r'[^\x20-\x7e]+', r'',h1[j])
            for j in range(len(h2)):
            #if "x" in h2[i]:
                h2[j] = re.sub(r'[^\x20-\x7e]+', r'',h2[j])
            for j in range(len(ref)):
                #if "x" in ref[i]:
                ref[j] = re.sub(r'[^\x20-\x7e]+', r'',ref[j])
            tags_all = tagger.tag_sents([h1,h2,ref])
            (t1_1,t1_2,t1_3,t1_4,h1_c,r_f) = (tag_matches(tags_all[0], tags_all[2]))#rset))
            (t2_1,t2_2,t2_3,t2_4,h2_c,r_f) = (tag_matches(tags_all[1], tags_all[2]))#rset))
            i+=1
            y = ( clf.decision_function([h1_1/lh1,h1_2/lh1,h1_3/lh1,h1_4/lh1,h1_1/lref,h1_2/lref,h1_3/lref,h1_4/lref,\
            (2*h1_1)/(lref*lh1),(2*h1_2)/(lref*lh1),(2*h1_3)/(lref*lh1),(2*h1_4)/(lref*lh1), (lh1-lref)/lref,(h1_c-r_f)/lref,\
            h2_1/lh2,h2_2/lh2,h2_3/lh2,h2_4/lh2,h2_1/lref,h2_2/lref,h2_3/lref,h2_4/lref,\
            (2*h2_1)/(lref*lh2),(2*h2_2)/(lref*lh2),(2*h2_3)/(lref*lh2),(2*h2_4)/(lref*lh2), (lh2-lref)/lref,(h2_c-r_f)/lref,\
            t1_1/lh1,t1_2/lh1,t1_3/lh1,t1_4/lh1,t1_1/lref,t1_2/lref,t1_3/lref,t1_4/lref,\
            (2*t1_1)/(lref*lh1),(2*t1_2)/(lref*lh1),(2*t1_3)/(lref*lh1),(2*t1_4)/(lref*lh1),\
            t2_1/lh2,t2_2/lh2,t2_3/lh2,t2_4/lh2,t2_1/lref,t2_2/lref,t2_3/lref,t2_4/lref,\
            (2*t2_1)/(lref*lh2),(2*t2_2)/(lref*lh2),(2*t2_3)/(lref*lh2),(2*t2_4)/(lref*lh2)]))
            #print y
            output_file.write("\n")
            if(y[0][0]<0 and y[0][2]>0):
              output_file.write("0")
            if(y[0][0]>0 and y[0][1]>0):
             output_file.write("-1")
            if(y[0][1]<0 and y[0][2]<0):
             output_file.write("1")
             #output_file.write("\n")
    #print op
    output_file.close()
    label_file.close()
# convention to allow import of this file as a module
if __name__ == '__main__':
    main()
