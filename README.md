Python Implementation of Evaluation Metrics for Machine Translation

Run as ‘python <filename>.py > <alignment_file>

Options:
-n num_sentences		take fewer lines from input files
-i input_file		    to give files other than default for candidate and reference sentences

meteor: Implements Simple METEOR metric to compute which of the two candidate sentences is a better translation
classifier: Trains a classifier using Support Vector Machines and labelled candidates-reference sentence set.

output_meteor: Output from meteor. Has one label from among {-1,0,1} corresponding to each line of input file.
output_classifier: Output from classifier. Has same format as output_meteor.

output files can be gives as input to compare-with-human-evaluation to get accuracy of evaluation performed. 

