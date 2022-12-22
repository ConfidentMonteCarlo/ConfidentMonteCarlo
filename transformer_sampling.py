from architecture import TransformerModel, PositionalEncoding, transformer_inference
import argparse
import os 

parser = argparse.ArgumentParser(description='Transformer sampling')
parser.add_argument('--modelpath', type=str, default='./mymodel', help='path of the model')
parser.add_argument('--n_samples', type=int, default=206848, help='number of samples')
parser.add_argument('--n_repetitions', type=int, default=1, help='number of samples')
parser.add_argument('--savepath', type=str, default='./repetitivesamples/', help='path of the sample file')
args = parser.parse_args()

model_eval = transformer_inference(args.modelpath)

for i in range(args.n_repetitions):
    tup = model_eval.generate_parallel(args.n_samples)
    sorted_by_logfreq = sorted(tup, key=lambda x: x[1])
    if args.n_repetitions > 1:
        savename = args.savepath + str(i) + '_transformer_samples.txt'
    else:
        savename = args.savepath
    with open(savename, 'w') as file:
        for i in range(len(sorted_by_logfreq)):
            file.write(sorted_by_logfreq[i][0] + '\t' + str(sorted_by_logfreq[i][1]) + '\n')




