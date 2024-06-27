# code modified with permission from cffan/neural_seq_decoder

import sys

modelName = 'speechBaseline4_transformer'

args = {}
args['outputDir'] = './logs/speech_logs/' + modelName
args['datasetPath'] = './data/ptDecoder_ctc'
args['seqLen'] = 150
args['maxTimeSeriesLen'] = 1200
args['batchSize'] = 64
args['lrMax'] = 0.001
args['warmup'] = 1600
args['nBatch'] = 20000 #3000
args['seed'] = 0
args['nClasses'] = 40
args['nInputFeatures'] = 256
args['dropout'] = 0.25
args['whiteNoiseSD'] = 0.8
args['constantOffsetSD'] = 0.2
args['gaussianSmoothWidth'] = 2.0
args['strideLen'] = 4
args['kernelLen'] = 32
args['l2_decay'] = 1e-5

from src.neural_decoder.neural_decoder_trainer_transformer import trainModel

breakpoint()
trainModel(args)
