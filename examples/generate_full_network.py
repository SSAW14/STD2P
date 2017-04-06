#! attach the parameters from trained model to baseline model

caffe_root = '../caffe-std2p/'

import sys
sys.path.insert(0,caffe_root + 'python')

import caffe
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m1', '--model_baseline_prototxt', default='./prototxt/fcn_16s_rgbd.prototxt')
    parser.add_argument('-m2', '--model_trained_prototxt', default='./prototxt/test_std2p.prototxt')
    parser.add_argument('-i1', '--input_baseline_model', default='./models/fcn-16s-rgbd-nyud2.caffemodel')
    parser.add_argument('-i2', '--input_trained_model', default='./models/train_iter_1000.caffemodel')
    parser.add_argument('-o', '--output_model', default='./models/full_std2p_nyud2.caffemodel')

    args = parser.parse_args()

    m1 = args.model_baseline_prototxt
    m2 = args.model_trained_prototxt
    input1 = args.input_baseline_model
    input2 = args.input_trained_model
    output = args.output_model

    net_input = caffe.Net(m2,input2,caffe.TEST)
    net_output = caffe.Net(m1,input1,caffe.TEST)

    print('baseline prototxt : ' + m1)
    print('trained prototxt : ' + m2)
    print('baseline model : ' + input1)
    print('trained model : ' + input2)
    print('saved model : ' + output)

    for k, v in net_input.params.items():
        for i in range(len(net_input.params[k])):
            net_output.params[k][i].data[...] = net_input.params[k][i].data

    net_output.save(output)

if __name__ == "__main__":
    main()

