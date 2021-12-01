import argparse
import os
from rknn.api import RKNN

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", '--onnx', type=str, default='yolov5s.onnx', help='weights path')  # from yolov5/models/
    parser.add_argument('--rknn', type=str, default='', help='保存路径')
    parser.add_argument("-p", '--precompile', action="store_true", help='是否是预编译模型')
    parser.add_argument("-o", '--original', action="store_true", help='是否是yolov5原生的模型')
    parser.add_argument("-bs", '--batch-size', type=int, default=1, help='batch size')
    opt = parser.parse_args()
    ONNX_MODEL = opt.onnx
    if opt.rknn:
        RKNN_MODEL = opt.rknn
    else:
        RKNN_MODEL = "%s.rknn" % os.path.splitext(ONNX_MODEL)[0]
    rknn = RKNN()
    print('--> config model')
    if opt.original:
        rknn.config(mean_values=[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                    std_values=[[255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0]],
                    batch_size=opt.batch_size)  # reorder_channel='0 1 2',
    else:
        rknn.config(channel_mean_value='0 0 0 255', reorder_channel='2 1 0', batch_size=opt.batch_size)
    # Load tensorflow model
    print('--> Loading model')
    ret = rknn.load_onnx(model=ONNX_MODEL)
    assert ret == 0, "Load onnx failed!"
    # Build model
    print('--> Building model')
    if opt.precompile:
        ret = rknn.build(do_quantization=True, dataset='./dataset.txt')  # pre_compile=True
    else:
        ret = rknn.build(do_quantization=True, dataset='./dataset.txt')
    assert ret == 0, "Build onnx failed!"
    # Export rknn model
    print('--> Export RKNN model')
    ret = rknn.export_rknn(RKNN_MODEL)
    assert ret == 0, "Export %s.rknn failed!" % opt.rknn
    print('done')
