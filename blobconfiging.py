import blobconverter
import sys, getopt

#---------------------------------------------------------
#%%%%%%% INTRO %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#---------------------------------------------------------
# pre structured call to blobconverter [1]
# with specific config for google's mediapine tflite configuration
# as defined by geax/pinto ([2],[3])
#
# [1] https://pypi.org/project/blobconverter/
# [2] https://github.com/geaxgx/depthai_blazepose
# [3] https://github.com/PINTO0309/tflite2tensorflow
#---------------------------------------------------------

#---------------------------------------------------------
#%%%%%%% VARIABLES OVERVIEW %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#---------------------------------------------------------
#
# number_shaves 
#       desired number of shaves for the ouput blob instance 
#       eg.{4,6,8}
#
# model_path 
#       path to the onnx model
#
# convertion_selector #
#       dummy variable to distinguish between parameters associated with detection and landmark models
#       with information from https://github.com/geaxgx/depthai_blazepose/blob/main/models/convert_models.sh
#       det - for detection
#       land - for landmarks
#---------------------------------------------------------

def mBlobConvertion(model_path, number_shaves, convertion_selector):

    if convertion_selector == 'det':

        print ('0 selected')

        blob_path = blobconverter.from_onnx(
            model=model_path,   
            optimizer_params=[
                "--data_type=FP16",
                "--mean_values=[127.5,127.5,127.5]",
                "--scale_values=[127.5,127.5,127.5]",
                "--reverse_input_channels"
            ],
            compile_params=[
                "-ip U8"
            ],
            output_dir="./",
            shaves=number_shaves,
        )

    elif convertion_selector == 'land':

        print ('1 selected')

        blob_path = blobconverter.from_onnx(
            model=model_path,   
            optimizer_params=[
                "--data_type=FP16",
                "--reverse_input_channels"
            ],
            compile_params=[
                "-ip fp16"
            ],
            output_dir="./",
            shaves=number_shaves,
        )

    elif convertion_selector == 'other':

        print ('2 selected')

        blob_path = blobconverter.from_onnx(
            model=model_path,   
            optimizer_params=[
                "--data_type=FP16",
            ],
            compile_params=[
                "-ip u8"
            ],
            output_dir="./",
            shaves=number_shaves,
        )
    elif convertion_selector == 'anotherother':

        print ('3 selected')

        blob_path = blobconverter.from_onnx(
            model=model_path,   
            optimizer_params=[
                "--data_type=FP16",
            ],
            compile_params=[
                "-ip u8 -op fp16"
            ],
            output_dir="./",
            shaves=number_shaves,
        )
    elif convertion_selector == 'anotheranotherother':

        print ('4 selected')

        blob_path = blobconverter.from_onnx(
            model=model_path,   
            optimizer_params=[
                "--data_type=FP16",
            ],
            compile_params=[
                "-ip fp32 -op fp16"
            ],
            output_dir="./",
            shaves=number_shaves,
        )
    elif convertion_selector == 'wally':

        print ('5 selected')

        blob_path = blobconverter.from_onnx(
            model=model_path,   
            optimizer_params=[
                "--data_type=FP16",
		"--input_shape=[1,16,2]",
		"--log_level=DEBUG",
            ],
            compile_params=[
                "-ip fp16 -op fp16"
            ],
            output_dir="./",
            shaves=number_shaves,
        )
    else:
        print ('not a valid option')

def main(argv):
    modelPath = ""
    numberShaves = '' 
    convertionSelector = ""

    try:
        opts, args = getopt.getopt(argv,"hi:n:c",["help", "imodel=","nshaves=","cselector="])
    except getopt.GetoptError:
        print ('blobconfiging.py -i <model_path> -n <number_shaves> -c <convertion_selector>')
        sys.exit(2)
    for opt, arg in opts:
        print (opt)
        print (arg)
        if opt == '-h':
            print ('blobconfiging.py -i <model_path> -n <number_shaves> -c <convertion_selector>')
            sys.exit()
        elif opt in ("-i", "--imodel"):
            modelPath = arg
        elif opt in ("-n", "--nshaves"):
            numberShaves = arg
        elif opt in ("-c", "--cselector"):
            convertionSelector = arg
       
    print ('Input model path is', modelPath)
    print ('Number shaves is', numberShaves)
    print ('Convertion mode selected is', convertionSelector)

    mBlobConvertion(modelPath, numberShaves, convertionSelector)

    print ('over and out')

if __name__ == "__main__":  
    main(sys.argv[1:])
