# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm
import pickle
import json
import os
import sys

from pathlib import Path

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemo

# constants
WINDOW_NAME = "COCO detections"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # To use demo for Panoptic-DeepLab, please uncomment the following two lines.
    # from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa
    # add_panoptic_deeplab_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations."
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)

    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"

        assert len(args.input)==1, "You must provide one input directory"
        assert os.path.isdir(args.input[0]), "Input must point to a directory"
        
        logger.info("Input is directory, recursively getting all directories in {}".format(args.input[0]))
        subdirs = [x[0] for x in os.walk(args.input[0])]
        logger.info("Found {}".format(subdirs)) 

        for subdir in tqdm.tqdm(subdirs, position=0):
            # check for .result_directory which indicates that this is not an input directory
            # but an output directory from a previous run
            if os.path.exists(subdir+"/.result_directory"):
                logger.info("{} is a result directory, will be skipped".format(subdir))
                continue

            png_files = glob.glob(subdir+"/*.png")
            logger.info("Found {} files in {}".format(len(png_files), subdir))
            output_dir = subdir+"/"+args.output
            if len(png_files)>0:
                if os.path.exists(output_dir):
                    if os.path.exists(output_dir+"/.result_directory"):
                        logger.info("{} was already processed, will be skipped".format(subdir))
                        continue
                    else:
                        logger.info("{} already contains the output directory, will be processed anyway and files may be overwritten".format(subdir))
                else:
                    os.mkdir(output_dir)
            for png_file in tqdm.tqdm(png_files, position=1):
                img = read_image(png_file, format="BGR")
                start_time = time.time()
                predictions, visualized_output = demo.run_on_image(img)
                # logger.info(
                #     "{}: {} in {:.2f}s".format(
                #         png_file,
                #         "detected {} instances".format(len(predictions["instances"]))
                #         if "instances" in predictions
                #         else "finished",
                #         time.time() - start_time,
                #     )
                # )
                out_filename = os.path.join(output_dir, os.path.basename(png_file))
                out_filename_res = os.path.join(output_dir, os.path.basename(png_file) + ".res")
                visualized_output.save(out_filename)
                with open(out_filename_res, 'w') as file:
                    file.writelines("#  Number of instances\n")
                    file.writelines("{}\n".format(len(predictions["instances"])))
                    file.writelines("#  For each instanc: box | score | class\n")
                    for i in range(len(predictions["instances"])):
                        file.writelines("{}, {}, {}, {} | {} | {}\n".format(
                            predictions["instances"][i].pred_boxes.tensor[0][0], predictions["instances"][i].pred_boxes.tensor[0][1], predictions["instances"][i].pred_boxes.tensor[0][2], predictions["instances"][i].pred_boxes.tensor[0][3], predictions["instances"][i].scores[0], predictions["instances"][i].pred_classes[0]))
            # create .result_directory file
            Path(output_dir+"/.result_directory").touch()