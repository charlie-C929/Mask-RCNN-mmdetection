#
import argparse
import torch


def parser_args():
    parser = argparse.ArgumentParser(description="modify the finetune model")
    parser.add_argument("checkpoint",help="checkpoint needs to be modified")

    args = parser.parse_args()
    return args


args = parser_args()

model = torch.load(args.checkpoint)
# # Remove the previous training parameters. 
# del model['iteration']
# del model['scheduler']
# del model['optimizer']
# Remove the output layers in COCO, these are the mismatched layers you saw.
#Second stage prediction
del model["state_dict"]["roi_head.bbox_head.fc_cls.weight"]
del model["state_dict"]["roi_head.bbox_head.fc_cls.bias"]
del model["state_dict"]["roi_head.bbox_head.fc_reg.weight"]
del model["state_dict"]["roi_head.bbox_head.fc_reg.bias"]
#mask prediction
del model["state_dict"]["roi_head.mask_head.conv_logits.weight"]
del model["state_dict"]["roi_head.mask_head.conv_logits.bias"]
# RPN

#save the model
torch.save(model, "modified_model.pth")
