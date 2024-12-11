import torch.nn as nn
import torch

class G_IoU(nn.Module):
    def __init__(self):
        super(G_IoU, self).__init__()
    def inter_area(self, label_coords, output_coords):
        l_x1, l_y1, l_x2, l_y2 = label_coords
        o_x1, o_y1, o_x2, o_y2 = output_coords
        x_dist = torch.min(l_x2, o_x2) - torch.max(l_x1, o_x1)
        y_dist = torch.min(l_y2, o_y2) - torch.max(l_y1, o_y1)
        if (x_dist <= 0 or y_dist <= 0):
            return x_dist * y_dist * 0
        else:
            return x_dist * y_dist

    def area_from_coords(self, coords):
        x1, y1, x2, y2 = coords
        return(torch.abs((x2-x1)*(y2-y1)))

    def IoU(self, o_coords, l_coords):
        inter = self.inter_area(l_coords, o_coords)
        total_area = self.area_from_coords(o_coords) + self.area_from_coords(l_coords) - inter
        return 1 - inter/total_area


    def forward(self, outputs, labels):

        iou_list = [self.IoU(o_coords, l_coords) for o_coords, l_coords in zip(outputs, labels)]
        return torch.mean(torch.stack(iou_list, 0))
        
def inter_area(label_coords, output_coords):
        l_x1, l_y1, l_x2, l_y2 = label_coords
        o_x1, o_y1, o_x2, o_y2 = output_coords
        x_dist = torch.min(l_x2, o_x2) - torch.max(l_x1, o_x1)
        y_dist = torch.min(l_y2, o_y2) - torch.max(l_y1, o_y1)
        if (x_dist <= 0 or y_dist <= 0):
            return x_dist * y_dist * 0
        else:
            return x_dist * y_dist

def area_from_coords(coords):
        x1, y1, x2, y2 = coords
        return(torch.abs((x2-x1)*(y2-y1)))

def IoU(out_coords, label_coords):
    running_iou = 0
    pairs = zip(out_coords, label_coords)
    for l_coords, o_coords in pairs:
        inter = inter_area(l_coords, o_coords)
        total_area = area_from_coords(o_coords) + area_from_coords(l_coords) - inter
        running_iou += 1-inter/total_area
    return running_iou/len(out_coords)
